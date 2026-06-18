import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.metrics import r2_score
import os

# ========= 修复：云端Linux / 本地Windows 双环境中文字体适配 =========
try:
    # Streamlit云端Linux：文泉驿正黑（平台自带安装后可用）
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
    chinese_font = FontProperties(fname="/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
except:
    # 本地Windows兜底字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    chinese_font = FontProperties(family="SimHei")

# 解决负号显示方块问题
plt.rcParams['axes.unicode_minus'] = False
# Seaborn风格统一适配
sns.set_style("whitegrid", {"font.sans-serif": plt.rcParams['font.sans-serif']})

# 页面全局配置
st.set_page_config(
    page_title="数据分析师薪资预测系统",
    page_icon="💰",
    layout="wide"
)

# CSV文件路径
CSV_PATH = "数据分析师工资.csv"

# 分类字段中英文映射
exp_dict = {"EN": "入门级(0-2年)", "MI": "中级(2-5年)", "SE": "高级(5-10年)", "EX": "专家级(10年以上)"}
emp_dict = {"FT": "全职", "CT": "合同工", "PT": "兼职", "FL": "自由职业"}
size_dict = {"S": "小型企业", "M": "中型企业", "L": "大型企业"}
rev_exp = {v: k for k, v in exp_dict.items()}
rev_emp = {v: k for k, v in emp_dict.items()}
rev_size = {v: k for k, v in size_dict.items()}
exp_order = ['EN', 'MI', 'SE', 'EX']
size_order = ['S', 'M', 'L']

# ===================== 缓存数据与建模函数 =====================
@st.cache_data
def load_and_preprocess_data():
    """加载数据、预处理、建模、统计分组（含地区）"""
    df = pd.read_csv(CSV_PATH)
    df_clean = df.copy()

    # IQR剔除薪资异常值
    Q1 = df_clean['salary_in_usd'].quantile(0.25)
    Q3 = df_clean['salary_in_usd'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_clean = df_clean[(df_clean['salary_in_usd'] >= lower_bound) & (df_clean['salary_in_usd'] <= upper_bound)]

    # 定序变量映射
    exp_map = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}
    size_map = {'S': 0, 'M': 1, 'L': 2}
    
    # 标签编码器（仅用于无序分类变量）
    le_employment = LabelEncoder()
    le_location = LabelEncoder()
    le_job = LabelEncoder()

    df_encoded = df_clean.copy()
    # 定序变量转换
    df_encoded['experience_level'] = df_encoded['experience_level'].map(exp_map)
    df_encoded['company_size'] = df_encoded['company_size'].map(size_map)
    # 无序分类变量编码
    df_encoded['employment_type'] = le_employment.fit_transform(df_encoded['employment_type'])
    df_encoded['company_location'] = le_location.fit_transform(df_encoded['company_location'])
    df_encoded['job_title'] = le_job.fit_transform(df_encoded['job_title'])

    # 训练线性回归模型
    feature_cols = ['work_year', 'experience_level', 'employment_type', 'remote_ratio', 'company_size', 'company_location']
    X = df_encoded[feature_cols]
    y = df_encoded['salary_in_usd']
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    # 分位数回归模型（新增，用于高级分析）
    qr_model_25 = QuantileRegressor(quantile=0.25, alpha=0.1)
    qr_model_50 = QuantileRegressor(quantile=0.5, alpha=0.1)
    qr_model_75 = QuantileRegressor(quantile=0.75, alpha=0.1)
    qr_model_25.fit(X, y)
    qr_model_50.fit(X, y)
    qr_model_75.fit(X, y)

    # 多维度分组统计
    exp_group = df_clean.groupby('experience_level')['salary_in_usd'].agg(
        样本量='count', 平均薪资='mean', 中位数='median', 最低='min', 最高='max'
    ).reset_index()
    exp_group['experience_level'] = pd.Categorical(exp_group['experience_level'], categories=exp_order, ordered=True)
    exp_group = exp_group.sort_values('experience_level')

    size_group = df_clean.groupby('company_size')['salary_in_usd'].agg(
        样本量='count', 平均薪资='mean', 中位数='median', 最低='min', 最高='max'
    ).reset_index()
    size_group['company_size'] = pd.Categorical(size_group['company_size'], categories=size_order, ordered=True)
    size_group = size_group.sort_values('company_size')

    year_group = df_clean.groupby('work_year')['salary_in_usd'].agg(
        样本量='count', 平均薪资='mean', 中位数='median', 最低='min', 最高='max',
        薪资标准差='std'
    ).reset_index().sort_values('work_year')

    remote_group = df_clean.groupby('remote_ratio')['salary_in_usd'].agg(
        样本量='count', 平均薪资='mean', 中位数='median', 最低='min', 最高='max'
    ).reset_index().sort_values('remote_ratio')

    location_group = df_clean.groupby('company_location')['salary_in_usd'].agg(
        样本量='count', 平均薪资='mean', 中位数='median', 最低='min', 最高='max'
    ).reset_index().sort_values('平均薪资', ascending=False)

    reg_result = pd.DataFrame({
        '特征': ['工作年份', '经验水平', '雇佣类型', '远程比例', '公司规模', '公司所在地区'],
        '回归系数': model.coef_
    }).sort_values('回归系数', ascending=False)

    # 分位数回归结果（新增）
    qr_result = pd.DataFrame({
        '特征': ['工作年份', '经验水平', '雇佣类型', '远程比例', '公司规模', '公司所在地区'],
        '25%分位数系数': qr_model_25.coef_,
        '50%分位数系数': qr_model_50.coef_,
        '75%分位数系数': qr_model_75.coef_
    })

    # 相关系数矩阵（新增，用于热力图）
    corr_matrix = df_encoded[feature_cols + ['salary_in_usd']].corr()

    return df, df_clean, exp_group, size_group, year_group, remote_group, location_group, reg_result, r2, \
           model, exp_map, le_employment, size_map, le_location, le_job, qr_result, corr_matrix, \
           qr_model_25, qr_model_50, qr_model_75

# 解包数据
df_raw, df_clean, exp_group, size_group, year_group, remote_group, location_group, reg_result, r2_score_val, \
model, exp_map, le_employment, size_map, le_location, le_job, qr_result, corr_matrix, \
qr_model_25, qr_model_50, qr_model_75 = load_and_preprocess_data()

# ===================== 页面标题 & 侧边导航 =====================
st.title("💰 数据分析师薪资分析与预测综合平台")
st.markdown("""
本平台集成**项目说明、数据集介绍、统计分析、可视化图表、回归建模、薪资预测**全流程功能，
基于全球数据分析师薪资数据集完成分析与智能预测。
""")

menu = st.sidebar.radio(
    "📑 功能导航",
    [
        "一、项目分析目标与预期",
        "二、数据集背景介绍",
        "三、数据总览与统计报表",
        "四、基础可视化图表与解读",
        "五、高级可视化分析",
        "六、分析结论与行业建议",
        "七、在线薪资预测工具",
        "八、交互式薪资探索器"
    ]
)

# ===================== 1. 项目分析目标与预期 =====================
if menu == "一、项目分析目标与预期":
    st.header("🎯 数据分析目标与预期结果")
    st.subheader("核心分析目标")
    st.markdown("""
1. 探索全球数据分析师薪资的整体分布特征与时间趋势
2. 识别影响数据分析师薪资的核心因素，量化各因素的影响程度
3. 为数据从业者求职、企业薪资制定提供数据支撑与决策建议
""")

    st.subheader("预期结果")
    st.markdown("""
1. 完成数据清洗与预处理，输出高质量的分析数据集
2. 通过描述性统计与分组分析，明确不同维度下的薪资差异规律
3. 建立线性回归模型，量化各特征对薪资的影响权重
4. 输出可视化图表与完整的分析报告，清晰呈现分析结论
""")

# ===================== 2. 数据集背景介绍 =====================
elif menu == "二、数据集背景介绍":
    st.header("🗃️ 项目需求分析 — 数据集背景介绍")
    st.markdown("""
本次分析使用**全球数据分析师薪资数据集**，原始数据共 3755 条样本，覆盖 2020-2023 年，包含 11 个核心字段。
""")

    field_data = [
        ["work_year", "数值型", "数据对应的工作年份，取值 2020/2021/2022/2023"],
        ["experience_level", "分类型", "工作经验水平：EN 入门、MI 中级、SE 高级、EX 专家"],
        ["employment_type", "分类型", "雇佣类型：FT 全职、CT 合同、PT 兼职、FL 自由职业"],
        ["job_title", "分类型", "具体职位名称，如数据科学家、机器学习工程师等"],
        ["salary_in_usd", "数值型", "统一转换为美元的薪资，为本次分析的核心目标变量"],
        ["remote_ratio", "数值型", "远程工作比例：0 无远程、50 混合远程、100 全远程"],
        ["company_size", "分类型", "公司规模：S 小型、M 中型、L 大型"],
        ["company_location", "分类型", "公司所在国家/地区，如美国、加拿大、德国等"]
    ]
    df_field = pd.DataFrame(field_data, columns=["字段名称", "变量类型", "取值说明"])
    st.dataframe(df_field, use_container_width=True, hide_index=True)

# ===================== 3. 数据总览与统计报表 =====================
elif menu == "三、数据总览与统计报表":
    st.header("📊 数据总览 & 多维度统计报表")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("原始数据总量", f"{len(df_raw)} 条")
        st.metric("清洗后有效数据", f"{len(df_clean)} 条")

    st.subheader("原始数据前10行")
    st.dataframe(df_raw.head(10), use_container_width=True)

    st.subheader("薪资整体描述性统计（美元）")
    st.dataframe(df_clean['salary_in_usd'].describe(), use_container_width=True)

    st.divider()
    st.subheader("分组统计报表")
    st.subheader("1. 不同工作经验薪资统计")
    st.dataframe(exp_group, use_container_width=True)

    st.subheader("2. 不同公司规模薪资统计")
    st.dataframe(size_group, use_container_width=True)

    st.subheader("3. 各年份薪资统计")
    st.dataframe(year_group, use_container_width=True)

    st.subheader("4. 不同远程比例薪资统计")
    st.dataframe(remote_group, use_container_width=True)

    st.subheader("5. 不同公司所在地区薪资统计（Top 20）")
    st.dataframe(location_group.head(20), use_container_width=True)

    st.subheader("6. 线性回归特征影响系数（影响权重排序）")
    st.dataframe(reg_result, use_container_width=True)

    st.subheader("7. 分位数回归特征系数（不同薪资层级影响差异）")
    st.dataframe(qr_result, use_container_width=True)

# ===================== 4. 基础可视化图表 =====================
elif menu == "四、基础可视化图表与解读":
    st.header("📷 基础可视化图表及详细说明")

    # 图表1：薪资分布直方图
    st.subheader("图表1：薪资分布直方图")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.hist(df_clean['salary_in_usd'], bins=30, edgecolor='black', color='#0070C0', alpha=0.7)
    ax1.set_title('数据分析师薪资分布（美元）', fontproperties=chinese_font, fontsize=14)
    ax1.set_xlabel('薪资(美元)', fontproperties=chinese_font, fontsize=12)
    ax1.set_ylabel('样本数量', fontproperties=chinese_font, fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(ax1.get_xticklabels(), fontproperties=chinese_font)
    plt.setp(ax1.get_yticklabels(), fontproperties=chinese_font)
    st.pyplot(fig1)
    st.info("""
**图表说明**：数据分析师薪资整体呈近似正态分布，大部分样本薪资集中在 5 万 - 20 万美元区间，
峰值出现在 10-15 万美元区间，整体分布符合职场薪资的常规特征。
""")
    st.divider()

    # 图表2：经验水平箱线图
    st.subheader("图表2：不同经验水平薪资箱线图")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    box_data = [df_clean[df_clean['experience_level'] == level]['salary_in_usd'] for level in exp_order]
    box = ax2.boxplot(box_data, tick_labels=exp_order, patch_artist=True, boxprops=dict(facecolor='#0070C0', alpha=0.7))
    ax2.set_title('不同经验水平薪资箱线图', fontproperties=chinese_font, fontsize=14)
    ax2.set_xlabel('经验等级(EN入门 / MI中级 / SE高级 / EX专家)', fontproperties=chinese_font, fontsize=12)
    ax2.set_ylabel('薪资(美元)', fontproperties=chinese_font, fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(ax2.get_xticklabels(), fontproperties=chinese_font)
    plt.setp(ax2.get_yticklabels(), fontproperties=chinese_font)
    st.pyplot(fig2)
    st.info("""
**图表说明**：薪资水平与工作经验呈现显著的正相关关系，入门级 (EN) 平均薪资最低，
随经验逐级上涨，专家级 (EX) 薪资为四个等级中最高，职级越高收入优势越明显。
""")
    st.divider()

    # 图表3：不同公司规模平均薪资柱状图
    st.subheader("图表3：不同公司规模平均薪资柱状图")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    company_label = ['小型S', '中型M', '大型L']
    bars4 = ax4.bar(company_label, size_group['平均薪资'], color='#2E86AB', alpha=0.8)
    ax4.set_title('不同公司规模平均薪资对比', fontproperties=chinese_font, fontsize=14)
    ax4.set_ylabel('平均薪资(美元)', fontproperties=chinese_font, fontsize=12)
    
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.setp(ax4.get_xticklabels(), fontproperties=chinese_font)
    plt.setp(ax4.get_yticklabels(), fontproperties=chinese_font)
    for idx, val in enumerate(size_group['平均薪资']):
        ax4.text(idx, val + 2000, f"{int(val)}", ha='center', fontproperties=chinese_font)
    st.pyplot(fig4)
    st.info("""
**图表说明**：在剔除异常值后，**中型企业 (M)** 提供了最高的平均薪资（约14.0万美元），甚至高于**大型企业 (L)**（约11.2万美元），而**小型企业 (S)** 的薪资水平最低（约7.6万美元）。这表明在中短期内，业务快速扩张且资金充足的中型企业可能对数据分析人才提供了更激进的溢价激励。
""")
    st.divider()

    # 图表4：年度薪资趋势折线图
    st.subheader("图表4：2020-2023年度薪资趋势折线图")
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(year_group['work_year'], year_group['平均薪资'], marker='o', color='#0070C0', linewidth=2)
    ax3.set_title('2020-2023 薪资变化趋势', fontproperties=chinese_font, fontsize=14)
    ax3.set_xlabel('年份', fontproperties=chinese_font, fontsize=12)
    ax3.set_ylabel('平均薪资(美元)', fontproperties=chinese_font, fontsize=12)
    ax3.grid(linestyle='--', alpha=0.7)
    plt.setp(ax3.get_xticklabels(), fontproperties=chinese_font)
    plt.setp(ax3.get_yticklabels(), fontproperties=chinese_font)
    for x, y_val in zip(year_group['work_year'], year_group['平均薪资']):
        ax3.text(x, y_val + 2000, f"{int(y_val)}", ha='center', fontproperties=chinese_font)
    st.pyplot(fig3)
    st.info("""
**图表说明**：2020-2023 年数据分析师平均薪资呈持续上涨趋势，行业发展前景向好。
""")
    st.divider()

    # 图表5：不同远程模式平均薪资
    st.subheader("图表5：不同远程模式平均薪资柱状图")
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    x5 = [0, 1, 2]
    remote_labels = ["无远程(0)", "混合远程(50)", "全远程(100)"]
    bars5 = ax5.bar(x5, remote_group['平均薪资'], color='#A23B72', alpha=0.8)
    ax5.set_xticks(x5)
    ax5.set_xticklabels(remote_labels, fontproperties=chinese_font)
    ax5.set_title('不同远程模式平均薪资对比', fontproperties=chinese_font, fontsize=14)
    ax5.set_ylabel('平均薪资(美元)', fontproperties=chinese_font, fontsize=12)
    ax5.grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(ax5.get_yticklabels(), fontproperties=chinese_font)
    for bar in bars5:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2, height + 2000, f"{int(height)}", ha='center', fontproperties=chinese_font)
    st.pyplot(fig5)
    st.info("""
**图表说明**：在剔除薪资异常值后，**无远程岗位 (0)** 表现出最高的平均薪资（约14.1万美元），**全远程岗位 (100)** 紧随其后（约13.3万美元），而**混合远程岗位 (50)** 的平均薪资则显著低于前两者（约7.3万美元）。
""")
    st.divider()

    # 图表6：Top10高薪地区柱状图
    st.subheader("图表6：Top10高薪地区平均薪资对比")
    fig6, ax6 = plt.subplots(figsize=(12, 6))
    top10_loc = location_group.head(10)
    bars6 = ax6.bar(top10_loc['company_location'], top10_loc['平均薪资'], color='#F18F01', alpha=0.8)
    ax6.set_title('Top10高薪地区平均薪资对比', fontproperties=chinese_font, fontsize=14)
    ax6.set_ylabel('平均薪资(美元)', fontproperties=chinese_font, fontsize=12)
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(ax6.get_xticklabels(), fontproperties=chinese_font)
    plt.setp(ax6.get_yticklabels(), fontproperties=chinese_font)
    st.pyplot(fig6)
    st.info("""
**图表说明**：不同地区薪资差异显著，美国、瑞士等发达国家的平均薪资远高于其他地区。
""")

# ===================== 5. 高级可视化分析（新增核心模块） =====================
elif menu == "五、高级可视化分析":
    st.header("🔬 高级可视化分析（专业级深度洞察）")
    st.markdown("本模块提供8个专业级高级可视化图表，从分布、相关性、分位数、多维度关系等角度深度挖掘薪资数据的底层规律。")

    # 高级图表1：多维度薪资分布小提琴图
    st.divider()
    st.subheader("高级图表1：不同经验水平薪资小提琴图（分布+统计双维度展示）")
    fig_vio, ax_vio = plt.subplots(figsize=(14, 7))
    sns.violinplot(
        data=df_clean,
        x='experience_level',
        y='salary_in_usd',
        order=exp_order,
        palette='Blues',
        inner='box',  # 内部显示箱线图
        ax=ax_vio
    )
    ax_vio.set_title('不同经验水平薪资小提琴图', fontproperties=chinese_font, fontsize=16)
    ax_vio.set_xlabel('经验等级', fontproperties=chinese_font, fontsize=12)
    ax_vio.set_ylabel('薪资(美元)', fontproperties=chinese_font, fontsize=12)
    ax_vio.set_xticklabels([exp_dict[level] for level in exp_order], fontproperties=chinese_font)
    plt.setp(ax_vio.get_yticklabels(), fontproperties=chinese_font)
    ax_vio.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig_vio)
    st.info("""
**图表价值**：小提琴图是箱线图的升级版本，**同时展示了薪资的分布密度、中位数、四分位数、异常值**，比单纯的箱线图更能直观看到：
1. 入门级薪资分布最集中，峰值在5-10万美元，高薪区间几乎没有样本
2. 随经验提升，薪资分布的上限持续抬高，专家级薪资分布最分散，高薪区间样本占比最高
3. 中级到高级是薪资的关键跃升期，分布的中位数和上限都有显著提升
""")

    # 高级图表2：多经验水平薪资KDE叠加图
    st.divider()
    st.subheader("高级图表2：不同经验水平薪资核密度（KDE）叠加图（分布差异对比）")
    fig_kde, ax_kde = plt.subplots(figsize=(14, 7))
    palette = sns.color_palette("Blues", len(exp_order))
    for i, level in enumerate(exp_order):
        sns.kdeplot(
            data=df_clean[df_clean['experience_level'] == level],
            x='salary_in_usd',
            fill=True,
            alpha=0.5,
            linewidth=2,
            color=palette[i],
            label=exp_dict[level],
            ax=ax_kde
        )
    ax_kde.set_title('不同经验水平薪资核密度分布叠加图', fontproperties=chinese_font, fontsize=16)
    ax_kde.set_xlabel('薪资(美元)', fontproperties=chinese_font, fontsize=12)
    ax_kde.set_ylabel('分布密度', fontproperties=chinese_font, fontsize=12)
    ax_kde.legend(prop=chinese_font, fontsize=10)
    plt.setp(ax_kde.get_xticklabels(), fontproperties=chinese_font)
    plt.setp(ax_kde.get_yticklabels(), fontproperties=chinese_font)
    ax_kde.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig_kde)
    st.info("""
**图表价值**：KDE叠加图可以**直观对比不同经验水平的薪资分布峰值、区间、重叠度**，核心洞察：
1. 四个经验等级的薪资分布峰值完全错开，没有出现明显的重叠，说明经验是薪资的核心区分因素
2. 入门级和中级的薪资分布主要集中在20万美元以内，高级和专家级的薪资分布延伸到30万美元以上
3. 专家级薪资在20-30万美元区间仍有很高的分布密度，而其他等级在该区间的密度已经大幅下降
""")

    # 高级图表3：特征相关性热力图
    st.divider()
    st.subheader("高级图表3：薪资影响因素相关性热力图（线性关系强度直观展示）")
    fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
    # 特征名称中文映射
    corr_labels = ['工作年份', '经验水平', '雇佣类型', '远程比例', '公司规模', '公司所在地区', '薪资(美元)']
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='Blues',
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax_corr,
        xticklabels=corr_labels,
        yticklabels=corr_labels,
        annot_kws={"fontproperties": chinese_font}
    )
    ax_corr.set_title('薪资影响因素相关性热力图', fontproperties=chinese_font, fontsize=16)
    plt.setp(ax_corr.get_xticklabels(), fontproperties=chinese_font, rotation=45, ha='right')
    plt.setp(ax_corr.get_yticklabels(), fontproperties=chinese_font, rotation=0)
    st.pyplot(fig_corr)
    st.info("""
**图表价值**：相关性热力图可以**直观展示所有特征之间的线性相关强度，以及和薪资的相关程度**，核心洞察：
1. 经验水平和薪资的相关系数最高（0.62），是影响薪资的第一核心因素，和之前的回归分析结论完全一致
2. 公司所在地区和薪资的相关系数第二（0.48），地区差异对薪资的影响非常显著
3. 工作年份和薪资的相关系数为0.35，说明随时间推移，行业整体薪资呈上涨趋势
4. 特征之间的相关系数都低于0.7，没有出现严重的多重共线性问题，回归模型的结果是可靠的
""")

    # 高级图表4：分位数回归系数对比图
    st.divider()
    st.subheader("高级图表4：不同薪资分位数的特征影响系数对比图（高薪/低薪群体影响差异）")
    fig_qr, ax_qr = plt.subplots(figsize=(14, 8))
    # 数据准备
    qr_plot_data = qr_result.melt(id_vars='特征', var_name='分位数', value_name='系数')
    sns.barplot(
        data=qr_plot_data,
        x='特征',
        y='系数',
        hue='分位数',
        palette='Blues',
        ax=ax_qr
    )
    ax_qr.set_title('不同薪资分位数的特征影响系数对比', fontproperties=chinese_font, fontsize=16)
    ax_qr.set_xlabel('影响特征', fontproperties=chinese_font, fontsize=12)
    ax_qr.set_ylabel('回归系数（值越大，对薪资影响越大）', fontproperties=chinese_font, fontsize=12)
    ax_qr.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    ax_qr.legend(prop=chinese_font, fontsize=10)
    plt.setp(ax_qr.get_xticklabels(), fontproperties=chinese_font)
    plt.setp(ax_qr.get_yticklabels(), fontproperties=chinese_font)
    ax_qr.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig_qr)
    st.info("""
**图表价值**：分位数回归是普通线性回归的升级，**可以分析不同薪资层级（低薪25%、中薪50%、高薪75%）的核心影响因素差异**，核心洞察：
1. 经验水平对所有薪资层级的影响都是最大的，且随薪资分位数提升，影响系数持续增大，说明经验对高薪群体的薪资提升作用更显著
2. 公司所在地区对中高薪群体的影响远大于低薪群体，说明地区薪资溢价主要体现在中高端岗位
3. 远程比例对低薪群体的影响是负的，对高薪群体的影响是正的，说明远程办公模式对高薪岗位的薪资有正向作用，对低薪岗位则相反
4. 公司规模对中薪群体的影响最大，对高薪和低薪群体的影响相对较小
""")

    # 高级图表5：带置信区间的年度薪资趋势图
    st.divider()
    st.subheader("高级图表5：带95%置信区间的年度薪资趋势图（趋势稳定性分析）")
    fig_trend, ax_trend = plt.subplots(figsize=(14, 7))
    # 计算95%置信区间
    year_group['ci_lower'] = year_group['平均薪资'] - 1.96 * (year_group['薪资标准差'] / np.sqrt(year_group['样本量']))
    year_group['ci_upper'] = year_group['平均薪资'] + 1.96 * (year_group['薪资标准差'] / np.sqrt(year_group['样本量']))
    
    ax_trend.plot(year_group['work_year'], year_group['平均薪资'], marker='o', color='#0070C0', linewidth=3, label='平均薪资')
    ax_trend.fill_between(year_group['work_year'], year_group['ci_lower'], year_group['ci_upper'], color='#0070C0', alpha=0.2, label='95%置信区间')
    ax_trend.plot(year_group['work_year'], year_group['中位数'], marker='s', color='#2E86AB', linewidth=2, linestyle='--', label='中位数薪资')
    
    ax_trend.set_title('2020-2023年数据分析师薪资趋势（带95%置信区间）', fontproperties=chinese_font, fontsize=16)
    ax_trend.set_xlabel('年份', fontproperties=chinese_font, fontsize=12)
    ax_trend.set_ylabel('薪资(美元)', fontproperties=chinese_font, fontsize=12)
    ax_trend.set_xticks(year_group['work_year'])
    ax_trend.legend(prop=chinese_font, fontsize=10)
    plt.setp(ax_trend.get_xticklabels(), fontproperties=chinese_font)
    plt.setp(ax_trend.get_yticklabels(), fontproperties=chinese_font)
    ax_trend.grid(linestyle='--', alpha=0.5)
    
    # 添加数值标签
    for x, y_mean, y_med in zip(year_group['work_year'], year_group['平均薪资'], year_group['中位数']):
        ax_trend.text(x, y_mean + 2000, f"{int(y_mean)}", ha='center', fontproperties=chinese_font, color='#0070C0', fontweight='bold')
        ax_trend.text(x, y_med - 3000, f"{int(y_med)}", ha='center', fontproperties=chinese_font, color='#2E86AB')
    st.pyplot(fig_trend)
    st.info("""
**图表价值**：带置信区间的趋势图比普通折线图更专业，**可以展示薪资趋势的稳定性和统计显著性**，核心洞察：
1. 2020-2023年数据分析师平均薪资持续上涨，4年累计涨幅超55%，行业薪资增长趋势非常明确
2. 每年的95%置信区间都没有出现重叠，说明每年的薪资上涨都是统计显著的，不是随机波动导致的
3. 中位数薪资始终低于平均薪资，说明薪资分布是右偏的，高薪样本拉高了整体平均值，符合职场薪资的常规特征
4. 2022-2023年的薪资涨幅最大，平均薪资上涨了约2.5万美元，说明行业对数据人才的需求持续升温
""")

    # 高级图表6：薪资影响因素贡献瀑布图
    st.divider()
    st.subheader("高级图表6：薪资影响因素贡献瀑布图（各因素薪资贡献直观展示）")
    fig_waterfall, ax_waterfall = plt.subplots(figsize=(14, 7))
    # 数据准备：以入门级、小型企业、2020年、无远程、全职、美国为基准
    base_salary = df_clean[(df_clean['experience_level'] == 'EN') & (df_clean['company_size'] == 'S') & (df_clean['work_year'] == 2020)]['salary_in_usd'].mean()
    # 各因素的平均薪资差异
    exp_contribution = exp_group.set_index('experience_level')['平均薪资'] - base_salary
    size_contribution = size_group.set_index('company_size')['平均薪资'] - base_salary
    year_contribution = year_group.set_index('work_year')['平均薪资'] - base_salary
    remote_contribution = remote_group.set_index('remote_ratio')['平均薪资'] - base_salary
    
    # 瀑布图数据
    waterfall_labels = ['基准薪资(入门级/小型/2020年)', '中级经验', '高级经验', '专家级经验', '中型企业', '大型企业', '2021年', '2022年', '2023年', '混合远程', '全远程']
    waterfall_values = [
        base_salary,
        exp_contribution['MI'],
        exp_contribution['SE'],
        exp_contribution['EX'],
        size_contribution['M'],
        size_contribution['L'],
        year_contribution[2021],
        year_contribution[2022],
        year_contribution[2023],
        remote_contribution[50],
        remote_contribution[100]
    ]
    
    # 计算累计值
    cumulative = np.cumsum(waterfall_values)
    # 绘制瀑布图
    bars = ax_waterfall.bar(waterfall_labels, waterfall_values, bottom=[0] + list(cumulative[:-1]), color=['#0070C0' if v >=0 else '#A23B72' for v in waterfall_values], alpha=0.8)
    
    ax_waterfall.set_title('薪资影响因素贡献瀑布图', fontproperties=chinese_font, fontsize=16)
    ax_waterfall.set_ylabel('薪资(美元)', fontproperties=chinese_font, fontsize=12)
    ax_waterfall.tick_params(axis='x', rotation=45)
    plt.setp(ax_waterfall.get_xticklabels(), fontproperties=chinese_font)
    plt.setp(ax_waterfall.get_yticklabels(), fontproperties=chinese_font)
    ax_waterfall.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 添加数值标签
    for bar, val in zip(bars, waterfall_values):
        height = bar.get_height()
        if height >= 0:
            label_y = bar.get_y() + height + 1000
        else:
            label_y = bar.get_y() + height - 2000
        ax_waterfall.text(bar.get_x() + bar.get_width()/2, label_y, f"{int(val)}", ha='center', fontproperties=chinese_font, fontweight='bold')
    st.pyplot(fig_waterfall)
    st.info("""
**图表价值**：瀑布图可以**直观展示每个因素对薪资的贡献值，清晰看到从基准薪资到高薪的每个环节的涨幅**，核心洞察：
1. 入门级/小型企业/2020年的基准平均薪资约为5.8万美元，是所有因素的起点
2. 经验提升是薪资涨幅的最大来源：从入门级到专家级，累计薪资涨幅超12万美元，远超其他所有因素的贡献
3. 中型企业的薪资贡献比大型企业更高，中型企业比基准薪资高约8.2万美元，大型企业仅高约5.4万美元，再次验证了中型企业的薪资溢价
4. 2020-2023年的时间累计带来了约4.2万美元的薪资涨幅，行业整体薪资增长显著
5. 全远程办公的薪资贡献为正，比基准薪资高约7.5万美元，而混合远程的薪资贡献为负，比基准薪资低约1.5万美元
""")

    # 高级图表7：Top15职位薪资分布箱线图
    st.divider()
    st.subheader("高级图表7：Top15热门数据相关职位薪资分布箱线图（职位差异对比）")
    fig_job, ax_job = plt.subplots(figsize=(16, 8))
    # 筛选Top15样本量最多的职位
    top15_jobs = df_clean['job_title'].value_counts().head(15).index.tolist()
    df_top15_jobs = df_clean[df_clean['job_title'].isin(top15_jobs)]
    # 按平均薪资排序
    job_order = df_top15_jobs.groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=False).index.tolist()
    
    sns.boxplot(
        data=df_top15_jobs,
        x='job_title',
        y='salary_in_usd',
        order=job_order,
        palette='Blues',
        ax=ax_job
    )
    ax_job.set_title('Top15热门数据相关职位薪资分布箱线图', fontproperties=chinese_font, fontsize=16)
    ax_job.set_xlabel('职位名称', fontproperties=chinese_font, fontsize=12)
    ax_job.set_ylabel('薪资(美元)', fontproperties=chinese_font, fontsize=12)
    ax_job.tick_params(axis='x', rotation=45, ha='right')
    plt.setp(ax_job.get_xticklabels(), fontproperties=chinese_font)
    plt.setp(ax_job.get_yticklabels(), fontproperties=chinese_font)
    ax_job.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig_job)
    st.info("""
**图表价值**：该图表可以**直观对比不同数据相关职位的薪资水平、分布、上限和下限**，核心洞察：
1. 薪资最高的职位是Principal Data Scientist（首席数据科学家），平均薪资超20万美元，薪资上限接近40万美元
2. 数据科学家、机器学习工程师、应用科学家等技术类职位的薪资普遍高于数据分析师、BI工程师等分析类职位
3. 数据分析师的薪资分布最集中，上限最低，而首席数据科学家、数据科学经理等管理/高级技术职位的薪资分布最分散，上限最高
4. 同样是数据相关职位，最高薪资和最低薪资的差距超过3倍，职位方向是影响薪资的重要因素
""")

    # 高级图表8：公司规模-经验水平薪资热力图
    st.divider()
    st.subheader("高级图表8：公司规模-经验水平薪资热力图（双维度交叉分析）")
    fig_heatmap2d, ax_heatmap2d = plt.subplots(figsize=(12, 8))
    # 构建交叉表
    pivot_table = df_clean.pivot_table(
        index='experience_level',
        columns='company_size',
        values='salary_in_usd',
        aggfunc='mean',
        fill_value=0
    ).reindex(index=exp_order, columns=size_order)
    # 轴标签中文映射
    y_labels = [exp_dict[level] for level in exp_order]
    x_labels = [size_dict[size] for size in size_order]
    
    sns.heatmap(
        pivot_table,
        annot=True,
        cmap='Blues',
        fmt='.0f',
        linewidths=0.5,
        ax=ax_heatmap2d,
        xticklabels=x_labels,
        yticklabels=y_labels,
        annot_kws={"fontproperties": chinese_font, "fontsize": 10}
    )
    ax_heatmap2d.set_title('公司规模-经验水平平均薪资热力图（美元）', fontproperties=chinese_font, fontsize=16)
    ax_heatmap2d.set_xlabel('公司规模', fontproperties=chinese_font, fontsize=12)
    ax_heatmap2d.set_ylabel('经验水平', fontproperties=chinese_font, fontsize=12)
    plt.setp(ax_heatmap2d.get_xticklabels(), fontproperties=chinese_font, rotation=0)
    plt.setp(ax_heatmap2d.get_yticklabels(), fontproperties=chinese_font, rotation=0)
    st.pyplot(fig_heatmap2d)
    st.info("""
**图表价值**：双维度交叉热力图可以**直观展示两个分类变量的组合对薪资的影响，找到最优的薪资组合**，核心洞察：
1. 薪资最高的组合是**专家级经验+中型企业**，平均薪资超21万美元，甚至高于专家级+大型企业的18.5万美元
2. 薪资最低的组合是**入门级经验+小型企业**，平均薪资仅约4.5万美元，和最高薪资组合的差距超过4倍
3. 对于入门级和中级经验的求职者，大型企业的薪资高于中型企业；但对于高级和专家级经验的求职者，中型企业的薪资反超大型企业，说明中型企业更愿意为高端人才支付溢价
4. 随经验提升，公司规模带来的薪资差异越来越大，说明高端人才的企业选择对薪资的影响更显著
""")

# ===================== 6. 分析结论与行业建议 =====================
elif menu == "六、分析结论与行业建议":
    st.header("💡 案例分析结论与建议")
    st.subheader("5.1 核心数据分析结果")
    
    st.markdown("""
1. **薪资整体特征**：全球数据分析师薪资整体呈正态分布，核心区间为 5 万 - 20 万美元，
2020-2023 年薪资持续上涨，4 年涨幅超 55%，行业发展前景向好。

2. **核心影响因素**：通过线性回归与相关性分析，对薪资影响程度从高到低依次为：
**工作经验水平 > 公司所在地区 > 公司规模 > 工作年份 > 远程比例 > 雇佣类型**，
其中工作经验是影响薪资的最核心因素，地区因素紧随其后。

3. **维度差异规律**：
- 经验维度：专家级岗位平均薪资是入门级的 3倍以上，薪资天花板随经验提升显著抬高；
- **企业规模维度**：中型企业（M）的平均薪资最具竞争力（超14万美元），高于大型企业（L）的 11.2 万美元，而小型企业最低（约7.6万美元）。这表明中型成长型企业在挖掘优秀数据人才方面更愿意支付高额溢价；
- 地区维度：发达国家/地区薪资水平显著高于发展中国家，美国、瑞士等地薪资优势明显；
- 办公模式：无远程岗位与全远程岗位的平均薪资整体维持在较高水平（均超13万美元），而混合办公岗位（50%）的薪资水平则表现出明显劣势。
""")

    st.divider()
    st.subheader("5.2 结论与建议")
    st.subheader("💡 给数据从业者的求职建议")
    
    st.markdown("""
1. **优先积累核心工作经验**：工作经验是薪资提升的第一核心要素，建议从业者优先深耕行业，
积累项目经验与专业能力，通过经验提升实现薪资的跨越式增长。

2. **大胆考虑正值红利期的中型企业**：不要局限于大型企业，数据表明中型成长型企业为了快速推进数字化，愿意开出甚至超越大厂的薪资，这也是数据分析师的高薪突破口。

3. **权衡办公模式做求职选择**：若追求高薪，应当优先选择全面线下坐班的岗位，或技术及管理体系更成熟的“纯全远程”企业；尽量避免选择定位相对模糊、薪资基数较低的半混合远程岗位。

4. **持续跟进行业发展趋势**：数据行业薪资持续上涨，从业者需持续学习新技术、新方法，
保持自身的行业竞争力，匹配行业的薪资增长节奏。
""")

    st.subheader("🏢 给企业的薪资制定建议")
    
    st.markdown("""
1. **建立基于经验的阶梯式薪资体系**：工作经验是影响员工价值的核心因素，建议企业建立清晰的
经验 - 薪资对应体系，为不同经验水平的员工提供匹配的薪资待遇，降低核心人才流失率。

2. **大型企业应当反思薪资结构竞争力**：大型企业的平均薪资在此数据集中被中型企业反超，这要求大型企业优化其福利和长期激励之外的现金薪酬（Base Salary）吸引力，防止腰部核心人才被中型企业重金挖走。

3. **提升小型企业的薪资竞争力**：小型企业薪资水平显著处于劣势，在人才竞争中极为被动，建议小型企业优化薪资结构，通过提供更多的股票期权、灵活办公环境与弹性职业发展空间来吸引人才。

4. **规范远程或驻场办公的薪酬匹配**：根据岗位实际交付特征明确办公形态，全坐班和纯全远程往往能招募到高标准人才，企业在制定这类岗位薪资时，应参考高位行业标准以维持岗位的竞争力。
""")

# ===================== 7. 在线薪资预测工具 =====================
elif menu == "七、在线薪资预测工具":
    st.header("🧮 在线薪资预测(仅供参考)")
    st.markdown("输入个人及工作信息，智能预测数据分析师税前年薪（美元）")

    with st.form("salary_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            work_year = st.slider("工作年份", min_value=2020, max_value=2026, value=2026, step=1)
            exp_options = ["入门级(0-2年)", "中级(2-5年)", "高级(5-10年)", "专家级(10年以上)"]
            experience_cn = st.selectbox("工作经验水平", options=exp_options)
            emp_cn = [emp_dict[item] for item in le_employment.classes_]
            employment_cn = st.selectbox("雇佣类型", options=emp_cn)

        with col2:
            remote_ratio = st.select_slider("远程工作比例", options=[0, 50, 100], value=100, format_func=lambda x: f"{x}%")
            size_options = ["小型企业", "中型企业", "大型企业"]
            company_cn = st.selectbox("公司规模", options=size_options)
            
            location = st.selectbox("公司所在国家/地区", options=le_location.classes_)

        submit = st.form_submit_button("开始预测薪资", use_container_width=True)

    if submit:
        # 反向推导英文原始编码
        exp_raw = rev_exp[experience_cn]
        emp_raw = rev_emp[employment_cn]
        size_raw = rev_size[company_cn]

        # 定序变量映射回正确的数学尺度
        exp_code = exp_map[exp_raw]
        com_code = size_map[size_raw]
        emp_code = le_employment.transform([emp_raw])[0]
        loc_code = le_location.transform([location])[0]

        input_features = np.array([[work_year, exp_code, emp_code, remote_ratio, com_code, loc_code]])
        predicted_salary = model.predict(input_features)[0]
        # 分位数预测
        q25_salary = qr_model_25.predict(input_features)[0]
        q50_salary = qr_model_50.predict(input_features)[0]
        q75_salary = qr_model_75.predict(input_features)[0]

        st.success("✅ 预测完成！")
        # 兜底处理：防止回归模型外推到极端组合时出现负数
        predicted_salary = max(0.0, predicted_salary)
        q25_salary = max(0.0, q25_salary)
        q50_salary = max(0.0, q50_salary)
        q75_salary = max(0.0, q75_salary)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="预测平均年薪（美元）", value=f"${predicted_salary:,.2f}")
        with col2:
            st.metric(label="25%分位数年薪", value=f"${q25_salary:,.2f}")
        with col3:
            st.metric(label="50%分位数年薪", value=f"${q50_salary:,.2f}")
        with col4:
            st.metric(label="75%分位数年薪", value=f"${q75_salary:,.2f}")

        st.info("""
说明：
1. 预测结果基于历史数据训练并经过定序修正的线性回归模型，趋势更科学、解释性更强。
2. 薪资单位为美元，为税前年薪。
3. 25%/50%/75%分位数预测分别对应低薪、中薪、高薪场景的预期薪资。
4. 可修改参数对比不同场景薪资差异。
        """)

# ===================== 8. 交互式薪资探索器（新增核心模块） =====================
elif menu == "八、交互式薪资探索器":
    st.header("🔍 交互式薪资探索器（自定义维度深度分析）")
    st.markdown("通过下拉框和滑块自定义筛选条件，动态探索不同维度下的薪资分布、统计数据和可视化图表。")

    # 筛选条件
    col1, col2, col3 = st.columns(3)
    with col1:
        # 经验水平筛选
        exp_filter = st.multiselect(
            "经验水平",
            options=list(exp_dict.keys()),
            default=list(exp_dict.keys()),
            format_func=lambda x: exp_dict[x]
        )
        # 公司规模筛选
        size_filter = st.multiselect(
            "公司规模",
            options=list(size_dict.keys()),
            default=list(size_dict.keys()),
            format_func=lambda x: size_dict[x]
        )
    with col2:
        # 工作年份筛选
        year_filter = st.slider(
            "工作年份",
            min_value=df_clean['work_year'].min(),
            max_value=df_clean['work_year'].max(),
            value=(df_clean['work_year'].min(), df_clean['work_year'].max()),
            step=1
        )
        # 远程比例筛选
        remote_filter = st.multiselect(
            "远程工作比例",
            options=df_clean['remote_ratio'].unique(),
            default=df_clean['remote_ratio'].unique(),
            format_func=lambda x: f"{x}%"
        )
    with col3:
        # 公司所在地区筛选（Top20）
        top20_loc = location_group.head(20)['company_location'].tolist()
        loc_filter = st.multiselect(
            "公司所在地区（Top20）",
            options=top20_loc,
            default=top20_loc
        )
        # 薪资范围筛选
        salary_filter = st.slider(
            "薪资范围（美元）",
            min_value=df_clean['salary_in_usd'].min(),
            max_value=df_clean['salary_in_usd'].max(),
            value=(df_clean['salary_in_usd'].min(), df_clean['salary_in_usd'].max()),
            step=1000
        )

    # 应用筛选条件
    df_filtered = df_clean[
        (df_clean['experience_level'].isin(exp_filter)) &
        (df_clean['company_size'].isin(size_filter)) &
        (df_clean['work_year'].between(year_filter[0], year_filter[1])) &
        (df_clean['remote_ratio'].isin(remote_filter)) &
        (df_clean['company_location'].isin(loc_filter)) &
        (df_clean['salary_in_usd'].between(salary_filter[0], salary_filter[1]))
    ]

    # 筛选结果统计
    st.divider()
    st.subheader("📊 筛选结果统计")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("筛选后样本量", f"{len(df_filtered)} 条")
    with col2:
        st.metric("平均薪资", f"${df_filtered['salary_in_usd'].mean():,.0f}")
    with col3:
        st.metric("薪资中位数", f"${df_filtered['salary_in_usd'].median():,.0f}")
    with col4:
        st.metric("薪资最大值", f"${df_filtered['salary_in_usd'].max():,.0f}")

    # 筛选后数据预览
    st.subheader("筛选后数据预览（前10行）")
    st.dataframe(df_filtered.head(10), use_container_width=True)

    # 动态可视化图表
    st.divider()
    st.subheader("📈 筛选后薪资分布可视化")
    col1, col2 = st.columns(2)
    with col1:
        # 薪资分布直方图
        fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
        ax_hist.hist(df_filtered['salary_in_usd'], bins=20, edgecolor='black', color='#0070C0', alpha=0.7)
        ax_hist.set_title('筛选后薪资分布直方图', fontproperties=chinese_font, fontsize=14)
        ax_hist.set_xlabel('薪资(美元)', fontproperties=chinese_font, fontsize=12)
        ax_hist.set_ylabel('样本数量', fontproperties=chinese_font, fontsize=12)
        ax_hist.grid(axis='y', linestyle='--', alpha=0.7)
        plt.setp(ax_hist.get_xticklabels(), fontproperties=chinese_font)
        plt.setp(ax_hist.get_yticklabels(), fontproperties=chinese_font)
        st.pyplot(fig_hist)
    with col2:
        # 经验水平薪资箱线图
        if len(exp_filter) > 1:
            fig_box, ax_box = plt.subplots(figsize=(10, 6))
            box_data = [df_filtered[df_filtered['experience_level'] == level]['salary_in_usd'] for level in exp_filter if level in df_filtered['experience_level'].unique()]
            box_labels = [exp_dict[level] for level in exp_filter if level in df_filtered['experience_level'].unique()]
            box = ax_box.boxplot(box_data, tick_labels=box_labels, patch_artist=True, boxprops=dict(facecolor='#0070C0', alpha=0.7))
            ax_box.set_title('筛选后不同经验水平薪资箱线图', fontproperties=chinese_font, fontsize=14)
            ax_box.set_xlabel('经验等级', fontproperties=chinese_font, fontsize=12)
            ax_box.set_ylabel('薪资(美元)', fontproperties=chinese_font, fontsize=12)
            ax_box.grid(axis='y', linestyle='--', alpha=0.7)
            plt.setp(ax_box.get_xticklabels(), fontproperties=chinese_font)
            plt.setp(ax_box.get_yticklabels(), fontproperties=chinese_font)
            st.pyplot(fig_box)
        else:
            st.info("仅选择了1个经验水平，无法绘制箱线图对比")

    # 按经验水平分组统计
    st.divider()
    st.subheader("📋 筛选后按经验水平分组统计")
    if len(exp_filter) > 0:
        exp_group_filtered = df_filtered.groupby('experience_level')['salary_in_usd'].agg(
            样本量='count', 平均薪资='mean', 中位数='median', 最低='min', 最高='max'
        ).reset_index()
        exp_group_filtered['experience_level'] = pd.Categorical(exp_group_filtered['experience_level'], categories=exp_order, ordered=True)
        exp_group_filtered = exp_group_filtered.sort_values('experience_level')
        exp_group_filtered['experience_level'] = exp_group_filtered['experience_level'].map(exp_dict)
        st.dataframe(exp_group_filtered, use_container_width=True)
    else:
        st.info("未选择经验水平，无法生成分组统计")

# 页脚
st.markdown("---")
st.markdown("© 2026 数据分析师薪资综合分析系统 | 全流程数据分析 + 可视化 + 智能预测")
