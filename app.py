import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.metrics import r2_score
import os

# ========= 修复：云端Linux / 本地Windows 双环境中文字体适配 =========
try:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
    chinese_font = FontProperties(fname="/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
except:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    chinese_font = FontProperties(family="SimHei")
plt.rcParams['axes.unicode_minus'] = False

# ========= 自定义 CSS + JavaScript：侧边栏拖动时文字自适应 =========
st.markdown("""
<style>
    section[data-testid="stSidebar"] {
        min-width: 180px;
        max-width: 400px;
    }
    section[data-testid="stSidebar"] > div {
        padding: 2rem 1rem !important;
    }
    .stRadio label {
        font-size: var(--sidebar-font-size, 16px) !important;
        line-height: 1.6 !important;
        padding: 0.4rem 0.6rem !important;
        border-radius: 6px;
        transition: background-color 0.2s;
    }
    .stRadio label:hover {
        background-color: rgba(255, 255, 255, 0.08);
    }
    .css-1d391kg, .css-1v3fvcr {
        font-size: calc(var(--sidebar-font-size, 16px) * 1.3) !important;
    }
</style>
<script>
    function initSidebarResize() {
        const sidebar = document.querySelector('section[data-testid="stSidebar"]');
        if (!sidebar) {
            setTimeout(initSidebarResize, 200);
            return;
        }
        function updateFontSize() {
            const width = sidebar.offsetWidth;
            let size = 12 + (width - 180) * (16 / 220);
            size = Math.min(28, Math.max(12, size));
            sidebar.style.setProperty('--sidebar-font-size', size + 'px');
        }
        const observer = new ResizeObserver(() => { updateFontSize(); });
        observer.observe(sidebar);
        updateFontSize();
        window.addEventListener('resize', updateFontSize);
    }
    initSidebarResize();
</script>
""", unsafe_allow_html=True)

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

    Q1 = df_clean['salary_in_usd'].quantile(0.25)
    Q3 = df_clean['salary_in_usd'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_clean = df_clean[(df_clean['salary_in_usd'] >= lower_bound) & (df_clean['salary_in_usd'] <= upper_bound)]

    exp_map = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}
    size_map = {'S': 0, 'M': 1, 'L': 2}
    le_employment = LabelEncoder()
    le_location = LabelEncoder()
    le_job = LabelEncoder()

    df_encoded = df_clean.copy()
    df_encoded['experience_level'] = df_encoded['experience_level'].map(exp_map)
    df_encoded['company_size'] = df_encoded['company_size'].map(size_map)
    df_encoded['employment_type'] = le_employment.fit_transform(df_encoded['employment_type'])
    df_encoded['company_location'] = le_location.fit_transform(df_encoded['company_location'])
    df_encoded['job_title'] = le_job.fit_transform(df_encoded['job_title'])

    feature_cols = ['work_year', 'experience_level', 'employment_type', 'remote_ratio', 'company_size', 'company_location']
    X = df_encoded[feature_cols]
    y = df_encoded['salary_in_usd']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)
    r2 = r2_score(y, y_pred)

    qr_model_25 = QuantileRegressor(quantile=0.25, alpha=0.01, solver='highs')
    qr_model_50 = QuantileRegressor(quantile=0.5, alpha=0.01, solver='highs')
    qr_model_75 = QuantileRegressor(quantile=0.75, alpha=0.01, solver='highs')
    qr_model_25.fit(X_scaled, y)
    qr_model_50.fit(X_scaled, y)
    qr_model_75.fit(X_scaled, y)

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

    qr_result = pd.DataFrame({
        '特征': ['工作年份', '经验水平', '雇佣类型', '远程比例', '公司规模', '公司所在地区'],
        '25%分位数系数': qr_model_25.coef_,
        '50%分位数系数': qr_model_50.coef_,
        '75%分位数系数': qr_model_75.coef_
    })

    corr_matrix = df_encoded[feature_cols + ['salary_in_usd']].corr()

    return df, df_clean, exp_group, size_group, year_group, remote_group, location_group, reg_result, r2, \
           model, exp_map, le_employment, size_map, le_location, le_job, qr_result, corr_matrix, \
           qr_model_25, qr_model_50, qr_model_75, scaler

# 解包数据
df_raw, df_clean, exp_group, size_group, year_group, remote_group, location_group, reg_result, r2_score_val, \
model, exp_map, le_employment, size_map, le_location, le_job, qr_result, corr_matrix, \
qr_model_25, qr_model_50, qr_model_75, scaler = load_and_preprocess_data()

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

# ===================== 4. 基础可视化图表（已缩小并调整布局） =====================
elif menu == "四、基础可视化图表与解读":
    st.header("📷 基础可视化图表及详细说明")

    # 辅助函数：显示图表+说明，右侧显示代码
    def show_chart_with_code(fig, info_text, code_str):
        col_chart, col_code = st.columns([2, 1])
        with col_chart:
            st.pyplot(fig)
            st.info(info_text)
        with col_code:
            with st.expander("📄 查看代码", expanded=True):
                st.code(code_str, language="python")

    # 图表1：薪资分布直方图
    st.subheader("图表1：薪资分布直方图")
    fig1, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax1.hist(df_clean['salary_in_usd'], bins=30, edgecolor='black', color='#0070C0', alpha=0.7)
    ax1.set_title('数据分析师薪资分布（美元）', fontproperties=chinese_font, fontsize=12)
    ax1.set_xlabel('薪资(美元)', fontproperties=chinese_font, fontsize=10)
    ax1.set_ylabel('样本数量', fontproperties=chinese_font, fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(ax1.get_xticklabels(), fontproperties=chinese_font, fontsize=8)
    plt.setp(ax1.get_yticklabels(), fontproperties=chinese_font, fontsize=8)
    fig1.tight_layout()
    code1 = """
fig1, ax1 = plt.subplots(figsize=(7.5, 4.5))
ax1.hist(df_clean['salary_in_usd'], bins=30, edgecolor='black', color='#0070C0', alpha=0.7)
ax1.set_title('数据分析师薪资分布（美元）', fontproperties=chinese_font, fontsize=12)
ax1.set_xlabel('薪资(美元)', fontproperties=chinese_font, fontsize=10)
ax1.set_ylabel('样本数量', fontproperties=chinese_font, fontsize=10)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
plt.setp(ax1.get_xticklabels(), fontproperties=chinese_font, fontsize=8)
plt.setp(ax1.get_yticklabels(), fontproperties=chinese_font, fontsize=8)
fig1.tight_layout()
st.pyplot(fig1)
"""
    info1 = "**图表说明**：数据分析师薪资整体呈近似正态分布，大部分样本薪资集中在 5 万 - 20 万美元区间，峰值出现在 10-15 万美元区间，整体分布符合职场薪资的常规特征。"
    show_chart_with_code(fig1, info1, code1)
    st.divider()

    # 图表2：经验水平箱线图
    st.subheader("图表2：不同经验水平薪资箱线图")
    fig2, ax2 = plt.subplots(figsize=(7.5, 4.5))
    box_data = [df_clean[df_clean['experience_level'] == level]['salary_in_usd'] for level in exp_order]
    ax2.boxplot(box_data, tick_labels=exp_order, patch_artist=True, boxprops=dict(facecolor='#0070C0', alpha=0.7))
    ax2.set_title('不同经验水平薪资箱线图', fontproperties=chinese_font, fontsize=12)
    ax2.set_xlabel('经验等级(EN入门 / MI中级 / SE高级 / EX专家)', fontproperties=chinese_font, fontsize=10)
    ax2.set_ylabel('薪资(美元)', fontproperties=chinese_font, fontsize=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(ax2.get_xticklabels(), fontproperties=chinese_font, fontsize=8)
    plt.setp(ax2.get_yticklabels(), fontproperties=chinese_font, fontsize=8)
    fig2.tight_layout()
    code2 = """
fig2, ax2 = plt.subplots(figsize=(7.5, 4.5))
box_data = [df_clean[df_clean['experience_level'] == level]['salary_in_usd'] for level in exp_order]
ax2.boxplot(box_data, tick_labels=exp_order, patch_artist=True, boxprops=dict(facecolor='#0070C0', alpha=0.7))
ax2.set_title('不同经验水平薪资箱线图', fontproperties=chinese_font, fontsize=12)
ax2.set_xlabel('经验等级(EN入门 / MI中级 / SE高级 / EX专家)', fontproperties=chinese_font, fontsize=10)
ax2.set_ylabel('薪资(美元)', fontproperties=chinese_font, fontsize=10)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
plt.setp(ax2.get_xticklabels(), fontproperties=chinese_font, fontsize=8)
plt.setp(ax2.get_yticklabels(), fontproperties=chinese_font, fontsize=8)
fig2.tight_layout()
st.pyplot(fig2)
"""
    info2 = "**图表说明**：薪资水平与工作经验呈现显著的正相关关系，入门级 (EN) 平均薪资最低，随经验逐级上涨，专家级 (EX) 薪资为四个等级中最高，职级越高收入优势越明显。"
    show_chart_with_code(fig2, info2, code2)
    st.divider()

    # 图表3：不同公司规模平均薪资柱状图
    st.subheader("图表3：不同公司规模平均薪资柱状图")
    fig4, ax4 = plt.subplots(figsize=(7.5, 4.5))
    company_label = ['小型S', '中型M', '大型L']
    bars4 = ax4.bar(company_label, size_group['平均薪资'], color='#2E86AB', alpha=0.8)
    ax4.set_title('不同公司规模平均薪资对比', fontproperties=chinese_font, fontsize=12)
    ax4.set_ylabel('平均薪资(美元)', fontproperties=chinese_font, fontsize=10)
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(ax4.get_xticklabels(), fontproperties=chinese_font, fontsize=8)
    plt.setp(ax4.get_yticklabels(), fontproperties=chinese_font, fontsize=8)
    for idx, val in enumerate(size_group['平均薪资']):
        ax4.text(idx, val + 2000, f"{int(val)}", ha='center', fontproperties=chinese_font, fontsize=8)
    fig4.tight_layout()
    code4 = """
fig4, ax4 = plt.subplots(figsize=(7.5, 4.5))
company_label = ['小型S', '中型M', '大型L']
bars4 = ax4.bar(company_label, size_group['平均薪资'], color='#2E86AB', alpha=0.8)
ax4.set_title('不同公司规模平均薪资对比', fontproperties=chinese_font, fontsize=12)
ax4.set_ylabel('平均薪资(美元)', fontproperties=chinese_font, fontsize=10)
ax4.grid(axis='y', linestyle='--', alpha=0.7)
plt.setp(ax4.get_xticklabels(), fontproperties=chinese_font, fontsize=8)
plt.setp(ax4.get_yticklabels(), fontproperties=chinese_font, fontsize=8)
for idx, val in enumerate(size_group['平均薪资']):
    ax4.text(idx, val + 2000, f"{int(val)}", ha='center', fontproperties=chinese_font, fontsize=8)
fig4.tight_layout()
st.pyplot(fig4)
"""
    info4 = "**图表说明**：在剔除异常值后，**中型企业 (M)** 提供了最高的平均薪资（约14.0万美元），甚至高于**大型企业 (L)**（约11.2万美元），而**小型企业 (S)** 的薪资水平最低（约7.6万美元）。这表明在中短期内，业务快速扩张且资金充足的中型企业可能对数据分析人才提供了更激进的溢价激励。"
    show_chart_with_code(fig4, info4, code4)
    st.divider()

    # 图表4：年度薪资趋势折线图
    st.subheader("图表4：2020-2023年度薪资趋势折线图")
    fig3, ax3 = plt.subplots(figsize=(7.5, 4.5))
    ax3.plot(year_group['work_year'], year_group['平均薪资'], marker='o', color='#0070C0', linewidth=2)
    ax3.set_title('2020-2023 薪资变化趋势', fontproperties=chinese_font, fontsize=12)
    ax3.set_xlabel('年份', fontproperties=chinese_font, fontsize=10)
    ax3.set_ylabel('平均薪资(美元)', fontproperties=chinese_font, fontsize=10)
    ax3.grid(linestyle='--', alpha=0.7)
    plt.setp(ax3.get_xticklabels(), fontproperties=chinese_font, fontsize=8)
    plt.setp(ax3.get_yticklabels(), fontproperties=chinese_font, fontsize=8)
    for x, y_val in zip(year_group['work_year'], year_group['平均薪资']):
        ax3.text(x, y_val + 2000, f"{int(y_val)}", ha='center', fontproperties=chinese_font, fontsize=8)
    fig3.tight_layout()
    code3 = """
fig3, ax3 = plt.subplots(figsize=(7.5, 4.5))
ax3.plot(year_group['work_year'], year_group['平均薪资'], marker='o', color='#0070C0', linewidth=2)
ax3.set_title('2020-2023 薪资变化趋势', fontproperties=chinese_font, fontsize=12)
ax3.set_xlabel('年份', fontproperties=chinese_font, fontsize=10)
ax3.set_ylabel('平均薪资(美元)', fontproperties=chinese_font, fontsize=10)
ax3.grid(linestyle='--', alpha=0.7)
plt.setp(ax3.get_xticklabels(), fontproperties=chinese_font, fontsize=8)
plt.setp(ax3.get_yticklabels(), fontproperties=chinese_font, fontsize=8)
for x, y_val in zip(year_group['work_year'], year_group['平均薪资']):
    ax3.text(x, y_val + 2000, f"{int(y_val)}", ha='center', fontproperties=chinese_font, fontsize=8)
fig3.tight_layout()
st.pyplot(fig3)
"""
    info3 = "**图表说明**：2020-2023 年数据分析师平均薪资呈持续上涨趋势，行业发展前景向好。"
    show_chart_with_code(fig3, info3, code3)
    st.divider()

    # 图表5：不同远程模式平均薪资柱状图
    st.subheader("图表5：不同远程模式平均薪资柱状图")
    fig5, ax5 = plt.subplots(figsize=(7.5, 4.5))
    x5 = [0, 1, 2]
    remote_labels = ["无远程(0)", "混合远程(50)", "全远程(100)"]
    bars5 = ax5.bar(x5, remote_group['平均薪资'], color='#A23B72', alpha=0.8)
    ax5.set_xticks(x5)
    ax5.set_xticklabels(remote_labels, fontproperties=chinese_font, fontsize=8)
    ax5.set_title('不同远程模式平均薪资对比', fontproperties=chinese_font, fontsize=12)
    ax5.set_ylabel('平均薪资(美元)', fontproperties=chinese_font, fontsize=10)
    ax5.grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(ax5.get_yticklabels(), fontproperties=chinese_font, fontsize=8)
    for bar in bars5:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2, height + 2000, f"{int(height)}", ha='center', fontproperties=chinese_font, fontsize=8)
    fig5.tight_layout()
    code5 = """
fig5, ax5 = plt.subplots(figsize=(7.5, 4.5))
x5 = [0, 1, 2]
remote_labels = ["无远程(0)", "混合远程(50)", "全远程(100)"]
bars5 = ax5.bar(x5, remote_group['平均薪资'], color='#A23B72', alpha=0.8)
ax5.set_xticks(x5)
ax5.set_xticklabels(remote_labels, fontproperties=chinese_font, fontsize=8)
ax5.set_title('不同远程模式平均薪资对比', fontproperties=chinese_font, fontsize=12)
ax5.set_ylabel('平均薪资(美元)', fontproperties=chinese_font, fontsize=10)
ax5.grid(axis='y', linestyle='--', alpha=0.7)
plt.setp(ax5.get_yticklabels(), fontproperties=chinese_font, fontsize=8)
for bar in bars5:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2, height + 2000, f"{int(height)}", ha='center', fontproperties=chinese_font, fontsize=8)
fig5.tight_layout()
st.pyplot(fig5)
"""
    info5 = "**图表说明**：在剔除薪资异常值后，**无远程岗位 (0)** 表现出最高的平均薪资（约14.1万美元），**全远程岗位 (100)** 紧随其后（约13.3万美元），而**混合远程岗位 (50)** 的平均薪资则显著低于前两者（约7.3万美元）。"
    show_chart_with_code(fig5, info5, code5)
    st.divider()

    # 图表6：Top10高薪地区柱状图
    st.subheader("图表6：Top10高薪地区平均薪资对比")
    fig6, ax6 = plt.subplots(figsize=(7.5, 4.5))
    top10_loc = location_group.head(10)
    bars6 = ax6.bar(top10_loc['company_location'], top10_loc['平均薪资'], color='#F18F01', alpha=0.8)
    ax6.set_title('Top10高薪地区平均薪资对比', fontproperties=chinese_font, fontsize=12)
    ax6.set_ylabel('平均薪资(美元)', fontproperties=chinese_font, fontsize=10)
    labels = ax6.get_xticklabels()
    plt.setp(labels, rotation=45, ha='right', fontproperties=chinese_font, fontsize=8)
    ax6.grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(ax6.get_yticklabels(), fontproperties=chinese_font, fontsize=8)
    fig6.tight_layout()
    code6 = """
fig6, ax6 = plt.subplots(figsize=(7.5, 4.5))
top10_loc = location_group.head(10)
bars6 = ax6.bar(top10_loc['company_location'], top10_loc['平均薪资'], color='#F18F01', alpha=0.8)
ax6.set_title('Top10高薪地区平均薪资对比', fontproperties=chinese_font, fontsize=12)
ax6.set_ylabel('平均薪资(美元)', fontproperties=chinese_font, fontsize=10)
labels = ax6.get_xticklabels()
plt.setp(labels, rotation=45, ha='right', fontproperties=chinese_font, fontsize=8)
ax6.grid(axis='y', linestyle='--', alpha=0.7)
plt.setp(ax6.get_yticklabels(), fontproperties=chinese_font, fontsize=8)
fig6.tight_layout()
st.pyplot(fig6)
"""
    info6 = "**图表说明**：不同地区薪资差异显著，美国、瑞士等发达国家的平均薪资远高于其他地区。"
    show_chart_with_code(fig6, info6, code6)

# ===================== 5. 高级可视化分析（已缩小并调整布局） =====================
elif menu == "五、高级可视化分析":
    st.header("🔬 高级可视化分析（专业级深度洞察）")
    st.markdown("本模块使用原生Matplotlib绘制专业图表，深度挖掘薪资数据底层规律。")

    def show_chart_with_code(fig, info_text, code_str):
        col_chart, col_code = st.columns([2, 1])
        with col_chart:
            st.pyplot(fig)
            st.info(info_text)
        with col_code:
            with st.expander("📄 查看代码", expanded=True):
                st.code(code_str, language="python")

    st.divider()
    st.subheader("高级图表1：特征相关性热力图")
    fig_corr, ax_corr = plt.subplots(figsize=(7.5, 4.5))
    corr_labels = ['工作年份', '经验水平', '雇佣类型', '远程比例', '公司规模', '公司所在地区', '薪资(美元)']
    im = ax_corr.imshow(corr_matrix, cmap="Blues", vmin=-1, vmax=1)
    ax_corr.set_xticks(np.arange(len(corr_labels)))
    ax_corr.set_yticks(np.arange(len(corr_labels)))
    ax_corr.set_xticklabels(corr_labels, fontproperties=chinese_font, fontsize=8, rotation=45, ha='right')
    ax_corr.set_yticklabels(corr_labels, fontproperties=chinese_font, fontsize=8)
    ax_corr.set_title('薪资影响因素相关性热力图', fontproperties=chinese_font, fontsize=12)
    for i in range(len(corr_labels)):
        for j in range(len(corr_labels)):
            ax_corr.text(j, i, f"{corr_matrix.iloc[i,j]:.2f}", ha="center", va="center",
                         fontproperties=chinese_font, fontsize=7, color="black")
    fig_corr.colorbar(im, ax=ax_corr)
    fig_corr.tight_layout()
    code_corr = """
fig_corr, ax_corr = plt.subplots(figsize=(7.5, 4.5))
corr_labels = ['工作年份', '经验水平', '雇佣类型', '远程比例', '公司规模', '公司所在地区', '薪资(美元)']
im = ax_corr.imshow(corr_matrix, cmap="Blues", vmin=-1, vmax=1)
ax_corr.set_xticks(np.arange(len(corr_labels)))
ax_corr.set_yticks(np.arange(len(corr_labels)))
ax_corr.set_xticklabels(corr_labels, fontproperties=chinese_font, fontsize=8, rotation=45, ha='right')
ax_corr.set_yticklabels(corr_labels, fontproperties=chinese_font, fontsize=8)
ax_corr.set_title('薪资影响因素相关性热力图', fontproperties=chinese_font, fontsize=12)
for i in range(len(corr_labels)):
    for j in range(len(corr_labels)):
        ax_corr.text(j, i, f"{corr_matrix.iloc[i,j]:.2f}", ha="center", va="center",
                     fontproperties=chinese_font, fontsize=7, color="black")
fig_corr.colorbar(im, ax=ax_corr)
fig_corr.tight_layout()
st.pyplot(fig_corr)
"""
    info_corr = "**图表说明**：热力图展示了各特征与薪资之间的两两相关性。其中**经验水平**（0.48）和**工作年份**（0.39）与薪资呈中等正相关，是影响薪资的最强因素；**公司所在地区**（0.16）和**公司规模**（0.13）也有一定正向影响；而**雇佣类型**和**远程比例**与薪资的相关性较弱（接近0）。这表明工作经验积累和时间推移是薪资增长的核心驱动力。"
    show_chart_with_code(fig_corr, info_corr, code_corr)

    st.divider()
    st.subheader("高级图表2：不同薪资分位数的标准化影响系数对比（分组散点连线图）")
    fig_qr, ax_qr = plt.subplots(figsize=(7.5, 4.5))
    features = qr_result['特征'].tolist()
    q25 = qr_result['25%分位数系数'].tolist()
    q50 = qr_result['50%分位数系数'].tolist()
    q75 = qr_result['75%分位数系数'].tolist()
    y_pos = np.arange(len(features))

    ax_qr.hlines(y=y_pos, xmin=q25, xmax=q75, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
    ax_qr.scatter(q25, y_pos - 0.15, color='#1f77b4', label='25% 低薪', s=50, zorder=5, marker='o')
    ax_qr.scatter(q50, y_pos, color='#ff7f0e', label='50% 中薪', s=60, zorder=5, marker='D')
    ax_qr.scatter(q75, y_pos + 0.15, color='#2ca02c', label='75% 高薪', s=50, zorder=5, marker='^')
    ax_qr.axvline(x=0, color='red', linestyle='-', linewidth=1, alpha=0.3, label='无影响基准')
    ax_qr.set_yticks(y_pos)
    ax_qr.set_yticklabels(features, fontproperties=chinese_font, fontsize=9)
    ax_qr.set_xlabel('标准化回归系数（Beta权重）', fontproperties=chinese_font, fontsize=10)
    ax_qr.set_title('分位数回归系数对比', fontproperties=chinese_font, fontsize=12)

    # ========== 修改图例位置：放在图表右侧外部，不遮挡数据 ==========
    ax_qr.legend(prop=chinese_font, loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8)
    # 调整布局，为右侧图例留出空间
    fig_qr.subplots_adjust(right=0.75)

    ax_qr.grid(axis='x', linestyle='--', alpha=0.5)
    ax_qr.autoscale(axis='x')
    fig_qr.tight_layout()
    code_qr = """
fig_qr, ax_qr = plt.subplots(figsize=(7.5, 4.5))
features = qr_result['特征'].tolist()
q25 = qr_result['25%分位数系数'].tolist()
q50 = qr_result['50%分位数系数'].tolist()
q75 = qr_result['75%分位数系数'].tolist()
y_pos = np.arange(len(features))

ax_qr.hlines(y=y_pos, xmin=q25, xmax=q75, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
ax_qr.scatter(q25, y_pos - 0.15, color='#1f77b4', label='25% 低薪', s=50, zorder=5, marker='o')
ax_qr.scatter(q50, y_pos, color='#ff7f0e', label='50% 中薪', s=60, zorder=5, marker='D')
ax_qr.scatter(q75, y_pos + 0.15, color='#2ca02c', label='75% 高薪', s=50, zorder=5, marker='^')
ax_qr.axvline(x=0, color='red', linestyle='-', linewidth=1, alpha=0.3, label='无影响基准')
ax_qr.set_yticks(y_pos)
ax_qr.set_yticklabels(features, fontproperties=chinese_font, fontsize=9)
ax_qr.set_xlabel('标准化回归系数（Beta权重）', fontproperties=chinese_font, fontsize=10)
ax_qr.set_title('分位数回归系数对比', fontproperties=chinese_font, fontsize=12)
# 图例放在右侧外部
ax_qr.legend(prop=chinese_font, loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8)
fig_qr.subplots_adjust(right=0.75)
ax_qr.grid(axis='x', linestyle='--', alpha=0.5)
ax_qr.autoscale(axis='x')
fig_qr.tight_layout()
st.pyplot(fig_qr)
"""
    info_qr = "**图表说明**：该图展示了每个特征在不同薪资分位数（低薪25%、中薪50%、高薪75%）下的标准化回归系数，灰色虚线连接了低分位到高分位的系数变化。**经验水平**和**工作年份**在三个分位数下都保持最高的正向影响，且高薪群体（75%）受经验影响更大（系数更高），说明经验积累对顶尖薪资的拉动作用尤为明显。**公司所在地区**和**公司规模**的影响在中等薪资水平时较强，而**远程比例**和**雇佣类型**的系数接近零或略负，表明其对薪资分层贡献较小。"
    show_chart_with_code(fig_qr, info_qr, code_qr)
    st.dataframe(qr_result, use_container_width=True)

    st.divider()
    st.subheader("高级图表3：带95%置信区间年度薪资趋势")
    fig_trend, ax_trend = plt.subplots(figsize=(7.5, 4.5))
    year_group['ci_lower'] = year_group['平均薪资'] - 1.96 * (year_group['薪资标准差'] / np.sqrt(year_group['样本量']))
    year_group['ci_upper'] = year_group['平均薪资'] + 1.96 * (year_group['薪资标准差'] / np.sqrt(year_group['样本量']))
    ax_trend.plot(year_group['work_year'], year_group['平均薪资'], marker='o', c='#0070C0', linewidth=2, label='平均薪资')
    ax_trend.fill_between(year_group['work_year'], year_group['ci_lower'], year_group['ci_upper'], alpha=0.2, color='#0070C0', label='95% CI')
    ax_trend.plot(year_group['work_year'], year_group['中位数'], marker='s', c='#2E86AB', linestyle='--', label='中位数')
    ax_trend.set_title('2020-2023薪资趋势（置信区间）', fontproperties=chinese_font, fontsize=12)
    ax_trend.set_xlabel('年份', fontproperties=chinese_font, fontsize=10)
    ax_trend.set_ylabel('薪资(美元)', fontproperties=chinese_font, fontsize=10)
    ax_trend.set_xticks(year_group['work_year'])
    ax_trend.legend(prop=chinese_font, fontsize=8)
    ax_trend.grid(linestyle='--', alpha=0.5)
    fig_trend.tight_layout()
    code_trend = """
fig_trend, ax_trend = plt.subplots(figsize=(7.5, 4.5))
year_group['ci_lower'] = year_group['平均薪资'] - 1.96 * (year_group['薪资标准差'] / np.sqrt(year_group['样本量']))
year_group['ci_upper'] = year_group['平均薪资'] + 1.96 * (year_group['薪资标准差'] / np.sqrt(year_group['样本量']))
ax_trend.plot(year_group['work_year'], year_group['平均薪资'], marker='o', c='#0070C0', linewidth=2, label='平均薪资')
ax_trend.fill_between(year_group['work_year'], year_group['ci_lower'], year_group['ci_upper'], alpha=0.2, color='#0070C0', label='95% CI')
ax_trend.plot(year_group['work_year'], year_group['中位数'], marker='s', c='#2E86AB', linestyle='--', label='中位数')
ax_trend.set_title('2020-2023薪资趋势（置信区间）', fontproperties=chinese_font, fontsize=12)
ax_trend.set_xlabel('年份', fontproperties=chinese_font, fontsize=10)
ax_trend.set_ylabel('薪资(美元)', fontproperties=chinese_font, fontsize=10)
ax_trend.set_xticks(year_group['work_year'])
ax_trend.legend(prop=chinese_font, fontsize=8)
ax_trend.grid(linestyle='--', alpha=0.5)
fig_trend.tight_layout()
st.pyplot(fig_trend)
"""
    info_trend = "**图表说明**：折线图展示了2020年至2023年数据分析师的平均薪资和中位数薪资的逐年变化，同时用浅蓝色阴影表示平均薪资的95%置信区间（反映年度内薪资的波动范围）。整体趋势表明，平均薪资从约6.5万美元稳步增长至近10万美元，涨幅超过50%，行业处于高速增长期。置信区间逐年收窄，说明薪资分布趋于集中，市场定价更加成熟稳定。"
    show_chart_with_code(fig_trend, info_trend, code_trend)

    st.divider()
    st.subheader("高级图表4：公司规模-经验薪资二维热力图")
    fig_heatmap2d, ax_heatmap2d = plt.subplots(figsize=(7.5, 4.5))
    pivot_table = df_clean.pivot_table(index='experience_level', columns='company_size', values='salary_in_usd', aggfunc='mean').reindex(index=exp_order, columns=size_order)
    im2 = ax_heatmap2d.imshow(pivot_table.values, cmap="Blues")
    ax_heatmap2d.set_xticks(np.arange(len(size_order)))
    ax_heatmap2d.set_yticks(np.arange(len(exp_order)))
    ax_heatmap2d.set_xticklabels([size_dict[s] for s in size_order], fontproperties=chinese_font, fontsize=8)
    ax_heatmap2d.set_yticklabels([exp_dict[e] for e in exp_order], fontproperties=chinese_font, fontsize=8)
    ax_heatmap2d.set_title('公司规模-经验水平平均薪资热力图', fontproperties=chinese_font, fontsize=12)
    for i in range(len(exp_order)):
        for j in range(len(size_order)):
            val = int(pivot_table.iloc[i,j])
            ax_heatmap2d.text(j,i,str(val),ha="center",va="center",fontproperties=chinese_font, fontsize=7)
    fig_heatmap2d.colorbar(im2, ax=ax_heatmap2d)
    fig_heatmap2d.tight_layout()
    code_heat = """
fig_heatmap2d, ax_heatmap2d = plt.subplots(figsize=(7.5, 4.5))
pivot_table = df_clean.pivot_table(index='experience_level', columns='company_size', values='salary_in_usd', aggfunc='mean').reindex(index=exp_order, columns=size_order)
im2 = ax_heatmap2d.imshow(pivot_table.values, cmap="Blues")
ax_heatmap2d.set_xticks(np.arange(len(size_order)))
ax_heatmap2d.set_yticks(np.arange(len(exp_order)))
ax_heatmap2d.set_xticklabels([size_dict[s] for s in size_order], fontproperties=chinese_font, fontsize=8)
ax_heatmap2d.set_yticklabels([exp_dict[e] for e in exp_order], fontproperties=chinese_font, fontsize=8)
ax_heatmap2d.set_title('公司规模-经验水平平均薪资热力图', fontproperties=chinese_font, fontsize=12)
for i in range(len(exp_order)):
    for j in range(len(size_order)):
        val = int(pivot_table.iloc[i,j])
        ax_heatmap2d.text(j,i,str(val),ha="center",va="center",fontproperties=chinese_font, fontsize=7)
fig_heatmap2d.colorbar(im2, ax=ax_heatmap2d)
fig_heatmap2d.tight_layout()
st.pyplot(fig_heatmap2d)
"""
    info_heat = "**图表说明**：热力图展示了不同经验水平与公司规模组合下的平均薪资（美元）。从横向看，**中型企业（M）** 在每一经验等级上都提供了最高的平均薪资，尤其在中级（MI）和高级（SE）阶段，其薪资远超大型和小型企业；从纵向看，**专家级（EX）** 在任何规模的企业中都是薪资最高的群体。这一结果进一步验证了中型企业对数据人才的积极溢价策略，以及经验积累带来的薪资跃升效应。"
    show_chart_with_code(fig_heatmap2d, info_heat, code_heat)

# ===================== 6. 分析结论与行业建议 =====================
elif menu == "六、分析结论与行业建议":
    st.header("💡 案例分析结论与建议")
    st.subheader("6.1 核心数据分析结果")
    
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
    st.subheader("6.2 结论与建议")
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
        exp_raw = rev_exp[experience_cn]
        emp_raw = rev_emp[employment_cn]
        size_raw = rev_size[company_cn]

        exp_code = exp_map[exp_raw]
        com_code = size_map[size_raw]
        emp_code = le_employment.transform([emp_raw])[0]
        loc_code = le_location.transform([location])[0]

        input_features = np.array([[work_year, exp_code, emp_code, remote_ratio, com_code, loc_code]])
        input_features_scaled = scaler.transform(input_features)

        predicted_salary = model.predict(input_features_scaled)[0]
        q25_salary = qr_model_25.predict(input_features_scaled)[0]
        q50_salary = qr_model_50.predict(input_features_scaled)[0]
        q75_salary = qr_model_75.predict(input_features_scaled)[0]

        st.success("✅ 预测完成！")
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

# ===================== 8. 交互式薪资探索器 =====================
elif menu == "八、交互式薪资探索器":
    st.header("🔍 交互式薪资探索器（自定义维度深度分析）")
    st.markdown("通过下拉框和滑块自定义筛选条件，动态探索不同维度下的薪资分布、统计数据和可视化图表。")

    col1, col2, col3 = st.columns(3)
    with col1:
        exp_filter = st.multiselect(
            "经验水平",
            options=list(exp_dict.keys()),
            default=list(exp_dict.keys()),
            format_func=lambda x: exp_dict[x]
        )
        size_filter = st.multiselect(
            "公司规模",
            options=list(size_dict.keys()),
            default=list(size_dict.keys()),
            format_func=lambda x: size_dict[x]
        )
    with col2:
        year_filter = st.slider(
            "工作年份",
            min_value=df_clean['work_year'].min(),
            max_value=df_clean['work_year'].max(),
            value=(df_clean['work_year'].min(), df_clean['work_year'].max()),
            step=1
        )
        remote_filter = st.multiselect(
            "远程工作比例",
            options=df_clean['remote_ratio'].unique(),
            default=df_clean['remote_ratio'].unique(),
            format_func=lambda x: f"{x}%"
        )
    with col3:
        top20_loc = location_group.head(20)['company_location'].tolist()
        loc_filter = st.multiselect(
            "公司所在地区（Top20）",
            options=top20_loc,
            default=top20_loc
        )
        salary_filter = st.slider(
            "薪资范围（美元）",
            min_value=df_clean['salary_in_usd'].min(),
            max_value=df_clean['salary_in_usd'].max(),
            value=(df_clean['salary_in_usd'].min(), df_clean['salary_in_usd'].max()),
            step=1000
        )

    df_filtered = df_clean[
        (df_clean['experience_level'].isin(exp_filter)) &
        (df_clean['company_size'].isin(size_filter)) &
        (df_clean['work_year'].between(year_filter[0], year_filter[1])) &
        (df_clean['remote_ratio'].isin(remote_filter)) &
        (df_clean['company_location'].isin(loc_filter)) &
        (df_clean['salary_in_usd'].between(salary_filter[0], salary_filter[1]))
    ]

    st.divider()
    st.subheader("📊 筛选结果统计")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("筛选后样本量", f"{len(df_filtered)} 条")
    with col2:
        avg_sal = df_filtered['salary_in_usd'].mean() if len(df_filtered) > 0 else 0
        st.metric("平均薪资", f"${avg_sal:,.0f}")
    with col3:
        med_sal = df_filtered['salary_in_usd'].median() if len(df_filtered) > 0 else 0
        st.metric("薪资中位数", f"${med_sal:,.0f}")
    with col4:
        max_sal = df_filtered['salary_in_usd'].max() if len(df_filtered) > 0 else 0
        st.metric("薪资最大值", f"${max_sal:,.0f}")

    st.subheader("筛选后数据预览（前10行）")
    st.dataframe(df_filtered.head(10), use_container_width=True)

    st.divider()
    st.subheader("📈 筛选后薪资分布可视化")
    col1, col2 = st.columns(2)
    with col1:
        fig_hist, ax_hist = plt.subplots(figsize=(7.5, 4.5))
        ax_hist.hist(df_filtered['salary_in_usd'], bins=20, edgecolor='black', color='#0070C0', alpha=0.7)
        ax_hist.set_title('筛选后薪资分布直方图', fontproperties=chinese_font, fontsize=12)
        ax_hist.set_xlabel('薪资(美元)', fontproperties=chinese_font, fontsize=10)
        ax_hist.set_ylabel('样本数量', fontproperties=chinese_font, fontsize=10)
        ax_hist.grid(axis='y', linestyle='--', alpha=0.7)
        plt.setp(ax_hist.get_xticklabels(), fontproperties=chinese_font, fontsize=8)
        plt.setp(ax_hist.get_yticklabels(), fontproperties=chinese_font, fontsize=8)
        fig_hist.tight_layout()
        st.pyplot(fig_hist)
    with col2:
        if len(exp_filter) > 1 and len(df_filtered) > 0:
            exist_levels = [l for l in exp_filter if l in df_filtered['experience_level'].unique()]
            if len(exist_levels) > 1:
                fig_box, ax_box = plt.subplots(figsize=(7.5, 4.5))
                box_data = [df_filtered[df_filtered['experience_level'] == level]['salary_in_usd'] for level in exist_levels]
                box_labels = [exp_dict[level] for level in exist_levels]
                box = ax_box.boxplot(box_data, tick_labels=box_labels, patch_artist=True, boxprops=dict(facecolor='#0070C0', alpha=0.7))
                ax_box.set_title('筛选后不同经验水平薪资箱线图', fontproperties=chinese_font, fontsize=12)
                ax_box.set_xlabel('经验等级', fontproperties=chinese_font, fontsize=10)
                ax_box.set_ylabel('薪资(美元)', fontproperties=chinese_font, fontsize=10)
                ax_box.grid(axis='y', linestyle='--', alpha=0.7)
                plt.setp(ax_box.get_xticklabels(), fontproperties=chinese_font, fontsize=8)
                plt.setp(ax_box.get_yticklabels(), fontproperties=chinese_font, fontsize=8)
                fig_box.tight_layout()
                st.pyplot(fig_box)
            else:
                st.info("当前筛选下有效经验等级不足2个，无法绘制对比箱线图")
        else:
            st.info("仅选择了1个经验水平，无法绘制箱线图对比")

    st.divider()
    st.subheader("📋 筛选后按经验水平分组统计")
    if len(exp_filter) > 0 and len(df_filtered) > 0:
        exp_group_filtered = df_filtered.groupby('experience_level')['salary_in_usd'].agg(
            样本量='count', 平均薪资='mean', 中位数='median', 最低='min', 最高='max'
        ).reset_index()
        exp_group_filtered['experience_level'] = pd.Categorical(exp_group_filtered['experience_level'], categories=exp_order, ordered=True)
        exp_group_filtered = exp_group_filtered.sort_values('experience_level')
        exp_group_filtered['experience_level'] = exp_group_filtered['experience_level'].map(exp_dict)
        st.dataframe(exp_group_filtered, use_container_width=True)
    else:
        st.info("未选择经验水平或无筛选数据，无法生成分组统计")

# 页脚
st.markdown("---")
st.markdown("© 2026 数据分析师薪资综合分析系统 | 全流程数据分析 + 可视化 + 智能预测")
