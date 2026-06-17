import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
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

    # ✨ 核心修复 1：在函数内部必须明确定义这两个映射字典
    exp_map = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}
    size_map = {'S': 0, 'M': 1, 'L': 2}
    
    # 标签编码器（仅用于无序分类变量）
    le_employment = LabelEncoder()
    le_location = LabelEncoder()

    df_encoded = df_clean.copy()
    # ✨ 核心修复 2：使用上面定义好的 map 进行转换
    df_encoded['experience_level'] = df_encoded['experience_level'].map(exp_map)
    df_encoded['company_size'] = df_encoded['company_size'].map(size_map)
    df_encoded['employment_type'] = le_employment.fit_transform(df_encoded['employment_type'])
    df_encoded['company_location'] = le_location.fit_transform(df_encoded['company_location'])

    # 训练线性回归模型
    feature_cols = ['work_year', 'experience_level', 'employment_type', 'remote_ratio', 'company_size', 'company_location']
    X = df_encoded[feature_cols]
    y = df_encoded['salary_in_usd']
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    # 多维度分组统计
    exp_order = ['EN', 'MI', 'SE', 'EX']
    exp_group = df_clean.groupby('experience_level')['salary_in_usd'].agg(
        样本量='count', 平均薪资='mean', 中位数='median', 最低='min', 最高='max'
    ).reset_index()
    exp_group['experience_level'] = pd.Categorical(exp_group['experience_level'], categories=exp_order, ordered=True)
    exp_group = exp_group.sort_values('experience_level')

    size_order = ['S', 'M', 'L']
    size_group = df_clean.groupby('company_size')['salary_in_usd'].agg(
        样本量='count', 平均薪资='mean', 中位数='median', 最低='min', 最高='max'
    ).reset_index()
    size_group['company_size'] = pd.Categorical(size_group['company_size'], categories=size_order, ordered=True)
    size_group = size_group.sort_values('company_size')

    year_group = df_clean.groupby('work_year')['salary_in_usd'].agg(
        样本量='count', 平均薪资='mean', 中位数='median', 最低='min', 最高='max'
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

    # ✨ 核心修复 3：确保 return 出来的变量数量（14个）和内容完全正确
    return df, df_clean, exp_group, size_group, year_group, remote_group, location_group, reg_result, r2, \
           model, exp_map, le_employment, size_map, le_location

# ✨ 核心修复 4：确保外部接收（解包）的变量名字、数量与 return 完美一一对应！
df_raw, df_clean, exp_group, size_group, year_group, remote_group, location_group, reg_result, r2_score_val, \
model, exp_map, le_employment, size_map, le_location = load_and_preprocess_data()

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
        "四、可视化图表与解读",
        "五、分析结论与行业建议",
        "六、在线薪资预测工具"
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

# ===================== 4. 可视化图表 =====================
elif menu == "四、可视化图表与解读":
    st.header("📷 数据可视化图表及详细说明")
    exp_order = ['EN', 'MI', 'SE', 'EX']

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

    # ✨ 核心修复点 1：修正公司规模柱状图说明，使之与截图真实数据匹配
    st.subheader("图表3：不同公司规模平均薪资柱状图")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    company_label = ['小型S', '中型M', '大型L']
    bars4 = ax4.bar(company_label, size_group['平均薪资'], color='#2E86AB', alpha=0.8)
    ax4.set_title('不同公司规模平均薪资对比', fontproperties=chinese_font, fontsize=14)
    ax4.set_ylabel('平均薪资(美元)', fontproperties=chinese_font, fontsize=12)
    ax4.grid(axis='
