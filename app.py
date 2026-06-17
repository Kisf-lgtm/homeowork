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

    # 内部无序映射
    exp_map = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}
    size_map = {'S': 0, 'M': 1, 'L': 2}
    
    le_employment = LabelEncoder()
    le_location = LabelEncoder()

    df_encoded = df_clean.copy()
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

    # 将英文规模标签替换为可读中文，方便后续代码动态提取
    size_group['company_size_cn'] = size_group['company_size'].map({"S": "小型企业(S)", "M": "中型企业(M)", "L": "大型企业(L)"})

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

    return df, df_clean, exp_group, size_group, year_group, remote_group, location_group, reg_result, r2, \
           model, exp_map, le_employment, size_map, le_location

# 解包变量
df_raw, df_clean, exp_group, size_group, year_group, remote_group, location_group, reg_result, r2_score_val, \
model, exp_map, le_employment, size_map, le_location = load_and_preprocess_data()

# ✨ 核心改进：在外部计算当前清洗数据的动态事实结论，消除硬编码错误！
highest_size_row = size_group.loc[size_group['平均薪资'].idxmax()]
lowest_size_row = size_group.loc[size_group['平均薪资'].idxmin()]
highest_size_name = highest_size_row['company_size_cn']
highest_size_val = int(highest_size_row['平均薪资'])
lowest_size_name = lowest_size_row['company_size_cn']
lowest_size_val = int(lowest_size_row['平均薪资'])
size_gap_ratio = ((highest_size_val - lowest_size_val) / lowest_size_val) * 100

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
1. 完成数据清洗与预处理，
