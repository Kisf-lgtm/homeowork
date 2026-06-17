import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ===================== 🛡️ 彻底修复：字体与图形死锁防御 =====================
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 自动寻找系统内可用中文黑体，找不到则安全降级，防止云端环境因为字体加载而白屏
supported_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'DejaVu Sans', 'sans-serif']
for font in supported_fonts:
    try:
        plt.rcParams['font.sans-serif'] = [font]
        # 测试绘图是否会崩溃
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'test')
        plt.close(fig)
        break
    except:
        continue

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
    """加载数据、预处理、建模、统计分组"""
    # 建立兜底空数据，防止文件不存在时白屏
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        st.error(f"⚠️ 无法读取数据源文件 '{CSV_PATH}'，请检查文件是否存在！错误信息: {e}")
        st.stop()

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

    # 映射中文名
    size_group['company_size_cn'] = size_group['company_size'].map({"S": "小型企业(S)", "M": "中型企业(M)", "L": "大型企业(L)"})

    year_group = df_clean.groupby('work_year')['salary_in_usd'].agg(
        样本量='count', 平均薪资
