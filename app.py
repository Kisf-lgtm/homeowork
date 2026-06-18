import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ========= 云端/本地中文字体适配 =========
try:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
    chinese_font = FontProperties(fname="/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
except:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    chinese_font = FontProperties(family="SimHei")
plt.rcParams['axes.unicode_minus'] = False

# 页面全局配置
st.set_page_config(
    page_title="数据分析师薪资预测系统",
    page_icon="💰",
    layout="wide"
)

CSV_PATH = "数据分析师工资.csv"

# 中英文映射
exp_dict = {"EN": "入门级(0-2年)", "MI": "中级(2-5年)", "SE": "高级(5-10年)", "EX": "专家级(10年以上)"}
emp_dict = {"FT": "全职", "CT": "合同工", "PT": "兼职", "FL": "自由职业"}
size_dict = {"S": "小型企业", "M": "中型企业", "L": "大型企业"}
rev_exp = {v: k for k, v in exp_dict.items()}
rev_emp = {v: k for k, v in emp_dict.items()}
rev_size = {v: k for k, v in size_dict.items()}

# 数据预处理&建模
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv(CSV_PATH)
    df_clean = df.copy()

    # IQR剔除薪资异常值
    Q1 = df_clean['salary_in_usd'].quantile(0.25)
    Q3 = df_clean['salary_in_usd'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_clean = df_clean[(df_clean['salary_in_usd'] >= lower_bound) & (df_clean['salary_in_usd'] <= upper_bound)]
    min_real_salary = df_clean['salary_in_usd'].min()

    # 有序分类映射
    exp_map = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}
    size_map = {'S': 0, 'M': 1, 'L': 2}
    df_clean['exp_code'] = df_clean['experience_level'].map(exp_map)
    df_clean['size_code'] = df_clean['company_size'].map(size_map)

    # 区分数值特征与无序分类特征
    num_features = ["work_year", "exp_code", "remote_ratio", "size_code"]
    cat_features = ["employment_type", "job_title", "company_location", "employee_residence"]
    X_raw = df_clean[num_features + cat_features]
    y = df_clean['salary_in_usd']

    # OneHot处理无序分类，规避LabelEncoder排序误差
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
        ],
        remainder="passthrough"
    )
    X_processed = preprocessor.fit_transform(X_raw)
    model = LinearRegression()
    model.fit(X_processed, y)
    y_pred = model.predict(X_processed)
    r2 = r2_score(y, y_pred)

    # 各类分组统计
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

    job_title_group = df_clean.groupby('job_title')['salary_in_usd'].agg(
        样本量='count', 平均薪资='mean', 中位数='median', 最低='min', 最高='max'
    ).reset_index().sort_values('平均薪资', ascending=False)

    # 下拉框唯一值
    emp_unique = sorted(df_clean['employment_type'].unique())
    loc_unique = sorted(df_clean['company_location'].unique())
    res_unique = sorted(df_clean['job_title'].unique())
    job_unique = sorted(df_clean['job_title'].unique())

    return (df, df_clean, exp_group, size_group, year_group, remote_group, location_group, job_title_group,
            r2, model, preprocessor, exp_map, size_map, min_real_salary,
            emp_unique, loc_unique, res_unique, job_unique)

# 解包变量
(df_raw, df_clean, exp_group, size_group, year_group, remote_group, location_group, job_title_group,
 r2_score_val, model, preprocessor, exp_map, size_map, min_salary,
 emp_list, loc_list, res_list, job_list) = load_and_preprocess_data()

# 侧边导航
st.title("💰 数据分析师薪资分析与预测综合平台")
st.markdown("本平台基于2020-2023全球薪资数据集，支持2020-2026年份薪资预测，2024-2026为行业趋势外推结果。")
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

# 1 项目目标
if menu == "一、项目分析目标与预期":
    st.header("🎯 数据分析目标与预期结果")
    st.subheader("核心分析目标")
    st.markdown("""
1. 探索2020-2023全球数据分析师薪资分布与年度变化趋势
2. 量化工作经验、地区、岗位等因素对薪资的影响程度
3. 提供求职与企业薪酬制定的数据参考，支持2020-2026薪资预测
""")
    st.subheader("预期结果")
    st.markdown("""
1. 完成数据清洗，剔除极端薪资异常值
2. 多维度分组统计，展示薪资差异规律
3. 构建回归模型，支持历史年份与未来趋势预测
4. 可视化图表输出完整行业分析报告
""")

# 2 数据集介绍
elif menu == "二、数据集背景介绍":
    st.header("🗃️ 数据集背景介绍")
    st.markdown("原始数据共3755条，时间范围2020-2023，共11个字段，统一以美元薪资为分析目标。")
    field_data = [
        ["work_year", "数值", "数据年份2020/2021/2022/2023，预测支持外推至2026"],
        ["experience_level", "分类", "EN入门/MI中级/SE高级/EX专家"],
        ["employment_type", "分类", "FT全职/CT合同/PT兼职/FL自由职业"],
        ["job_title", "分类", 93类数据相关岗位],
        ["salary_in_usd", "数值", 统一美元年薪，分析目标],
        ["remote_ratio", "数值", 0线下/50混合/100全远程],
        ["company_size", "分类", S小型/M中型/L大型],
        ["company_location", "分类", 企业所在国家地区],
        ["employee_residence", "分类", 员工居住国家地区]
    ]
    df_field = pd.DataFrame(field_data, columns=["字段名称", "类型", "说明"])
    st.dataframe(df_field, use_container_width=True, hide_index=True)

# 3 数据总览
elif menu == "三、数据总览与统计报表":
    st.header("📊 数据总览 & 统计报表")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("原始数据量", f"{len(df_raw)} 条")
        st.metric("清洗后数据", f"{len(df_clean)} 条")
    with col2:
        st.metric("模型R²", f"{r2_score_val:.4f}")
        st.metric("平均年薪", f"${df_clean['salary_in_usd']:.2f}")
    st.subheader("原始数据前10行")
    st.dataframe(df_raw.head(10), use_container_width=True)
    st.subheader("薪资描述统计")
    st.dataframe(df_clean['salary_in_usd'].describe(), use_container_width=True)
    st.divider()
    st.subheader("1. 经验薪资分布")
    st.dataframe(exp_group, use_container_width=True)
    st.subheader("2. 企业规模薪资")
    st.dataframe(size_group, use_container_width=True)
    st.subheader("3. 年度薪资趋势")
    st.dataframe(year_group, use_container_width=True)
    st.subheader("4. 远程模式薪资")
    st.dataframe(remote_group, use_container_width=True)
    st.subheader("5. 地区薪资TOP20")
    st.dataframe(location_group.head(20), use_container_width=True)
    st.subheader("6. 岗位薪资TOP20")
    st.dataframe(job_title_group.head(20))

# 4 可视化模块
elif menu == "四、可视化图表与解读":
    st.header("📷 可视化图表")
    exp_order = ['EN', 'MI', 'SE', 'EX']
    # 薪资分布直方图
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.hist(df_clean['salary_in_usd'], bins=30, edgecolor='black', color='#0070C0', alpha=0.7)
    ax1.set_title('薪资分布（美元）', fontproperties=chinese_font)
    ax1.set_xlabel('年薪美元', fontproperties=chinese_font)
    ax1.set_ylabel('样本数', fontproperties=chinese_font)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig1)
    st.info("薪资集中在5万-20万美元，近似正态分布")
    st.divider()
    # 经验箱线图
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    box_data = [df_clean[df_clean['experience_level'] == lvl]['salary_in_usd'] for lvl in exp_order]
    ax2.boxplot(box_data, tick_labels=exp_order, patch_artist=True, boxprops=dict(facecolor='#0070C0', alpha=0.7))
    ax2.set_title("不同经验薪资箱线图", fontproperties=chinese_font)
    ax2.set_xlabel("经验等级", fontproperties=chinese_font)
    ax2.set_ylabel("年薪美元", fontproperties=chinese_font)
    st.pyplot(fig2)
    st.info("薪资随经验显著提升，专家级薪资上限最高")
    st.divider()
    # 企业规模柱状图
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    company_label = ['小型S', '中型M', '大型L']
    bars4 = ax4.bar(company_label, size_group['平均薪资'], color='#2E86AB', alpha=0.8)
    for idx, val in enumerate(size_group['平均薪资']):
        ax4.text(idx, val + 2000, f"{int(val)}", ha='center', fontproperties=chinese_font)
    ax4.set_title("各规模企业平均薪资", fontproperties=chinese_font)
    ax4.set_ylabel("平均年薪", fontproperties=chinese_font)
    st.pyplot(fig4)
    st.info("中型企业薪资均值高于大型企业，小企业薪资最低")
    st.divider()
    # 年度趋势
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(year_group['work_year'], year_group['平均薪资'], marker='o', linewidth=2)
    for x, y_val in zip(year_group['work_year'], year_group['平均薪资']):
        ax3.text(x, y_val + 2000, f"{int(y_val)}", ha='center', fontproperties=chinese_font)
    ax3.set_title("2020-2023薪资趋势", fontproperties=chinese_font)
    ax3.set_xlabel("年份", fontproperties=chinese_font)
    ax3.set_ylabel("平均年薪", fontproperties=chinese_font)
    st.pyplot(fig3)
    st.info("2020至2023薪资逐年上涨，行业持续扩容")
    st.divider()
    # 远程模式
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    x5 = [0, 1, 2]
    remote_labels = ["线下0%", "混合50%", "全远程100%"]
    bars5 = ax5.bar(x5, remote_group['平均薪资'], color='#A23B72', alpha=0.8)
    ax5.set_xticks(x5)
    ax5.set_xticklabels(remote_labels, fontproperties=chinese_font)
    for bar in bars5:
        h = bar.get_height()
        ax5.text(bar.get_x()+bar.get_width()/2, h+2000, f"{int(h)}", ha='center', fontproperties=chinese_font)
    ax5.set_title("远程模式薪资对比", fontproperties=chinese_font)
    st.pyplot(fig5)
    st.info("线下、全远程薪资高于混合远程岗位")
    st.divider()
    # Top10地区
    fig6, ax6 = plt.subplots(figsize=(12, 6))
    top10_loc = location_group.head(10)
    bars6 = ax6.bar(top10_loc['company_location'], top10_loc['平均薪资'], color='#F18F01', alpha=0.8)
    ax6.tick_params(axis='x', rotation=45)
    ax6.set_title("高薪地区TOP10", fontproperties=chinese_font)
    ax6.set_ylabel("平均年薪", fontproperties=chinese_font)
    st.pyplot(fig6)
    st.info("欧美发达国家薪资水平显著领先")

# 5 分析结论
elif menu == "五、分析结论与行业建议":
    st.header("💡 分析结论与行业建议")
    st.subheader("核心结论")
    st.markdown("""
1. 薪资趋势：2020-2023持续上涨，模型可外推至2026年，但远期仅作参考；
2. 核心变量：工作经验是薪资第一影响因素，专家薪资约为入门3倍；
3. 企业差异：中型企业薪资＞大型＞小型；
4. 地域差异：欧美地区薪资远高于其他国家；
5. 办公：纯线下、全远程薪资优于混合岗位。
""")
    st.subheader("求职者建议")
    st.markdown("""
1. 优先积累项目经验，是薪资提升核心；
2. 不要只盯大厂，中型成长企业薪资溢价更高；
3. 优先线下/纯远程岗位，避开混合远程；
4. 预测2024-2026薪资持续走高，长期深耕数据赛道。
""")
    st.subheader("企业薪酬建议")
    st.markdown("""
1. 搭建分经验阶梯薪资；
2. 大厂优化基础薪资，防止人才流失；
3. 小企业用期权、弹性办公弥补底薪短板；
4. 区分远程/线下岗位薪酬标准。
""")

# 6 薪资预测（年份2020-2026）
elif menu == "六、在线薪资预测工具":
    st.header("🧮 在线薪资预测（支持2020-2026）")
    st.markdown("提示：2024-2026为历史趋势外推，预测结果仅作行业参考")
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            # 年份改为2020~2026
            work_year = st.slider("工作年份", min_value=2020, max_value=2026, value=2023, step=1)
            exp_options = ["入门级(0-2年)", "中级(2-5年)", "高级(5-10年)", "专家级(10年以上)"]
            exp_cn = st.selectbox("工作经验", options=exp_options)
            emp_opt = [emp_dict[t] for t in emp_list]
            emp_cn = st.selectbox("雇佣类型", options=emp_opt)
            job = st.selectbox("岗位名称", options=job_list)
        with col2:
            remote = st.select_slider("远程比例", options=[0,50,100], format_func=lambda x:f"{x}%")
            size_opt = ["小型企业", "中型企业", "大型企业"]
            size_cn = st.selectbox("企业规模", options=size_opt)
            comp_loc = st.selectbox("企业地区", options=loc_list)
            emp_res = st.selectbox("员工居住地", options=res_list)
        submit = st.form_submit_button("预测薪资")
    if submit:
        exp_raw = rev_exp[exp_cn]
        emp_raw = rev_emp[emp_cn]
        size_raw = rev_size[size_cn]
        exp_code = exp_map[exp_raw]
        size_code = size_map[size_raw]
        # 构造输入数据
        input_df = pd.DataFrame({
            "work_year":[work_year],
            "exp_code":[exp_code],
            "remote_ratio":[remote],
            "size_code":[size_code],
            "employment_type":[emp_raw],
            "job_title":[job],
            "company_location":[comp_loc],
            "employee_residence":[emp_res]
        })
        X_in = preprocessor.transform(input_df)
        pred = model.predict(X_in)[0]
        # 下限为数据集真实最低工资，杜绝0
        pred = max(pred, min_salary)
        # 匹配样本查询
        match = df_clean[(df_clean['experience_level']==exp_raw)&
                          (df_clean['company_size']==size_raw)&
                          (df_clean['employment_type']==emp_raw)&
                          (df_clean['job_title']==job)]
        st.success("✅ 预测完成")
        st.metric("预测税前年薪(美元)", f"${pred:,.2f}")
        if len(match) < 5:
            st.warning(f"⚠️ 当前组合仅{len(match)}条历史样本，预测可信度较低；2024-2026无真实数据，仅趋势估算")
        else:
            st.info(f"同岗位同经验样本量：{len(match)}，均值${match['salary_in_usd']:.2f}，中位数${match['salary_in_usd']:.2f}")
        st.markdown("""
说明：
1. 原始数据仅2020-2023，2024-2026为线性趋势外推，仅供行业参考；
2. 采用OneHot编码，避免地区/岗位编码失真，不会出现0薪资；
3. 预测下限为数据集真实最低工资，入门岗位也会输出合理薪资。
""")

# 页脚
st.markdown("---")
st.markdown("© 2026 数据分析师薪资综合分析系统 | 支持2020-2026年份预测")
