import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ========= 中文字体适配（云端Linux/本地Windows） =========
try:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
    chinese_font = FontProperties(fname="/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
except:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    chinese_font = FontProperties(family="SimHei")
plt.rcParams['axes.unicode_minus'] = False

# 页面基础配置
st.set_page_config(
    page_title="数据分析师薪资预测系统",
    page_icon="💰",
    layout="wide"
)

# 数据文件路径
CSV_PATH = "数据分析师工资.csv"

# 中英文映射字典
exp_dict = {"EN": "入门级(0-2年)", "MI": "中级(2-5年)", "SE": "高级(5-10年)", "EX": "专家级(10年以上)"}
emp_dict = {"FT": "全职", "CT": "合同工", "PT": "兼职", "FL": "自由职业"}
size_dict = {"S": "小型企业", "M": "中型企业", "L": "大型企业"}
rev_exp = {v: k for k, v in exp_dict.items()}
rev_emp = {v: k for k, v in emp_dict.items()}
rev_size = {v: k for k, v in size_dict.items()}

# 数据加载与建模缓存函数
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv(CSV_PATH)
    df_clean = df.copy()

    # IQR法剔除薪资极端异常值
    Q1 = df_clean['salary_in_usd'].quantile(0.25)
    Q3 = df_clean['salary_in_usd'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_clean = df_clean[(df_clean['salary_in_usd'] >= lower_bound) & (df_clean['salary_in_usd'] <= upper_bound)]
    min_real_salary = df_clean['salary_in_usd'].min()

    # 有序分类变量数字映射
    exp_map = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}
    size_map = {'S': 0, 'M': 1, 'L': 2}
    df_clean['exp_code'] = df_clean['experience_level'].map(exp_map)
    df_clean['size_code'] = df_clean['company_size'].map(size_map)

    # 区分数值特征与无序分类特征
    num_features = ["work_year", "exp_code", "remote_ratio", "size_code"]
    cat_features = ["employment_type", "job_title", "company_location", "employee_residence"]
    X_raw = df_clean[num_features + cat_features]
    y = df_clean['salary_in_usd']

    # 独热编码处理无序分类（解决LabelEncoder逻辑错误，避免负薪资）
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
        ],
        remainder="passthrough"
    )
    X_processed = preprocessor.fit_transform(X_raw)
    model = LinearRegression()
    model.fit(X_processed, y)
    y_pred = model.predict(X_raw)
    r2 = r2_score(y, y_pred)

    # 各维度分组统计
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

    # 下拉框唯一取值列表
    emp_unique = sorted(df_clean['employment_type'].unique())
    loc_unique = sorted(df_clean['company_location'].unique())
    res_unique = sorted(df_clean['employee_residence'].unique())
    job_unique = sorted(df_clean['job_title'].unique())

    return (df, df_clean, exp_group, size_group, year_group, remote_group, location_group, job_title_group,
            r2, model, preprocessor, exp_map, size_map, min_real_salary,
            emp_unique, loc_unique, res_unique, job_unique)

# 解包模型与统计数据
(df_raw, df_clean, exp_group, size_group, year_group, remote_group, location_group, job_title_group,
 r2_score_val, model, preprocessor, exp_map, size_map, min_salary,
 emp_list, loc_list, res_list, job_list) = load_and_preprocess_data()

# 页面标题
st.title("💰 数据分析师薪资分析与预测综合平台")
st.markdown("本系统基于2020-2023真实薪资数据集，支持2020至2026年薪资预测，2024-2026为行业趋势外推，仅供参考")

# 侧边导航菜单
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

# 1. 项目目标页面
if menu == "一、项目分析目标与预期":
    st.header("🎯 数据分析目标与预期结果")
    st.subheader("核心分析目标")
    st.markdown("""
1. 探索2020-2023全球数据分析师薪资分布与年度变化趋势
2. 量化工作经验、地区、岗位等因素对薪资的影响权重
3. 为求职者、企业薪酬制定提供真实数据支撑
""")
    st.subheader("预期输出")
    st.markdown("""
1. 完成数据清洗，剔除极端薪资异常样本
2. 多维度分组统计，展示不同条件薪资差异
3. 构建线性回归模型，支持历史与未来薪资预测
4. 可视化图表输出完整行业分析报告
""")

# 2. 数据集介绍页面
elif menu == "二、数据集背景介绍":
    st.header("🗃️ 数据集背景介绍")
    st.markdown("原始数据共3755条记录，采集年份2020-2023，包含11个业务字段，统一以美元年薪作为分析目标。")
    # 全部使用英文半角逗号，字符串双引号闭合，无中文标点
    field_data = [
        ["work_year", "数值", "数据年份，原始仅2020-2023，预测支持外推至2026"],
        ["experience_level", "分类", "经验等级：EN入门/MI中级/SE高级/EX专家"],
        ["employment_type", "分类", "雇佣类型：FT全职/CT合同/PT兼职/FL自由职业"],
        ["job_title", "分类", "93类数据相关岗位名称"],
        ["salary_in_usd", "数值", "统一换算后的美元年薪，核心分析字段"],
        ["remote_ratio", "数值", "远程比例：0线下/50混合/100全远程"],
        ["company_size", "分类", "企业规模：S小型/M中型/L大型"],
        ["company_location", "分类", "企业所在国家/地区"],
        ["employee_residence", "分类", "员工居住国家/地区"]
    ]
    df_field = pd.DataFrame(field_data, columns=["字段名称", "数据类型", "字段说明"])
    st.dataframe(df_field, use_container_width=True, hide_index=True)

# 3. 数据总览与统计
elif menu == "三、数据总览与统计报表":
    st.header("📊 数据总览 & 多维度统计报表")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("原始数据总量", f"{len(df_raw)} 条")
        st.metric("清洗后有效数据", f"{len(df_clean)} 条")
    with col2:
        st.metric("模型拟合度R²", f"{r2_score_val:.4f}")
        st.metric("平均年薪(美元)", f"${df_clean['salary_in_usd']:.2f}")

    st.subheader("原始数据前10行")
    st.dataframe(df_raw.head(10), use_container_width=True)
    st.subheader("薪资整体描述统计")
    st.dataframe(df_clean['salary_in_usd'].describe(), use_container_width=True)
    st.divider()
    st.subheader("1. 不同经验薪资统计")
    st.dataframe(exp_group, use_container_width=True)
    st.subheader("2. 不同企业规模薪资统计")
    st.dataframe(size_group, use_container_width=True)
    st.subheader("3. 各年份薪资统计")
    st.dataframe(year_group, use_container_width=True)
    st.subheader("4. 不同远程模式薪资统计")
    st.dataframe(remote_group, use_container_width=True)
    st.subheader("5. 高薪地区TOP20")
    st.dataframe(location_group.head(20), use_container_width=True)
    st.subheader("6. 高薪岗位TOP20")
    st.dataframe(job_title_group.head(20))

# 4. 可视化图表页面
elif menu == "四、可视化图表与解读":
    st.header("📷 数据可视化图表及解读")
    exp_order = ['EN', 'MI', 'SE', 'EX']

    # 薪资分布直方图
    st.subheader("图表1：薪资整体分布直方图")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.hist(df_clean['salary_in_usd'], bins=30, edgecolor='black', color='#0070C0', alpha=0.7)
    ax1.set_title("全球数据分析师薪资分布（美元）", fontproperties=chinese_font)
    ax1.set_xlabel("年薪（美元）", fontproperties=chinese_font)
    ax1.set_ylabel("样本数量", fontproperties=chinese_font)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(ax1.get_xticklabels(), fontproperties=chinese_font)
    plt.setp(ax1.get_yticklabels(), fontproperties=chinese_font)
    st.pyplot(fig1)
    st.info("薪资近似正态分布，大部分岗位年薪集中在5万至20万美元区间")
    st.divider()

    # 经验箱线图
    st.subheader("图表2：不同经验薪资箱线图")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    box_data = [df_clean[df_clean['experience_level'] == lvl]['salary_in_usd'] for lvl in exp_order]
    ax2.boxplot(box_data, tick_labels=exp_order, patch_artist=True, boxprops=dict(facecolor='#0070C0', alpha=0.7))
    ax2.set_title("各经验等级薪资对比", fontproperties=chinese_font)
    ax2.set_xlabel("经验等级", fontproperties=chinese_font)
    ax2.set_ylabel("年薪（美元）", fontproperties=chinese_font)
    plt.setp(ax2.get_xticklabels(), fontproperties=chinese_font)
    plt.setp(ax2.get_yticklabels(), fontproperties=chinese_font)
    st.pyplot(fig2)
    st.info("薪资随工作经验稳步提升，专家级薪资上限远高于入门人员")
    st.divider()

    # 企业规模柱状图
    st.subheader("图表3：不同企业平均薪资对比")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    company_label = ["小型企业", "中型企业", "大型企业"]
    bars4 = ax4.bar(company_label, size_group['平均薪资'], color='#2E86AB', alpha=0.8)
    ax4.set_title("大中小企业平均年薪", fontproperties=chinese_font)
    ax4.set_ylabel("平均年薪（美元）", fontproperties=chinese_font)
    for idx, val in enumerate(size_group['平均薪资']):
        ax4.text(idx, val + 2000, f"{int(val)}", ha='center', fontproperties=chinese_font)
    plt.setp(ax4.get_xticklabels(), fontproperties=chinese_font)
    plt.setp(ax4.get_yticklabels(), fontproperties=chinese_font)
    st.pyplot(fig4)
    st.info("中型企业平均薪资最高，高于大型企业，小型企业薪资偏低")
    st.divider()

    # 年度趋势图
    st.subheader("图表4：2020-2023薪资年度趋势")
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(year_group['work_year'], year_group['平均薪资'], marker='o', linewidth=2, color='#0070C0')
    ax3.set_title("历年薪资变化趋势", fontproperties=chinese_font)
    ax3.set_xlabel("年份", fontproperties=chinese_font)
    ax3.set_ylabel("平均年薪（美元）", fontproperties=chinese_font)
    for x, y_val in zip(year_group['work_year'], year_group['work_year']):
        ax3.text(x, y_val + 2000, f"{int(y_val)}", ha='center', fontproperties=chinese_font)
    plt.setp(ax3.get_xticklabels(), fontproperties=chinese_font)
    plt.setp(ax3.get_yticklabels(), fontproperties=chinese_font)
    st.pyplot(fig3)
    st.info("2020至2023薪资持续上涨，行业需求逐年扩大")
    st.divider()

    # 远程模式柱状图
    st.subheader("图表5：不同远程模式薪资对比")
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    x5 = [0, 1, 2]
    remote_labels = ["线下0%", "混合50%", "全远程100%"]
    bars5 = ax5.bar(x5, remote_group['平均薪资'], color='#A23B72', alpha=0.8)
    ax5.set_xticks(x5)
    ax5.set_xticklabels(remote_labels, fontproperties=chinese_font)
    ax5.set_title("远程办公薪资差异", fontproperties=chinese_font)
    ax5.set_ylabel("平均年薪（美元）", fontproperties=chinese_font)
    for bar in bars5:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2, height + 2000, f"{int(height)}", ha='center', fontproperties=chinese_font)
    plt.setp(ax5.get_yticklabels(), fontproperties=chinese_font)
    st.pyplot(fig5)
    st.info("线下坐班、纯远程岗位薪资更高，混合远程岗位薪资偏低")
    st.divider()

    # 高薪地区柱状图
    st.subheader("图表6：全球高薪地区TOP10")
    fig6, ax6 = plt.subplots(figsize=(12, 6))
    top10_loc = location_group.head(10)
    bars6 = ax6.bar(top10_loc['company_location'], top10_loc['平均薪资'], color='#F18F01', alpha=0.8)
    ax6.set_title("高薪国家/地区排名", fontproperties=chinese_font)
    ax6.set_ylabel("平均年薪（美元）", fontproperties=chinese_font)
    ax6.tick_params(axis='x', rotation=45)
    plt.setp(ax6.get_xticklabels(), fontproperties=chinese_font)
    plt.setp(ax6.get_yticklabels(), fontproperties=chinese_font)
    st.pyplot(fig6)
    st.info("欧美发达国家薪资水平大幅领先其他地区")

# 5. 分析结论页面
elif menu == "五、分析结论与行业建议":
    st.header("💡 分析结论与行业建议")
    st.subheader("一、核心数据结论")
    st.markdown("""
1. 薪资趋势：2020-2023年薪持续上涨，模型可外推至2026年，远期仅作参考；
2. 核心影响因素：工作经验是薪资第一决定要素，其次为地区、企业规模、岗位；
3. 企业差异：中型企业平均薪资 > 大型企业 > 小型企业；
4. 地域差异：欧美国家薪资远高于亚非拉发展地区；
5. 办公模式：线下、纯远程岗位薪资优于混合远程岗位。
""")
    st.subheader("二、给求职者的建议")
    st.markdown("""
1. 优先积累项目实战经验，经验是薪资提升核心突破口；
2. 不要只瞄准大厂，中型成长企业薪资溢价更高；
3. 求职优先选择线下或纯远程岗位，避开混合远程低薪岗；
4. 长期深耕数据赛道，行业薪资持续上行。
""")
    st.subheader("三、给企业薪酬方案建议")
    st.markdown("""
1. 搭建分经验阶梯薪资制度，留住资深数据人才；
2. 大型企业需提高基础年薪，防止核心人才被中型公司挖走；
3. 小型企业可搭配股权、弹性办公弥补底薪劣势；
4. 区分远程/线下岗位薪酬标准，匹配市场行情。
""")

# 6. 薪资预测工具（2020-2026年份）
elif menu == "六、在线薪资预测工具":
    st.header("🧮 在线薪资预测工具")
    st.markdown("提示：原始数据仅2020-2023，2024至2026为趋势外推，预测结果仅作行业参考")
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            work_year = st.slider("工作年份", min_value=2020, max_value=2026, value=2023, step=1)
            exp_options = ["入门级(0-2年)", "中级(2-5年)", "高级(5-10年)", "专家级(10年以上)"]
            exp_cn = st.selectbox("工作经验", options=exp_options)
            emp_opt = [emp_dict[t] for t in emp_list]
            emp_cn = st.selectbox("雇佣类型", options=emp_opt)
            job = st.selectbox("岗位名称", options=job_list)
        with col2:
            remote = st.select_slider("远程比例", options=[0, 50, 100], format_func=lambda x: f"{x}%")
            size_opt = ["小型企业", "中型企业", "大型企业"]
            size_cn = st.selectbox("企业规模", options=size_opt)
            comp_loc = st.selectbox("企业所在地区", options=loc_list)
            emp_res = st.selectbox("员工居住地", options=res_list)
        submit = st.form_submit_button("开始预测薪资")

    if submit:
        exp_raw = rev_exp[exp_cn]
        emp_raw = rev_emp[emp_cn]
        size_raw = rev_size[size_cn]
        exp_code = exp_map[exp_raw]
        size_code = size_map[size_raw]
        input_df = pd.DataFrame({
            "work_year": [work_year],
            "exp_code": [exp_code],
            "remote_ratio": [remote],
            "size_code": [size_code],
            "employment_type": [emp_raw],
            "job_title": [job],
            "company_location": [comp_loc],
            "employee_residence": [emp_res]
        })
        X_in = preprocessor.transform(input_df)
        pred = model.predict(X_in)[0]
        # 兜底为数据集真实最低工资，不会输出0
        pred = max(pred, min_salary)

        # 匹配同条件历史数据
        match_data = df_clean[(df_clean['experience_level'] == exp_raw) &
                              (df_clean['company_size'] == size_raw) &
                              (df_clean['employment_type'] == emp_raw) &
                              (df_clean['job_title'] == job)]
        st.success("✅ 预测完成")
        st.metric("预测税前年薪（美元）", f"${pred:,.2f}")

        if len(match_data) < 5:
            st.warning(f"⚠️ 当前组合在历史数据仅{len(match_data)}条样本，预测可信度较低，2024-2026无真实数据仅供参考")
        else:
            st.info(f"📊 同场景历史数据：样本{len(match_data)}条，平均年薪${match_data['平均薪资']:.2f}，中位数${match_data['中位数']:.2f}")

        st.markdown("""
预测说明：
1. 模型采用独热编码处理地区、岗位，消除编码失真，不会出现0薪资；
2. 预测下限为数据集真实最低工资，入门岗位也会输出合理薪资；
3. 2024-2026属于趋势外推，实际薪资会受市场行情波动影响。
""")

# 页面底部
st.markdown("---")
st.markdown("© 2026 数据分析师薪资综合分析系统 | 支持2020-2026年份预测")
