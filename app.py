import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os

# ===================== 全局基础配置 =====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False
st.set_page_config(
    page_title="数据分析师薪资预测系统",
    page_icon="💰",
    layout="wide"
)

# ========== 请务必修改为你本地CSV真实路径 ==========
# 原来错误写法
# CSV_PATH = r"C:\Users\mao16\Desktop\案例报告数据\数据分析师工资.csv"
# 修改为云端可用写法
CSV_PATH = "数据分析师工资.csv"

# 分类字段中英文映射
exp_dict = {"EN": "入门级(0-2年)", "MI": "中级(2-5年)", "SE": "高级(5-10年)", "EX": "专家级(10年以上)"}
emp_dict = {"FT": "全职", "CT": "合同工", "PT": "兼职", "FL": "自由职业"}
size_dict = {"S": "小型企业", "M": "中型企业", "L": "大型企业"}
rev_exp = {v: k for k, v in exp_dict.items()}
rev_emp = {v: k for k, v in emp_dict.items()}
rev_size = {v: k for k, v in size_dict.items()}

# ===================== 缓存数据与模型（仅加载一次） =====================
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

    # 标签编码（加入地区编码）
    le_experience = LabelEncoder()
    le_employment = LabelEncoder()
    le_company = LabelEncoder()
    le_location = LabelEncoder()

    df_encoded = df_clean.copy()
    df_encoded['experience_level'] = le_experience.fit_transform(df_encoded['experience_level'])
    df_encoded['employment_type'] = le_employment.fit_transform(df_encoded['employment_type'])
    df_encoded['company_size'] = le_company.fit_transform(df_encoded['company_size'])
    df_encoded['company_location'] = le_location.fit_transform(df_encoded['company_location'])

    # 训练线性回归模型（特征加入地区）
    feature_cols = ['work_year', 'experience_level', 'employment_type', 'remote_ratio', 'company_size', 'company_location']
    X = df_encoded[feature_cols]
    y = df_encoded['salary_in_usd']
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)


# joblib.dump(model, "salary_prediction_model.pkl")
# joblib.dump(le_experience, "experience_level_label_encoder.pkl")
# joblib.dump(le_employment, "employment_type_label_encoder.pkl")
# joblib.dump(le_company, "company_size_label_encoder.pkl")
# joblib.dump(le_location, "location_label_encoder.pkl")

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

    # 新增地区分组统计
    location_group = df_clean.groupby('company_location')['salary_in_usd'].agg(
        样本量='count', 平均薪资='mean', 中位数='median', 最低='min', 最高='max'
    ).reset_index().sort_values('平均薪资', ascending=False)

    reg_result = pd.DataFrame({
        '特征': ['工作年份', '经验水平', '雇佣类型', '远程比例', '公司规模', '公司所在地区'],
        '回归系数': model.coef_
    }).sort_values('回归系数', ascending=False)

    return df, df_clean, exp_group, size_group, year_group, remote_group, location_group, reg_result, r2, \
           model, le_experience, le_employment, le_company, le_location

# 执行数据加载
df_raw, df_clean, exp_group, size_group, year_group, remote_group, location_group, reg_result, r2_score_val, \
model, le_experience, le_employment, le_company, le_location = load_and_preprocess_data()

# ===================== 页面主标题 & 侧边导航 =====================
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

# ===================== 1. 项目分析目标与预期结果 =====================
if menu == "一、项目分析目标与预期":
    st.header("2.2 数据分析目标与预期结果")
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
    st.header("三、项目需求分析 — 3.1 数据集背景介绍")
    st.markdown("""
本次分析使用**全球数据分析师薪资数据集**，原始数据共 3755 条样本，覆盖 2020-2023 年，包含 11 个核心字段。
""")

    # 字段说明表格（补充地区字段）
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
    with col2:
        st.metric("模型决定系数 R²", f"{r2_score_val:.4f}")

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

# ===================== 4. 可视化图表 + 对应解读 =====================
elif menu == "四、可视化图表与解读":
    st.header("📷 数据可视化图表及详细说明")
    exp_order = ['EN', 'MI', 'SE', 'EX']

    # 图表1
    st.subheader("图表1：薪资分布直方图")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.hist(df_clean['salary_in_usd'], bins=30, edgecolor='black', color='#0070C0', alpha=0.7)
    ax1.set_title('数据分析师薪资分布（美元）')
    ax1.set_xlabel('薪资(美元)')
    ax1.set_ylabel('样本数量')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig1)
    st.info("""
**图表说明**：数据分析师薪资整体呈近似正态分布，大部分样本薪资集中在 5 万 - 20 万美元区间，
峰值出现在 10-15 万美元区间，整体分布符合职场薪资的常规特征。
""")
    st.divider()

    # 图表2
    st.subheader("图表2：不同经验水平薪资箱线图")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    box_data = [df_clean[df_clean['experience_level'] == level]['salary_in_usd'] for level in exp_order]
    ax2.boxplot(box_data, tick_labels=exp_order, patch_artist=True, boxprops=dict(facecolor='#0070C0', alpha=0.7))
    ax2.set_title('不同经验水平薪资箱线图')
    ax2.set_xlabel('经验等级(EN入门 / MI中级 / SE高级 / EX专家)')
    ax2.set_ylabel('薪资(美元)')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig2)
    st.info("""
**图表说明**：薪资水平与工作经验呈现显著的正相关关系，入门级 (EN) 平均薪资仅约 6 万美元，
专家级 (EX) 平均薪资可达 20 万美元以上，同时薪资的离散程度也随经验提升而增大，
高级别岗位的薪资天花板更高。
""")
    st.divider()

    # 图表3
    st.subheader("图表3：不同公司规模平均薪资柱状图")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    company_label = ['小型S', '中型M', '大型L']
    ax4.bar(company_label, size_group['平均薪资'], color='#2E86AB', alpha=0.8)
    ax4.set_title('不同公司规模平均薪资对比')
    ax4.set_ylabel('平均薪资(美元)')
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    for idx, val in enumerate(size_group['平均薪资']):
        ax4.text(idx, val + 2000, f"{int(val)}", ha='center')
    st.pyplot(fig4)
    st.info("""
**图表说明**：公司规模与薪资水平呈正相关，大型公司 (L) 平均薪资约 14.5 万美元，
中型公司 (M) 约 13.8 万美元，小型公司 (S) 仅约 10.5 万美元，
大型企业的薪资竞争力显著高于中小微企业。
""")
    st.divider()

    # 图表4
    st.subheader("图表4：2020-2023年度薪资趋势折线图")
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(year_group['work_year'], year_group['平均薪资'], marker='o', color='#0070C0', linewidth=2)
    ax3.set_title('2020-2023 薪资变化趋势')
    ax3.set_xlabel('年份')
    ax3.set_ylabel('平均薪资(美元)')
    ax3.grid(linestyle='--', alpha=0.7)
    for x, y_val in zip(year_group['work_year'], year_group['平均薪资']):
        ax3.text(x, y_val + 2000, f"{int(y_val)}", ha='center')
    st.pyplot(fig3)
    st.info("""
**图表说明**：2020-2023 年数据分析师平均薪资呈持续上涨趋势，
从 2020 年的约 9.5 万美元上涨至 2023 年的约 14.8 万美元，
4 年涨幅超 55%，行业薪资增长势头强劲。
""")
    st.divider()

    # 图表5
    st.subheader("图表5：不同远程模式平均薪资柱状图")
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    x5 = [0, 1, 2]
    remote_labels = ["无远程(0)", "混合远程(50)", "全远程(100)"]
    bars = ax5.bar(x5, remote_group['平均薪资'], color='#A23B72', alpha=0.8)
    ax5.set_xticks(x5)
    ax5.set_xticklabels(remote_labels)
    ax5.set_title('不同远程模式平均薪资对比')
    ax5.set_ylabel('平均薪资(美元)')
    ax5.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2, height + 2000, f"{int(height)}", ha='center')
    st.pyplot(fig5)
    st.info("""
**图表说明**：全远程 (100%) 岗位的平均薪资显著高于无远程 (0%) 和混合远程 (50%) 岗位，
全远程岗位平均薪资约 14.2 万美元，无远程岗位仅约 13.1 万美元，
远程办公模式与更高的薪资水平存在关联。
""")
    st.divider()

    # 图表6：新增地区薪资对比
    st.subheader("图表6：Top10高薪地区平均薪资对比")
    fig6, ax6 = plt.subplots(figsize=(12, 6))
    top10_loc = location_group.head(10)
    ax6.bar(top10_loc['company_location'], top10_loc['平均薪资'], color='#F18F01', alpha=0.8)
    ax6.set_title('Top10高薪地区平均薪资对比')
    ax6.set_ylabel('平均薪资(美元)')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig6)
    st.info("""
**图表说明**：不同地区薪资差异显著，美国、瑞士等发达国家的平均薪资远高于其他地区，
地区经济水平与薪资水平存在强相关性。
""")

# ===================== 5. 分析结论与行业建议 =====================
elif menu == "五、分析结论与行业建议":
    st.header("五、案例分析结论与建议")
    st.subheader("5.1 核心数据分析结果")
    st.markdown("""
1. **薪资整体特征**：全球数据分析师薪资整体呈正态分布，核心区间为 5 万 - 20 万美元，
2020-2023 年薪资持续上涨，4 年涨幅超 55%，行业发展前景向好。

2. **核心影响因素**：通过线性回归分析，对薪资影响程度从高到低依次为：
**工作经验水平 > 公司所在地区 > 公司规模 > 工作年份 > 远程比例 > 雇佣类型**，
其中工作经验是影响薪资的最核心因素，地区因素紧随其后。

3. **维度差异规律**：
- 经验维度：专家级岗位平均薪资是入门级的 3 倍以上，薪资天花板随经验提升显著抬高；
- 企业维度：大型企业薪资竞争力显著高于中小微企业，平均薪资差距超 40%；
- 地区维度：发达国家/地区薪资水平显著高于发展中国家，美国、瑞士等地薪资优势明显；
- 办公模式：全远程岗位薪资高于非远程岗位，远程办公已成为行业高薪岗位的常见特征。
""")

    st.divider()
    st.subheader("5.2 结论与建议")
    st.subheader("💡 给数据从业者的求职建议")
    st.markdown("""
1. **优先积累核心工作经验**：工作经验是薪资提升的第一核心要素，建议从业者优先深耕行业，
积累项目经验与专业能力，通过经验提升实现薪资的跨越式增长。

2. **优先选择大型企业平台与高薪地区**：大型企业和发达国家/地区的薪资水平更高，
建议求职者在职业发展中优先考虑大型企业、一线城市或高薪地区的岗位机会。

3. **关注远程办公岗位机会**：全远程岗位不仅薪资水平更高，同时能提供更灵活的工作模式，
建议具备独立工作能力的从业者，优先关注全远程的高薪岗位机会。

4. **持续跟进行业发展趋势**：数据行业薪资持续上涨，从业者需持续学习新技术、新方法，
保持自身的行业竞争力，匹配行业的薪资增长节奏。
""")

    st.subheader("🏢 给企业的薪资制定建议")
    st.markdown("""
1. **建立基于经验的阶梯式薪资体系**：工作经验是影响员工价值的核心因素，建议企业建立清晰的
经验 - 薪资对应体系，为不同经验水平的员工提供匹配的薪资待遇，降低核心人才流失率。

2. **参考地区薪资水平制定薪酬策略**：地区差异对薪资影响显著，建议企业根据所在地区的行业水平，
制定有竞争力的薪酬方案，同时可以通过远程办公模式，扩大人才招聘的地域范围。

3. **提升中小微企业的薪资竞争力**：中小微企业薪资水平显著低于大型企业，在人才竞争中处于劣势，
建议中小微企业优化薪资结构，通过灵活的薪酬体系与职业发展机会，提升对人才的吸引力。

4. **灵活应用远程办公模式**：全远程岗位已成为行业高薪岗位的常见特征，建议企业根据岗位特性，
灵活采用远程办公模式，不仅能提升岗位的薪资竞争力，同时能扩大人才招聘的地域范围。
""")

# ===================== 6. 在线薪资预测工具（含地区） =====================
elif menu == "六、在线薪资预测工具":
    st.header("🎯 在线薪资预测")
    st.markdown("输入个人及工作信息，智能预测数据分析师税前年薪（美元）")

    with st.form("salary_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            work_year = st.slider("工作年份", min_value=2020, max_value=2025, value=2023, step=1)
            exp_cn = [exp_dict[item] for item in le_experience.classes_]
            experience_cn = st.selectbox("工作经验水平", options=exp_cn)
            emp_cn = [emp_dict[item] for item in le_employment.classes_]
            employment_cn = st.selectbox("雇佣类型", options=emp_cn)

        with col2:
            remote_ratio = st.select_slider("远程工作比例", options=[0, 50, 100], value=100, format_func=lambda x: f"{x}%")
            size_cn = [size_dict[item] for item in le_company.classes_]
            company_cn = st.selectbox("公司规模", options=size_cn)
            location = st.selectbox("公司所在国家/地区", options=le_location.classes_)

        submit = st.form_submit_button("开始预测薪资", use_container_width=True)

    if submit:
        exp_raw = rev_exp[experience_cn]
        emp_raw = rev_emp[employment_cn]
        size_raw = rev_size[company_cn]

        exp_code = le_experience.transform([exp_raw])[0]
        emp_code = le_employment.transform([emp_raw])[0]
        com_code = le_company.transform([size_raw])[0]
        loc_code = le_location.transform([location])[0]

        input_features = np.array([[work_year, exp_code, emp_code, remote_ratio, com_code, loc_code]])
        predicted_salary = model.predict(input_features)[0]

        st.success("✅ 预测完成！")
        st.metric(label="预测税前年薪（美元）", value=f"${predicted_salary:,.2f}")

        st.info("""
说明：
1. 预测结果基于历史数据训练的线性回归模型，仅供参考
2. 薪资单位为美元，为税前年薪
3. 可修改参数对比不同场景薪资差异
        """)

# 页脚
st.markdown("---")
st.markdown("© 2025 数据分析师薪资综合分析系统 | 全流程数据分析 + 可视化 + 智能预测")
