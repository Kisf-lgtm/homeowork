import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ========= 云端Linux / 本地Windows 双环境中文字体适配 =========
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

# 分类字段中英文映射
exp_dict = {"EN": "入门级(0-2年)", "MI": "中级(2-5年)", "SE": "高级(5-10年)", "EX": "专家级(10年以上)"}
emp_dict = {"FT": "全职", "CT": "合同工", "PT": "兼职", "FL": "自由职业"}
size_dict = {"S": "小型企业", "M": "中型企业", "L": "大型企业"}
rev_exp = {v: k for k, v in exp_dict.items()}
rev_emp = {v: k for k, v in emp_dict.items()}
rev_size = {v: k for k, v in size_dict.items()}

# ===================== 缓存数据与建模函数【修复核心：OneHot替代LabelEncoder】 =====================
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
    min_real_salary = df_clean['salary_in_usd'].min()  # 记录真实最低薪资，替代0兜底

    # 定序变量有序映射（经验、公司规模为有序分类，可数字编码）
    exp_map = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}
    size_map = {'S': 0, 'M': 1, 'L': 2}
    df_clean['exp_code'] = df_clean['experience_level'].map(exp_map)
    df_clean['size_code'] = df_clean['company_size'].map(size_map)

    # 特征拆分：有序数值 + 无序分类（需要独热编码）
    num_features = ["work_year", "exp_code", "remote_ratio", "size_code"]
    cat_features = ["employment_type", "job_title", "company_location", "employee_residence"]
    X_raw = df_clean[num_features + cat_features]
    y = df_clean['salary_in_usd']

    # 独热转换器，仅对无序分类编码
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
        ],
        remainder="passthrough"
    )
    X_processed = preprocessor.fit_transform(X_raw)

    # 训练线性回归
    model = LinearRegression()
    model.fit(X_processed, y)
    y_pred = model.predict(X_processed)
    r2 = r2_score(y, y_pred)

    # 分组统计（保留原有全部报表）
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

    # 原始分类唯一值，供给预测下拉框
    emp_unique = sorted(df_clean['employment_type'].unique())
    loc_unique = sorted(df_clean['company_location'].unique())
    res_unique = sorted(df_clean['employee_residence'].unique())
    job_unique = sorted(df_clean['job_title'].unique())

    return (df, df_clean, exp_group, size_group, year_group, remote_group, location_group, job_title_group,
            r2, model, preprocessor, exp_map, size_map, min_real_salary,
            emp_unique, loc_unique, res_unique, job_unique)

# 解包数据
(df_raw, df_clean, exp_group, size_group, year_group, remote_group, location_group, job_title_group,
 r2_score_val, model, preprocessor, exp_map, size_map, min_salary,
 emp_list, loc_list, res_list, job_list) = load_and_preprocess_data()

# ===================== 页面标题 & 侧边导航 =====================
st.title("💰 数据分析师薪资分析与预测综合平台")
st.markdown("""
本平台集成**项目说明、数据集介绍、统计分析、可视化图表、回归建模、薪资预测**全流程功能，
基于全球3755条2020-2023年数据分析师真实薪资数据集完成分析与智能预测。
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

# ===================== 1~5页面逻辑完全保留（省略，和你原代码一致，不用改动） =====================
if menu == "一、项目分析目标与预期":
    st.header("🎯 数据分析目标与预期结果")
    st.subheader("核心分析目标")
    st.markdown("""
1. 探索全球数据分析师薪资的整体分布特征与2020-2023年时间趋势
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
        ["company_location", "分类型", "公司所在国家/地区"],
        ["employee_residence", "分类型", "员工居住国家/地区"]
    ]
    df_field = pd.DataFrame(field_data, columns=["字段名称", "变量类型", "取值说明"])
    st.dataframe(df_field, use_container_width=True, hide_index=True)

elif menu == "三、数据总览与统计报表":
    st.header("📊 数据总览 & 多维度统计报表")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("原始数据总量", f"{len(df_raw)} 条")
        st.metric("清洗后有效数据", f"{len(df_clean)} 条")
    with col2:
        st.metric("模型R²得分", f"{r2_score_val:.4f}")
        st.metric("薪资平均水平", f"${df_clean['salary_in_usd'].mean():,.2f}")
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
    st.subheader("6. 不同职位薪资统计（Top 20）")
    st.dataframe(job_title_group.head(20))

elif menu == "四、可视化图表与解读":
    st.header("📷 数据可视化图表及详细说明")
    exp_order = ['EN', 'MI', 'SE', 'EX']
    # 图表1：薪资分布直方图
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.hist(df_clean['salary_in_usd'], bins=30, edgecolor='black', color='#0070C0', alpha=0.7)
    ax1.set_title('数据分析师薪资分布（美元）', fontproperties=chinese_font, fontsize=14)
    ax1.set_xlabel('薪资(美元)', fontproperties=chinese_font, fontsize=12)
    ax1.set_ylabel('样本数量', fontproperties=chinese_font, fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(ax1.get_xticklabels(), fontproperties=chinese_font)
    plt.setp(ax1.get_yticklabels(), fontproperties=chinese_font)
    st.pyplot(fig1)
    st.info("薪资整体近似正态分布，集中在5万-20万美元区间。")
    st.divider()
    # 图表2：经验箱线图
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    box_data = [df_clean[df_clean['experience_level'] == lvl]['salary_in_usd'] for lvl in exp_order]
    ax2.boxplot(box_data, tick_labels=exp_order, patch_artist=True, boxprops=dict(facecolor='#0070C0', alpha=0.7))
    ax2.set_title('不同经验水平薪资箱线图', fontproperties=chinese_font)
    ax2.set_xlabel('经验等级(EN入门 / MI中级 / SE高级 / EX专家)', fontproperties=chinese_font)
    ax2.set_ylabel('薪资(美元)', fontproperties=chinese_font)
    st.pyplot(fig2)
    st.info("薪资随工作经验逐级显著上涨，专家级薪资天花板最高。")
    st.divider()
    # 图表3 公司规模柱状图
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    company_label = ['小型S', '中型M', '大型L']
    bars4 = ax4.bar(company_label, size_group['平均薪资'], color='#2E86AB', alpha=0.8)
    ax4.set_title('不同公司规模平均薪资对比', fontproperties=chinese_font)
    ax4.set_ylabel('平均薪资(美元)', fontproperties=chinese_font)
    for idx, val in enumerate(size_group['平均薪资']):
        ax4.text(idx, val + 2000, f"{int(val)}", ha='center', fontproperties=chinese_font)
    st.pyplot(fig4)
    st.info("中型企业平均薪资高于大厂，小型企业薪资最低。")
    st.divider()
    # 图表4 年度趋势折线
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(year_group['work_year'], year_group['平均薪资'], marker='o', color='#0070C0', linewidth=2)
    ax3.set_title('2020-2023 薪资变化趋势', fontproperties=chinese_font)
    ax3.set_xlabel('年份', fontproperties=chinese_font)
    ax3.set_ylabel('平均薪资(美元)', fontproperties=chinese_font)
    for x, y_val in zip(year_group['work_year'], year_group['平均薪资']):
        ax3.text(x, y_val + 2000, f"{int(y_val)}", ha='center', fontproperties=chinese_font)
    st.pyplot(fig3)
    st.info("2020-2023薪资持续上涨，行业需求旺盛。")
    st.divider()
    # 图表5 远程模式柱状图
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    x5 = [0, 1, 2]
    remote_labels = ["无远程(0)", "混合远程(50)", "全远程(100)"]
    bars5 = ax5.bar(x5, remote_group['平均薪资'], color='#A23B72', alpha=0.8)
    ax5.set_xticks(x5)
    ax5.set_xticklabels(remote_labels, fontproperties=chinese_font)
    ax5.set_title('不同远程模式平均薪资对比', fontproperties=chinese_font)
    ax5.set_ylabel('平均薪资(美元)', fontproperties=chinese_font)
    for bar in bars5:
        h = bar.get_height()
        ax5.text(bar.get_x()+bar.get_width()/2, h+2000, f"{int(h)}", ha='center', fontproperties=chinese_font)
    st.pyplot(fig5)
    st.info("线下坐班、全远程薪资更高，混合远程薪资偏低。")
    st.divider()
    # 图表6 Top10地区
    fig6, ax6 = plt.subplots(figsize=(12, 6))
    top10_loc = location_group.head(10)
    bars6 = ax6.bar(top10_loc['company_location'], top10_loc['平均薪资'], color='#F18F01', alpha=0.8)
    ax6.set_title('Top10高薪地区平均薪资对比', fontproperties=chinese_font)
    ax6.set_ylabel('平均薪资(美元)', fontproperties=chinese_font)
    ax6.tick_params(axis='x', rotation=45)
    st.pyplot(fig6)
    st.info("欧美发达国家薪资显著高于其他地区。")

elif menu == "五、分析结论与行业建议":
    st.header("💡 案例分析结论与建议")
    st.subheader("5.1 核心数据分析结果")
    st.markdown("""
1. **薪资整体特征**：全球数据分析师薪资整体呈正态分布，核心区间为 5 万 - 20 万美元，
2020-2023 年薪资持续上涨，行业发展前景向好。
2. **核心影响因素**：工作经验是薪资第一核心变量，其次是所在地区、公司规模、职位类型。
3. **维度差异规律**：
- 经验维度：专家级薪资是入门级3倍以上；
- 企业规模：中型企业薪资＞大型企业＞小型企业；
- 地区：欧美国家薪资优势极大；
- 办公模式：纯线下、全远程薪资高于混合办公。
""")
    st.divider()
    st.subheader("5.2 求职&企业建议")
    st.subheader("💡 给从业者的求职建议")
    st.markdown("""
1. 优先积累项目经验，经验是薪资提升最关键抓手；
2. 中型成长企业薪资溢价更高，不要只盯着大厂；
3. 高薪岗位优先选择线下或纯全远程，谨慎混合远程岗；
4. 优先选择欧美地区高薪岗位，提升收入上限。
""")
    st.subheader("🏢 给企业的薪资制定建议")
    st.markdown("""
1. 建立经验阶梯薪资体系，留住资深数据人才；
2. 大厂需提升基础现金薪资，防止人才被中型企业挖走；
3. 小企业搭配股权、弹性办公弥补基础薪资劣势；
4. 区分线下/远程岗位薪酬标准，匹配市场行情。
""")

# ===================== 六、在线薪资预测工具【完全修复，解决0薪资bug】 =====================
elif menu == "六、在线薪资预测工具":
    st.header("🧮 在线薪资预测")
    st.markdown("输入工作信息预测税前年薪（美元），所有选项严格匹配2020-2023真实数据集范围")

    with st.form("salary_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            # 修复1：年份锁定数据集区间2020-2023，移除无依据外推
            work_year = st.slider("工作年份", min_value=2020, max_value=2023, value=2023, step=1)
            exp_options = ["入门级(0-2年)", "中级(2-5年)", "高级(5-10年)", "专家级(10年以上)"]
            experience_cn = st.selectbox("工作经验水平", options=exp_options)
            # 雇佣类型下拉（匹配原始数据）
            emp_cn_options = [emp_dict[t] for t in emp_list]
            employment_cn = st.selectbox("雇佣类型", options=emp_cn_options)
            job_title = st.selectbox("职位名称", options=job_list)

        with col2:
            remote_ratio = st.select_slider("远程工作比例", options=[0, 50, 100], value=100, format_func=lambda x: f"{x}%")
            size_options = ["小型企业", "中型企业", "大型企业"]
            company_cn = st.selectbox("公司规模", options=size_options)
            company_location = st.selectbox("公司所在国家/地区", options=loc_list)
            employee_residence = st.selectbox("员工居住国家/地区", options=res_list)

        submit = st.form_submit_button("开始预测薪资", use_container_width=True)

    if submit:
        # 转换输入为原始编码
        exp_raw = rev_exp[experience_cn]
        emp_raw = rev_emp[employment_cn]
        size_raw = rev_size[company_cn]
        exp_code = exp_map[exp_raw]
        size_code = size_map[size_raw]

        # 构造单行输入DataFrame（和训练集特征顺序完全一致）
        input_df = pd.DataFrame({
            "work_year": [work_year],
            "exp_code": [exp_code],
            "remote_ratio": [remote_ratio],
            "size_code": [size_code],
            "employment_type": [emp_raw],
            "job_title": [job_title],
            "company_location": [company_location],
            "employee_residence": [employee_residence]
        })
        # 独热编码转换输入
        X_input = preprocessor.transform(input_df)
        pred_salary = model.predict(X_input)[0]

        # 修复2：兜底改为数据集真实最低薪资，不再强制置0
        pred_salary = max(pred_salary, min_salary)

        # 校验同条件样本量，提示稀疏风险
        match_data = df_clean[
            (df_clean['work_year'] == work_year) &
            (df_clean['experience_level'] == exp_raw) &
            (df_clean['company_size'] == size_raw) &
            (df_clean['employment_type'] == emp_raw) &
            (df_clean['job_title'] == job_title) &
            (df_clean['company_location'] == company_location)
        ]

        st.success("✅ 预测完成！")
        st.metric(label="预测税前年薪（美元）", value=f"${pred_salary:,.2f}")

        # 稀疏样本警告
        if len(match_data) < 5:
            st.warning(f"""
            ⚠️ 风险提示：当前特征组合在原始数据集中仅 {len(match_data)} 条样本，样本量过少，预测结果参考性有限！
            建议更换热门地区/热门职位/全职岗位，提升预测可信度。
            """)
        else:
            st.info(f"""
            📊 同场景真实数据参考：
            匹配样本量：{len(match_data)} 条
            同场景平均薪资：${match_data['salary_in_usd'].mean():,.2f}
            同场景薪资中位数：${match_data['salary_in_usd'].median():,.2f}
            """)

        st.info("""
📝 预测说明：
1. 模型基于2020-2023真实薪资训练，年份仅支持2020~2023，无超期外推；
2. 无序地区/职位采用独热编码，消除原有LabelEncoder逻辑错误；
3. 预测下限为数据集真实最低薪资，不会再出现0美元；
4. 小众地区+冷门职位+入门级组合样本稀少，预测仅供参考。
        """)

# 页脚
st.markdown("---")
st.markdown("© 2026 数据分析师薪资综合分析系统 | 基于2020-2023全球真实薪资数据集")
