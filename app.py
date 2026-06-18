import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Font adapt for Linux cloud and Windows local
try:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
    chinese_font = FontProperties(fname="/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
except:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    chinese_font = FontProperties(family="SimHei")
plt.rcParams['axes.unicode_minus'] = False

# Page config
st.set_page_config(
    page_title="Data Analyst Salary Prediction System",
    page_icon="💰",
    layout="wide"
)

CSV_PATH = "数据分析师工资.csv"

# Mapping dict
exp_dict = {"EN": "Entry(0-2y)", "MI": "Mid(2-5y)", "SE": "Senior(5-10y)", "EX": "Expert(10+y)"}
emp_dict = {"FT": "Full-time", "CT": "Contract", "PT": "Part-time", "FL": "Freelance"}
size_dict = {"S": "Small", "M": "Medium", "L": "Large"}
rev_exp = {v: k for k, v in exp_dict.items()}
rev_emp = {v: k for k, v in emp_dict.items()}
rev_size = {v: k for k, v in size_dict.items()}

@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv(CSV_PATH)
    df_clean = df.copy()

    # Remove outlier by IQR
    Q1 = df_clean['salary_in_usd'].quantile(0.25)
    Q3 = df_clean['salary_in_usd'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_clean = df_clean[(df_clean['salary_in_usd'] >= lower_bound) & (df_clean['salary_in_usd'] <= upper_bound)]
    min_real_salary = df_clean['salary_in_usd'].min()

    # Ordinal mapping
    exp_map = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}
    size_map = {'S': 0, 'M': 1, 'L': 2}
    df_clean['exp_code'] = df_clean['experience_level'].map(exp_map)
    df_clean['size_code'] = df_clean['company_size'].map(size_map)

    num_features = ["work_year", "exp_code", "remote_ratio", "size_code"]
    cat_features = ["employment_type", "job_title", "company_location", "employee_residence"]
    X_raw = df_clean[num_features + cat_features]
    y = df_clean['salary_in_usd']

    # OneHot for unordered category
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

    # Group stat
    exp_order = ['EN', 'MI', 'SE', 'EX']
    exp_group = df_clean.groupby('experience_level')['salary_in_usd'].agg(
        count='count', mean_salary='mean', median_salary='median', min_salary='min', max_salary='max'
    ).reset_index()
    exp_group['experience_level'] = pd.Categorical(exp_group['experience_level'], categories=exp_order, ordered=True)
    exp_group = exp_group.sort_values('experience_level')

    size_order = ['S', 'M', 'L']
    size_group = df_clean.groupby('company_size')['salary_in_usd'].agg(
        count='count', mean_salary='mean', median_salary='median', min_salary='min', max_salary='max'
    ).reset_index()
    size_group['company_size'] = pd.Categorical(size_group['company_size'], categories=size_order, ordered=True)
    size_group = size_group.sort_values('company_size')

    year_group = df_clean.groupby('work_year')['salary_in_usd'].agg(
        count='count', mean_salary='mean', median_salary='median', min_salary='min', max_salary='max'
    ).reset_index().sort_values('work_year')

    remote_group = df_clean.groupby('remote_ratio')['salary_in_usd'].agg(
        count='count', mean_salary='mean', median_salary='median', min_salary='min', max_salary='max'
    ).reset_index().sort_values('remote_ratio')

    location_group = df_clean.groupby('company_location')['salary_in_usd'].agg(
        count='count', mean_salary='mean', median_salary='median', min_salary='min', max_salary='max'
    ).reset_index().sort_values('mean_salary', ascending=False)

    job_title_group = df_clean.groupby('job_title')['salary_in_usd'].agg(
        count='count', mean_salary='mean', median_salary='median', min_salary='min', max_salary='max'
    ).reset_index().sort_values('mean_salary', ascending=False)

    emp_unique = sorted(df_clean['employment_type'].unique())
    loc_unique = sorted(df_clean['company_location'].unique())
    res_unique = sorted(df_clean['employee_residence'].unique())
    job_unique = sorted(df_clean['job_title'].unique())

    return (df, df_clean, exp_group, size_group, year_group, remote_group, location_group, job_title_group,
            r2, model, preprocessor, exp_map, size_map, min_real_salary,
            emp_unique, loc_unique, res_unique, job_unique)

(df_raw, df_clean, exp_group, size_group, year_group, remote_group, location_group, job_title_group,
 r2_score_val, model, preprocessor, exp_map, size_map, min_salary,
 emp_list, loc_list, res_list, job_list) = load_and_preprocess_data()

# Page Header
st.title("💰 Data Analyst Salary Analysis & Prediction Platform")
st.markdown("Dataset range:2020-2023, prediction support 2020-2026, data after 2023 is trend extrapolation")

menu = st.sidebar.radio(
    "Menu",
    [
        "1.Project Objective",
        "2.Dataset Intro",
        "3.Data Overview & Stat",
        "4.Visualization",
        "5.Analysis Conclusion & Suggestion",
        "6.Salary Predict Tool"
    ]
)

if menu == "1.Project Objective":
    st.header("Project Analysis Target")
    st.subheader("Core Goals")
    st.markdown("""
1. Explore global data analyst salary distribution and yearly trend from 2020 to 2023
2. Quantify the impact of each feature on salary
3. Provide data reference for job seekers and enterprise salary setting
""")
    st.subheader("Expected Output")
    st.markdown("""
1. Clean dataset by removing abnormal salary values
2. Multi-dimensional group statistics of salary difference
3. Build linear regression prediction model
4. Generate visual analysis report
""")

elif menu == "2.Dataset Intro":
    st.header("Dataset Background")
    st.markdown("Raw data contains 3755 records from 2020 to 2023 with 11 columns")
    # 修复：所有描述字符串添加成对双引号，消除未闭合报错
    field_data = [
        ["work_year", "numeric", "year of data, support predict to 2026"],
        ["experience_level", "category", "EN/MI/SE/EX four experience grade"],
        ["employment_type", "category", "FT/CT/PT/FL work type"],
        ["job_title", "category", "93 kinds of data related jobs"],
        ["salary_in_usd", "numeric", "target salary converted to USD"],
        ["remote_ratio", "numeric", "0/50/100 remote work proportion"],
        ["company_size", "category", "S small/M medium/L large enterprise"],
        ["company_location", "category", "country of company"],
        ["employee_residence", "category", "country of employee"]
    ]
    df_field = pd.DataFrame(field_data, columns=["Column Name", "Type", "Description"])
    st.dataframe(df_field, use_container_width=True, hide_index=True)

elif menu == "3.Data Overview & Stat":
    st.header("Data Overview & Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Raw Data Count", f"{len(df_raw)}")
        st.metric("Cleaned Data Count", f"{len(df_clean)}")
    with col2:
        st.metric("Model R2", f"{r2_score_val:.4f}")
        st.metric("Avg Salary USD", f"${df_clean['salary_in_usd']:.2f}")
    st.subheader("First 10 rows raw data")
    st.dataframe(df_raw.head(10), use_container_width=True)
    st.subheader("Salary describe stat")
    st.dataframe(df_clean['salary_in_usd'].describe(), use_container_width=True)
    st.divider()
    st.subheader("1.Salary group by experience")
    st.dataframe(exp_group, use_container_width=True)
    st.subheader("2.Salary group by company size")
    st.dataframe(size_group, use_container_width=True)
    st.subheader("3.Salary group by year")
    st.dataframe(year_group, use_container_width=True)
    st.subheader("4.Salary group by remote ratio")
    st.dataframe(remote_group, use_container_width=True)
    st.subheader("5.Top20 region salary")
    st.dataframe(location_group.head(20), use_container_width=True)
    st.subheader("6.Top20 job salary")
    st.dataframe(job_title_group.head(20))

elif menu == "4.Visualization":
    st.header("Data Visualization Chart")
    exp_order = ['EN', 'MI', 'SE', 'EX']
    # Salary hist
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.hist(df_clean['salary_in_usd'], bins=30, edgecolor='black', color='#0070C0', alpha=0.7)
    ax1.set_title('Salary Distribution(USD)', fontproperties=chinese_font)
    ax1.set_xlabel('Annual Salary USD', fontproperties=chinese_font)
    ax1.set_ylabel('Record Count', fontproperties=chinese_font)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig1)
    st.info("Salary mainly distributed between 50000-200000 USD, approximate normal distribution")
    st.divider()
    # Experience boxplot
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    box_data = [df_clean[df_clean['experience_level'] == lvl]['salary_in_usd'] for lvl in exp_order]
    ax2.boxplot(box_data, tick_labels=exp_order, patch_artist=True, boxprops=dict(facecolor='#0070C0', alpha=0.7))
    ax2.set_title("Salary By Experience Level", fontproperties=chinese_font)
    ax2.set_xlabel("Experience Grade", fontproperties=chinese_font)
    ax2.set_ylabel("Annual Salary USD", fontproperties=chinese_font)
    st.pyplot(fig2)
    st.info("Salary rises significantly with work experience")
    st.divider()
    # Company size bar
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    company_label = ['Small','Medium','Large']
    bars4 = ax4.bar(company_label, size_group['mean_salary'], color='#2E86AB', alpha=0.8)
    for idx, val in enumerate(size_group['mean_salary']):
        ax4.text(idx, val + 2000, f"{int(val)}", ha='center', fontproperties=chinese_font)
    ax4.set_title("Avg Salary By Company Scale", fontproperties=chinese_font)
    ax4.set_ylabel("Average Salary USD", fontproperties=chinese_font)
    st.pyplot(fig4)
    st.info("Medium companies offer higher average salary than large companies")
    st.divider()
    # Year trend
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(year_group['work_year'], year_group['mean_salary'], marker='o', linewidth=2)
    for x, y_val in zip(year_group['work_year'], year_group['mean_salary']):
        ax3.text(x, y_val + 2000, f"{int(y_val)}", ha='center', fontproperties=chinese_font)
    ax3.set_title("2020-2023 Salary Trend", fontproperties=chinese_font)
    ax3.set_xlabel("Year", fontproperties=chinese_font)
    ax3.set_ylabel("Average Salary USD", fontproperties=chinese_font)
    st.pyplot(fig3)
    st.info("Salary keep growing year by year")
    st.divider()
    # Remote mode
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    x5 = [0, 1, 2]
    remote_labels = ["On-site(0)","Hybrid(50)","Full Remote(100)"]
    bars5 = ax5.bar(x5, remote_group['mean_salary'], color='#A23B72', alpha=0.8)
    ax5.set_xticks(x5)
    ax5.set_xticklabels(remote_labels, fontproperties=chinese_font)
    for bar in bars5:
        h = bar.get_height()
        ax5.text(bar.get_x()+bar.get_width()/2, h+2000, f"{int(h)}", ha='center', fontproperties=chinese_font)
    ax5.set_title("Remote Mode Salary Comparison", fontproperties=chinese_font)
    st.pyplot(fig5)
    st.info("On-site and full remote salary higher than hybrid posts")
    st.divider()
    # Top10 region
    fig6, ax6 = plt.subplots(figsize=(12, 6))
    top10_loc = location_group.head(10)
    bars6 = ax6.bar(top10_loc['company_location'], top10_loc['mean_salary'], color='#F18F01', alpha=0.8)
    ax6.tick_params(axis='x', rotation=45)
    ax6.set_title("Top10 High Salary Regions", fontproperties=chinese_font)
    ax6.set_ylabel("Average Salary USD", fontproperties=chinese_font)
    st.pyplot(fig6)
    st.info("European and American countries have obvious salary advantages")

elif menu == "5.Analysis Conclusion & Suggestion":
    st.header("Analysis Result & Industry Advice")
    st.subheader("Core Conclusion")
    st.markdown("""
1. Salary trend: keep rising from 2020 to 2023, prediction to 2026 is only reference
2. Top impact factor: work experience is the most important variable
3. Company scale: Medium > Large > Small average salary
4. Region gap: developed countries have much higher income
5. Remote policy: on-site / full remote better than hybrid
""")
    st.subheader("Advice for Job Seekers")
    st.markdown("""
1. Accumulate project experience to raise salary quickly
2. Do not only focus on big firms, medium enterprises have higher salary premium
3. Prioritize on-site or full remote positions
4. Data industry has stable upward trend for long-term career
""")
    st.subheader("Advice for Companies")
    st.markdown("""
1. Build tiered salary standard based on experience
2. Large enterprises need to improve base salary to retain core staff
3. Small companies can use equity and flexible schedule to compensate low pay
4. Set differentiated salary for remote and offline roles
""")

elif menu == "6.Salary Predict Tool":
    st.header("Online Salary Predictor")
    st.markdown("Note: data only contains 2020-2023 records, 2024-2026 are trend extrapolation")
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            work_year = st.slider("Work Year", min_value=2020, max_value=2026, value=2023, step=1)
            exp_options = ["Entry(0-2y)", "Mid(2-5y)", "Senior(5-10y)", "Expert(10+y)"]
            exp_cn = st.selectbox("Experience Level", options=exp_options)
            emp_opt = [emp_dict[t] for t in emp_list]
            emp_cn = st.selectbox("Employment Type", options=emp_opt)
            job = st.selectbox("Job Title", options=job_list)
        with col2:
            remote = st.select_slider("Remote Ratio", options=[0,50,100], format_func=lambda x:f"{x}%")
            size_opt = ["Small","Medium","Large"]
            size_cn = st.selectbox("Company Scale", options=size_opt)
            comp_loc = st.selectbox("Company Location", options=loc_list)
            emp_res = st.selectbox("Employee Residence", options=res_list)
        submit = st.form_submit_button("Predict Salary")
    if submit:
        exp_raw = rev_exp[exp_cn]
        emp_raw = rev_emp[emp_cn]
        size_raw = rev_size[size_cn]
        exp_code = exp_map[exp_raw]
        size_code = size_map[size_raw]
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
        # Limit salary to real minimum to avoid zero
        pred = max(pred, min_salary)
        match = df_clean[(df_clean['experience_level']==exp_raw)&
                          (df_clean['company_size']==size_raw)&
                          (df_clean['employment_type']==emp_raw)&
                          (df_clean['job_title']==job)]
        st.success("Prediction Complete")
        st.metric("Predicted Annual Salary(USD)", f"${pred:,.2f}")
        if len(match) < 5:
            st.warning(f"Warning: only {len(match)} matched records in dataset, prediction reference value is low")
        else:
            st.info(f"Matched records: {len(match)}, average ${match['mean_salary']:.2f}, median ${match['salary_in_usd']:.2f}")
        st.markdown("""
Tips:
1. Original dataset only covers 2020-2023
2. Unordered fields use OneHotEncoder to avoid model bias
3. Prediction floor set to real minimum salary in dataset, no zero output
""")

# Footer
st.markdown("---")
st.markdown("© 2026 Data Analyst Salary Analysis System | Support 2020-2026 salary prediction")
