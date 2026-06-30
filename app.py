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

# ========= 1. 全局配置 =========
# 中文字体适配（自动切换）
try:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
    chinese_font = FontProperties(fname="/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
except:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    chinese_font = FontProperties(family="SimHei")
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="数据分析师薪资预测系统", page_icon="💰", layout="wide")

CSV_PATH = "数据分析师工资.csv"

# 分类映射
exp_dict = {"EN": "入门级(0-2年)", "MI": "中级(2-5年)", "SE": "高级(5-10年)", "EX": "专家级(10年以上)"}
emp_dict = {"FT": "全职", "CT": "合同工", "PT": "兼职", "FL": "自由职业"}
size_dict = {"S": "小型企业", "M": "中型企业", "L": "大型企业"}
rev_exp = {v: k for k, v in exp_dict.items()}
rev_emp = {v: k for k, v in emp_dict.items()}
rev_size = {v: k for k, v in size_dict.items()}
exp_order = ['EN', 'MI', 'SE', 'EX']
size_order = ['S', 'M', 'L']

# 图表基础尺寸配置（宽, 高）
BASE_FIGSIZE = {
    'default': (8, 4),
    'small': (6, 3.5),
    'medium': (9, 5),
    'large': (10, 6),
    'heatmap': (9, 6),
}

# ========= 2. 辅助绘图函数 =========
def plot_histogram(data, title, xlabel, ylabel, bins=30, color='#0070C0', figsize=None):
    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE['default'])
    ax.hist(data, bins=bins, edgecolor='black', color=color, alpha=0.7)
    ax.set_title(title, fontproperties=chinese_font, fontsize=14)
    ax.set_xlabel(xlabel, fontproperties=chinese_font, fontsize=12)
    ax.set_ylabel(ylabel, fontproperties=chinese_font, fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(ax.get_xticklabels(), fontproperties=chinese_font)
    plt.setp(ax.get_yticklabels(), fontproperties=chinese_font)
    fig.tight_layout()
    return fig

def plot_boxplot(data_list, labels, title, xlabel, ylabel, color='#0070C0', figsize=None):
    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE['default'])
    ax.boxplot(data_list, tick_labels=labels, patch_artist=True,
               boxprops=dict(facecolor=color, alpha=0.7))
    ax.set_title(title, fontproperties=chinese_font, fontsize=14)
    ax.set_xlabel(xlabel, fontproperties=chinese_font, fontsize=12)
    ax.set_ylabel(ylabel, fontproperties=chinese_font, fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(ax.get_xticklabels(), fontproperties=chinese_font)
    plt.setp(ax.get_yticklabels(), fontproperties=chinese_font)
    fig.tight_layout()
    return fig

def plot_bar(x, height, labels, title, ylabel, color='#2E86AB', figsize=None, value_format="{:.0f}"):
    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE['default'])
    bars = ax.bar(x, height, color=color, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontproperties=chinese_font)
    ax.set_title(title, fontproperties=chinese_font, fontsize=14)
    ax.set_ylabel(ylabel, fontproperties=chinese_font, fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(ax.get_yticklabels(), fontproperties=chinese_font)
    for bar, val in zip(bars, height):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02*max(height),
                value_format.format(val), ha='center', fontproperties=chinese_font)
    fig.tight_layout()
    return fig

def plot_line(x, y, title, xlabel, ylabel, color='#0070C0', marker='o', figsize=None, y2=None, y2_label=None):
    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE['default'])
    ax.plot(x, y, marker=marker, color=color, linewidth=2, label='平均薪资')
    if y2 is not None:
        ax.plot(x, y2, marker='s', color='#2E86AB', linestyle='--', label=y2_label)
    ax.set_title(title, fontproperties=chinese_font, fontsize=14)
    ax.set_xlabel(xlabel, fontproperties=chinese_font, fontsize=12)
    ax.set_ylabel(ylabel, fontproperties=chinese_font, fontsize=12)
    ax.grid(linestyle='--', alpha=0.7)
    plt.setp(ax.get_xticklabels(), fontproperties=chinese_font)
    plt.setp(ax.get_yticklabels(), fontproperties=chinese_font)
    if y2 is not None:
        ax.legend(prop=chinese_font)
    fig.tight_layout()
    return fig

def plot_correlation(corr_matrix, labels, title, figsize=None):
    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE['heatmap'])
    im = ax.imshow(corr_matrix, cmap="Blues", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontproperties=chinese_font)
    ax.set_yticklabels(labels, fontproperties=chinese_font)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title(title, fontproperties=chinese_font, fontsize=16)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr_matrix.iloc[i,j]:.2f}", ha="center", va="center",
                    fontproperties=chinese_font, color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig

def plot_qr_coefficients(qr_result, features, figsize=None):
    """绘制分位数回归系数散点连线图"""
    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE['large'])
    q25 = qr_result['25%分位数系数'].tolist()
    q50 = qr_result['50%分位数系数'].tolist()
    q75 = qr_result['75%分位数系数'].tolist()
    y_pos = np.arange(len(features))

    ax.hlines(y=y_pos, xmin=q25, xmax=q75, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
    ax.scatter(q25, y_pos - 0.15, color='#1f77b4', label='25% 低薪分位数', s=90, zorder=5, marker='o')
    ax.scatter(q50, y_pos, color='#ff7f0e', label='50% 中等薪资分位数', s=120, zorder=5, marker='D')
    ax.scatter(q75, y_pos + 0.15, color='#2ca02c', label='75% 高薪分位数', s=90, zorder=5, marker='^')
    ax.axvline(x=0, color='red', linestyle='-', linewidth=1, alpha=0.3, label='无影响基准')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontproperties=chinese_font, fontsize=12)
    ax.set_xlabel('标准化回归系数（Beta权重）\n（数值越大该特征对薪资提升越强，负值代表压低薪资）',
                  fontproperties=chinese_font, fontsize=12)
    ax.set_title('各特征在不同薪资分位数下的标准化影响系数深度对比', fontproperties=chinese_font, fontsize=16)
    ax.legend(prop=chinese_font, loc='lower right')
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.autoscale(axis='x')
    fig.tight_layout()
    return fig

def plot_heatmap_2d(pivot_table, xlabels, ylabels, title, figsize=None):
    fig, ax = plt.subplots(figsize=figsize or BASE_FIGSIZE['heatmap'])
    im = ax.imshow(pivot_table.values, cmap="Blues")
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels, fontproperties=chinese_font)
    ax.set_yticklabels(ylabels, fontproperties=chinese_font)
    ax.set_title(title, fontproperties=chinese_font, fontsize=16)
    for i in range(len(ylabels)):
        for j in range(len(xlabels)):
            val = int(pivot_table.iloc[i,j])
            ax.text(j, i, str(val), ha="center", va="center", fontproperties=chinese_font)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig

def show_chart_with_details(fig, info_text, code_str, zoom_factor=1.5):
    """显示图表，并附带说明、代码展示和放大查看功能"""
    st.pyplot(fig)
    st.info(info_text)
    with st.expander("📄 查看此图表的代码"):
        st.code(code_str, language="python")
    # 放大查看
    with st.expander("🔍 放大查看（更大尺寸）"):
        # 重新生成图表（乘上缩放系数）
        # 由于我们不能直接修改fig的尺寸，需要重新绘制，所以这里使用缓存机制：我们将绘图函数作为参数传递，但实现复杂。
        # 简便方法：在外部调用时传入放大尺寸，这里我们只负责显示，不负责生成。
        # 因此我们调整设计：在调用show_chart_with_details时，如果zoom_factor>1，会重新绘制一次大图？
        # 为简化，我们这里只显示一个提醒，实际放大功能由调用者实现（见下文）。
        st.warning("点击展开后，下方将显示放大版本（请等待重新绘制）")
        # 由于该函数不包含绘图逻辑，我们需在调用处处理。
        # 因此实际使用中，我们会把绘图代码和放大绘图都放在外部。
        # 这里留空，实际由外层代码处理。

# 为了简化，我们放弃使用这个通用函数，而是采用更直接的方式：在每个图表后面单独处理。
# 但为了代码优化，我们仍然使用辅助绘图函数，并在外层直接调用两次（正常和放大）。
# 下面我们采用更简洁的方式：在显示正常图后，用一个expander显示放大图（重新调用绘图函数并传更大的figsize）。

# ========= 3. 数据加载与建模 =========
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv(CSV_PATH)
    df_clean = df.copy()

    Q1 = df_clean['salary_in_usd'].quantile(0.25)
    Q3 = df_clean['salary_in_usd'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df_clean = df_clean[(df_clean['salary_in_usd'] >= lower) & (df_clean['salary_in_usd'] <= upper)]

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

    # 分组统计
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
        样本量='count', 平均薪资='mean', 中位数='median', 最低='min', 最高='max', 薪资标准差='std'
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

    return (df, df_clean, exp_group, size_group, year_group, remote_group, location_group,
            reg_result, r2, model, exp_map, le_employment, size_map, le_location, le_job,
            qr_result, corr_matrix, qr_model_25, qr_model_50, qr_model_75, scaler)

# 解包数据
(df_raw, df_clean, exp_group, size_group, year_group, remote_group, location_group,
 reg_result, r2_score_val, model, exp_map, le_employment, size_map, le_location, le_job,
 qr_result, corr_matrix, qr_model_25, qr_model_50, qr_model_75, scaler) = load_and_preprocess_data()

# ========= 4. 页面布局 =========
st.title("💰 数据分析师薪资分析与预测综合平台")
st.markdown("集成**项目说明、数据集介绍、统计分析、可视化图表、回归建模、薪资预测**全流程功能。")

menu = st.sidebar.radio(
    "📑 功能导航",
    ["一、项目分析目标与预期", "二、数据集背景介绍", "三、数据总览与统计报表",
     "四、基础可视化图表与解读", "五、高级可视化分析", "六、分析结论与行业建议",
     "七、在线薪资预测工具", "八、交互式薪资探索器"]
)

# ========= 5. 各菜单页面 =========
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

elif menu == "二、数据集背景介绍":
    st.header("🗃️ 项目需求分析 — 数据集背景介绍")
    st.markdown("本次分析使用**全球数据分析师薪资数据集**，原始数据共 3755 条样本，覆盖 2020-2023 年，包含 11 个核心字段。")
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

# ========= 四、基础可视化图表与解读 =========
elif menu == "四、基础可视化图表与解读":
    st.header("📷 基础可视化图表及详细说明")

    # 辅助函数：显示图表+说明+代码+放大
    def display_chart(fig, info, code, zoom_fig_func):
        st.pyplot(fig)
        st.info(info)
        with st.expander("📄 查看此图表的代码"):
            st.code(code, language="python")
        with st.expander("🔍 放大查看"):
            st.pyplot(zoom_fig_func())
            st.caption("（放大版，尺寸约为原始1.5倍）")

    # 图表1：薪资分布直方图
    st.subheader("图表1：薪资分布直方图")
    fig1 = plot_histogram(df_clean['salary_in_usd'],
                          title='数据分析师薪资分布（美元）',
                          xlabel='薪资(美元)', ylabel='样本数量',
                          figsize=BASE_FIGSIZE['medium'])
    code1 = '''
fig = plot_histogram(df_clean['salary_in_usd'],
                     title='数据分析师薪资分布（美元）',
                     xlabel='薪资(美元)', ylabel='样本数量',
                     figsize=BASE_FIGSIZE['medium'])
st.pyplot(fig)
'''
    info1 = "**图表说明**：数据分析师薪资整体呈近似正态分布，大部分样本薪资集中在 5 万 - 20 万美元区间，峰值出现在 10-15 万美元区间。"
    def zoom1():
        return plot_histogram(df_clean['salary_in_usd'],
                              title='数据分析师薪资分布（美元）',
                              xlabel='薪资(美元)', ylabel='样本数量',
                              figsize=(BASE_FIGSIZE['medium'][0]*1.5, BASE_FIGSIZE['medium'][1]*1.5))
    display_chart(fig1, info1, code1, zoom1)
    st.divider()

    # 图表2：经验水平箱线图
    st.subheader("图表2：不同经验水平薪资箱线图")
    box_data = [df_clean[df_clean['experience_level'] == level]['salary_in_usd'] for level in exp_order]
    fig2 = plot_boxplot(box_data, exp_order,
                        title='不同经验水平薪资箱线图',
                        xlabel='经验等级(EN入门 / MI中级 / SE高级 / EX专家)',
                        ylabel='薪资(美元)',
                        figsize=BASE_FIGSIZE['medium'])
    code2 = '''
box_data = [df_clean[df_clean['experience_level'] == level]['salary_in_usd'] for level in exp_order]
fig = plot_boxplot(box_data, exp_order,
                   title='不同经验水平薪资箱线图',
                   xlabel='经验等级(EN入门 / MI中级 / SE高级 / EX专家)',
                   ylabel='薪资(美元)',
                   figsize=BASE_FIGSIZE['medium'])
st.pyplot(fig)
'''
    info2 = "**图表说明**：薪资水平与工作经验呈现显著的正相关关系，入门级 (EN) 平均薪资最低，随经验逐级上涨，专家级 (EX) 薪资最高。"
    def zoom2():
        return plot_boxplot(box_data, exp_order,
                            title='不同经验水平薪资箱线图',
                            xlabel='经验等级(EN入门 / MI中级 / SE高级 / EX专家)',
                            ylabel='薪资(美元)',
                            figsize=(BASE_FIGSIZE['medium'][0]*1.5, BASE_FIGSIZE['medium'][1]*1.5))
    display_chart(fig2, info2, code2, zoom2)
    st.divider()

    # 图表3：公司规模平均薪资柱状图
    st.subheader("图表3：不同公司规模平均薪资柱状图")
    company_label = ['小型S', '中型M', '大型L']
    fig3 = plot_bar(range(len(size_group)), size_group['平均薪资'], company_label,
                    title='不同公司规模平均薪资对比',
                    ylabel='平均薪资(美元)',
                    color='#2E86AB',
                    figsize=BASE_FIGSIZE['medium'])
    code3 = '''
company_label = ['小型S', '中型M', '大型L']
fig = plot_bar(range(len(size_group)), size_group['平均薪资'], company_label,
               title='不同公司规模平均薪资对比',
               ylabel='平均薪资(美元)',
               color='#2E86AB',
               figsize=BASE_FIGSIZE['medium'])
st.pyplot(fig)
'''
    info3 = "**图表说明**：中型企业 (M) 平均薪资最高（约14.0万美元），大型企业 (L) 次之（约11.2万美元），小型企业 (S) 最低（约7.6万美元）。"
    def zoom3():
        return plot_bar(range(len(size_group)), size_group['平均薪资'], company_label,
                        title='不同公司规模平均薪资对比',
                        ylabel='平均薪资(美元)',
                        color='#2E86AB',
                        figsize=(BASE_FIGSIZE['medium'][0]*1.5, BASE_FIGSIZE['medium'][1]*1.5))
    display_chart(fig3, info3, code3, zoom3)
    st.divider()

    # 图表4：年度薪资趋势
    st.subheader("图表4：2020-2023年度薪资趋势折线图")
    fig4 = plot_line(year_group['work_year'], year_group['平均薪资'],
                     title='2020-2023 薪资变化趋势',
                     xlabel='年份', ylabel='平均薪资(美元)',
                     figsize=BASE_FIGSIZE['medium'])
    # 添加数据标签
    for x, y in zip(year_group['work_year'], year_group['平均薪资']):
        fig4.axes[0].text(x, y+2000, f"{int(y)}", ha='center', fontproperties=chinese_font)
    fig4.tight_layout()
    code4 = '''
fig = plot_line(year_group['work_year'], year_group['平均薪资'],
                title='2020-2023 薪资变化趋势',
                xlabel='年份', ylabel='平均薪资(美元)',
                figsize=BASE_FIGSIZE['medium'])
for x, y in zip(year_group['work_year'], year_group['平均薪资']):
    fig.axes[0].text(x, y+2000, f"{int(y)}", ha='center', fontproperties=chinese_font)
fig.tight_layout()
st.pyplot(fig)
'''
    info4 = "**图表说明**：2020-2023 年数据分析师平均薪资持续上涨，行业发展前景向好。"
    def zoom4():
        fig = plot_line(year_group['work_year'], year_group['平均薪资'],
                        title='2020-2023 薪资变化趋势',
                        xlabel='年份', ylabel='平均薪资(美元)',
                        figsize=(BASE_FIGSIZE['medium'][0]*1.5, BASE_FIGSIZE['medium'][1]*1.5))
        for x, y in zip(year_group['work_year'], year_group['平均薪资']):
            fig.axes[0].text(x, y+2000, f"{int(y)}", ha='center', fontproperties=chinese_font)
        fig.tight_layout()
        return fig
    display_chart(fig4, info4, code4, zoom4)
    st.divider()

    # 图表5：远程模式平均薪资
    st.subheader("图表5：不同远程模式平均薪资柱状图")
    remote_labels = ["无远程(0)", "混合远程(50)", "全远程(100)"]
    fig5 = plot_bar(range(3), remote_group['平均薪资'], remote_labels,
                    title='不同远程模式平均薪资对比',
                    ylabel='平均薪资(美元)',
                    color='#A23B72',
                    figsize=BASE_FIGSIZE['medium'])
    code5 = '''
remote_labels = ["无远程(0)", "混合远程(50)", "全远程(100)"]
fig = plot_bar(range(3), remote_group['平均薪资'], remote_labels,
               title='不同远程模式平均薪资对比',
               ylabel='平均薪资(美元)',
               color='#A23B72',
               figsize=BASE_FIGSIZE['medium'])
st.pyplot(fig)
'''
    info5 = "**图表说明**：无远程岗位 (0) 平均薪资最高（约14.1万美元），全远程 (100) 次之（约13.3万美元），混合远程 (50) 最低（约7.3万美元）。"
    def zoom5():
        return plot_bar(range(3), remote_group['平均薪资'], remote_labels,
                        title='不同远程模式平均薪资对比',
                        ylabel='平均薪资(美元)',
                        color='#A23B72',
                        figsize=(BASE_FIGSIZE['medium'][0]*1.5, BASE_FIGSIZE['medium'][1]*1.5))
    display_chart(fig5, info5, code5, zoom5)
    st.divider()

    # 图表6：Top10高薪地区
    st.subheader("图表6：Top10高薪地区平均薪资对比")
    top10 = location_group.head(10)
    fig6 = plot_bar(range(10), top10['平均薪资'], top10['company_location'],
                    title='Top10高薪地区平均薪资对比',
                    ylabel='平均薪资(美元)',
                    color='#F18F01',
                    figsize=BASE_FIGSIZE['large'])
    # 旋转x轴标签
    fig6.axes[0].tick_params(axis='x', rotation=45)
    fig6.tight_layout()
    code6 = '''
top10 = location_group.head(10)
fig = plot_bar(range(10), top10['平均薪资'], top10['company_location'],
               title='Top10高薪地区平均薪资对比',
               ylabel='平均薪资(美元)',
               color='#F18F01',
               figsize=BASE_FIGSIZE['large'])
fig.axes[0].tick_params(axis='x', rotation=45)
fig.tight_layout()
st.pyplot(fig)
'''
    info6 = "**图表说明**：不同地区薪资差异显著，美国、瑞士等发达国家的平均薪资远高于其他地区。"
    def zoom6():
        fig = plot_bar(range(10), top10['平均薪资'], top10['company_location'],
                       title='Top10高薪地区平均薪资对比',
                       ylabel='平均薪资(美元)',
                       color='#F18F01',
                       figsize=(BASE_FIGSIZE['large'][0]*1.5, BASE_FIGSIZE['large'][1]*1.5))
        fig.axes[0].tick_params(axis='x', rotation=45)
        fig.tight_layout()
        return fig
    display_chart(fig6, info6, code6, zoom6)

# ========= 五、高级可视化分析 =========
elif menu == "五、高级可视化分析":
    st.header("🔬 高级可视化分析（专业级深度洞察）")
    st.markdown("本模块使用原生Matplotlib绘制专业图表，深度挖掘薪资数据底层规律。")

    # 复用显示函数
    def display_chart(fig, info, code, zoom_fig_func):
        st.pyplot(fig)
        st.info(info)
        with st.expander("📄 查看此图表的代码"):
            st.code(code, language="python")
        with st.expander("🔍 放大查看"):
            st.pyplot(zoom_fig_func())
            st.caption("（放大版，尺寸约为原始1.5倍）")

    # 高级图表1：相关性热力图
    st.divider()
    st.subheader("高级图表1：特征相关性热力图")
    corr_labels = ['工作年份', '经验水平', '雇佣类型', '远程比例', '公司规模', '公司所在地区', '薪资(美元)']
    fig1 = plot_correlation(corr_matrix, corr_labels,
                            title='薪资影响因素相关性热力图',
                            figsize=BASE_FIGSIZE['heatmap'])
    code1 = '''
corr_labels = ['工作年份', '经验水平', '雇佣类型', '远程比例', '公司规模', '公司所在地区', '薪资(美元)']
fig = plot_correlation(corr_matrix, corr_labels,
                       title='薪资影响因素相关性热力图',
                       figsize=BASE_FIGSIZE['heatmap'])
st.pyplot(fig)
'''
    info1 = "**图表说明**：经验水平（0.48）和工作年份（0.39）与薪资呈中等正相关，是影响薪资的最强因素；公司所在地区（0.16）和公司规模（0.13）也有一定正向影响；雇佣类型和远程比例相关性较弱。"
    def zoom1():
        return plot_correlation(corr_matrix, corr_labels,
                                title='薪资影响因素相关性热力图',
                                figsize=(BASE_FIGSIZE['heatmap'][0]*1.5, BASE_FIGSIZE['heatmap'][1]*1.5))
    display_chart(fig1, info1, code1, zoom1)

    # 高级图表2：分位数回归系数
    st.divider()
    st.subheader("高级图表2：不同薪资分位数的标准化影响系数对比（分组散点连线图）")
    features = qr_result['特征'].tolist()
    fig2 = plot_qr_coefficients(qr_result, features, figsize=BASE_FIGSIZE['large'])
    code2 = '''
features = qr_result['特征'].tolist()
fig = plot_qr_coefficients(qr_result, features, figsize=BASE_FIGSIZE['large'])
st.pyplot(fig)
'''
    info2 = "**图表说明**：经验水平和工作年份在三个分位数下都保持最高的正向影响，且高薪群体（75%）受经验影响更大；公司所在地区和公司规模的影响在中等薪资水平时较强，远程比例和雇佣类型贡献较小。"
    def zoom2():
        return plot_qr_coefficients(qr_result, features, figsize=(BASE_FIGSIZE['large'][0]*1.5, BASE_FIGSIZE['large'][1]*1.5))
    display_chart(fig2, info2, code2, zoom2)
    st.dataframe(qr_result, use_container_width=True)

    # 高级图表3：带置信区间的年度趋势
    st.divider()
    st.subheader("高级图表3：带95%置信区间年度薪资趋势")
    # 计算置信区间
    year_group['ci_lower'] = year_group['平均薪资'] - 1.96 * (year_group['薪资标准差'] / np.sqrt(year_group['样本量']))
    year_group['ci_upper'] = year_group['平均薪资'] + 1.96 * (year_group['薪资标准差'] / np.sqrt(year_group['样本量']))
    fig3, ax = plt.subplots(figsize=BASE_FIGSIZE['large'])
    ax.plot(year_group['work_year'], year_group['平均薪资'], marker='o', c='#0070C0', linewidth=3, label='平均薪资')
    ax.fill_between(year_group['work_year'], year_group['ci_lower'], year_group['ci_upper'], alpha=0.2, color='#0070C0', label='95%置信区间')
    ax.plot(year_group['work_year'], year_group['中位数'], marker='s', c='#2E86AB', linestyle='--', label='中位数薪资')
    ax.set_title('2020-2023薪资趋势（置信区间）', fontproperties=chinese_font, fontsize=16)
    ax.set_xlabel('年份', fontproperties=chinese_font)
    ax.set_ylabel('薪资(美元)', fontproperties=chinese_font)
    ax.set_xticks(year_group['work_year'])
    ax.legend(prop=chinese_font)
    ax.grid(linestyle='--', alpha=0.5)
    fig3.tight_layout()
    code3 = '''
year_group['ci_lower'] = year_group['平均薪资'] - 1.96 * (year_group['薪资标准差'] / np.sqrt(year_group['样本量']))
year_group['ci_upper'] = year_group['平均薪资'] + 1.96 * (year_group['薪资标准差'] / np.sqrt(year_group['样本量']))
fig, ax = plt.subplots(figsize=BASE_FIGSIZE['large'])
ax.plot(year_group['work_year'], year_group['平均薪资'], marker='o', c='#0070C0', linewidth=3, label='平均薪资')
ax.fill_between(year_group['work_year'], year_group['ci_lower'], year_group['ci_upper'], alpha=0.2, color='#0070C0', label='95%置信区间')
ax.plot(year_group['work_year'], year_group['中位数'], marker='s', c='#2E86AB', linestyle='--', label='中位数薪资')
ax.set_title('2020-2023薪资趋势（置信区间）', fontproperties=chinese_font, fontsize=16)
ax.set_xlabel('年份', fontproperties=chinese_font)
ax.set_ylabel('薪资(美元)', fontproperties=chinese_font)
ax.set_xticks(year_group['work_year'])
ax.legend(prop=chinese_font)
ax.grid(linestyle='--', alpha=0.5)
fig.tight_layout()
st.pyplot(fig)
'''
    info3 = "**图表说明**：平均薪资从约6.5万美元稳步增长至近10万美元，涨幅超50%，置信区间逐年收窄，表明市场定价更加成熟稳定。"
    def zoom3():
        fig, ax = plt.subplots(figsize=(BASE_FIGSIZE['large'][0]*1.5, BASE_FIGSIZE['large'][1]*1.5))
        ax.plot(year_group['work_year'], year_group['平均薪资'], marker='o', c='#0070C0', linewidth=3, label='平均薪资')
        ax.fill_between(year_group['work_year'], year_group['ci_lower'], year_group['ci_upper'], alpha=0.2, color='#0070C0', label='95%置信区间')
        ax.plot(year_group['work_year'], year_group['中位数'], marker='s', c='#2E86AB', linestyle='--', label='中位数薪资')
        ax.set_title('2020-2023薪资趋势（置信区间）', fontproperties=chinese_font, fontsize=16)
        ax.set_xlabel('年份', fontproperties=chinese_font)
        ax.set_ylabel('薪资(美元)', fontproperties=chinese_font)
        ax.set_xticks(year_group['work_year'])
        ax.legend(prop=chinese_font)
        ax.grid(linestyle='--', alpha=0.5)
        fig.tight_layout()
        return fig
    display_chart(fig3, info3, code3, zoom3)

    # 高级图表4：公司规模-经验热力图
    st.divider()
    st.subheader("高级图表4：公司规模-经验薪资二维热力图")
    pivot = df_clean.pivot_table(index='experience_level', columns='company_size', values='salary_in_usd', aggfunc='mean').reindex(index=exp_order, columns=size_order)
    fig4 = plot_heatmap_2d(pivot,
                           xlabels=[size_dict[s] for s in size_order],
                           ylabels=[exp_dict[e] for e in exp_order],
                           title='公司规模-经验水平平均薪资热力图',
                           figsize=BASE_FIGSIZE['heatmap'])
    code4 = '''
pivot = df_clean.pivot_table(index='experience_level', columns='company_size', values='salary_in_usd', aggfunc='mean').reindex(index=exp_order, columns=size_order)
fig = plot_heatmap_2d(pivot,
                      xlabels=[size_dict[s] for s in size_order],
                      ylabels=[exp_dict[e] for e in exp_order],
                      title='公司规模-经验水平平均薪资热力图',
                      figsize=BASE_FIGSIZE['heatmap'])
st.pyplot(fig)
'''
    info4 = "**图表说明**：中型企业（M）在每一经验等级上平均薪资最高，尤其在MI和SE阶段远超其他规模；专家级（EX）在所有规模中都是最高薪资群体。"
    def zoom4():
        return plot_heatmap_2d(pivot,
                               xlabels=[size_dict[s] for s in size_order],
                               ylabels=[exp_dict[e] for e in exp_order],
                               title='公司规模-经验水平平均薪资热力图',
                               figsize=(BASE_FIGSIZE['heatmap'][0]*1.5, BASE_FIGSIZE['heatmap'][1]*1.5))
    display_chart(fig4, info4, code4, zoom4)

# ========= 六、分析结论与建议 =========
elif menu == "六、分析结论与行业建议":
    st.header("💡 案例分析结论与建议")
    st.subheader("6.1 核心数据分析结果")
    st.markdown("""
1. **薪资整体特征**：全球数据分析师薪资整体呈正态分布，核心区间为 5 万 - 20 万美元，2020-2023 年薪资持续上涨，4 年涨幅超 55%。
2. **核心影响因素**：**工作经验水平 > 公司所在地区 > 公司规模 > 工作年份 > 远程比例 > 雇佣类型**。
3. **维度差异规律**：
   - 经验：专家级平均薪资是入门级的3倍以上；
   - 企业规模：中型企业平均薪资最具竞争力（超14万美元）；
   - 地区：美国、瑞士等发达国家薪资优势明显；
   - 办公模式：无远程与全远程薪资较高（超13万美元），混合远程较低。
""")
    st.divider()
    st.subheader("6.2 结论与建议")
    st.subheader("💡 给数据从业者的求职建议")
    st.markdown("""
1. **优先积累核心工作经验**。
2. **考虑中型企业**，其薪资竞争力强。
3. **权衡办公模式**，优先选择全线下或成熟的全远程岗位。
4. **持续学习**，匹配行业薪资增长。
""")
    st.subheader("🏢 给企业的薪资制定建议")
    st.markdown("""
1. **建立阶梯式薪资体系**。
2. **大型企业反思薪资结构**，提升现金薪酬吸引力。
3. **小型企业优化薪资结构**，提供股票期权等非现金激励。
4. **规范远程/驻场薪酬匹配**，参考行业高位标准。
""")

# ========= 七、在线薪资预测 =========
elif menu == "七、在线薪资预测工具":
    st.header("🧮 在线薪资预测(仅供参考)")
    st.markdown("输入个人及工作信息，智能预测数据分析师税前年薪（美元）")
    with st.form("salary_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            work_year = st.slider("工作年份", 2020, 2026, 2026, 1)
            exp_cn = st.selectbox("工作经验水平", options=list(exp_dict.values()))
            emp_cn = st.selectbox("雇佣类型", options=list(emp_dict.values()))
        with col2:
            remote_ratio = st.select_slider("远程工作比例", options=[0, 50, 100], value=100, format_func=lambda x: f"{x}%")
            size_cn = st.selectbox("公司规模", options=list(size_dict.values()))
            location = st.selectbox("公司所在国家/地区", options=le_location.classes_)
        submit = st.form_submit_button("开始预测薪资", use_container_width=True)

    if submit:
        exp_raw = rev_exp[exp_cn]
        emp_raw = rev_emp[emp_cn]
        size_raw = rev_size[size_cn]
        exp_code = exp_map[exp_raw]
        com_code = size_map[size_raw]
        emp_code = le_employment.transform([emp_raw])[0]
        loc_code = le_location.transform([location])[0]

        input_features = np.array([[work_year, exp_code, emp_code, remote_ratio, com_code, loc_code]])
        input_scaled = scaler.transform(input_features)

        pred = model.predict(input_scaled)[0]
        q25 = qr_model_25.predict(input_scaled)[0]
        q50 = qr_model_50.predict(input_scaled)[0]
        q75 = qr_model_75.predict(input_scaled)[0]

        st.success("✅ 预测完成！")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("预测平均年薪（美元）", f"${max(0,pred):,.2f}")
        col2.metric("25%分位数年薪", f"${max(0,q25):,.2f}")
        col3.metric("50%分位数年薪", f"${max(0,q50):,.2f}")
        col4.metric("75%分位数年薪", f"${max(0,q75):,.2f}")
        st.info("预测基于历史数据训练，薪资单位为美元，税前年薪。")

# ========= 八、交互式薪资探索器 =========
elif menu == "八、交互式薪资探索器":
    st.header("🔍 交互式薪资探索器（自定义维度深度分析）")
    st.markdown("通过下拉框和滑块自定义筛选条件，动态探索不同维度下的薪资分布、统计数据和可视化图表。")

    col1, col2, col3 = st.columns(3)
    with col1:
        exp_filter = st.multiselect("经验水平", options=list(exp_dict.keys()), default=list(exp_dict.keys()), format_func=lambda x: exp_dict[x])
        size_filter = st.multiselect("公司规模", options=list(size_dict.keys()), default=list(size_dict.keys()), format_func=lambda x: size_dict[x])
    with col2:
        year_range = st.slider("工作年份", min_value=df_clean['work_year'].min(), max_value=df_clean['work_year'].max(),
                               value=(df_clean['work_year'].min(), df_clean['work_year'].max()), step=1)
        remote_filter = st.multiselect("远程工作比例", options=df_clean['remote_ratio'].unique(), default=df_clean['remote_ratio'].unique(),
                                       format_func=lambda x: f"{x}%")
    with col3:
        top_locs = location_group.head(20)['company_location'].tolist()
        loc_filter = st.multiselect("公司所在地区（Top20）", options=top_locs, default=top_locs)
        salary_range = st.slider("薪资范围（美元）", min_value=df_clean['salary_in_usd'].min(), max_value=df_clean['salary_in_usd'].max(),
                                 value=(df_clean['salary_in_usd'].min(), df_clean['salary_in_usd'].max()), step=1000)

    df_filtered = df_clean[
        (df_clean['experience_level'].isin(exp_filter)) &
        (df_clean['company_size'].isin(size_filter)) &
        (df_clean['work_year'].between(year_range[0], year_range[1])) &
        (df_clean['remote_ratio'].isin(remote_filter)) &
        (df_clean['company_location'].isin(loc_filter)) &
        (df_clean['salary_in_usd'].between(salary_range[0], salary_range[1]))
    ]

    st.divider()
    st.subheader("📊 筛选结果统计")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("筛选后样本量", f"{len(df_filtered)} 条")
    col2.metric("平均薪资", f"${df_filtered['salary_in_usd'].mean():,.0f}" if len(df_filtered)>0 else "$0")
    col3.metric("薪资中位数", f"${df_filtered['salary_in_usd'].median():,.0f}" if len(df_filtered)>0 else "$0")
    col4.metric("薪资最大值", f"${df_filtered['salary_in_usd'].max():,.0f}" if len(df_filtered)>0 else "$0")

    st.subheader("筛选后数据预览（前10行）")
    st.dataframe(df_filtered.head(10), use_container_width=True)

    st.divider()
    st.subheader("📈 筛选后薪资分布可视化")
    col1, col2 = st.columns(2)
    with col1:
        fig = plot_histogram(df_filtered['salary_in_usd'],
                             title='筛选后薪资分布直方图',
                             xlabel='薪资(美元)', ylabel='样本数量',
                             figsize=BASE_FIGSIZE['small'])
        st.pyplot(fig)
    with col2:
        if len(exp_filter) > 1 and len(df_filtered) > 0:
            exist_levels = [l for l in exp_filter if l in df_filtered['experience_level'].unique()]
            if len(exist_levels) > 1:
                box_data = [df_filtered[df_filtered['experience_level'] == level]['salary_in_usd'] for level in exist_levels]
                labels = [exp_dict[l] for l in exist_levels]
                fig = plot_boxplot(box_data, labels,
                                   title='筛选后不同经验水平薪资箱线图',
                                   xlabel='经验等级', ylabel='薪资(美元)',
                                   figsize=BASE_FIGSIZE['small'])
                st.pyplot(fig)
            else:
                st.info("有效经验等级不足2个，无法绘制箱线图")
        else:
            st.info("仅选择1个经验水平，无法绘制箱线图")

    st.divider()
    st.subheader("📋 筛选后按经验水平分组统计")
    if len(exp_filter) > 0 and len(df_filtered) > 0:
        group = df_filtered.groupby('experience_level')['salary_in_usd'].agg(
            样本量='count', 平均薪资='mean', 中位数='median', 最低='min', 最高='max'
        ).reset_index()
        group['experience_level'] = pd.Categorical(group['experience_level'], categories=exp_order, ordered=True)
        group = group.sort_values('experience_level')
        group['experience_level'] = group['experience_level'].map(exp_dict)
        st.dataframe(group, use_container_width=True)
    else:
        st.info("无筛选数据或未选择经验水平")

# 页脚
st.markdown("---")
st.markdown("© 2026 数据分析师薪资综合分析系统 | 全流程数据分析 + 可视化 + 智能预测")
