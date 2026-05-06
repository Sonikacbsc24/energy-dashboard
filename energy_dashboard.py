import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Energy Consumption Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }
    .block-container { padding-top: 1.5rem; }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        color: #e0e0e0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #f5a623;
        font-family: 'IBM Plex Mono', monospace;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #8a9ab5;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 4px;
    }

    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #f5a623;
        border-left: 4px solid #f5a623;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
    }

    .insight-box {
        background: #0d1b2a;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        color: #cdd9e5;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    .insight-box strong { color: #f5a623; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #0d1b2a;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8a9ab5;
        background: transparent;
        border-radius: 6px;
        padding: 0.4rem 1rem;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background: #f5a623 !important;
        color: #0d1b2a !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #f5a623, #e8920f);
        color: #0d1b2a;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: 0.05em;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #ffc15e, #f5a623);
        color: #000;
    }

    .pred-result {
        background: linear-gradient(135deg, #1a3a1a, #0d2a0d);
        border: 1px solid #2a6a2a;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        color: #7fff7f;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.1rem;
        text-align: center;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Energy Dashboard")
    st.markdown("---")
    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="Upload your household_power_consumption.csv"
    )
    st.markdown("---")
    st.markdown("""
    **Expected Columns:**
    - `Date`, `Time`
    - `Global_active_power`
    - `Global_reactive_power`
    - `Voltage`, `Global_intensity`
    - `Sub_metering_1/2/3`
    """)
    st.markdown("---")
    sample_rows = st.slider("Sample rows for heavy charts", 1000, 20000, 5000, step=1000)

# ─── Load & Process Data ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_process(file):
    df = pd.read_csv(file, encoding='latin1')
    df.rename(columns={'ï»¿Date': 'Date'}, inplace=True)

    # Datetime parsing
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
    df.drop(['Date', 'Time'], axis=1, inplace=True, errors='ignore')
    df.set_index('Datetime', inplace=True)

    # Numeric conversion
    cols = ['Global_active_power','Global_reactive_power','Voltage',
            'Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3']
    df.replace('?', np.nan, inplace=True)
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # Time features
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday
    df['year'] = df.index.year
    df['DayOfWeek'] = df.index.day_name()
    df['Hour'] = df.index.hour
    df['MonthName'] = df.index.month_name()

    # Lag features
    df['lag_1'] = df['Global_active_power'].shift(1)
    df['lag_2'] = df['Global_active_power'].shift(2)
    df.dropna(inplace=True)

    # Encoding for ML
    le = LabelEncoder()
    df['Month_enc'] = le.fit_transform(df['MonthName'])
    dow_map = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}
    df['DayOfWeek_enc'] = df['DayOfWeek'].map(dow_map)

    return df

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("# Household Energy Consumption Dashboard")
st.markdown("*End-to-end analysis, visualization & ML predictions — upload your CSV to begin.*")

if not uploaded:
    st.info(" Upload your `household_power_consumption.csv` in the sidebar to get started.")
    st.stop()

with st.spinner("Processing data…"):
    df = load_and_process(uploaded)

# ─── KPI Cards ────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
def kpi(col, value, label):
    col.markdown(f"""
    <div class="metric-card">
        <div class="value">{value}</div>
        <div class="label">{label}</div>
    </div>""", unsafe_allow_html=True)

kpi(c1, f"{len(df):,}", "Total Records")
kpi(c2, f"{df['Global_active_power'].mean():.3f} kW", "Avg Active Power")
kpi(c3, f"{df['Global_active_power'].max():.2f} kW", "Peak Power")
kpi(c4, f"{df['Voltage'].mean():.1f} V", "Avg Voltage")
kpi(c5, f"{df.index.year.nunique()}", "Years Covered")

st.markdown("")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Overview", " Patterns", " Outliers & Scaling", " Predict Power", " Predict Sub-Metering"
])

plt_kwargs = dict(facecolor='#0d1b2a')
sns.set_style("dark")
plt.rcParams.update({
    'figure.facecolor': '#0d1b2a',
    'axes.facecolor': '#131f30',
    'axes.edgecolor': '#2a4a6a',
    'text.color': '#cdd9e5',
    'axes.labelcolor': '#cdd9e5',
    'xtick.color': '#8a9ab5',
    'ytick.color': '#8a9ab5',
    'grid.color': '#1e3a5f',
    'axes.titlecolor': '#f5a623',
})

# ══════════════════════════════════════════════
# TAB 1 – OVERVIEW
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Global Active Power Over Time</div>', unsafe_allow_html=True)

    sel = df.iloc[:sample_rows].copy()
    fig, ax = plt.subplots(figsize=(14, 4), facecolor='#0d1b2a')
    ax.set_facecolor('#131f30')
    ax.plot(sel.index, sel['Global_active_power'], color='#f5a623', linewidth=0.6, alpha=0.85)
    ax.fill_between(sel.index, sel['Global_active_power'], alpha=0.12, color='#f5a623')
    ax.set_title(f'Global Active Power (first {sample_rows:,} records)', color='#f5a623')
    ax.set_xlabel('Datetime'); ax.set_ylabel('kW')
    ax.grid(alpha=0.3)
    st.pyplot(fig); plt.close()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Voltage vs Active Power</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0d1b2a')
        ax.set_facecolor('#131f30')
        sample = df.sample(min(3000, len(df)), random_state=42)
        ax.scatter(sample['Voltage'], sample['Global_active_power'],
                   alpha=0.3, s=5, color='#4fc3f7')
        ax.set_xlabel('Voltage (V)'); ax.set_ylabel('Global Active Power (kW)')
        ax.set_title('Voltage vs Active Power', color='#f5a623')
        ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown('<div class="section-header">Intensity vs Active Power</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0d1b2a')
        ax.set_facecolor('#131f30')
        ax.scatter(sample['Global_intensity'], sample['Global_active_power'],
                   alpha=0.3, s=5, color='#ff7043')
        ax.set_xlabel('Global Intensity (A)'); ax.set_ylabel('Global Active Power (kW)')
        ax.set_title('Intensity vs Active Power', color='#f5a623')
        ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">Bubble Plot: Active Power vs Voltage</div>', unsafe_allow_html=True)
    bubble_df = df.iloc[:min(8000, len(df))]
    fig, ax = plt.subplots(figsize=(12, 5), facecolor='#0d1b2a')
    ax.set_facecolor('#131f30')
    ax.scatter(bubble_df['Global_active_power'], bubble_df['Voltage'],
               s=bubble_df['Global_intensity'] * 4,
               color='#ce93d8', alpha=0.35, edgecolors='#7b1fa2', linewidths=0.3)
    ax.set_title('Bubble Plot: Active Power vs Voltage (bubble size = Intensity)', color='#f5a623')
    ax.set_xlabel('Global Active Power (kW)'); ax.set_ylabel('Voltage (V)')
    ax.grid(alpha=0.3)
    st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">Raw Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(100), use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 – PATTERNS
# ══════════════════════════════════════════════
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Avg Power by Day of Week</div>', unsafe_allow_html=True)
        avg_day = df.groupby('DayOfWeek')['Global_active_power'].mean().reindex(
            ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
        fig, ax = plt.subplots(figsize=(7, 4), facecolor='#0d1b2a')
        ax.set_facecolor('#131f30')
        colors = cm.viridis(np.linspace(0.2, 0.9, 7))
        ax.bar(avg_day.index, avg_day.values, color=colors)
        ax.set_title('Avg Active Power — Day of Week', color='#f5a623')
        ax.set_xlabel('Day'); ax.set_ylabel('kW')
        plt.xticks(rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown('<div class="section-header">Avg Power by Hour of Day</div>', unsafe_allow_html=True)
        avg_hour = df.groupby('Hour')['Global_active_power'].mean()
        fig, ax = plt.subplots(figsize=(7, 4), facecolor='#0d1b2a')
        ax.set_facecolor('#131f30')
        colors_h = cm.coolwarm(np.linspace(0.1, 0.9, 24))
        ax.bar(avg_hour.index, avg_hour.values, color=colors_h)
        ax.set_title('Avg Active Power — Hour of Day', color='#f5a623')
        ax.set_xlabel('Hour'); ax.set_ylabel('kW')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">Total Sub-Metering by Month</div>', unsafe_allow_html=True)
    month_order = ['January','February','March','April','May','June',
                   'July','August','September','October','November','December']
    total_sub = df.groupby('MonthName')[['Sub_metering_1','Sub_metering_2','Sub_metering_3']].sum()
    total_sub = total_sub.reindex([m for m in month_order if m in total_sub.index])
    fig, ax = plt.subplots(figsize=(12, 5), facecolor='#0d1b2a')
    ax.set_facecolor('#131f30')
    total_sub.plot(kind='bar', stacked=True, ax=ax,
                   color=['#f5a623','#4fc3f7','#ce93d8'])
    ax.set_title('Total Sub-Metering Energy by Month', color='#f5a623')
    ax.set_xlabel('Month'); ax.set_ylabel('Watt-hours')
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(['Sub_metering_1','Sub_metering_2','Sub_metering_3'],
              facecolor='#1a1a2e', edgecolor='#2a4a6a', labelcolor='#cdd9e5')
    st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
    numeric_cols = ['Global_active_power','Global_reactive_power','Voltage',
                    'Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3','hour']
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 6), facecolor='#0d1b2a')
    ax.set_facecolor('#0d1b2a')
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
                linewidths=0.5, linecolor='#0d1b2a',
                annot_kws={"size": 8, "color": "white"})
    ax.set_title('Feature Correlation Matrix', color='#f5a623')
    st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════
# TAB 3 – OUTLIERS & SCALING
# ══════════════════════════════════════════════
with tab3:
    numerical_features = ['Global_active_power','Global_reactive_power','Voltage',
                          'Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3']

    st.markdown('<div class="section-header">IQR Outlier Summary</div>', unsafe_allow_html=True)
    outlier_data = []
    for col in numerical_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lb = Q1 - 1.5 * IQR
        ub = Q3 + 1.5 * IQR
        n_out = ((df[col] < lb) | (df[col] > ub)).sum()
        outlier_data.append({'Feature': col, 'Q1': round(Q1,4), 'Q3': round(Q3,4),
                             'IQR': round(IQR,4), 'Lower Bound': round(lb,4),
                             'Upper Bound': round(ub,4), 'Outliers': n_out})
    st.dataframe(pd.DataFrame(outlier_data), use_container_width=True)

    st.markdown('<div class="section-header">Boxplot (Sample)</div>', unsafe_allow_html=True)
    box_df = df[numerical_features].iloc[2000:6000]
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0d1b2a')
    ax.set_facecolor('#131f30')
    bp = ax.boxplot([box_df[c].dropna().values for c in numerical_features],
                    vert=False, patch_artist=True,
                    labels=numerical_features,
                    medianprops=dict(color='#f5a623', linewidth=2))
    colors_b = ['#4fc3f7','#ff7043','#ce93d8','#81c784','#ffb74d','#e57373','#80cbc4']
    for patch, color in zip(bp['boxes'], colors_b):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_title('Feature Distribution Boxplot', color='#f5a623')
    ax.grid(axis='x', alpha=0.3)
    st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">MinMax vs Standard Scaling</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    feat_sel = st.selectbox("Select feature to compare scaling", numerical_features)

    mm = MinMaxScaler()
    std = StandardScaler()
    mm_vals = mm.fit_transform(df[[feat_sel]]).flatten()
    std_vals = std.fit_transform(df[[feat_sel]]).flatten()

    with col1:
        fig, ax = plt.subplots(figsize=(5, 3), facecolor='#0d1b2a')
        ax.set_facecolor('#131f30')
        ax.hist(mm_vals[:10000], bins=50, color='#4fc3f7', alpha=0.8, edgecolor='#0d1b2a')
        ax.set_title(f'MinMax Scaled — {feat_sel}', color='#f5a623')
        ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5, 3), facecolor='#0d1b2a')
        ax.set_facecolor('#131f30')
        ax.hist(std_vals[:10000], bins=50, color='#ff7043', alpha=0.8, edgecolor='#0d1b2a')
        ax.set_title(f'Standard Scaled — {feat_sel}', color='#f5a623')
        ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════
# TAB 4 – PREDICT OVERALL POWER
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Model: Predict Global Active Power</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">Uses <strong>Linear Regression</strong> on electrical parameters + time features to predict total household power consumption.</div>', unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def train_power_model(df_key):
        X = df[['Voltage','Global_intensity','Sub_metering_1','Sub_metering_2',
                'Sub_metering_3','DayOfWeek_enc','Month_enc','Hour']]
        y = df['Global_active_power']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return model, X_test, y_test, y_pred, X.columns.tolist()

    model1, X_test1, y_test1, y_pred1, feat_names1 = train_power_model(len(df))

    mse1 = mean_squared_error(y_test1, y_pred1)
    mae1 = mean_absolute_error(y_test1, y_pred1)
    r21  = r2_score(y_test1, y_pred1)

    m1, m2, m3 = st.columns(3)
    kpi(m1, f"{mse1:.4f}", "MSE")
    kpi(m2, f"{mae1:.4f}", "MAE")
    kpi(m3, f"{r21:.4f}", "R² Score")
    st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Actual vs Predicted</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0d1b2a')
        ax.set_facecolor('#131f30')
        sample_idx = np.random.choice(len(y_test1), min(300, len(y_test1)), replace=False)
        ax.scatter(y_test1.values[sample_idx], y_pred1[sample_idx],
                   alpha=0.4, s=10, color='#4fc3f7')
        lims = [min(y_test1.min(), y_pred1.min()), max(y_test1.max(), y_pred1.max())]
        ax.plot(lims, lims, '--', color='#f5a623', linewidth=1.5, label='Perfect fit')
        ax.set_xlabel('Actual (kW)'); ax.set_ylabel('Predicted (kW)')
        ax.set_title('Actual vs Predicted Power', color='#f5a623')
        ax.legend(facecolor='#1a1a2e', labelcolor='#cdd9e5')
        ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown('<div class="section-header">Feature Coefficients</div>', unsafe_allow_html=True)
        coef_df = pd.DataFrame({'Feature': feat_names1, 'Coefficient': model1.coef_}).sort_values('Coefficient')
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0d1b2a')
        ax.set_facecolor('#131f30')
        colors_c = ['#ff7043' if c < 0 else '#4fc3f7' for c in coef_df['Coefficient']]
        ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors_c)
        ax.set_title('Feature Coefficients', color='#f5a623')
        ax.axvline(0, color='#f5a623', linewidth=0.8, linestyle='--')
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">Try a Custom Prediction</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        voltage   = st.number_input("Voltage (V)", value=240.0, step=1.0)
        intensity = st.number_input("Global Intensity (A)", value=5.0, step=0.5)
    with c2:
        sub1 = st.number_input("Sub_metering_1 (Wh)", value=1.0, step=0.5)
        sub2 = st.number_input("Sub_metering_2 (Wh)", value=1.0, step=0.5)
    with c3:
        sub3    = st.number_input("Sub_metering_3 (Wh)", value=2.0, step=0.5)
        hour_in = st.slider("Hour of Day", 0, 23, 22)
    with c4:
        dow_in  = st.selectbox("Day of Week", ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
        month_in = st.selectbox("Month", list(range(1, 13)), index=9)

    dow_map2 = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}

    if st.button(" Predict Power Consumption"):
        new_val = [[voltage, intensity, sub1, sub2, sub3, dow_map2[dow_in], month_in, hour_in]]
        pred = model1.predict(new_val)[0]
        energy_kwh = pred * (1/60)
        st.markdown(f"""
        <div class="pred-result">
            Predicted Global Active Power: <strong>{pred:.4f} kW</strong><br>
            Predicted Energy (1 min interval): <strong>{energy_kwh:.5f} kWh</strong>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 5 – PREDICT SUB-METERING 3 (LAUNDRY)
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Model: Predict Sub_metering_3 (Laundry / Water Heater)</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">Uses <strong>Linear Regression</strong> with engineered features (lag values, hour², hour×weekday) to predict washing machine / laundry area energy.</div>', unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def train_sub3_model(df_key):
        d = df.copy()
        d['Hour_squared']   = d['Hour'] ** 2
        d['Hour_DayOfWeek'] = d['Hour'] * d['DayOfWeek_enc']
        d['Prev_1'] = d['Sub_metering_3'].shift(1).fillna(0)
        d['Prev_2'] = d['Sub_metering_3'].shift(2).fillna(0)
        feats = ['Global_active_power','Global_reactive_power','Voltage','Global_intensity',
                 'Sub_metering_1','Sub_metering_2','Hour','DayOfWeek_enc','year','Month_enc',
                 'Hour_squared','Hour_DayOfWeek','Prev_1','Prev_2']
        X = d[feats].dropna()
        y = d.loc[X.index, 'Sub_metering_3']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return model, X_test, y_test, y_pred, feats

    model2, X_test2, y_test2, y_pred2, feat_names2 = train_sub3_model(len(df))

    mse2 = mean_squared_error(y_test2, y_pred2)
    mae2 = mean_absolute_error(y_test2, y_pred2)
    r22  = r2_score(y_test2, y_pred2)

    m1, m2, m3 = st.columns(3)
    kpi(m1, f"{mse2:.4f}", "MSE")
    kpi(m2, f"{mae2:.4f}", "MAE")
    kpi(m3, f"{r22:.4f}", "R² Score")
    st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Actual vs Predicted Sub_metering_3</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0d1b2a')
        ax.set_facecolor('#131f30')
        sidx = np.random.choice(len(y_test2), min(300, len(y_test2)), replace=False)
        ax.scatter(y_test2.values[sidx], y_pred2[sidx], alpha=0.4, s=10, color='#ce93d8')
        lims2 = [min(y_test2.min(), y_pred2.min()), max(y_test2.max(), y_pred2.max())]
        ax.plot(lims2, lims2, '--', color='#f5a623', linewidth=1.5, label='Perfect fit')
        ax.set_xlabel('Actual (Wh)'); ax.set_ylabel('Predicted (Wh)')
        ax.set_title('Actual vs Predicted Sub_metering_3', color='#f5a623')
        ax.legend(facecolor='#1a1a2e', labelcolor='#cdd9e5')
        ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown('<div class="section-header">Sub_metering_3 Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0d1b2a')
        ax.set_facecolor('#131f30')
        ax.hist(df['Sub_metering_3'].values[:20000], bins=50,
                color='#ce93d8', alpha=0.8, edgecolor='#0d1b2a')
        ax.set_title('Sub_metering_3 Distribution', color='#f5a623')
        ax.set_xlabel('Wh'); ax.set_ylabel('Frequency')
        ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">Try a Custom Sub-Metering Prediction</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        s_gap  = st.number_input("Global Active Power (kW)", value=1.8, step=0.1)
        s_grp  = st.number_input("Global Reactive Power (kVAR)", value=0.2, step=0.05)
    with c2:
        s_volt = st.number_input("Voltage (V) ", value=241.0, step=1.0)
        s_int  = st.number_input("Global Intensity (A) ", value=7.0, step=0.5)
    with c3:
        s_sm1  = st.number_input("Sub_metering_1", value=0.6, step=0.1)
        s_sm2  = st.number_input("Sub_metering_2", value=0.4, step=0.1)
        s_hr   = st.slider("Hour of Day ", 0, 23, 19)
    with c4:
        s_dow  = st.selectbox("Day of Week ", ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], index=4)
        s_yr   = st.number_input("Year", value=2025, step=1)
        s_mo   = st.selectbox("Month ", list(range(1,13)), index=9)

    if st.button(" Predict Sub-Metering 3"):
        dow_enc  = dow_map2[s_dow]
        h2       = s_hr ** 2
        h_dow    = s_hr * dow_enc
        new_val2 = [[s_gap, s_grp, s_volt, s_int, s_sm1, s_sm2,
                     s_hr, dow_enc, s_yr, s_mo, h2, h_dow, 0.35, 0.25]]
        pred2 = model2.predict(new_val2)[0]
        e2 = pred2 / 60
        st.markdown(f"""
        <div class="pred-result">
            Predicted Sub_metering_3: <strong>{pred2:.4f} Wh</strong><br>
            Energy for 1 minute: <strong>{e2:.5f} kWh</strong>
        </div>""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<center style='color:#8a9ab5; font-size:0.8rem'>Household Energy Consumption Dashboard · Built with Streamlit</center>", unsafe_allow_html=True)
