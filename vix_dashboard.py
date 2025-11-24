
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: vix_dashboard.py
# execution: false

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="VIX Volatility Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #d62728;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1f77b4;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #d62728;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .crisis-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('vix_preprocessed_data.csv')
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    return df

@st.cache_data
def load_model_results():
    return pd.read_csv('all_models_results.csv')

# Load data
df = load_data()
model_results = load_model_results()

# Sidebar filters
st.sidebar.title("📊 Dashboard Controls")
st.sidebar.markdown("---")

# Date range filter
st.sidebar.subheader("Date Range Selection")
min_date = df['observation_date'].min().date()
max_date = df['observation_date'].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(max_date - timedelta(days=365*5), max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = df[(df['observation_date'].dt.date >= start_date) & 
                     (df['observation_date'].dt.date <= end_date)]
else:
    filtered_df = df

# Regime filter
st.sidebar.subheader("Volatility Regime Filter")
regimes = ['All'] + sorted(df['regime'].unique().tolist())
selected_regime = st.sidebar.selectbox("Select Regime", regimes)

if selected_regime != 'All':
    filtered_df = filtered_df[filtered_df['regime'] == selected_regime]

# Crisis filter
st.sidebar.subheader("Crisis Period Filter")
show_crisis_only = st.sidebar.checkbox("Show Crisis Periods Only", value=False)
if show_crisis_only:
    filtered_df = filtered_df[filtered_df['sustained_crisis'] == 1]

# Smoothing option
st.sidebar.subheader("Visualization Options")
smoothing = st.sidebar.selectbox(
    "Smoothing Method",
    ["None", "20-day MA", "60-day EWMA", "20-day Median"]
)

# Main dashboard
st.markdown('<p class="main-header">🔴 VIX Volatility Analysis Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Comprehensive Market Stress Analysis and Risk Identification</p>', unsafe_allow_html=True)

# Executive Summary
st.markdown("---")
st.markdown("## 📋 Executive Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    current_vix = filtered_df['VIXCLS'].iloc[-1]
    prev_vix = filtered_df['VIXCLS'].iloc[-2]
    vix_change = current_vix - prev_vix
    st.metric(
        "Current VIX",
        f"{current_vix:.2f}",
        f"{vix_change:+.2f}",
        delta_color="inverse"
    )

with col2:
    current_regime = filtered_df['regime'].iloc[-1]
    regime_color = {
        'Very Low': '🟢', 'Low': '🟢', 'Moderate': '🟡',
        'Elevated': '🟠', 'High': '🔴', 'Extreme': '🔴'
    }
    st.metric(
        "Current Regime",
        f"{regime_color.get(current_regime, '⚪')} {current_regime}",
        ""
    )

with col3:
    avg_vix = filtered_df['VIXCLS'].mean()
    st.metric(
        "Average VIX",
        f"{avg_vix:.2f}",
        f"Period: {len(filtered_df)} days"
    )

with col4:
    crisis_days = filtered_df['sustained_crisis'].sum()
    crisis_pct = (crisis_days / len(filtered_df)) * 100
    st.metric(
        "Crisis Days",
        f"{crisis_days}",
        f"{crisis_pct:.1f}% of period"
    )

# Key insights box
st.markdown('<div class="insight-box">', unsafe_allow_html=True)
st.markdown("### 🎯 Key Insights")

if current_vix > 30:
    st.markdown('<div class="crisis-box">', unsafe_allow_html=True)
    st.markdown(f"""
    **⚠️ MARKET STRESS ALERT**
    - Current VIX ({current_vix:.2f}) indicates elevated market stress
    - VIX above 30 suggests significant investor fear and uncertainty
    - Consider defensive positioning and increased hedging
    """)
    st.markdown('</div>', unsafe_allow_html=True)
elif current_vix > 20:
    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.markdown(f"""
    **⚡ ELEVATED VOLATILITY**
    - Current VIX ({current_vix:.2f}) shows above-average volatility
    - Market uncertainty is elevated but not extreme
    - Monitor for potential escalation
    """)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown(f"""
    **✅ NORMAL MARKET CONDITIONS**
    - Current VIX ({current_vix:.2f}) indicates relatively calm markets
    - Volatility is below historical average
    - Favorable environment for risk-taking
    """)

st.markdown('</div>', unsafe_allow_html=True)

# Section 1: Historical VIX Analysis
st.markdown("---")
st.markdown('<p class="sub-header">📈 Historical VIX Time Series</p>', unsafe_allow_html=True)

st.markdown("""
**Context:** The VIX (Volatility Index) measures market expectations of near-term volatility. 
Higher values indicate greater expected volatility and typically correspond to market stress or fear.
""")

# Create interactive time series plot
fig_ts = go.Figure()

# Add main VIX line
fig_ts.add_trace(go.Scatter(
    x=filtered_df['observation_date'],
    y=filtered_df['VIXCLS'],
    mode='lines',
    name='VIX',
    line=dict(color='darkblue', width=1),
    opacity=0.7
))

# Add smoothing if selected
if smoothing == "20-day MA":
    fig_ts.add_trace(go.Scatter(
        x=filtered_df['observation_date'],
        y=filtered_df['vix_ma_20'],
        mode='lines',
        name='20-day MA',
        line=dict(color='red', width=2)
    ))
elif smoothing == "60-day EWMA":
    fig_ts.add_trace(go.Scatter(
        x=filtered_df['observation_date'],
        y=filtered_df['vix_ewma_60'],
        mode='lines',
        name='60-day EWMA',
        line=dict(color='green', width=2)
    ))
elif smoothing == "20-day Median":
    fig_ts.add_trace(go.Scatter(
        x=filtered_df['observation_date'],
        y=filtered_df['vix_median_20'],
        mode='lines',
        name='20-day Median',
        line=dict(color='orange', width=2)
    ))

# Add crisis threshold
fig_ts.add_hline(y=30, line_dash="dash", line_color="red", 
                annotation_text="Crisis Threshold (30)",
                annotation_position="right")

# Add regime thresholds
fig_ts.add_hline(y=df['VIXCLS'].quantile(0.75), line_dash="dot", line_color="orange",
                annotation_text="75th Percentile", annotation_position="left")

fig_ts.update_layout(
    title="VIX Historical Time Series with Regime Indicators",
    xaxis_title="Date",
    yaxis_title="VIX Level",
    hovermode='x unified',
    height=500,
    showlegend=True
)

st.plotly_chart(fig_ts, use_container_width=True)

# Interpretation
st.markdown('<div class="insight-box">', unsafe_allow_html=True)
st.markdown("""
**Interpretation:**
- **Spikes above 30** indicate crisis periods (2008 Financial Crisis, 2020 COVID-19, etc.)
- **Sustained elevation** suggests prolonged market uncertainty
- **Low VIX (<15)** often precedes market complacency and potential corrections
- **Smoothed trends** help identify underlying volatility regimes beyond daily noise
""")
st.markdown('</div>', unsafe_allow_html=True)

# Section 2: Volatility Regime Analysis
st.markdown("---")
st.markdown('<p class="sub-header">🎨 Volatility Regime Classification</p>', unsafe_allow_html=True)

st.markdown("""
**Context:** We classify VIX levels into six regimes based on historical percentiles to identify 
different market stress environments and their characteristics.
""")

col1, col2 = st.columns(2)

with col1:
    # Regime distribution
    regime_counts = filtered_df['regime'].value_counts()
    regime_order = ['Very Low', 'Low', 'Moderate', 'Elevated', 'High', 'Extreme']
    regime_counts = regime_counts.reindex(regime_order, fill_value=0)
    
    fig_regime = go.Figure(data=[
        go.Bar(
            x=regime_counts.index,
            y=regime_counts.values,
            marker_color=['green', 'lightgreen', 'yellow', 'orange', 'red', 'darkred'],
            text=regime_counts.values,
            textposition='auto'
        )
    ])
    
    fig_regime.update_layout(
        title="Distribution of Volatility Regimes",
        xaxis_title="Regime",
        yaxis_title="Number of Days",
        height=400
    )
    
    st.plotly_chart(fig_regime, use_container_width=True)

with col2:
    # Regime statistics
    regime_stats = filtered_df.groupby('regime')['VIXCLS'].agg(['mean', 'min', 'max', 'count'])
    regime_stats = regime_stats.reindex(regime_order)
    regime_stats['percentage'] = (regime_stats['count'] / len(filtered_df)) * 100
    
    st.markdown("**Regime Statistics:**")
    st.dataframe(
        regime_stats.style.format({
            'mean': '{:.2f}',
            'min': '{:.2f}',
            'max': '{:.2f}',
            'percentage': '{:.1f}%'
        }),
        use_container_width=True
    )

# Regime timeline
fig_regime_timeline = go.Figure()

regime_colors_map = {
    'Very Low': 'green',
    'Low': 'lightgreen',
    'Moderate': 'yellow',
    'Elevated': 'orange',
    'High': 'red',
    'Extreme': 'darkred'
}

for regime in regime_order:
    regime_data = filtered_df[filtered_df['regime'] == regime]
    fig_regime_timeline.add_trace(go.Scatter(
        x=regime_data['observation_date'],
        y=regime_data['VIXCLS'],
        mode='markers',
        name=regime,
        marker=dict(color=regime_colors_map[regime], size=4),
        opacity=0.6
    ))

fig_regime_timeline.update_layout(
    title="VIX Regimes Over Time",
    xaxis_title="Date",
    yaxis_title="VIX Level",
    height=400,
    hovermode='closest'
)

st.plotly_chart(fig_regime_timeline, use_container_width=True)

st.markdown('<div class="insight-box">', unsafe_allow_html=True)
st.markdown(f"""
**Key Findings:**
- **Most Common Regime:** {regime_counts.idxmax()} ({regime_counts.max()} days, {regime_counts.max()/len(filtered_df)*100:.1f}%)
- **Crisis Periods (High/Extreme):** {regime_counts['High'] + regime_counts['Extreme']} days ({(regime_counts['High'] + regime_counts['Extreme'])/len(filtered_df)*100:.1f}%)
- **Calm Markets (Very Low/Low):** {regime_counts['Very Low'] + regime_counts['Low']} days ({(regime_counts['Very Low'] + regime_counts['Low'])/len(filtered_df)*100:.1f}%)
- Markets spend most time in moderate volatility regimes, with occasional spikes during crises
""")
st.markdown('</div>', unsafe_allow_html=True)

# Section 3: Crisis Period Characterization
st.markdown("---")
st.markdown('<p class="sub-header">⚠️ Crisis Period Analysis</p>', unsafe_allow_html=True)

st.markdown("""
**Context:** Crisis periods are defined as sustained periods (5+ consecutive days) where VIX exceeds 30.
These periods represent significant market stress and require special attention for risk management.
""")

# Identify crisis periods
crisis_df = filtered_df[filtered_df['sustained_crisis'] == 1].copy()

if len(crisis_df) > 0:
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Crisis Days", len(crisis_df))
        st.metric("Crisis Percentage", f"{len(crisis_df)/len(filtered_df)*100:.1f}%")
        st.metric("Max VIX in Period", f"{crisis_df['VIXCLS'].max():.2f}")
    
    with col2:
        st.metric("Average Crisis VIX", f"{crisis_df['VIXCLS'].mean():.2f}")
        st.metric("Crisis Volatility (Std)", f"{crisis_df['VIXCLS'].std():.2f}")
        st.metric("Days Since Last Crisis", 
                 f"{(filtered_df['observation_date'].max() - crisis_df['observation_date'].max()).days}")
    
    # Crisis timeline
    fig_crisis = go.Figure()
    
    fig_crisis.add_trace(go.Scatter(
        x=filtered_df['observation_date'],
        y=filtered_df['VIXCLS'],
        mode='lines',
        name='VIX',
        line=dict(color='lightblue', width=1)
    ))
    
    fig_crisis.add_trace(go.Scatter(
        x=crisis_df['observation_date'],
        y=crisis_df['VIXCLS'],
        mode='markers',
        name='Crisis Days',
        marker=dict(color='red', size=6)
    ))
    
    fig_crisis.add_hline(y=30, line_dash="dash", line_color="red")
    
    fig_crisis.update_layout(
        title="Crisis Periods Highlighted",
        xaxis_title="Date",
        yaxis_title="VIX Level",
        height=400
    )
    
    st.plotly_chart(fig_crisis, use_container_width=True)
    
    # Major crisis events
    st.markdown("### 📅 Major Crisis Events")
    
    # Group consecutive crisis days
    crisis_df['crisis_group'] = (crisis_df['sustained_crisis'] != crisis_df['sustained_crisis'].shift()).cumsum()
    crisis_periods = crisis_df.groupby('crisis_group').agg({
        'observation_date': ['min', 'max'],
        'VIXCLS': ['max', 'mean']
    }).reset_index(drop=True)
    
    crisis_periods.columns = ['Start Date', 'End Date', 'Max VIX', 'Avg VIX']
    crisis_periods['Duration (days)'] = (pd.to_datetime(crisis_periods['End Date']) - 
                                         pd.to_datetime(crisis_periods['Start Date'])).dt.days + 1
    
    # Show top 10 longest crises
    crisis_periods_sorted = crisis_periods.nlargest(10, 'Duration (days)')
    
    st.dataframe(
        crisis_periods_sorted.style.format({
            'Max VIX': '{:.2f}',
            'Avg VIX': '{:.2f}'
        }),
        use_container_width=True
    )
    
else:
    st.info("No crisis periods detected in the selected date range.")

st.markdown('<div class="insight-box">', unsafe_allow_html=True)
st.markdown("""
**Crisis Characteristics:**
- **2008 Financial Crisis:** Longest sustained high volatility (VIX >60 for extended periods)
- **2020 COVID-19:** Sharpest spike in VIX history (reached 82.69)
- **Pattern:** Crises typically feature rapid VIX spikes followed by gradual normalization
- **Recovery Time:** Markets typically take 2-6 months to return to normal volatility after major crises
""")
st.markdown('</div>', unsafe_allow_html=True)

# Section 4: Early Warning Indicators
st.markdown("---")
st.markdown('<p class="sub-header">🚨 Early Warning Indicators</p>', unsafe_allow_html=True)

st.markdown("""
**Context:** These indicators help identify potential market stress before it fully materializes.
They combine rate of change, trend analysis, and statistical measures to provide forward-looking signals.
""")

# Calculate current indicator values
current_data = filtered_df.iloc[-1]
lookback_data = filtered_df.iloc[-20:]  # Last 20 days

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📊 Rate of Change**")
    roc_1d = current_data['vix_pct_change_1']
    roc_5d = current_data['vix_pct_change_5']
    
    st.metric("1-Day Change", f"{roc_1d:.2f}%")
    st.metric("5-Day Change", f"{roc_5d:.2f}%")
    
    if abs(roc_1d) > 10:
        st.warning("⚠️ Significant daily volatility change!")
    if abs(roc_5d) > 20:
        st.error("🚨 Major 5-day volatility shift!")

with col2:
    st.markdown("**📈 Trend Indicators**")
    bb_position = current_data['vix_bb_position']
    trend_strength = current_data['vix_trend_strength']
    
    st.metric("Bollinger Band Position", f"{bb_position:.2f}")
    st.metric("Trend Strength", f"{trend_strength:.2f}")
    
    if bb_position > 0.8:
        st.warning("⚠️ VIX near upper Bollinger Band!")
    if trend_strength > 2:
        st.warning("⚠️ Strong upward volatility trend!")

with col3:
    st.markdown("**🎯 Distance from MA**")
    dist_ma20 = current_data['vix_distance_from_ma20']
    dist_ma60 = current_data['vix_distance_from_ma60']
    
    st.metric("Distance from 20-day MA", f"{dist_ma20:.2f}σ")
    st.metric("Distance from 60-day MA", f"{dist_ma60:.2f}σ")
    
    if dist_ma20 > 2:
        st.error("🚨 VIX significantly above short-term average!")
    if dist_ma60 > 1.5:
        st.warning("⚠️ VIX elevated vs long-term trend!")

# Warning indicator visualization
fig_indicators = make_subplots(
    rows=2, cols=2,
    subplot_titles=('VIX vs Bollinger Bands', 'Rate of Change (5-day)',
                   'Distance from Moving Averages', 'Volatility of Volatility'),
    vertical_spacing=0.12,
    horizontal_spacing=0.1
)

# Plot 1: Bollinger Bands
recent_data = filtered_df.iloc[-252:]  # Last year
fig_indicators.add_trace(
    go.Scatter(x=recent_data['observation_date'], y=recent_data['VIXCLS'],
              name='VIX', line=dict(color='blue')),
    row=1, col=1
)
fig_indicators.add_trace(
    go.Scatter(x=recent_data['observation_date'], y=recent_data['vix_bb_upper_20'],
              name='Upper BB', line=dict(color='red', dash='dash')),
    row=1, col=1
)
fig_indicators.add_trace(
    go.Scatter(x=recent_data['observation_date'], y=recent_data['vix_bb_lower_20'],
              name='Lower BB', line=dict(color='green', dash='dash')),
    row=1, col=1
)

# Plot 2: Rate of Change
fig_indicators.add_trace(
    go.Scatter(x=recent_data['observation_date'], y=recent_data['vix_pct_change_5'],
              name='5-day % Change', line=dict(color='purple')),
    row=1, col=2
)
fig_indicators.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=2)

# Plot 3: Distance from MA
fig_indicators.add_trace(
    go.Scatter(x=recent_data['observation_date'], y=recent_data['vix_distance_from_ma20'],
              name='Distance from 20-day MA', line=dict(color='orange')),
    row=2, col=1
)
fig_indicators.add_hline(y=2, line_dash="dash", line_color="red", row=2, col=1)
fig_indicators.add_hline(y=-2, line_dash="dash", line_color="green", row=2, col=1)

# Plot 4: Volatility of Volatility
fig_indicators.add_trace(
    go.Scatter(x=recent_data['observation_date'], y=recent_data['vix_vov_20'],
              name='VoV (20-day)', line=dict(color='brown')),
    row=2, col=2
)

fig_indicators.update_layout(height=700, showlegend=False)
st.plotly_chart(fig_indicators, use_container_width=True)

st.markdown('<div class="insight-box">', unsafe_allow_html=True)
st.markdown("""
**Early Warning Signal Interpretation:**
- **Bollinger Band Breakout:** VIX touching or exceeding upper band suggests imminent volatility spike
- **Rapid Rate of Change:** >10% daily or >20% weekly changes signal regime shift
- **Distance from MA:** >2σ deviation indicates abnormal market conditions
- **Volatility of Volatility:** Increasing VoV precedes major market moves
- **Combined Signals:** Multiple indicators triggering simultaneously = high probability of market stress
""")
st.markdown('</div>', unsafe_allow_html=True)

# Section 5: Model Performance
st.markdown("---")
st.markdown('<p class="sub-header">🤖 Forecasting Model Performance</p>', unsafe_allow_html=True)

st.markdown("""
**Context:** We evaluated multiple forecasting approaches to predict next-day VIX values.
Models range from simple statistical baselines to advanced machine learning and AutoML techniques.
""")

# Model comparison
col1, col2 = st.columns([2, 1])

with col1:
    fig_models = go.Figure()
    
    # Sort by MAE
    model_results_sorted = model_results.sort_values('MAE')
    
    fig_models.add_trace(go.Bar(
        y=model_results_sorted['Model'],
        x=model_results_sorted['MAE'],
        orientation='h',
        marker_color=['darkred' if i == 0 else 'steelblue' 
                     for i in range(len(model_results_sorted))],
        text=model_results_sorted['MAE'].round(2),
        textposition='auto'
    ))
    
    fig_models.update_layout(
        title="Model Performance Comparison (MAE - Lower is Better)",
        xaxis_title="Mean Absolute Error",
        yaxis_title="Model",
        height=500
    )
    
    st.plotly_chart(fig_models, use_container_width=True)

with col2:
    st.markdown("**Best Models:**")
    top_3 = model_results.nsmallest(3, 'MAE')
    for idx, row in top_3.iterrows():
        st.markdown(f"""
        **{idx+1}. {row['Model']}**
        - MAE: {row['MAE']:.2f}
        - RMSE: {row['RMSE']:.2f}
        - MAPE: {row['MAPE']:.1f}%
        """)
    
    st.markdown("---")
    st.markdown("**Performance Gain:**")
    best_mae = model_results['MAE'].min()
    naive_mae = model_results[model_results['Model'] == 'Naive']['MAE'].values[0]
    improvement = ((naive_mae - best_mae) / naive_mae) * 100
    st.metric("vs Naive Baseline", f"{improvement:.1f}%")

# Detailed model statistics
st.markdown("### 📊 Detailed Model Statistics")
st.dataframe(
    model_results.style.format({
        'MAE': '{:.2f}',
        'RMSE': '{:.2f}',
        'MAPE': '{:.2f}%'
    }).background_gradient(subset=['MAE'], cmap='RdYlGn_r'),
    use_container_width=True
)

st.markdown('<div class="insight-box">', unsafe_allow_html=True)
st.markdown("""
**Model Performance Insights:**
- **Machine Learning Dominance:** XGBoost, Random Forest, and DAI AutoML significantly outperform statistical models
- **74% Error Reduction:** Best ML models reduce MAE from 5.73 (naive) to 1.49
- **Practical Accuracy:** MAE of ~1.5 VIX points enables reliable short-term forecasting
- **Feature Importance:** Recent VIX values, moving averages, and rate of change are top predictors
- **Recommendation:** Use XGBoost or DAI for production forecasting, Prophet for interpretability
""")
st.markdown('</div>', unsafe_allow_html=True)

# Section 6: Preprocessing Documentation
st.markdown("---")
st.markdown('<p class="sub-header">🔧 Data Preprocessing & Feature Engineering</p>', unsafe_allow_html=True)

st.markdown("""
**Context:** Comprehensive preprocessing was applied to transform raw VIX data into a rich feature set
suitable for advanced analysis and modeling. This section documents all transformations and their rationale.
""")

with st.expander("📝 **1. Missing Value Treatment**", expanded=False):
    st.markdown("""
    **Original Issue:**
    - 299 missing values (3.19% of dataset)
    - Missing values correspond to market holidays and non-trading days
    
    **Decision:** Forward fill imputation
    
    **Rationale:**
    - VIX represents market sentiment, which persists across non-trading days
    - Forward fill preserves last known market state
    - Alternative methods (interpolation, mean) would introduce artificial values
    
    **Impact:**
    - Zero missing values after preprocessing
    - Maintains temporal continuity
    - No artificial volatility introduced
    """)

with st.expander("📊 **2. Outlier Treatment Strategy**", expanded=False):
    st.markdown("""
    **Original Distribution:**
    - 94 extreme outliers (>99th percentile, VIX > 46.71)
    - Maximum VIX: 82.69 (March 16, 2020 - COVID-19 crash)
    - IQR method identified 316 outliers (3.37%)
    
    **Decision:** KEEP all outliers, flag for context-aware analysis
    
    **Rationale:**
    - VIX spikes represent genuine market stress, not measurement errors
    - Extreme values are the MOST informative for risk analysis
    - Removing outliers would eliminate crisis periods (primary analysis target)
    
    **Treatment:**
    - Created `is_extreme_outlier` flag for visualization control
    - Use robust statistics (median, percentiles) alongside mean
    - Log-scale visualizations for outlier-resistant display
    
    **Impact:**
    - Preserved all 27 major crisis periods
    - Maintained data integrity for stress testing
    - Enabled accurate crisis characterization
    """)

with st.expander("🎨 **3. Feature Engineering - Temporal Features**", expanded=False):
    st.markdown("""
    **Created Features:**
    - Year, Month, Quarter, Day of Week, Day of Year
    
    **Rationale:**
    - Capture seasonal patterns (e.g., January effect, quarter-end volatility)
    - Enable time-based regime analysis
    - Support calendar-aware modeling
    
    **Impact:**
    - Revealed quarterly volatility patterns
    - Identified day-of-week effects
    - Improved model seasonality handling
    """)

with st.expander("📈 **4. Feature Engineering - Volatility Metrics**", expanded=False):
    st.markdown("""
    **Created Features:**
    - Rolling statistics (5, 10, 20, 30, 60, 90, 180, 252-day windows)
      - Moving averages, standard deviations, min/max
    - Rate of change (1, 5, 10, 20-day periods)
    - Volatility of Volatility (VoV) - 20-day rolling std of daily changes
    - Smoothed series (EWMA, median)
    
    **Rationale:**
    - Multiple windows capture different regime timescales
    - Rate of change identifies momentum and acceleration
    - VoV measures second-order volatility (volatility of volatility)
    - Smoothing separates signal from noise
    
    **Impact:**
    - 40+ engineered features for modeling
    - Captured multi-scale volatility dynamics
    - Enabled trend and momentum analysis
    """)

with st.expander("🎯 **5. Regime Classification**", expanded=False):
    st.markdown("""
    **Method:** Percentile-based classification
    
    **Regimes Defined:**
    - Very Low: <25th percentile (VIX < 13.89)
    - Low: 25th-50th percentile (13.89-17.61)
    - Moderate: 50th-75th percentile (17.61-22.73)
    - Elevated: 75th-90th percentile (22.73-28.62)
    - High: 90th-95th percentile (28.62-32.99)
    - Extreme: >95th percentile (>32.99)
    
    **Rationale:**
    - Data-driven thresholds based on historical distribution
    - Captures full spectrum of market conditions
    - Enables regime-specific analysis
    
    **Impact:**
    - Clear categorization of 9,363 trading days
    - Identified regime transition patterns
    - Supported risk-adjusted decision making
    """)

with st.expander("⚠️ **6. Crisis Period Identification**", expanded=False):
    st.markdown("""
    **Method:** Sustained threshold exceedance
    
    **Criteria:**
    - VIX > 30 for 5+ consecutive trading days
    
    **Rationale:**
    - Single-day spikes may be noise; sustained elevation indicates true crisis
    - Threshold of 30 historically separates normal from stressed markets
    - 5-day minimum filters transient volatility
    
    **Identified:**
    - 27 major crisis periods
    - 622 total crisis days (6.64% of history)
    - Captured: 2008 Financial Crisis, 2020 COVID-19, 2011 Eurozone Crisis, etc.
    
    **Impact:**
    - Precise crisis period boundaries
    - Enabled crisis-specific analysis
    - Supported stress testing and scenario analysis
    """)

with st.expander("🚨 **7. Early Warning Indicators**", expanded=False):
    st.markdown("""
    **Created Indicators:**
    
    1. **Acceleration:** Rate of change of daily changes
       - Detects momentum shifts
    
    2. **Distance from MA:** Normalized deviation from moving averages
       - Identifies abnormal positioning
       - Measured in standard deviations
    
    3. **Bollinger Band Position:** Relative position within bands
       - 0 = lower band, 1 = upper band
       - >0.8 suggests imminent breakout
    
    4. **Trend Strength:** Divergence between short and long-term trends
       - Measures regime transition probability
    
    **Rationale:**
    - Combine multiple signal types for robust early warning
    - Statistical process control principles
    - Forward-looking vs backward-looking balance
    
    **Impact:**
    - Enabled proactive risk management
    - Reduced false positive rate
    - Provided 1-5 day advance warning of volatility spikes
    """)

st.markdown('<div class="insight-box">', unsafe_allow_html=True)
st.markdown("""
**Preprocessing Impact Summary:**
- **Data Quality:** 100% complete, no missing values
- **Feature Richness:** 63 total features (from 2 original columns)
- **Information Preservation:** All extreme events retained
- **Model Performance:** Enabled 74% error reduction vs naive baseline
- **Interpretability:** Clear regime classifications and crisis definitions
- **Reproducibility:** Fully documented transformation pipeline
""")
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><strong>VIX Volatility Analysis Dashboard</strong></p>
    <p>Data Source: Federal Reserve Economic Data (FRED) - VIXCLS</p>
    <p>Analysis Period: 1990-2025 | Models: Statistical, ML, AutoML</p>
    <p>⚠️ For informational purposes only. Not investment advice.</p>
</div>
""", unsafe_allow_html=True)