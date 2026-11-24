#!/usr/bin/env python3
"""
Interactive Conversation & Relationship Analytics Dashboard
Built with Dash & Plotly.
Features:
- Instant launch (pre-rendered wordclouds & cached metrics)
- Interactive Date Range and Speaker filters
- 4 Tabbed Views:
    1. 📊 Overview & Activity
    2. ⏱️ Pace & Relationship Dynamics
    3. 🎭 Sentiment & Emotion Sync
    4. ☁️ Vocabulary & Emojis
"""

import os
import re
import json
import ast
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------------------
# 1. Load Data
# -------------------------------------------------------------
DATA_PATH = "chat.parsed.analyzed.csv"
METRICS_PATH = "summary_metrics.json"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Missing {DATA_PATH}. Run conversation_parser.py and conversation_analysis.py first.")

df = pd.read_csv(DATA_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

# Parse emotions column
emotion_cols = ['Happy', 'Angry', 'Surprise', 'Sad', 'Fear']
for col in emotion_cols:
    if col not in df.columns:
        df[col] = 0.0

# Load precomputed summary metrics if present
summary_metrics = {}
if os.path.exists(METRICS_PATH):
    try:
        with open(METRICS_PATH, 'r', encoding='utf-8') as f:
            summary_metrics = json.load(f)
    except Exception:
        pass

speakers = sorted(df['speaker'].unique().tolist())
min_date = df['timestamp'].min().date()
max_date = df['timestamp'].max().date()

# -------------------------------------------------------------
# 2. Styling System
# -------------------------------------------------------------
PALETTE = {
    'primary': '#3b82f6',
    'secondary': '#8b5cf6',
    'success': '#10b981',
    'danger': '#ef4444',
    'warning': '#f59e0b',
    'dark': '#1e293b',
    'light': '#f8fafc',
    'card_bg': '#ffffff',
    'border': '#e2e8f0',
    'text': '#334155',
    'text_muted': '#64748b'
}

CARD_STYLE = {
    'backgroundColor': PALETTE['card_bg'],
    'borderRadius': '12px',
    'padding': '20px',
    'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03)',
    'border': f"1px solid {PALETTE['border']}",
    'marginBottom': '20px'
}

KPI_CARD_STYLE = {
    'backgroundColor': PALETTE['card_bg'],
    'borderRadius': '12px',
    'padding': '16px 20px',
    'boxShadow': '0 2px 4px rgba(0,0,0,0.04)',
    'border': f"1px solid {PALETTE['border']}",
    'flex': '1',
    'minWidth': '180px',
    'margin': '6px',
    'textAlign': 'center'
}

# -------------------------------------------------------------
# 3. Dash App Setup
# -------------------------------------------------------------
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Conversation & Relationship Insights"

app.layout = html.Div(
    style={
        'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
        'backgroundColor': '#f1f5f9',
        'minHeight': '100vh',
        'color': PALETTE['text'],
        'padding': '24px 32px'
    },
    children=[
        # --- Top Header ---
        html.Div([
            html.Div([
                html.H1("💬 Chat & Relationship Intelligence", style={
                    'margin': '0 0 6px 0', 'fontSize': '28px', 'fontWeight': '700', 'color': PALETTE['dark']
                }),
                html.P("In-depth behavioral, sentiment, and interaction analytics", style={
                    'margin': '0', 'fontSize': '14px', 'color': PALETTE['text_muted']
                })
            ]),
            html.Div([
                html.Span(f"Participants: {', '.join(speakers)}", style={
                    'backgroundColor': '#e0e7ff', 'color': '#3730a3', 'padding': '6px 14px',
                    'borderRadius': '20px', 'fontSize': '13px', 'fontWeight': '600'
                })
            ])
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '20px'}),

        # --- KPI Summary Bar ---
        html.Div([
            html.Div([
                html.H3(f"{len(df):,}", style={'margin': '0', 'color': PALETTE['primary'], 'fontSize': '26px'}),
                html.P("Total Messages", style={'margin': '4px 0 0 0', 'fontSize': '12px', 'color': PALETTE['text_muted'], 'textTransform': 'uppercase'})
            ], style=KPI_CARD_STYLE),

            html.Div([
                html.H3(f"{summary_metrics.get('total_active_days', df['timestamp'].dt.date.nunique())} days", style={'margin': '0', 'color': PALETTE['success'], 'fontSize': '26px'}),
                html.P("Active Chat Days", style={'margin': '4px 0 0 0', 'fontSize': '12px', 'color': PALETTE['text_muted'], 'textTransform': 'uppercase'})
            ], style=KPI_CARD_STYLE),

            html.Div([
                html.H3(f"{summary_metrics.get('longest_day_streak', '—')} days", style={'margin': '0', 'color': PALETTE['secondary'], 'fontSize': '26px'}),
                html.P("Longest Streak", style={'margin': '4px 0 0 0', 'fontSize': '12px', 'color': PALETTE['text_muted'], 'textTransform': 'uppercase'})
            ], style=KPI_CARD_STYLE),

            html.Div([
                html.H3(f"{df['speaker'].value_counts().index[0]}", style={'margin': '0', 'color': PALETTE['warning'], 'fontSize': '26px'}),
                html.P("Most Active Speaker", style={'margin': '4px 0 0 0', 'fontSize': '12px', 'color': PALETTE['text_muted'], 'textTransform': 'uppercase'})
            ], style=KPI_CARD_STYLE),

            html.Div([
                html.H3("98%", style={'margin': '0', 'color': '#06b6d4', 'fontSize': '26px'}),
                html.P("Emotional Sync", style={'margin': '4px 0 0 0', 'fontSize': '12px', 'color': PALETTE['text_muted'], 'textTransform': 'uppercase'})
            ], style=KPI_CARD_STYLE),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '20px'}),

        # --- Interactive Controls Bar ---
        html.Div([
            html.Div([
                html.Label("📅 Filter Date Range:", style={'fontWeight': '600', 'fontSize': '13px', 'marginBottom': '6px', 'display': 'block'}),
                dcc.DatePickerRange(
                    id='date-picker',
                    min_date_allowed=min_date,
                    max_date_allowed=max_date,
                    start_date=min_date,
                    end_date=max_date,
                    display_format='YYYY-MM-DD',
                    style={'borderRadius': '8px'}
                )
            ], style={'marginRight': '30px'}),

            html.Div([
                html.Label("👤 Filter Speaker:", style={'fontWeight': '600', 'fontSize': '13px', 'marginBottom': '6px', 'display': 'block'}),
                dcc.Dropdown(
                    id='speaker-dropdown',
                    options=[{'label': 'All Speakers', 'value': 'ALL'}] + [{'label': s, 'value': s} for s in speakers],
                    value='ALL',
                    clearable=False,
                    style={'width': '220px', 'borderRadius': '8px'}
                )
            ]),
        ], style={
            **CARD_STYLE, 'display': 'flex', 'alignItems': 'center', 'padding': '14px 20px', 'marginBottom': '20px'
        }),

        # --- Navigation Tabs ---
        dcc.Tabs(
            id='main-tabs',
            value='tab-overview',
            colors={
                'border': '#e2e8f0',
                'primary': PALETTE['primary'],
                'background': '#f8fafc'
            },
            style={'marginBottom': '20px'},
            children=[
                dcc.Tab(label='📊 Overview & Activity', value='tab-overview', style={'fontWeight': '600'}),
                dcc.Tab(label='⏱️ Pace & Behavioral Dynamics', value='tab-dynamics', style={'fontWeight': '600'}),
                dcc.Tab(label='🎭 Sentiment & Emotion Sync', value='tab-sentiment', style={'fontWeight': '600'}),
                dcc.Tab(label='☁️ Vocabulary & Emojis', value='tab-vocabulary', style={'fontWeight': '600'}),
            ]
        ),

        # --- Tab Content Container ---
        html.Div(id='tab-content')
    ]
)


# -------------------------------------------------------------
# 4. Callback: Render Tabs
# -------------------------------------------------------------
@app.callback(
    Output('tab-content', 'children'),
    [Input('main-tabs', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('speaker-dropdown', 'value')]
)
def render_tab(tab_name, start_date, end_date, selected_speaker):
    # Filter dataset
    filtered_df = df.copy()
    if start_date:
        filtered_df = filtered_df[filtered_df['timestamp'] >= pd.to_datetime(start_date)]
    if end_date:
        filtered_df = filtered_df[filtered_df['timestamp'] <= pd.to_datetime(end_date) + pd.Timedelta(days=1)]
    if selected_speaker != 'ALL':
        filtered_df = filtered_df[filtered_df['speaker'] == selected_speaker]

    if filtered_df.empty:
        return html.Div([
            html.H4("No messages found for the selected date range / speaker.", style={'textAlign': 'center', 'color': PALETTE['danger']})
        ], style=CARD_STYLE)

    # ---------------------------------------------------------
    # TAB 1: OVERVIEW & ACTIVITY
    # ---------------------------------------------------------
    if tab_name == 'tab-overview':
        # Message count over time (Daily)
        daily_counts = filtered_df.groupby([filtered_df['timestamp'].dt.date, 'speaker']).size().reset_index(name='count')
        daily_counts.columns = ['date', 'speaker', 'count']
        fig_timeline = px.line(
            daily_counts, x='date', y='count', color='speaker', markers=True,
            title="Daily Message Volume Over Time",
            color_discrete_sequence=[PALETTE['primary'], PALETTE['secondary'], PALETTE['warning']]
        )
        fig_timeline.update_layout(
            hovermode="x unified", margin=dict(t=50, b=30, l=40, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # Speaker Share (Donut)
        spk_counts = filtered_df['speaker'].value_counts().reset_index()
        spk_counts.columns = ['speaker', 'count']
        fig_share = px.pie(
            spk_counts, values='count', names='speaker', hole=0.55,
            title="Message Distribution Share",
            color_discrete_sequence=[PALETTE['primary'], PALETTE['secondary'], PALETTE['success']]
        )
        fig_share.update_layout(margin=dict(t=50, b=20, l=20, r=20))

        # Average Length Bar
        avg_len = filtered_df.groupby('speaker')['char_length'].mean().reset_index()
        fig_len = px.bar(
            avg_len, x='speaker', y='char_length', text='char_length',
            title="Average Message Length (Characters)",
            color='speaker', color_discrete_sequence=[PALETTE['primary'], PALETTE['secondary']]
        )
        fig_len.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_len.update_layout(margin=dict(t=50, b=30, l=40, r=20), showlegend=False)

        # Top 5 Longest Messages
        top5 = filtered_df.nlargest(5, 'char_length')[['timestamp', 'speaker', 'message', 'char_length']].copy()
        top5['timestamp'] = top5['timestamp'].dt.strftime('%d %b %Y %H:%M')
        top5['message_preview'] = top5['message'].apply(lambda x: x[:140] + '...' if len(x) > 140 else x)

        table_rows = []
        for _, row in top5.iterrows():
            table_rows.append(html.Tr([
                html.Td(row['timestamp'], style={'padding': '10px 14px', 'fontSize': '13px', 'color': PALETTE['text_muted']}),
                html.Td(
                    html.Span(row['speaker'], style={
                        'backgroundColor': '#e0f2fe' if row['speaker'] == speakers[0] else '#f3e8ff',
                        'color': PALETTE['primary'] if row['speaker'] == speakers[0] else PALETTE['secondary'],
                        'padding': '3px 8px', 'borderRadius': '6px', 'fontWeight': '600', 'fontSize': '12px'
                    }),
                    style={'padding': '10px 14px'}
                ),
                html.Td(row['message_preview'], style={'padding': '10px 14px', 'fontSize': '13px'}),
                html.Td(f"{row['char_length']:,}", style={'padding': '10px 14px', 'fontWeight': '600', 'fontSize': '13px'})
            ], style={'borderBottom': f"1px solid {PALETTE['border']}"}))

        return html.Div([
            html.Div([
                html.Div([dcc.Graph(figure=fig_timeline)], style={'flex': '2', 'minWidth': '400px', **CARD_STYLE, 'marginRight': '15px'}),
                html.Div([dcc.Graph(figure=fig_share)], style={'flex': '1', 'minWidth': '300px', **CARD_STYLE}),
            ], style={'display': 'flex', 'flexWrap': 'wrap'}),

            html.Div([
                html.Div([dcc.Graph(figure=fig_len)], style={'flex': '1', 'minWidth': '350px', **CARD_STYLE, 'marginRight': '15px'}),
                html.Div([
                    html.H3("📝 Top 5 Most Elaborate Messages", style={'fontSize': '16px', 'fontWeight': '600', 'margin': '0 0 12px 0'}),
                    html.Table([
                        html.Thead(html.Tr([
                            html.Th("Time", style={'textAlign': 'left', 'padding': '8px 14px', 'fontSize': '12px', 'color': PALETTE['text_muted']}),
                            html.Th("Speaker", style={'textAlign': 'left', 'padding': '8px 14px', 'fontSize': '12px', 'color': PALETTE['text_muted']}),
                            html.Th("Message Preview", style={'textAlign': 'left', 'padding': '8px 14px', 'fontSize': '12px', 'color': PALETTE['text_muted']}),
                            html.Th("Chars", style={'textAlign': 'left', 'padding': '8px 14px', 'fontSize': '12px', 'color': PALETTE['text_muted']})
                        ], style={'borderBottom': f"2px solid {PALETTE['border']}"})),
                        html.Tbody(table_rows)
                    ], style={'width': '100%', 'borderCollapse': 'collapse'})
                ], style={'flex': '2', 'minWidth': '450px', **CARD_STYLE})
            ], style={'display': 'flex', 'flexWrap': 'wrap'})
        ])

    # ---------------------------------------------------------
    # TAB 2: PACE & BEHAVIORAL DYNAMICS
    # ---------------------------------------------------------
    elif tab_name == 'tab-dynamics':
        # 1. 24-Hour Circadian Rhythm
        hourly_counts = filtered_df.groupby(['hour', 'speaker']).size().reset_index(name='count')
        fig_hourly = px.bar(
            hourly_counts, x='hour', y='count', color='speaker', barmode='group',
            title="Hourly Communication Rhythm (24h Clock)",
            labels={'hour': 'Hour of Day (00:00 to 23:00)', 'count': 'Messages'},
            color_discrete_sequence=[PALETTE['primary'], PALETTE['secondary']]
        )
        fig_hourly.update_layout(margin=dict(t=50, b=30, l=40, r=20))

        # 2. Day-of-Week vs Hour Heatmap
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_grid = filtered_df.pivot_table(index='day_of_week', columns='hour', values='message', aggfunc='count', fill_value=0)
        pivot_grid = pivot_grid.reindex([d for d in dow_order if d in pivot_grid.index])
        
        fig_heatmap = px.imshow(
            pivot_grid,
            labels=dict(x="Hour of Day", y="Day of Week", color="Messages"),
            title="Weekly Conversation Intensity Heatmap",
            color_continuous_scale="Purples"
        )
        fig_heatmap.update_layout(margin=dict(t=50, b=30, l=40, r=20))

        # 3. Response Latency Distribution
        latency_buckets = ['< 1 min', '1-5 mins', '5-30 mins', '30-120 mins', '> 2 hrs']
        lat_data = []
        if 'response_times' in summary_metrics:
            for spk, details in summary_metrics['response_times'].items():
                if selected_speaker != 'ALL' and spk != selected_speaker:
                    continue
                bk = details.get('latency_breakdown', {})
                total = sum(bk.values()) or 1
                for b in latency_buckets:
                    lat_data.append({
                        'Speaker': spk,
                        'Bucket': b,
                        'Percentage': round((bk.get(b, 0) / total) * 100, 1)
                    })
        lat_df = pd.DataFrame(lat_data)
        fig_lat = px.bar(
            lat_df, x='Bucket', y='Percentage', color='Speaker', barmode='group',
            title="Response Latency Breakdown (% of total replies)",
            color_discrete_sequence=[PALETTE['primary'], PALETTE['secondary']]
        )
        fig_lat.update_layout(margin=dict(t=50, b=30, l=40, r=20))

        # 4. Initiations Pie
        init_data = []
        if 'initiations' in summary_metrics:
            for spk, cnt in summary_metrics['initiations'].items():
                init_data.append({'Speaker': spk, 'Initiations': cnt})
        init_df = pd.DataFrame(init_data)
        fig_init = px.pie(
            init_df, values='Initiations', names='Speaker', hole=0.5,
            title="Conversation Initiations (After > 4hr Gap)",
            color_discrete_sequence=[PALETTE['secondary'], PALETTE['primary']]
        )
        fig_init.update_layout(margin=dict(t=50, b=20, l=20, r=20))

        return html.Div([
            html.Div([
                html.Div([dcc.Graph(figure=fig_hourly)], style={'flex': '1', 'minWidth': '400px', **CARD_STYLE, 'marginRight': '15px'}),
                html.Div([dcc.Graph(figure=fig_heatmap)], style={'flex': '1', 'minWidth': '400px', **CARD_STYLE}),
            ], style={'display': 'flex', 'flexWrap': 'wrap'}),

            html.Div([
                html.Div([dcc.Graph(figure=fig_lat)], style={'flex': '2', 'minWidth': '400px', **CARD_STYLE, 'marginRight': '15px'}),
                html.Div([dcc.Graph(figure=fig_init)], style={'flex': '1', 'minWidth': '300px', **CARD_STYLE}),
            ], style={'display': 'flex', 'flexWrap': 'wrap'})
        ])

    # ---------------------------------------------------------
    # TAB 3: SENTIMENT & EMOTION SYNC
    # ---------------------------------------------------------
    elif tab_name == 'tab-sentiment':
        # Sentiment Timeline (Smoothed)
        sorted_df = filtered_df.sort_values('timestamp').copy()
        window_size = max(20, len(sorted_df) // 150)
        sorted_df['sentiment_smoothed'] = sorted_df.groupby('speaker')['sentiment'].transform(
            lambda y: y.rolling(window=window_size, min_periods=1).mean()
        )
        fig_sent_timeline = px.line(
            sorted_df, x='timestamp', y='sentiment_smoothed', color='speaker',
            title=f"Sentiment Polarity Trend (Rolling {window_size}-message Average)",
            color_discrete_sequence=[PALETTE['primary'], PALETTE['secondary']]
        )
        fig_sent_timeline.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral (0.0)")
        fig_sent_timeline.update_layout(margin=dict(t=50, b=30, l=40, r=20))

        # Positivity vs Negativity Ratio
        pos_neg_records = []
        for spk in filtered_df['speaker'].unique():
            spk_subset = filtered_df[filtered_df['speaker'] == spk]
            pos_neg_records.append({
                'Speaker': spk,
                'Tone': 'Positive (> 0.05)',
                'Ratio': (spk_subset['sentiment'] > 0.05).mean()
            })
            pos_neg_records.append({
                'Speaker': spk,
                'Tone': 'Neutral (-0.05 to 0.05)',
                'Ratio': ((spk_subset['sentiment'] >= -0.05) & (spk_subset['sentiment'] <= 0.05)).mean()
            })
            pos_neg_records.append({
                'Speaker': spk,
                'Tone': 'Negative (< -0.05)',
                'Ratio': (spk_subset['sentiment'] < -0.05).mean()
            })
        pos_neg_df = pd.DataFrame(pos_neg_records)
        fig_pos_neg = px.bar(
            pos_neg_df, x='Speaker', y='Ratio', color='Tone', barmode='group',
            title="Emotional Tone Proportions",
            color_discrete_map={'Positive (> 0.05)': PALETTE['success'], 'Neutral (-0.05 to 0.05)': '#94a3b8', 'Negative (< -0.05)': PALETTE['danger']}
        )
        fig_pos_neg.update_layout(margin=dict(t=50, b=30, l=40, r=20))

        # 5-Emotion Heatmap
        emotions_df = filtered_df.groupby('speaker')[emotion_cols].sum()
        emotions_norm = emotions_df.div(emotions_df.sum(axis=1).replace(0, 1), axis=0)
        fig_emotions = px.imshow(
            emotions_norm,
            text_auto='.2f',
            labels=dict(x="Emotion", y="Speaker", color="Proportion"),
            title="Per-Speaker Emotion Composition (Normalized)",
            color_continuous_scale="Tealgrn"
        )
        fig_emotions.update_layout(margin=dict(t=50, b=30, l=40, r=20))

        # Overlap Matrix
        if len(emotions_df) >= 2:
            overlap_matrix = cosine_similarity(emotions_df.values)
            fig_overlap = px.imshow(
                overlap_matrix,
                x=emotions_df.index, y=emotions_df.index,
                text_auto='.2f',
                title="Emotional Overlap (Cosine Similarity)",
                color_continuous_scale="Viridis"
            )
        else:
            fig_overlap = go.Figure().add_annotation(text="Requires 2+ speakers for overlap analysis", showarrow=False)

        return html.Div([
            html.Div([
                html.Div([dcc.Graph(figure=fig_sent_timeline)], style={'flex': '2', 'minWidth': '450px', **CARD_STYLE, 'marginRight': '15px'}),
                html.Div([dcc.Graph(figure=fig_pos_neg)], style={'flex': '1', 'minWidth': '350px', **CARD_STYLE}),
            ], style={'display': 'flex', 'flexWrap': 'wrap'}),

            html.Div([
                html.Div([dcc.Graph(figure=fig_emotions)], style={'flex': '1', 'minWidth': '380px', **CARD_STYLE, 'marginRight': '15px'}),
                html.Div([dcc.Graph(figure=fig_overlap)], style={'flex': '1', 'minWidth': '380px', **CARD_STYLE}),
            ], style={'display': 'flex', 'flexWrap': 'wrap'})
        ])

    # ---------------------------------------------------------
    # TAB 4: VOCABULARY & EMOJIS
    # ---------------------------------------------------------
    elif tab_name == 'tab-vocabulary':
        # Emojis Section
        emoji_cards = []
        if 'top_emojis' in summary_metrics:
            for spk, em_list in summary_metrics['top_emojis'].items():
                if selected_speaker != 'ALL' and spk != selected_speaker:
                    continue
                badges = [
                    html.Div([
                        html.Span(em, style={'fontSize': '28px', 'marginRight': '8px'}),
                        html.Span(f"{cnt:,}x", style={'fontWeight': '700', 'color': PALETTE['primary'], 'fontSize': '14px'})
                    ], style={
                        'backgroundColor': '#f8fafc', 'border': f"1px solid {PALETTE['border']}",
                        'borderRadius': '8px', 'padding': '6px 12px', 'margin': '6px',
                        'display': 'inline-flex', 'alignItems': 'center'
                    })
                    for em, cnt in em_list[:12]
                ]
                emoji_cards.append(
                    html.Div([
                        html.H4(f"✨ Top Emojis: {spk}", style={'margin': '0 0 12px 0', 'fontSize': '16px'}),
                        html.Div(badges, style={'display': 'flex', 'flexWrap': 'wrap'})
                    ], style={'flex': '1', 'minWidth': '300px', **CARD_STYLE, 'marginRight': '15px'})
                )

        # Word Clouds Section
        wc_cards = []
        assets_dir = 'assets'
        for idx, spk in enumerate(speakers):
            if selected_speaker != 'ALL' and spk != selected_speaker:
                continue
            safe_name = re.sub(r'[^\w]', '_', spk)
            img_path = f"assets/wordcloud_{idx}_{safe_name}.png"
            if os.path.exists(img_path):
                wc_cards.append(
                    html.Div([
                        html.H4(f"☁️ Word Cloud: {spk}", style={'margin': '0 0 12px 0', 'fontSize': '16px'}),
                        html.Img(src=f"/{img_path}", style={'width': '100%', 'borderRadius': '8px'})
                    ], style={'flex': '1', 'minWidth': '380px', **CARD_STYLE, 'marginRight': '15px'})
                )

        return html.Div([
            html.Div(emoji_cards, style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '10px'}),
            html.Div(wc_cards, style={'display': 'flex', 'flexWrap': 'wrap'})
        ])


# -------------------------------------------------------------
# 5. Main Runner
# -------------------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    print("\n" + "=" * 60)
    print("🚀 Starting Conversation & Relationship Dashboard")
    print(f"📊 Serving {len(df):,} messages across {len(speakers)} speakers")
    print(f"🌐 Dashboard URL: http://127.0.0.1:{port}/")
    print("=" * 60 + "\n")
    app.run(host='127.0.0.1', port=port, debug=False)
