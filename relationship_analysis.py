#!/usr/bin/env python3
"""
Comprehensive Relationship Dynamics Report
Prints terminal analysis and generates relationship plots based on chat.parsed.analyzed.csv.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def print_section(title: str):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def generate_relationship_report(csv_path: str = "chat.parsed.analyzed.csv"):
    file_path = Path(csv_path)
    if not file_path.exists():
        print(f"Error: {csv_path} not found. Please run conversation_analysis.py first.")
        return

    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    speakers = df['speaker'].unique().tolist()

    # Load summary metrics if available
    summary = {}
    if Path('summary_metrics.json').exists():
        try:
            with open('summary_metrics.json', 'r', encoding='utf-8') as f:
                summary = json.load(f)
        except Exception:
            pass

    print_section("👥 PARTICIPANT OVERVIEW & ACTIVITY")
    print(f"Total Messages Analyzed: {len(df):,}")
    print(f"Active Conversation Period: {df['timestamp'].min().strftime('%d %b %Y')} to {df['timestamp'].max().strftime('%d %b %Y')}")
    
    dates = sorted(df['timestamp'].dt.date.unique())
    print(f"Total Active Days: {len(dates)} days")
    if 'longest_day_streak' in summary:
        print(f"Longest Daily Conversation Streak: {summary['longest_day_streak']} consecutive days")

    for spk in speakers:
        spk_df = df[df['speaker'] == spk]
        pct = (len(spk_df) / len(df)) * 100
        avg_len = spk_df['message'].str.len().mean()
        avg_words = spk_df['message'].str.split().str.len().mean()
        print(f"\n• {spk}:")
        print(f"  - Messages: {len(spk_df):,} ({pct:.1f}% share)")
        print(f"  - Avg Message Length: {avg_len:.1f} chars ({avg_words:.1f} words)")

    print_section("⏱️ RESPONSE TIMES & CONVERSATIONAL PACE")
    if 'response_times' in summary:
        for spk, stats in summary['response_times'].items():
            print(f"• {spk}:")
            print(f"  - Median Response Time: {stats['median_mins']:.1f} mins")
            print(f"  - Mean Response Time:   {stats['mean_mins']:.1f} mins")
            breakdown = stats.get('latency_breakdown', {})
            total_replies = sum(breakdown.values()) or 1
            print("  - Response Speed Breakdown:")
            for bucket in ['< 1 min', '1-5 mins', '5-30 mins', '30-120 mins', '> 2 hrs']:
                cnt = breakdown.get(bucket, 0)
                print(f"      {bucket:<12}: {cnt:>5} ({cnt/total_replies*100:5.1f}%)")

    print_section("🚀 INITIATIONS & BURSTINESS (DOUBLE-TEXTING)")
    if 'initiations' in summary:
        total_inits = sum(summary['initiations'].values()) or 1
        print("Who starts the conversation (after > 4hr gap)?")
        for spk, count in summary['initiations'].items():
            print(f"  - {spk:<10}: {count:>3} initiations ({count/total_inits*100:.1f}%)")

    if 'streaks' in summary:
        print("\nConsecutive message bursts (double-texting):")
        for spk, strk in summary['streaks'].items():
            print(f"  - {spk:<10}: Avg {strk['avg_streak']} msgs in a row (Max streak: {strk['max_streak']} msgs)")

    print_section("💖 SENTIMENT & EMOTIONAL SYNC")
    emotion_cols = ['Happy', 'Angry', 'Surprise', 'Sad', 'Fear']
    for spk in speakers:
        spk_df = df[df['speaker'] == spk]
        pos_ratio = (spk_df['sentiment'] > 0.05).mean() * 100
        neg_ratio = (spk_df['sentiment'] < -0.05).mean() * 100
        print(f"• {spk}: Avg Sentiment {spk_df['sentiment'].mean():.2f} (Positive: {pos_ratio:.1f}%, Negative: {neg_ratio:.1f}%)")

    # Emotional Overlap
    df_emotions = df.groupby('speaker')[emotion_cols].sum()
    overlap = df_emotions.apply(lambda x: x / x.sum() if x.sum() != 0 else x, axis=1)
    if len(speakers) >= 2:
        vec1 = overlap.loc[speakers[0]].values
        vec2 = overlap.loc[speakers[1]].values
        from numpy import dot
        from numpy.linalg import norm
        overlap_score = dot(vec1, vec2) / (norm(vec1) * norm(vec2)) if (norm(vec1) * norm(vec2)) != 0 else 0
        alignment = "High" if overlap_score > 0.75 else "Moderate" if overlap_score > 0.5 else "Low"
        print(f"\nEmotional Pattern Alignment: {overlap_score:.2f} ({alignment} Synchronization)")

    print_section("😊 EMOJI SIGNATURES")
    if 'top_emojis' in summary:
        for spk, em_list in summary['top_emojis'].items():
            em_str = " ".join([f"{em}({cnt})" for em, cnt in em_list[:8]])
            print(f"• {spk}: {em_str}")

    print("\n" + "=" * 60)
    print("✅ Full report complete.")


def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'chat.parsed.analyzed.csv'
    generate_relationship_report(csv_file)


if __name__ == '__main__':
    main()
