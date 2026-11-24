#!/usr/bin/env python3
"""
Enhanced Conversation Analysis Engine
Computes:
- VADER sentiment (with chat slang, emoji, and Hinglish lexicon enhancement)
- TextBlob polarity & subjectivity
- Fast, optimized multi-emotion scoring (Happy, Angry, Surprise, Sad, Fear)
- Emoji analytics (frequency, top emojis per speaker)
- Behavioral interaction metrics (response latency, initiations, double-text bursts)
- Circadian (24-hour) & day-of-week communication rhythms
- Pre-generated word clouds with normalized stopwords
- Exports enriched CSV and summary_metrics.json for instant dashboard loading
"""

import sys
import os
import re
import json
import argparse
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
from tqdm import tqdm

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import emoji
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS


# -------------------------------------------------------------
# 1. Custom Lexicons (Hinglish + Emojis)
# -------------------------------------------------------------
HINGLISH_SENTIMENT = {
    # Strong Positive
    'pyaar': 3.0, 'pyar': 3.0, 'love': 3.0, 'badiya': 2.5, 'badhiya': 2.5, 'mast': 2.8,
    'sahi': 1.8, 'shandar': 2.8, 'accha': 2.0, 'achha': 2.0, 'khoobsurat': 2.8, 'khubsurat': 2.8,
    'khush': 2.5, 'sukoon': 2.5, 'dost': 1.8, 'hasi': 2.0, 'maza': 2.5, 'mazza': 2.5,
    'tareef': 2.0, 'shukriya': 2.0, 'dhanyawad': 2.0, 'sweet': 2.2, 'cute': 2.2,
    # Mild Positive
    'theek': 1.0, 'thik': 1.0, 'hanji': 1.0, 'haanji': 1.0, 'wah': 1.5, 'waah': 1.8,
    'chalega': 1.0, 'badia': 2.2,
    # Negative / Anger / Frustration
    'bakwas': -2.5, 'bekaar': -2.5, 'bekar': -2.5, 'ganda': -2.5, 'gandi': -2.5,
    'dhokebaaz': -3.0, 'dhoke': -2.5, 'bewafa': -2.8, 'pagal': -1.5, 'pagli': -1.2,
    'ghanta': -2.0, 'rona': -2.2, 'dukh': -2.5, 'pareshan': -2.2, 'musibat': -2.5,
    'jhuth': -2.5, 'jhooth': -2.5, 'nafrat': -3.0, 'laat': -1.8, 'marenge': -1.8,
    'kalesh': -2.2, 'lafda': -2.0, 'jhagda': -2.2, 'chup': -1.5, 'hatt': -1.5, 'hat': -1.2,
    'gaddari': -2.5, 'korbe': -1.0,
    # Emojis for VADER
    '❤️': 3.2, '💖': 3.0, '💕': 3.0, '🥰': 3.0, '😍': 3.2, '😘': 2.8,
    '😂': 2.0, '🤣': 2.2, '😄': 1.8, '😊': 1.8, '😁': 1.8, '🥳': 2.5,
    '🔥': 2.0, '✨': 1.8, '👍': 1.5, '🎉': 2.2, '🥺': 0.8, '🥹': 1.5,
    '😭': -1.8, '😢': -2.0, '💔': -3.0, '😞': -2.0, '😔': -2.0, '😩': -1.8,
    '😡': -2.8, '😠': -2.5, '🤬': -3.5, '🖕': -3.0, '😤': -1.8, '😑': -1.5,
    '🤮': -2.8, '🤢': -2.5, '💀': 1.2, '😏': 1.0
}

EMOJI_EMOTIONS = {
    '😂': 'Happy', '🤣': 'Happy', '😊': 'Happy', '😄': 'Happy', '🥰': 'Happy',
    '😍': 'Happy', '❤️': 'Happy', '💖': 'Happy', '🎉': 'Happy', '✨': 'Happy',
    '🥳': 'Happy', '😁': 'Happy', '💃': 'Happy', '🕺': 'Happy', '👍': 'Happy',
    '😭': 'Sad', '😢': 'Sad', '🥺': 'Sad', '💔': 'Sad', '😞': 'Sad', '😔': 'Sad', '😩': 'Sad',
    '😡': 'Angry', '😠': 'Angry', '🤬': 'Angry', '🖕': 'Angry', '😤': 'Angry', '🤮': 'Angry',
    '😱': 'Fear', '😨': 'Fear', '😰': 'Fear', '😬': 'Fear', '🫣': 'Fear',
    '😮': 'Surprise', '😲': 'Surprise', '🤯': 'Surprise', '😳': 'Surprise', '🫢': 'Surprise'
}

EMOTION_COLS = ['Happy', 'Angry', 'Surprise', 'Sad', 'Fear']


# -------------------------------------------------------------
# 2. Fast O(1) Emotion Lexicon Setup
# -------------------------------------------------------------
def build_fast_emotion_lexicon():
    """Extract and compile text2emotion lexicon into an O(1) dictionary."""
    lexicon = {}
    try:
        import text2emotion as te
        import inspect
        src = inspect.getsource(te.get_emotion)
        match = re.search(r"df = \{'Word': (\[.*?\]),\s*'Emotion': (\[.*?\])\}", src, re.DOTALL)
        if match:
            import ast
            words = ast.literal_eval(match.group(1))
            emotions = ast.literal_eval(match.group(2))
            lexicon = dict(zip([w.lower() for w in words], emotions))
    except Exception as e:
        print(f"Warning: Could not extract full text2emotion dictionary: {e}")

    # Merge emoji emotion mappings
    lexicon.update(EMOJI_EMOTIONS)
    return lexicon


FAST_EMOTION_LEXICON = build_fast_emotion_lexicon()


def extract_emotions_fast(text: str) -> dict:
    text_lower = str(text).lower()
    counts = {e: 0.0 for e in EMOTION_COLS}
    tokens = re.findall(r'\w+|[^\w\s]', text_lower)
    total_matches = 0
    for t in tokens:
        if t in FAST_EMOTION_LEXICON:
            emotion = FAST_EMOTION_LEXICON[t]
            if emotion in counts:
                counts[emotion] += 1.0
                total_matches += 1

    if total_matches > 0:
        return {e: round(counts[e] / total_matches, 3) for e in EMOTION_COLS}
    return {e: 0.0 for e in EMOTION_COLS}


# -------------------------------------------------------------
# 3. Stopwords List (Normalized Lowercase)
# -------------------------------------------------------------
CLEAN_STOPWORDS = set(STOPWORDS)
ADDITIONAL_STOPWORDS = {
    # System & Media
    'omitted', 'sticker', 'image', 'video', 'audio', 'deleted', 'message', 'messages',
    'edited', 'waiting', 'calls', 'photo', 'gif', 'media', 'document', 'contact', 'call',
    'pm', 'am', 'http', 'https', 'www', 'com',
    # Common English chat fillers
    'yeah', 'oh', 'ok', 'okay', 'right', 'got', 'thats', 'what', 'will', 'shall', 'let',
    'sure', 'maybe', 'good', 'well', 'even', 'can', 'could', 'would', 'just', 'now', 'did',
    'like', 'also', 'dont', 'isnt', 'wont', 'hey', 'hmm', 'hmmm', 'pls', 'huh', 'bye',
    'wow', 'thanks', 'thank', 'great', 'yes', 'hi', 'yea', 'yep', 'nope', 'bro', 'dude',
    'guy', 'guys', 'lol', 'lmao', 'rofl',
    # Hindi / Hinglish pronouns, prepositions, auxiliaries
    'main', 'mein', 'hai', 'hu', 'hoon', 'hun', 'hain', 'thi', 'tha', 'the', 'hona',
    'kar', 'karo', 'karne', 'kiya', 'kiye', 'ke', 'ko', 'ka', 'ki', 'yeh', 'woh', 'ye',
    'wo', 'ho', 'ha', 'raha', 'rha', 'rahi', 'rhi', 'rhe', 'rahe', 'par', 'pe', 'jab',
    'tab', 'to', 'toh', 'tu', 'tum', 'mera', 'mere', 'meri', 'apna', 'apni', 'apne',
    'kuch', 'ab', 'abhi', 'kab', 'kabhi', 'koi', 'bhi', 'sab', 'sabko', 'aur', 'se',
    'bahut', 'bohot', 'thoda', 'thodi', 'kyu', 'kyun', 'kyon', 'kis', 'kaun', 'kaha',
    'kahan', 'aisa', 'aise', 'aisi', 'aisehi', 'kitna', 'kitni', 'kitne', 'uska', 'uske',
    'uski', 'unka', 'unke', 'unki', 'iska', 'iske', 'iski', 'inka', 'inke', 'inki',
    'inhe', 'unhe', 'vo', 'phir', 'fir', 'tak', 'tan', 'bahar', 'andar', 'na', 'nahi',
    'nhi', 'mat', 'kya', 'haan', 'han', 'hui', 'hua', 'hue', 'meko', 'teko', 'tujhe',
    'mujhe', 'ek', 'do', 'teen', 'char', 'aree', 'arey', 'yaar', 'bhai', 'behen', 'waise',
    'waisa', 'suno', 'bolo', 'dekh', 'dekho', 'dekhna', 'chal', 'chalo', 'chalega', 'gaya',
    'gayi', 'gaye', 'gya', 'gyi', 'gye', 'diya', 'diye', 'liya', 'liye', 'lekin', 'magar',
    'aaj', 'kal', 'din', 'saal', 'sal', 'mahina', 'time', 'samay', 'bar', 'baar', 'bas'
}
CLEAN_STOPWORDS.update(ADDITIONAL_STOPWORDS)


# -------------------------------------------------------------
# 4. Main Analysis Routine
# -------------------------------------------------------------
def analyze_conversation(input_path: str, output_path: str = 'chat.parsed.analyzed.csv'):
    print(f"📊 Loading parsed chat from: {input_path}")
    df = pd.read_csv(input_path)

    df['message'] = df['message'].fillna('').astype(str)
    df = df[df['message'].str.strip() != ''].copy()

    # Ensure timestamp is parsed properly (parsed CSV already uses YYYY-MM-DD HH:MM:SS format)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    print(f"✓ Loaded {len(df):,} valid messages from {df['speaker'].nunique()} speakers.")

    # ---------------------------------------------------------
    # A. Sentiment Analysis (VADER + Hinglish & TextBlob)
    # ---------------------------------------------------------
    print("🧠 Computing VADER + Hinglish sentiment scores...")
    vader = SentimentIntensityAnalyzer()
    vader.lexicon.update(HINGLISH_SENTIMENT)

    vader_compounds = []
    vader_positives = []
    vader_negatives = []
    textblob_polarities = []
    textblob_subjectivities = []

    for msg in tqdm(df['message'], desc="Sentiment"):
        scores = vader.polarity_scores(msg)
        vader_compounds.append(round(scores['compound'], 4))
        vader_positives.append(round(scores['pos'], 4))
        vader_negatives.append(round(scores['neg'], 4))

        tb = TextBlob(msg)
        textblob_polarities.append(round(tb.sentiment.polarity, 4))
        textblob_subjectivities.append(round(tb.sentiment.subjectivity, 4))

    df['vader_compound'] = vader_compounds
    df['vader_pos'] = vader_positives
    df['vader_neg'] = vader_negatives
    df['textblob_polarity'] = textblob_polarities
    df['textblob_subjectivity'] = textblob_subjectivities

    # Composite 'sentiment' column (VADER compound is primary for chat)
    df['sentiment'] = df['vader_compound']

    # ---------------------------------------------------------
    # B. Emoji Analytics
    # ---------------------------------------------------------
    print("😊 Extracting and analyzing emojis...")
    all_extracted_emojis = []
    for msg in df['message']:
        msg_emojis = [e['emoji'] for e in emoji.emoji_list(msg)]
        all_extracted_emojis.append(msg_emojis)

    df['emojis'] = all_extracted_emojis
    df['emoji_count'] = df['emojis'].apply(len)

    speaker_emojis = {}
    for spk in df['speaker'].unique():
        speaker_msgs_emojis = df[df['speaker'] == spk]['emojis']
        flat = [e for sublist in speaker_msgs_emojis for e in sublist]
        speaker_emojis[spk] = Counter(flat).most_common(15)

    # ---------------------------------------------------------
    # C. Fast Emotion Analysis
    # ---------------------------------------------------------
    print("🎭 Extracting multi-emotion vectors (Happy, Angry, Surprise, Sad, Fear)...")
    emotion_records = []
    for msg in tqdm(df['message'], desc="Emotions"):
        emotion_records.append(extract_emotions_fast(msg))

    emotion_df = pd.DataFrame(emotion_records)
    for col in EMOTION_COLS:
        df[col] = emotion_df[col]

    # Stringified dict column for backward compatibility
    df['emotions'] = emotion_df.apply(lambda r: str(r.to_dict()), axis=1)

    # ---------------------------------------------------------
    # D. Behavioral & Interaction Dynamics
    # ---------------------------------------------------------
    print("⏱️ Computing response latency, initiations, and conversational streaks...")
    SESSION_GAP_HOURS = 4.0
    response_times = {spk: [] for spk in df['speaker'].unique()}
    initiations = {spk: 0 for spk in df['speaker'].unique()}
    last_words = {spk: 0 for spk in df['speaker'].unique()}
    streaks = {spk: [] for spk in df['speaker'].unique()}

    prev_row = None
    curr_streak_speaker = None
    curr_streak_count = 0

    gap_seconds_list = []
    is_response_list = []
    is_initiation_list = []

    for i, row in df.iterrows():
        spk = row['speaker']
        ts = row['timestamp']

        if prev_row is None:
            initiations[spk] += 1
            curr_streak_speaker = spk
            curr_streak_count = 1
            gap_seconds_list.append(0.0)
            is_response_list.append(False)
            is_initiation_list.append(True)
        else:
            gap_sec = (ts - prev_row['timestamp']).total_seconds()
            gap_hours = gap_sec / 3600.0
            gap_seconds_list.append(gap_sec)

            if gap_hours >= SESSION_GAP_HOURS:
                initiations[spk] += 1
                last_words[prev_row['speaker']] += 1
                is_initiation_list.append(True)
            else:
                is_initiation_list.append(False)

            if spk != prev_row['speaker']:
                is_response_list.append(True)
                if gap_hours < 12.0:
                    response_times[spk].append(gap_sec)
                streaks[curr_streak_speaker].append(curr_streak_count)
                curr_streak_speaker = spk
                curr_streak_count = 1
            else:
                is_response_list.append(False)
                curr_streak_count += 1

        prev_row = row

    if curr_streak_speaker:
        streaks[curr_streak_speaker].append(curr_streak_count)
    if prev_row is not None:
        last_words[prev_row['speaker']] += 1

    df['gap_seconds'] = gap_seconds_list
    df['is_response'] = is_response_list
    df['is_initiation'] = is_initiation_list

    # Daily streaks
    unique_dates = sorted(df['timestamp'].dt.date.unique())
    longest_day_streak = 0
    curr_streak = 1
    for d_idx in range(1, len(unique_dates)):
        if (unique_dates[d_idx] - unique_dates[d_idx-1]).days == 1:
            curr_streak += 1
        else:
            longest_day_streak = max(longest_day_streak, curr_streak)
            curr_streak = 1
    longest_day_streak = max(longest_day_streak, curr_streak)

    # ---------------------------------------------------------
    # E. Pre-Generate Word Clouds into assets/
    # ---------------------------------------------------------
    assets_dir = Path('assets')
    assets_dir.mkdir(exist_ok=True)
    print("☁️ Pre-generating word clouds for each speaker...")
    for idx, speaker in enumerate(df['speaker'].unique()):
        spk_text = " ".join(df[df['speaker'] == speaker]['message'].astype(str))
        if spk_text.strip():
            safe_name = re.sub(r'[^\w]', '_', speaker)
            img_filename = f"wordcloud_{idx}_{safe_name}.png"
            img_path = assets_dir / img_filename

            wc = WordCloud(
                width=800,
                height=400,
                background_color='white',
                stopwords=CLEAN_STOPWORDS,
                collocations=False,
                max_words=120,
                relative_scaling=0.5,
                min_font_size=10
            ).generate(spk_text)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            plt.savefig(img_path, format='png', bbox_inches='tight', dpi=100)
            plt.close(fig)
            print(f"  ✓ Word cloud saved: {img_path}")

    # ---------------------------------------------------------
    # F. Save Enriched CSV & Summary Metrics
    # ---------------------------------------------------------
    # Save CSV
    df.to_csv(output_path, index=False)
    print(f"💾 Enriched data saved to: {output_path}")

    # Response latency categorization
    def categorize_latency(secs):
        if secs < 60:
            return '< 1 min'
        elif secs < 300:
            return '1-5 mins'
        elif secs < 1800:
            return '5-30 mins'
        elif secs < 7200:
            return '30-120 mins'
        else:
            return '> 2 hrs'

    latency_breakdown = {}
    for spk in df['speaker'].unique():
        cats = [categorize_latency(s) for s in response_times[spk]]
        latency_breakdown[spk] = dict(Counter(cats))

    # Compile summary JSON for dashboard
    summary = {
        'total_messages': len(df),
        'unique_speakers': df['speaker'].unique().tolist(),
        'date_range': [str(df['timestamp'].min()), str(df['timestamp'].max())],
        'total_active_days': len(unique_dates),
        'longest_day_streak': longest_day_streak,
        'speaker_stats': {},
        'initiations': {spk: int(cnt) for spk, cnt in initiations.items()},
        'last_words': {spk: int(cnt) for spk, cnt in last_words.items()},
        'streaks': {
            spk: {
                'avg_streak': round(float(np.mean(s)), 2) if len(s) else 0.0,
                'max_streak': int(np.max(s)) if len(s) else 0
            }
            for spk, s in streaks.items()
        },
        'response_times': {
            spk: {
                'median_mins': round(float(np.median(t)) / 60.0, 2) if len(t) else 0.0,
                'mean_mins': round(float(np.mean(t)) / 60.0, 2) if len(t) else 0.0,
                'sample_count': len(t),
                'latency_breakdown': latency_breakdown[spk]
            }
            for spk, t in response_times.items()
        },
        'top_emojis': {spk: [list(item) for item in ems] for spk, ems in speaker_emojis.items()}
    }

    for spk in df['speaker'].unique():
        spk_df = df[df['speaker'] == spk]
        summary['speaker_stats'][spk] = {
            'message_count': int(len(spk_df)),
            'avg_length': round(float(spk_df['message'].str.len().mean()), 1),
            'avg_sentiment': round(float(spk_df['sentiment'].mean()), 3),
            'positive_ratio': round(float((spk_df['sentiment'] > 0.05).mean()), 3),
            'negative_ratio': round(float((spk_df['sentiment'] < -0.05).mean()), 3),
            'neutral_ratio': round(float(((spk_df['sentiment'] >= -0.05) & (spk_df['sentiment'] <= 0.05)).mean()), 3),
            'total_emojis': int(spk_df['emoji_count'].sum())
        }

    summary_file = Path('summary_metrics.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"📊 Summary metrics saved to: {summary_file}")

    # ---------------------------------------------------------
    # G. Generate High-Res Static Summary Plots
    # ---------------------------------------------------------
    generate_static_plots(df, summary)

    print("\n✅ Conversation Analysis Complete!")
    return df, summary


def generate_static_plots(df: pd.DataFrame, summary: dict):
    print("📈 Generating high-resolution static charts...")
    sns.set_theme(style="whitegrid")

    # 1. Speaker Dominance
    plt.figure(figsize=(8, 5))
    spk_counts = df['speaker'].value_counts().reset_index()
    spk_counts.columns = ['speaker', 'count']
    ax = sns.barplot(data=spk_counts, x='speaker', y='count', hue='speaker', palette='Blues_r', legend=False)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=11, xytext=(0, 5), textcoords='offset points')
    plt.title('Speaker Dominance (Message Volume)', fontsize=14, fontweight='bold')
    plt.xlabel('Speaker', fontsize=12)
    plt.ylabel('Message Count', fontsize=12)
    plt.tight_layout()
    plt.savefig('speaker_dominance.png', dpi=150)
    plt.close()

    # 2. Sentiment Distribution
    plt.figure(figsize=(9, 5))
    sns.kdeplot(data=df, x='sentiment', hue='speaker', common_norm=False, fill=True, alpha=0.35)
    plt.title('Sentiment Density Distribution per Speaker', fontsize=14, fontweight='bold')
    plt.xlabel('Sentiment Polarity (-1.0 Negative to +1.0 Positive)', fontsize=12)
    plt.tight_layout()
    plt.savefig('sentiment_distribution.png', dpi=150)
    plt.close()

    # 3. Emotion Heatmap
    df_emotions = df.groupby('speaker')[EMOTION_COLS].sum()
    df_emotions_norm = df_emotions.div(df_emotions.sum(axis=1).replace(0, 1), axis=0)
    plt.figure(figsize=(8, 5))
    sns.heatmap(df_emotions_norm, annot=True, fmt='.2f', cmap='YlGnBu')
    plt.title('Emotion Composition per Speaker', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('emotion_heatmap.png', dpi=150)
    plt.close()

    # 4. Positivity vs Negativity
    pos_neg_records = []
    for spk, stats in summary['speaker_stats'].items():
        pos_neg_records.append({
            'speaker': spk,
            'Positive': stats['positive_ratio'],
            'Negative': stats['negative_ratio'],
            'Neutral': stats['neutral_ratio']
        })
    pos_neg_df = pd.DataFrame(pos_neg_records).melt(id_vars='speaker', var_name='Tone', value_name='Ratio')
    plt.figure(figsize=(8, 5))
    sns.barplot(data=pos_neg_df, x='speaker', y='Ratio', hue='Tone', palette=['#2ecc71', '#e74c3c', '#95a5a6'])
    plt.title('Positivity vs Negativity Proportions', fontsize=14, fontweight='bold')
    plt.ylabel('Proportion', fontsize=12)
    plt.tight_layout()
    plt.savefig('positivity_negativity.png', dpi=150)
    plt.close()

    # 5. Hourly Activity Distribution (24-Hour Circadian Rhythm)
    plt.figure(figsize=(10, 5))
    hourly_df = df.groupby(['hour', 'speaker']).size().unstack(fill_value=0)
    hourly_df.plot(kind='bar', stacked=False, figsize=(11, 5), colormap='tab10', width=0.8)
    plt.title('Activity by Hour of the Day (24h Clock)', fontsize=14, fontweight='bold')
    plt.xlabel('Hour (00:00 to 23:00)', fontsize=12)
    plt.ylabel('Messages', fontsize=12)
    plt.legend(title='Speaker')
    plt.tight_layout()
    plt.savefig('activity_hourly.png', dpi=150)
    plt.close()

    # 6. Response Time Breakdown
    latency_data = []
    order = ['< 1 min', '1-5 mins', '5-30 mins', '30-120 mins', '> 2 hrs']
    for spk, details in summary['response_times'].items():
        breakdown = details['latency_breakdown']
        total = sum(breakdown.values()) or 1
        for bucket in order:
            cnt = breakdown.get(bucket, 0)
            latency_data.append({
                'Speaker': spk,
                'Bucket': bucket,
                'Percentage': round((cnt / total) * 100, 1)
            })
    lat_df = pd.DataFrame(latency_data)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=lat_df, x='Bucket', y='Percentage', hue='Speaker', palette='Set2')
    plt.title('Response Latency Distribution (% of replies)', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage of Replies (%)', fontsize=12)
    plt.tight_layout()
    plt.savefig('response_times.png', dpi=150)
    plt.close()

    print("✓ Saved static charts: speaker_dominance.png, sentiment_distribution.png, emotion_heatmap.png, positivity_negativity.png, activity_hourly.png, response_times.png")


def main():
    parser = argparse.ArgumentParser(description="Enhanced Conversation Analysis Engine")
    parser.add_argument('positional_input', nargs='?', default=None, help="Input parsed CSV file (e.g. chat.parsed.csv)")
    parser.add_argument('-i', '--input', default=None, help="Input parsed CSV file")
    parser.add_argument('-o', '--output', default='chat.parsed.analyzed.csv', help="Output analyzed CSV file")

    args = parser.parse_args()
    input_file = args.input or args.positional_input or 'chat.parsed.csv'
    output_file = args.output

    analyze_conversation(input_file, output_file)


if __name__ == '__main__':
    main()
