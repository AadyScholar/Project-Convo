#!/usr/bin/env python3
"""
Universal WhatsApp Chat Parser
Supports iOS and Android export formats, cleans hidden unicode marks,
filters system notifications/media omissions, and formats timestamps.
"""

import re
import sys
import argparse
from pathlib import Path
import pandas as pd


# Regex patterns covering common WhatsApp export variants (iOS and Android, 12h and 24h)
PATTERNS = [
    # iOS: [11/06/26, 3:54:54 PM] Name: Message or [2026-06-11, 15:54:54] Name: Message
    re.compile(r'^\[(\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4},?\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?)\]\s+([^:]+?):\s*(.*)$'),
    # Android: 11/06/26, 3:54 PM - Name: Message or 11/06/2026, 15:54 - Name: Message
    re.compile(r'^(\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4},?\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?)\s+-\s+([^:]+?):\s*(.*)$'),
]

SYSTEM_PATTERNS = [
    'messages and calls are end-to-end encrypted',
    'this message was deleted',
    'you deleted this message',
    '<this message was edited>',
    'image omitted',
    'sticker omitted',
    'video omitted',
    'audio omitted',
    'gif omitted',
    'document omitted',
    'contact card omitted',
    'location omitted',
    'missed voice call',
    'missed video call',
    'security code changed',
    'changed the subject to',
    'changed the group description',
    'joined using this group\'s invite link',
]


def clean_line(line: str) -> str:
    """Strip hidden unicode marks, directional marks, and non-breaking spaces."""
    # LTR, RTL, pop directional, BOM
    for char in ('\u200e', '\u200f', '\u202a', '\u202c', '\u202e', '\ufeff'):
        line = line.replace(char, '')
    # Narrow no-break space & non-breaking space
    line = line.replace('\u202f', ' ').replace('\xa0', ' ')
    return line.strip()


def is_system_message(msg: str) -> bool:
    """Check if message is automated WhatsApp notification or media stub."""
    lower_msg = msg.lower().strip()
    return any(p in lower_msg for p in SYSTEM_PATTERNS) or lower_msg.startswith(('jpg', 'png', 'gif', 'mp4'))


def parse_chat(input_path: str, output_path: str) -> pd.DataFrame:
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input chat file not found: {input_path}")

    data = []
    current_ts = None
    current_speaker = None
    current_msg = []

    print(f"📖 Reading and parsing chat from: {input_path}")
    with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            cleaned = clean_line(line)
            if not cleaned:
                continue

            match = None
            for pattern in PATTERNS:
                m = pattern.match(cleaned)
                if m:
                    match = m
                    break

            if match:
                # Save the completed previous message
                if current_speaker and current_msg:
                    full_message = '\n'.join(current_msg).strip()
                    if full_message and not is_system_message(full_message):
                        data.append((current_ts, current_speaker, full_message))

                current_ts = match.group(1).strip()
                current_speaker = match.group(2).strip()
                current_msg = [match.group(3).strip()]
            else:
                # Continuation of multiline message
                if current_msg is not None:
                    current_msg.append(cleaned)

    # Append trailing message
    if current_speaker and current_msg:
        full_message = '\n'.join(current_msg).strip()
        if full_message and not is_system_message(full_message):
            data.append((current_ts, current_speaker, full_message))

    df = pd.DataFrame(data, columns=['raw_timestamp', 'speaker', 'message'])
    print(f"✓ Extracted {len(df)} candidate message rows.")

    # Parse and standardize timestamps
    print("⏳ Standardizing timestamps...")
    # Try exact format first for speed and consistency
    try:
        df['timestamp'] = pd.to_datetime(df['raw_timestamp'], format='%d/%m/%y, %I:%M:%S %p', errors='coerce')
    except Exception:
        df['timestamp'] = pd.NaT

    if df['timestamp'].isna().any():
        # Fallback to general parsing with dayfirst=True
        df['timestamp'] = df['timestamp'].fillna(
            pd.to_datetime(df['raw_timestamp'], format='mixed', dayfirst=True, errors='coerce')
        )

    # Drop any remaining unparseable rows
    df = df.dropna(subset=['timestamp', 'message'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Add useful temporal & textual features
    df['date'] = df['timestamp'].dt.date.astype(str)
    df['time'] = df['timestamp'].dt.time.astype(str)
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['char_length'] = df['message'].str.len()
    df['word_count'] = df['message'].str.split().str.len()

    # Format timestamp to standard readable string
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df = df.drop(columns=['raw_timestamp'])

    # Reorder columns
    col_order = ['timestamp', 'date', 'time', 'day_of_week', 'hour', 'speaker', 'message', 'char_length', 'word_count']
    df = df[col_order]

    # Save to CSV
    output_file = Path(output_path)
    df.to_csv(output_file, index=False)
    print(f"✅ Successfully parsed {len(df)} messages across {df['speaker'].nunique()} speakers.")
    print("Speaker Breakdown:")
    for spk, cnt in df['speaker'].value_counts().items():
        print(f"  - {spk}: {cnt:,} messages")
    print(f"💾 Saved cleanly to: {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Universal WhatsApp Chat Parser")
    parser.add_argument('positional_input', nargs='?', default=None, help="Input raw chat file (e.g. chat.txt)")
    parser.add_argument('-i', '--input', default=None, help="Input raw chat file")
    parser.add_argument('-o', '--output', default='chat.parsed.csv', help="Output parsed CSV file path")

    args = parser.parse_args()
    input_file = args.input or args.positional_input or 'chat.txt'
    output_file = args.output

    parse_chat(input_file, output_file)


if __name__ == '__main__':
    main()
