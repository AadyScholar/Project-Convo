# 💬 Project Convo: Conversation & Relationship Analytics

An end-to-end NLP and behavioral analytics pipeline that transforms exported chat logs (WhatsApp/text exports) into deep interaction insights, sentiment tracking, emotional synchronization metrics, and an interactive Plotly Dash web dashboard.

---

## 🌟 Key Features

* **Universal WhatsApp Parser**:
  * Seamlessly supports both iOS (`[DD/MM/YY, HH:MM:SS] Name: Msg`) and Android (`DD/MM/YY, HH:MM - Name: Msg`) formats.
  * Cleans hidden bidirectional marks (`\u200e`, `\u200f`) and non-breaking spaces (`\u202f`).
  * Intelligent multi-line handling and automated filtering of system notices/media stubs.
* **Dual Sentiment & Hinglish Support**:
  * Utilizes **VADER** augmented with custom **Hinglish/chat slang** lexicons and **emoji affective weighting**.
  * Measures **TextBlob** polarity and subjectivity.
* **High-Speed Multi-Emotion Scoring**:
  * Optimized $O(1)$ lexicon-based classifier capturing 5 fundamental emotions (*Happy, Angry, Surprise, Sad, Fear*) + emojis in milliseconds.
* **Deep Behavioral & Relationship Dynamics**:
  * **Response Latency**: Calculates median/mean reply times and speed distributions (`<1 min`, `1-5 mins`, `5-30 mins`, etc.).
  * **Conversation Initiations**: Identifies who starts chats after a quiet period (>4hr lull).
  * **Double-Texting (Burstiness)**: Tracks consecutive message streaks per speaker.
  * **Circadian & Weekly Rhythm**: 24-hour activity curves and 7-day $\times$ 24-hour weekly heatmaps.
  * **Streaks**: Tracks active chat days and longest daily conversation streaks.
* **Interactive Web Dashboard**:
  * Built with **Plotly Dash** featuring date-range filters, speaker selection, and 4 dedicated tabs:
    1. 📊 *Overview & Activity*
    2. ⏱️ *Pace & Behavioral Dynamics*
    3. 🎭 *Sentiment & Emotion Sync*
    4. ☁️ *Vocabulary & Emojis*

---

## 🚀 Quick Start

### 1. Prerequisites & Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/AadyScholar/Project-Convo.git
cd Project-Convo

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

Place your exported WhatsApp chat as `chat.txt` in the root folder, then run:

```bash
python run.py
```

This single command will:
1. Parse `chat.txt` into `chat.parsed.csv`
2. Run NLP sentiment, emotion, and behavioral analytics into `chat.parsed.analyzed.csv`
3. Generate summary metrics and high-res static charts
4. Pre-render word clouds in `assets/`
5. Print a terminal summary report
6. Automatically launch the interactive dashboard at **`http://127.0.0.1:8050/`**

---

## 🛠️ Modular Execution

You can also run specific pipeline components independently:

```bash
# Parse chat only
python conversation_parser.py chat.txt -o chat.parsed.csv

# Run analysis only
python conversation_analysis.py chat.parsed.csv -o chat.parsed.analyzed.csv

# Print terminal relationship report
python relationship_analysis.py chat.parsed.analyzed.csv

# Launch interactive dashboard
python conversation_dashboard.py
```

---

## 📁 Repository Structure

```
.
├── run.py                       # Unified pipeline runner CLI
├── conversation_parser.py       # Universal WhatsApp chat parser
├── conversation_analysis.py     # NLP, VADER, emotions & interaction engine
├── relationship_analysis.py     # Terminal relationship report generator
├── conversation_dashboard.py    # Interactive Plotly Dash dashboard
├── requirements.txt             # Python dependencies
├── command.txt                  # Quick command reference
├── assets/                      # Dashboard assets & generated word clouds
├── chat.txt                     # Sample chat log (anonymized)
├── chat.parsed.csv              # Structured parsed messages
├── chat.parsed.analyzed.csv     # Enriched dataset with sentiments & emotions
└── summary_metrics.json         # Precomputed metrics for fast dashboard loading
```

---

## 📊 Dashboard Preview

* **Pace & Rhythm**: Uncover when conversations peak during the day and who responds fastest.
* **Emotional Overlap**: Measure emotional synchronization between speakers using cosine similarity.
* **Emoji Signatures**: Discover each person's most distinctive and frequently sent emojis.
* **Vocabulary Word Clouds**: Visualize frequent topics with common conversational fillers removed.

---

## 📄 License

MIT License. Feel free to use and customize for your own conversations!
