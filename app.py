import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import emoji
import contractions
import re
import string
from num2words import num2words
import altair as alt
from typing import Dict, List, Union

# ----- Page Configuration -----
st.set_page_config(page_title="BERT Emotion Classifier", layout="wide")

# ----- Global Constants -----
MODEL_PATH = "bert-emotion-model"
EMOTION_LABELS = ["anger", "fear", "joy", "love", "sadness", "surprise"]

# Create label-to-ID and ID-to-label mappings for the model
label_to_id: Dict[str, int] = {label: i for i, label in enumerate(EMOTION_LABELS)}
id_to_label: Dict[int, str] = {i: label for i, label in enumerate(EMOTION_LABELS)}


# ----- Model Loading -----
@st.cache_resource
def load_classifier_pipeline() -> pipeline:
    """
    Loads and caches the text classification pipeline from a fine-tuned BERT model.
    The model and tokenizer are loaded from the local MODEL_PATH.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=len(id_to_label),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    # Create a pipeline for easy text classification
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
    )


# Attempt to load the model and handle potential errors
try:
    classifier = load_classifier_pipeline()
except OSError:
    st.error(
        f"Model not found. Please ensure the '{MODEL_PATH}' directory "
        f"is in the same location as your `app.py` file."
    )
    st.stop()


# ----- Text Preprocessing Dictionaries & Functions -----

# Dictionary for mapping common emoticons to descriptive text
EMOTICON_DICT: Dict[str, str] = {
    ":-)": "happy face", ":)": "happy face", ":-]": "happy face", ":]": "happy face",
    ":->": "happy face", ":>": "happy face", "8-)": "happy face", "8)": "happy face",
    ":-}": "happy face", ":}": "happy face", ":^)": "happy face", "=]": "happy face",
    "=)": "happy face", ": )": "happy face", ":-D": "laughing", ":D": "laughing",
    "8-D": "laughing", "8D": "laughing", "=D": "laughing", "=3": "laughing",
    "B^D": "laughing", "c:": "laughing", "C:": "laughing", "x-D": "laughing",
    "xD": "laughing", "X-D": "laughing", "XD": "laughing", ":-))": "Very happy",
    ":))": "Very happy", ":-(": "sad", ":(": "sad", ":-c": "sad", ":c": "sad",
    ":-<": "sad", ":<": "sad", ":-[": "sad", ":[": "sad", ":-||": "sad",
    ":{": "sad", ":@": "sad", ";(": "sad", ":'-(": "Crying", ":'(": "Crying",
    ":=": "Crying", ":'-)": "tears of happiness", ':"D': "tears of happiness",
    ">:(": "angry", ">:[": "angry", "D-'": "horror", "D:<": "horror",
    "D:": "horror", "D8": "horror", "D;": "horror", "D=": "horror", "DX": "horror",
    ":-O": "surprise", ":O": "surprise", ":-o": "surprise", ":o": "surprise",
    ":-0": "surprise", ":0": "surprise", "8-0": "surprise", ">:O": "surprise",
    "=O": "surprise", "=o": "surprise", "=0": "surprise", ":-3": "cutesy",
    ":3": "cutesy", "x3": "cutesy", "X3": "cutesy", ">:3": "evil cat", ":-*": "kiss",
    ":*": "kiss", ":x": "kiss", ";-)": "wink", ";)": "wink", "*-)": "wink",
    "*)": "wink", ";-]": "wink", ";]": "wink", ";^)": "wink", ";>": "wink",
    ":-,": "wink", ";D": "wink", ";3": "wink", ":-P": "cheeky", ":P": "cheeky",
    "X-P": "cheeky", "XP": "cheeky", "x-p": "cheeky", "xp": "cheeky", ":-p": "cheeky",
    ":p": "cheeky", ":-Þ": "cheeky", ":Þ": "cheeky", ":-þ": "cheeky", ":þ": "cheeky",
    ":-b": "cheeky", ":b": "cheeky", "d:": "cheeky", "=p": "cheeky", ">:b": "cheeky",
    ":-/": "skeptical", ":/": "skeptical", "',:^I": "skeptical", ">:\\": "skeptical",
    ">:\\/": "skeptical", ":\\": "skeptical", "=/": "skeptical", "=\\": "skeptical",
    ":L": "skeptical", "=L": "skeptical", ":S": "skeptical", ":-|": "no expression",
    ":|": "no expression", ":$": "embarrassed", "://)": "embarrassed", "://3": "embarrassed",
    ":-X": "sealed lips", ":X": "sealed lips", ":-#": "sealed lips", ":#": "sealed lips",
    ":-&": "sealed lips", ":&": "sealed lips", "O:-)": "innocent", "O:)": "innocent",
    "0:-3": "innocent", "0:3": "innocent", "0:-)": "innocent", "0:)": "innocent",
    "0;^)": "innocent", ">:-)": "devilish", ">:)": "devilish", "}:-)": "devilish",
    "}:)": "devilish", "3:-)": "devilish", "3:)": "devilish", ">;-)": "devilish",
    ">;)": "devilish", "|;-)": "Cool", "|-O": "Cool", "B-)": "Cool",
    ":-J": "Tongue in cheek", ":J": "Tongue in cheek", "#-)": "Partied all night",
    "%-)": "confused", "%)": "confused", ":-###..": "being sick", ":###..": "being sick",
    "<:-|": "dumb", "',:-|": "disbelief", "',:-l": "disbelief", ":E": "nervous",
    "8-X": "skull and crossbones", "8=X": "skull and crossbones", "x-3": "skull and crossbones",
    "x=3": "skull and crossbones", "~:>": "chicken",
}

# Dictionary for expanding common slang and abbreviations
ABBREVIATIONS_DICT: Dict[str, str] = {
    "afaik": "as far as I know", "asap": "as soon as possible", "atm": "at the moment",
    "brb": "be right back", "btw": "by the way", "idk": "I do not know",
    "imho": "in my humble opinion", "imo": "in my opinion", "ikr": "I know, right",
    "lol": "laughing out loud", "lmao": "laughing my ass off", "rofl": "rolling on the floor laughing",
    "omg": "oh my god", "ttyl": "talk to you later", "tbh": "to be honest",
    "smh": "shaking my head", "fyi": "for your information", "ftw": "for the win",
    "fomo": "fear of missing out", "bae": "significant other", "stan": "admire extremely",
    "yeet": "to throw with force", "sus": "suspicious", "woke": "aware", "fam": "family",
    "nvm": "never mind", "rly": "really", "omw": "on my way", "lmk": "let me know",
    "ty": "thank you", "tysm": "thank you so much", "lmfao": "laughing my fucking ass off",
    "bff": "best friend forever", "hbu": "how about you", "wyd": "what are you doing",
    "tbf": "to be fair", "tmi": "too much information", "wbu": "what about you",
    "bc": "because", "cuz": "because", "cu": "see you", "cya": "see you later",
    "jk": "just kidding", "srsly": "seriously", "omfg": "oh my fucking god",
    "hmu": "hit me up", "irl": "in real life", "np": "no problem", "idc": "I do not care",
    "bday": "birthday", "gr8": "great", "l8r": "later", "b4": "before",
    "wth": "what the hell", "wya": "where you at", "gtg": "got to go",
    "icymi": "in case you missed it", "smdh": "shaking my damn head", "tldr": "too long, did not read",
    "bfn": "bye for now", "ily": "I love you", "obvi": "obviously", "txt": "text",
    "msg": "message", "totes": "totally", "rn": "right now", "w8": "wait",
    "hbd": "happy birthday", "oic": "oh, I see", "sup": "what's up", "f2f": "face to face",
    "b/c": "because", "cfn": "call for now", "dm": "direct message", "ft": "featuring",
    "ic": "I see", "nbd": "no big deal", "obv": "obviously", "pov": "point of view",
    "stfu": "shut the fuck up", "thx": "thanks", "w/e": "whatever", "w/": "with",
    "w/o": "without", "yolo": "you only live once", "bcz": "because", "bcuz": "because",
    "cos": "because", "'cause": "because", "coz": "because", "u": "you", "r": "are",
    "ur": "your", "2day": "today", "brt": "be right there", "plz": "please",
    "l8": "late", "ya": "you", "m8": "mate", "cuzn": "because", "dunno": "do not know",
    "skibidi": "cool", "goat": "greatest of all time", "goated": "performed exceptionally",
    "cap": "lie", "no cap": "no lie", "bet": "okay", "lit": "amazing", "drip": "stylish",
    "clout": "influence", "shade": "disrespect", "fire": "cool", "savage": "ruthless",
    "skrrt": "accelerate", "sksksk": "laughter", "and i oop": "excuse me",
    "big mood": "strong feeling", "tfw": "that feeling when", "extra": "over the top",
    "thicc": "curvy", "vibe": "atmosphere", "simp": "overly submissive",
    "cringe": "embarrassing", "fr": "for real", "its": "it is", "fyp": "for you page",
    "uldate": "update", "🇵🇰": "pakistan", "band": "banned", "ui": "user interface",
    "cuzz": "because", "becuz": "because", "bcs": "because", "bck": "back",
    "msgs": "messages", "ngl": "not gonna lie", "lil": "little", "cn": "china",
    "uk": "united kingdom", "ccp": "chinese communist party", "otp": "one time password",
    "essy": "easy", "usa": "united states of america", "clurb": "club",
    "fye": "for your entertainment", "app": "application", "apps": "applications",
}


# --- Preprocessing Step Functions ---

def expand_contractions(text: str) -> str:
    """Expands contractions (e.g., "don't" -> "do not")."""
    return contractions.fix(text)


def convert_emojis_and_emoticons(text: str, emoticon_map: Dict[str, str]) -> str:
    """Converts emojis and emoticons into descriptive text."""
    # Convert emojis to text (e.g., 😊 -> 'smiling face with smiling eyes')
    text = emoji.demojize(text)
    text = re.sub(r":([^:]+):", lambda m: f" {m.group(1).replace('_', ' ')} ", text)

    # Convert emoticons to text using the dictionary
    keys_sorted = sorted(emoticon_map.keys(), key=len, reverse=True)
    pattern = re.compile(r'(?<!\w)(' + '|'.join(re.escape(k) for k in keys_sorted) + r')(?!\w)')
    return pattern.sub(lambda m: emoticon_map.get(m.group(0), m.group(0)), text)


def replace_slang(text: str, slang_map: Dict[str, str]) -> str:
    """Replaces slang and abbreviations with their full forms."""
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in slang_map.keys()) + r')\b', flags=re.IGNORECASE)
    
    def replacer(match: re.Match) -> str:
        word = match.group(0)
        replacement = slang_map.get(word.lower(), word)
        # Preserve capitalization if the original word was capitalized
        return replacement.capitalize() if word[0].isupper() else replacement
        
    return pattern.sub(replacer, text)


def convert_numbers_to_words(text: str) -> str:
    """Converts numerical digits into their word equivalents."""
    def _replacer(match: re.Match) -> str:
        num_str = match.group(0)
        try:
            # Handle integers
            return num2words(int(num_str))
        except ValueError:
            # Handle floats
            try:
                integer_part, decimal_part = num_str.split('.')
                words_int = num2words(int(integer_part))
                words_dec = ' '.join([num2words(int(digit)) for digit in decimal_part])
                return f"{words_int} point {words_dec}"
            except Exception:
                return num_str  # Return original string if conversion fails
                
    return re.sub(r"\d+\.?\d*", _replacer, text)


def remove_punctuation_and_normalize(text: str) -> str:
    """Removes punctuation and normalizes whitespace."""
    translator = str.maketrans('', '', string.punctuation)
    cleaned_text = text.translate(translator)
    return re.sub(r'\s+', ' ', cleaned_text).strip()


def compress_repeated_characters(text: str) -> str:
    """Compresses sequences of 3 or more repeated characters to 2 (e.g., "soooo" -> "soo")."""
    repeat_pattern = re.compile(r"([A-Za-z])\1{2,}")
    return repeat_pattern.sub(r"\1\1", text)


# --- Main Preprocessing Pipeline ---

def preprocess_text(text_input: str) -> Dict[str, str]:
    """
    Runs the input text through the complete preprocessing pipeline and returns a
    dictionary showing the output of each step.
    """
    processing_steps: Dict[str, str] = {}
    processing_steps["Original"] = text_input
    
    # 1. Convert to lowercase for consistency
    text = text_input.lower()
    processing_steps["1. Lowercasing"] = text
    
    # 2. Expand contractions
    text = expand_contractions(text)
    processing_steps["2. Contraction Expansion"] = text
    
    # 3. Convert emojis and emoticons
    text = convert_emojis_and_emoticons(text, EMOTICON_DICT)
    processing_steps["3. Emoji & Emoticon Conversion"] = text
    
    # 4. Replace slang
    text = replace_slang(text, ABBREVIATIONS_DICT)
    processing_steps["4. Slang Replacement"] = text
    
    # 5. Convert numbers to words
    text = convert_numbers_to_words(text)
    processing_steps["5. Number Conversion"] = text
    
    # 6. Remove punctuation
    text = remove_punctuation_and_normalize(text)
    processing_steps["6. Punctuation Removal"] = text
    
    # 7. Compress repeated characters
    final_text = compress_repeated_characters(text)
    processing_steps["7. Repeat Compression (Final Output)"] = final_text
    
    return processing_steps


# ----- Utility Function -----
@st.cache_data
def convert_dataframe_to_csv(df: pd.DataFrame) -> bytes:
    """Converts a DataFrame to a CSV string for downloading."""
    # Use 'utf-8-sig' to ensure correct character encoding in Excel
    return df.to_csv(index=False).encode('utf-8-sig')


# ----- Streamlit UI -----
st.title("Text Emotion Classification with BERT")
st.markdown(
    "This application uses a fine-tuned BERT model to classify emotions from English text "
    f"into six categories: **{', '.join(EMOTION_LABELS)}**."
)

# Create tabs for different analysis modes
# tab_single, tab_batch = st.tabs(["Single Text Analysis", "CSV File Analysis"])

selected_tab = st.radio(
    "Navigation",
    options=["Single Text Analysis", "CSV File Analysis"],
    horizontal=True,
    label_visibility="collapsed"  # This hides the "Navigation" label
)

# --- Tab 1: Single Text Analysis ---
if selected_tab == "Single Text Analysis":
    st.header("Analyze a Single Piece of Text")
    text_input = st.text_area("Type or paste your text here:", height=150, key="single_text_input")

    if st.button("Classify Text", key="classify_single_button"):
        if text_input:
            with st.spinner("Analyzing..."):
                # Run preprocessing and get the final text for the model
                processing_steps = preprocess_text(text_input)
                final_processed_text = processing_steps["7. Repeat Compression (Final Output)"]

                # Get model predictions
                model_output = classifier(final_processed_text)[0]
                probabilities: Dict[str, float] = {d["label"]: d["score"] for d in model_output}
                predicted_emotion: str = max(probabilities, key=probabilities.get)

            # Display results
            st.subheader("Prediction Results")
            st.success(f"**Detected Emotion:** {predicted_emotion.capitalize()}")

            # Create two columns for results display
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Emotion Probabilities")
                df_probs = pd.DataFrame.from_dict(probabilities, orient="index", columns=["Probability"])
                df_probs = df_probs.sort_values(by="Probability", ascending=False)
                st.dataframe(df_probs.style.format("{:.2%}"))

            with col2:
                st.subheader("Probability Visualization")
                df_chart = df_probs.reset_index().rename(columns={"index": "Emotion"})

                # Create a bar chart with Altair
                chart = alt.Chart(df_chart).mark_bar().encode(
                    x=alt.X('Probability:Q', title='Probability', axis=alt.Axis(format='%')),
                    y=alt.Y('Emotion:N', sort='-x', title='Emotion'),
                    color=alt.condition(
                        alt.datum.Emotion == predicted_emotion,
                        alt.value('orange'),  # Highlight the predicted emotion
                        alt.value('lightgray')
                    ),
                    tooltip=['Emotion', alt.Tooltip('Probability', format='.2%')]
                ).properties(
                    title='Emotion Probability Distribution'
                )
                st.altair_chart(chart, use_container_width=True)

            # Show the preprocessing steps in an expander
            with st.expander("View Preprocessing Steps"):
                df_steps = pd.DataFrame.from_dict(processing_steps, orient='index', columns=['Result'])
                st.table(df_steps.rename_axis('Step'))
        else:
            st.warning("Please enter some text to analyze.")


# --- Tab 2: CSV File Analysis ---
elif selected_tab == "CSV File Analysis":
    st.header("Upload a CSV File for Batch Analysis")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Read the uploaded CSV file
            df = pd.read_csv(uploaded_file, encoding='utf-8')
            st.markdown("**Data Preview:**")
            st.dataframe(df.head())

            # Let the user select the column containing the text
            text_column_name = st.selectbox("Select the column containing the text to analyze:", df.columns.tolist())

            if st.button("Analyze CSV File", key="classify_batch_button"):
                if text_column_name:
                    with st.spinner(f"Analyzing {len(df)} rows... This may take a moment."):
                        predictions: List[Union[str, None]] = []

                        # Iterate through each text entry in the selected column
                        for text in df[text_column_name]:
                            # Ensure the entry is a valid, non-empty string
                            if isinstance(text, str) and text.strip():
                                final_processed_text = preprocess_text(text)["7. Repeat Compression (Final Output)"]
                                model_output = classifier(final_processed_text)[0]
                                predicted_emotion = max(model_output, key=lambda x: x['score'])['label']
                                predictions.append(predicted_emotion)
                            else:
                                predictions.append(None)  # Append None for empty or invalid rows

                        # Add the predictions as a new column to the DataFrame
                        results_df = df.copy()
                        results_df['predicted_emotion'] = predictions

                    st.subheader("Batch Analysis Results")
                    st.dataframe(results_df)

                    # --- Visualization of Results ---
                    st.subheader("Predicted Emotion Distribution")
                    emotion_counts = results_df['predicted_emotion'].value_counts().reset_index()
                    emotion_counts.columns = ['Emotion', 'Count']

                    # Create a donut chart with Altair
                    pie_chart = alt.Chart(emotion_counts).mark_arc(innerRadius=50).encode(
                        theta=alt.Theta(field="Count", type="quantitative"),
                        color=alt.Color(field="Emotion", type="nominal", title="Emotion"),
                        tooltip=['Emotion', 'Count']
                    ).properties(
                        title='Distribution of Emotions in the Dataset'
                    )
                    st.altair_chart(pie_chart, use_container_width=True)

                    # --- Download Button ---
                    csv_to_download = convert_dataframe_to_csv(results_df)
                    st.download_button(
                        label="Download Analysis Results (CSV)",
                        data=csv_to_download,
                        file_name="emotion_prediction_results.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")