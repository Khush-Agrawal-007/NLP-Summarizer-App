# app/services/summarizer.py
import string
from collections import Counter

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

from app.utils.model_utils import load_model_tokenizer, generate_summary
from app.core.config import settings


class SummarizerService:
    """General-purpose text summarizer using BART with abstractive and extractive modes."""

    def __init__(self):
        self.model_name = settings.MODEL_NAME
        self.min_length = settings.MIN_SUMMARY_LENGTH
        self.max_length = settings.MAX_SUMMARY_LENGTH
        self.model, self.tokenizer, self.device = load_model_tokenizer(self.model_name)
        self.stop_words = set(stopwords.words('english'))

    # ------------------------------------------------------------------
    #  Abstractive Summarization
    # ------------------------------------------------------------------

    def abstractive_summarize(self, text: str) -> str:
        """Generate an abstractive summary, handling long texts via chunking."""
        if not text or not text.strip():
            return "No text provided for summarization."

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        # Leave a small buffer for special tokens
        max_chunk_length = min(self.tokenizer.model_max_length - 50, 1024)

        # Text fits in a single forward pass
        if len(tokens) <= max_chunk_length:
            return self._run_summary(text, len(tokens))

        # Text is too long — split into sentence chunks and summarize each
        chunks = self._split_into_chunks(text, max_chunk_length)
        chunk_summaries = []
        for chunk in chunks:
            chunk_token_count = len(self.tokenizer.encode(chunk, add_special_tokens=False))
            result = self._run_summary(chunk, chunk_token_count)
            if result:
                chunk_summaries.append(result)

        if not chunk_summaries:
            return "Error: Unable to generate summary from chunks."

        combined = " ".join(chunk_summaries)

        # If the combined chunk summaries are still too long, run one final pass
        combined_tokens = self.tokenizer.encode(combined, add_special_tokens=False)
        if len(combined_tokens) > max_chunk_length:
            return self._run_summary(combined, len(combined_tokens))

        return combined

    def _run_summary(self, text: str, token_count: int) -> str:
        """Run beam-search generation on a single text block with dynamic length bounds."""
        max_model_half = self.tokenizer.model_max_length // 2
        dynamic_max = min(max(self.max_length, int(token_count * 0.4)), max_model_half)
        dynamic_min = min(self.min_length, dynamic_max - 10)

        try:
            return generate_summary(
                model=self.model,
                tokenizer=self.tokenizer,
                text=text,
                device=self.device,
                max_length=dynamic_max,
                min_length=dynamic_min,
            )
        except Exception as e:
            print(f"Summarization error: {e}")
            return ""

    def _split_into_chunks(self, text: str, max_chunk_length: int) -> list:
        """Split text into token-safe chunks aligned on sentence boundaries."""
        sentences = sent_tokenize(text)
        chunks, current_chunk, current_length = [], [], 0

        for sentence in sentences:
            sentence_tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
            if current_length + len(sentence_tokens) <= max_chunk_length:
                current_chunk.append(sentence)
                current_length += len(sentence_tokens)
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence_tokens)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    # ------------------------------------------------------------------
    #  Extractive Summarization
    # ------------------------------------------------------------------

    def extractive_summarize(self, text: str, num_sentences: int = 10) -> str:
        """Extract the top-scoring sentences from the text as a summary."""
        if not text or not text.strip():
            return "No text provided for summarization."

        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text  # Text is already short enough

        # Build a word-frequency table (stop words and punctuation excluded)
        words = word_tokenize(text.lower())
        filtered = [w for w in words if w.isalnum() and w not in self.stop_words and w not in string.punctuation]
        word_freq = Counter(filtered)

        # Score each sentence by normalised word frequency
        scores = {}
        for i, sentence in enumerate(sentences):
            sentence_words = word_tokenize(sentence.lower())
            total = sum(word_freq[w] for w in sentence_words if w in word_freq)
            scores[i] = total / len(sentence_words) if sentence_words else 0

        # Pick top-N sentences and preserve their original document order
        top_indices = sorted(sorted(scores, key=scores.get, reverse=True)[:num_sentences])
        return ' '.join(sentences[i] for i in top_indices)

    # ------------------------------------------------------------------
    #  Document Comparison
    # ------------------------------------------------------------------

    def compare_summaries(self, summaries: list) -> dict:
        """Find common themes and unique keywords across multiple summaries."""
        all_keywords = []
        per_summary_keywords = []

        for summary in summaries:
            words = word_tokenize(summary.lower())
            filtered = [w for w in words if w.isalnum() and w not in self.stop_words and len(w) > 3]
            top_keywords = [w for w, _ in Counter(filtered).most_common(10)]
            per_summary_keywords.append(set(top_keywords))
            all_keywords.extend(top_keywords)

        keyword_freq = Counter(all_keywords)

        # A theme is "common" if it appears in at least 2 documents
        common_themes = [w for w, count in keyword_freq.most_common(10) if count >= 2]

        # Unique points: keywords that are not shared as common themes (up to 5 per doc)
        unique_points = {}
        for i, keywords in enumerate(per_summary_keywords):
            unique = [kw for kw in keywords if keyword_freq[kw] <= 2 and kw not in common_themes]
            unique_points[f"Document {i+1}"] = unique[:5]

        return {
            "common_themes": common_themes[:8],
            "unique_points": unique_points,
        }