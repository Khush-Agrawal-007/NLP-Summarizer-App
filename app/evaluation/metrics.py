# app/evaluation/metrics.py
"""Evaluation metrics for summarization quality.

Supports ROUGE, BLEU, and BARTScore (computed directly via transformers).
BARTScore measures the log-likelihood of a reference given a generated summary.
"""
import traceback
from typing import Dict

import torch
import nltk
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. BARTScore will be skipped.")

# Ensure the punkt tokenizer is present before any scoring runs
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class SummarizationEvaluator:
    """Computes ROUGE, BLEU, and BARTScore for a generated vs. reference summary pair."""

    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )
        self.smoothing = SmoothingFunction()

        # BARTScore model — loaded once per evaluator instance
        self.bart_tokenizer = None
        self.bart_model = None
        self.bart_device = None

        if TRANSFORMERS_AVAILABLE:
            self._load_bart_model()

    def _load_bart_model(self):
        """Load the BART model used for BARTScore computation."""
        model_name = 'facebook/bart-large-cnn'
        try:
            print(f"Loading BARTScore model: {model_name}")
            self.bart_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bart_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.bart_device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.bart_device == "cuda":
                self.bart_model = self.bart_model.to(self.bart_device)
            self.bart_model.eval()
            print("BARTScore model loaded.")
        except Exception as e:
            print(f"Warning: BARTScore model failed to load: {e}")
            self.bart_tokenizer = self.bart_model = None

    # ------------------------------------------------------------------
    #  Individual Metrics
    # ------------------------------------------------------------------

    def evaluate_rouge(self, generated: str, reference: str) -> Dict[str, float]:
        """Return ROUGE-1, ROUGE-2, and ROUGE-L precision / recall / F1 scores."""
        scores = self.rouge_scorer.score(reference, generated)
        return {
            'rouge1_precision': scores['rouge1'].precision,
            'rouge1_recall':    scores['rouge1'].recall,
            'rouge1_fmeasure':  scores['rouge1'].fmeasure,
            'rouge2_precision': scores['rouge2'].precision,
            'rouge2_recall':    scores['rouge2'].recall,
            'rouge2_fmeasure':  scores['rouge2'].fmeasure,
            'rougeL_precision': scores['rougeL'].precision,
            'rougeL_recall':    scores['rougeL'].recall,
            'rougeL_fmeasure':  scores['rougeL'].fmeasure,
        }

    def evaluate_bleu(self, generated: str, reference: str) -> Dict[str, float]:
        """Return BLEU-1, BLEU-2, and BLEU-4 scores."""
        gen_tokens = word_tokenize(generated.lower())
        ref_tokens = word_tokenize(reference.lower())
        kwargs = dict(smoothing_function=self.smoothing.method1)
        return {
            'bleu1': sentence_bleu([ref_tokens], gen_tokens, weights=(1, 0, 0, 0), **kwargs),
            'bleu2': sentence_bleu([ref_tokens], gen_tokens, weights=(0.5, 0.5, 0, 0), **kwargs),
            'bleu4': sentence_bleu([ref_tokens], gen_tokens, weights=(0.25, 0.25, 0.25, 0.25), **kwargs),
        }

    def evaluate_bart(self, generated: str, reference: str) -> Dict[str, float]:
        """Return BARTScore = -loss(reference | generated).

        A higher (less negative) value means the reference is more probable
        given the generated text. Returns 0.0 if the BART model is unavailable.
        """
        if not TRANSFORMERS_AVAILABLE or self.bart_model is None:
            return {'bart_score': 0.0}

        try:
            source = self.bart_tokenizer(
                generated, return_tensors="pt", truncation=True, max_length=1024
            )
            target = self.bart_tokenizer(
                reference, return_tensors="pt", truncation=True, max_length=1024
            )

            source = {k: v.to(self.bart_device) for k, v in source.items()}
            target_ids = target["input_ids"].to(self.bart_device)

            with torch.no_grad():
                outputs = self.bart_model(
                    input_ids=source["input_ids"],
                    attention_mask=source.get("attention_mask"),
                    labels=target_ids,
                    return_dict=True,
                )

            # Negative loss = log-likelihood (higher is better)
            return {'bart_score': float(-outputs.loss.item())}

        except Exception as e:
            print(f"BARTScore error: {e}")
            traceback.print_exc()
            return {'bart_score': 0.0}

    # ------------------------------------------------------------------
    #  Combined Evaluation
    # ------------------------------------------------------------------

    def evaluate_text_summary(self, generated: str, reference: str) -> Dict[str, float]:
        """Run all metrics and return a combined score dictionary.

        Overall score weights:
          With BARTScore:    ROUGE-1 20%, ROUGE-2 20%, ROUGE-L 15%, BLEU-4 15%, BART 30%
          Without BARTScore: ROUGE-1 30%, ROUGE-2 30%, ROUGE-L 20%, BLEU-4 20%
        """
        rouge = self.evaluate_rouge(generated, reference)
        bleu  = self.evaluate_bleu(generated, reference)
        bart  = self.evaluate_bart(generated, reference)

        all_scores = {**rouge, **bleu, **bart}

        bart_score = bart.get('bart_score', 0.0)
        if bart_score != 0.0:
            # Normalise negative BARTScore to [0, 1] range
            normalized_bart = (
                1 / (1 + abs(bart_score)) if bart_score < 0 else min(bart_score, 1.0)
            )
            overall = (
                rouge['rouge1_fmeasure'] * 0.20 +
                rouge['rouge2_fmeasure'] * 0.20 +
                rouge['rougeL_fmeasure'] * 0.15 +
                bleu['bleu4']            * 0.15 +
                normalized_bart          * 0.30
            )
        else:
            overall = (
                rouge['rouge1_fmeasure'] * 0.30 +
                rouge['rouge2_fmeasure'] * 0.30 +
                rouge['rougeL_fmeasure'] * 0.20 +
                bleu['bleu4']            * 0.20
            )

        all_scores['overall_score'] = overall
        return all_scores


def format_evaluation_report(scores: Dict[str, float]) -> str:
    """Format the scores dictionary into a human-readable text report."""
    report = "📊 Evaluation Report\n" + "=" * 50 + "\n\n"

    if 'rouge1_fmeasure' in scores:
        report += (
            "ROUGE Scores:\n"
            f"  ROUGE-1 F1: {scores['rouge1_fmeasure']:.4f}\n"
            f"  ROUGE-2 F1: {scores['rouge2_fmeasure']:.4f}\n"
            f"  ROUGE-L F1: {scores['rougeL_fmeasure']:.4f}\n\n"
        )

    if 'bleu4' in scores:
        report += (
            "BLEU Scores:\n"
            f"  BLEU-1: {scores['bleu1']:.4f}\n"
            f"  BLEU-2: {scores['bleu2']:.4f}\n"
            f"  BLEU-4: {scores['bleu4']:.4f}\n\n"
        )

    if 'bart_score' in scores:
        report += f"BART Score:\n  BARTScore: {scores['bart_score']:.4f}\n\n"

    if 'overall_score' in scores:
        report += f"Overall Score: {scores['overall_score']:.4f}\n"

    return report
