use std::collections::HashSet;

use crate::backprop::cosine;
use crate::encoder::Encoder;

pub const MIN_QUERY_CONFIDENCE: f32 = 0.25;
const MAX_ANSWER_WORDS: usize = 10;

#[derive(Clone, Debug, PartialEq)]
pub struct DecodedAnswer {
    pub answer: String,
    pub confidence: f32,
}

pub fn decode_answer(output: &[f32], encoder: &Encoder, question: &str) -> Option<DecodedAnswer> {
    if output.iter().all(|value| value.abs() <= f32::EPSILON) {
        return None;
    }

    let query_words = normalized_words(question)
        .into_iter()
        .collect::<HashSet<_>>();
    let mut candidates = encoder
        .known_words()
        .into_iter()
        .filter(|word| !query_words.contains(word))
        .filter(|word| !is_stopword(word))
        .filter_map(|word| {
            let vector = encoder.encode_deterministic(&word);
            if vector.iter().all(|value| value.abs() <= f32::EPSILON) {
                return None;
            }
            let score = cosine(output, &vector);
            score.is_finite().then_some((word, score))
        })
        .collect::<Vec<_>>();

    candidates.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let best_score = candidates.first().map(|(_, score)| *score)?;
    if best_score < MIN_QUERY_CONFIDENCE {
        return None;
    }

    let threshold = (best_score * 0.25).max(MIN_QUERY_CONFIDENCE * 0.25);
    let mut words = candidates
        .iter()
        .filter(|(_, score)| *score >= threshold)
        .take(MAX_ANSWER_WORDS)
        .map(|(word, _)| word.clone())
        .collect::<Vec<_>>();

    if words.len() < 3 {
        for (word, score) in &candidates {
            if words.len() >= 3 || *score <= 0.0 {
                break;
            }
            if !words.contains(word) {
                words.push(word.clone());
            }
        }
    }

    if words.is_empty() {
        return None;
    }

    Some(DecodedAnswer {
        answer: words.join(" "),
        confidence: best_score.clamp(0.0, 1.0),
    })
}

fn normalized_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter_map(|raw| {
            let cleaned = raw
                .chars()
                .filter(|ch| ch.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect::<String>();

            if cleaned.is_empty() {
                None
            } else {
                Some(cleaned)
            }
        })
        .collect()
}

fn is_stopword(word: &str) -> bool {
    matches!(
        word,
        "a" | "an"
            | "and"
            | "are"
            | "as"
            | "at"
            | "be"
            | "by"
            | "for"
            | "from"
            | "in"
            | "is"
            | "it"
            | "of"
            | "on"
            | "or"
            | "the"
            | "to"
            | "was"
            | "were"
            | "what"
            | "when"
            | "where"
            | "who"
            | "why"
            | "with"
    )
}
