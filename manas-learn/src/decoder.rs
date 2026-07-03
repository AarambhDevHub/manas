use std::collections::{HashMap, HashSet};

use crate::backprop::cosine;
use crate::encoder::{ANSWER_CODEC_MARKER, ANSWER_COUNT_SCALE, ANSWER_ID_SCALE, Encoder};

pub const MIN_QUERY_CONFIDENCE: f32 = 0.25;
const MAX_ANSWER_WORDS: usize = 6;
const MAX_LEGACY_EMBEDDING_WORDS: usize = 1;
const MIN_PACKED_ACTIVATION: f32 = 0.20;
const PACKED_ROUND_TOLERANCE: f32 = 0.35;
const MIN_EMBEDDING_RATIO: f32 = 0.72;

#[derive(Clone, Debug, PartialEq)]
pub struct DecodedAnswer {
    pub answer: String,
    pub confidence: f32,
}

pub fn decode_answer(output: &[f32], encoder: &Encoder, question: &str) -> Option<DecodedAnswer> {
    if output.iter().all(|value| value.abs() <= f32::EPSILON) {
        return None;
    }

    decode_packed_answer(output, encoder, question)
        .or_else(|| decode_embedding_answer(output, encoder, question))
}

fn decode_packed_answer(
    output: &[f32],
    encoder: &Encoder,
    question: &str,
) -> Option<DecodedAnswer> {
    if output.len() < 3 || output[0] >= ANSWER_CODEC_MARKER * MIN_PACKED_ACTIVATION {
        return None;
    }

    let activation = output[0] / ANSWER_CODEC_MARKER;
    if !activation.is_finite() || activation < MIN_PACKED_ACTIVATION {
        return None;
    }

    let count_value = output[1] / activation * ANSWER_COUNT_SCALE;
    let count = rounded_usize(count_value)?;
    if count == 0 || count > output.len().saturating_sub(2) || count > MAX_ANSWER_WORDS {
        return None;
    }

    let id_to_word = encoder
        .known_word_ids()
        .into_iter()
        .collect::<HashMap<u32, String>>();
    let query_words = normalized_words(question)
        .into_iter()
        .collect::<HashSet<_>>();

    let mut words = Vec::with_capacity(count);
    for slot in 0..count {
        let id_value = output[slot + 2] / activation * ANSWER_ID_SCALE;
        let encoded_id = rounded_u32(id_value)?;
        let word_id = encoded_id.checked_sub(1)?;
        let word = id_to_word.get(&word_id)?;
        if !query_words.contains(word) && !is_stopword(word) {
            words.push(word.clone());
        }
    }

    if words.is_empty() {
        return None;
    }

    Some(DecodedAnswer {
        answer: words.join(" "),
        confidence: activation.clamp(0.0, 1.0),
    })
}

fn decode_embedding_answer(
    output: &[f32],
    encoder: &Encoder,
    question: &str,
) -> Option<DecodedAnswer> {
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

    let threshold = (best_score * MIN_EMBEDDING_RATIO).max(MIN_QUERY_CONFIDENCE);
    let words = candidates
        .iter()
        .filter(|(_, score)| *score >= threshold)
        .take(MAX_LEGACY_EMBEDDING_WORDS)
        .map(|(word, _)| word.clone())
        .collect::<Vec<_>>();

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

fn rounded_usize(value: f32) -> Option<usize> {
    if !value.is_finite() {
        return None;
    }

    let rounded = value.round();
    if (value - rounded).abs() > PACKED_ROUND_TOLERANCE || rounded < 0.0 {
        None
    } else {
        Some(rounded as usize)
    }
}

fn rounded_u32(value: f32) -> Option<u32> {
    if !value.is_finite() {
        return None;
    }

    let rounded = value.round();
    if (value - rounded).abs() > PACKED_ROUND_TOLERANCE
        || rounded < 0.0
        || rounded > u32::MAX as f32
    {
        None
    } else {
        Some(rounded as u32)
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_answer_decodes_exact_meaningful_words() {
        let mut encoder = Encoder::with_dim(32);
        let output = encoder.encode_answer("small animal with fur");

        let decoded = decode_answer(&output, &encoder, "What is a cat?").unwrap();

        assert_eq!(decoded.answer, "small animal fur");
        assert_eq!(decoded.confidence, 1.0);
    }
}
