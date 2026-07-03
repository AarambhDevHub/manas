use manas_core::Neuron;

const SECONDS_PER_DAY: u64 = 86_400;

const TIMELESS_WORDS: &[&str] = &[
    "theorem",
    "law",
    "always",
    "definition",
    "formula",
    "proof",
    "principle",
];
const TIMELESS_PHRASES: &[&str] = &["defined as"];
const FAST_WORDS: &[&str] = &[
    "released", "release", "version", "launched", "update", "updated", "news",
];
const FAST_PHRASES: &[&str] = &["last month", "this year"];
const REALTIME_WORDS: &[&str] = &[
    "today", "breaking", "live", "latest", "now", "current", "score", "scores", "stock", "price",
    "market",
];

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FreshnessCategory {
    Timeless = 0,
    Slow = 1,
    Fast = 2,
    Realtime = 3,
}

impl FreshnessCategory {
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Timeless,
            2 => Self::Fast,
            3 => Self::Realtime,
            _ => Self::Slow,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Timeless => "Timeless",
            Self::Slow => "Slow",
            Self::Fast => "Fast",
            Self::Realtime => "Realtime",
        }
    }
}

impl From<u8> for FreshnessCategory {
    fn from(value: u8) -> Self {
        Self::from_u8(value)
    }
}

impl From<FreshnessCategory> for u8 {
    fn from(category: FreshnessCategory) -> Self {
        category as u8
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FreshnessWarning {
    pub category: FreshnessCategory,
    pub age_days: u64,
}

pub fn detect_freshness(text: &str) -> FreshnessCategory {
    let normalized = normalize_text(text);

    if has_any_word(&normalized, REALTIME_WORDS) {
        return FreshnessCategory::Realtime;
    }

    if has_any_word(&normalized, FAST_WORDS) || has_any_phrase(&normalized, FAST_PHRASES) {
        return FreshnessCategory::Fast;
    }

    if has_any_word(&normalized, TIMELESS_WORDS) || has_any_phrase(&normalized, TIMELESS_PHRASES) {
        return FreshnessCategory::Timeless;
    }

    FreshnessCategory::Slow
}

pub fn freshness_age_days(neuron: &Neuron, now_secs: u64) -> u64 {
    now_secs.saturating_sub(neuron.born_at) / SECONDS_PER_DAY
}

pub fn is_stale(neuron: &Neuron, now_secs: u64) -> bool {
    let age_days = freshness_age_days(neuron, now_secs);
    match FreshnessCategory::from(neuron.freshness_category) {
        FreshnessCategory::Timeless => false,
        FreshnessCategory::Slow => age_days > 365,
        FreshnessCategory::Fast => age_days > 30,
        FreshnessCategory::Realtime => age_days > 1,
    }
}

pub fn staleness_warning(neuron: &Neuron, now_secs: u64) -> Option<FreshnessWarning> {
    if !is_stale(neuron, now_secs) {
        return None;
    }

    Some(FreshnessWarning {
        category: FreshnessCategory::from(neuron.freshness_category),
        age_days: freshness_age_days(neuron, now_secs),
    })
}

fn normalize_text(text: &str) -> String {
    let mut normalized = String::with_capacity(text.len());
    let mut previous_was_space = true;

    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            normalized.push(ch.to_ascii_lowercase());
            previous_was_space = false;
        } else if !previous_was_space {
            normalized.push(' ');
            previous_was_space = true;
        }
    }

    normalized.trim().to_string()
}

fn has_any_word(normalized: &str, keywords: &[&str]) -> bool {
    normalized
        .split_whitespace()
        .any(|word| keywords.contains(&word))
}

fn has_any_phrase(normalized: &str, phrases: &[&str]) -> bool {
    let bounded = format!(" {normalized} ");
    phrases
        .iter()
        .any(|phrase| bounded.contains(&format!(" {phrase} ")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use manas_core::Network;

    const NOW: u64 = 1_800_000_000;
    const DAY: u64 = 86_400;

    #[test]
    fn detects_timeless_keywords() {
        assert_eq!(
            detect_freshness("The Pythagorean theorem states that a²+b²=c²"),
            FreshnessCategory::Timeless
        );
        assert_eq!(
            detect_freshness("Water is always composed of hydrogen and oxygen"),
            FreshnessCategory::Timeless
        );
    }

    #[test]
    fn detects_realtime_keywords() {
        assert_eq!(
            detect_freshness("Breaking news: the stock market fell today"),
            FreshnessCategory::Realtime
        );
        assert_eq!(
            detect_freshness("Live scores updated every minute"),
            FreshnessCategory::Realtime
        );
    }

    #[test]
    fn detects_fast_keywords() {
        assert_eq!(
            detect_freshness("Rust 2.0 was released last month"),
            FreshnessCategory::Fast
        );
    }

    #[test]
    fn defaults_to_slow() {
        assert_eq!(
            detect_freshness("Paris is a city in France"),
            FreshnessCategory::Slow
        );
    }

    #[test]
    fn word_matching_does_not_use_substrings() {
        assert_eq!(
            detect_freshness("Stockholm is the capital of Sweden"),
            FreshnessCategory::Slow
        );
    }

    #[test]
    fn unknown_category_defaults_to_slow() {
        assert_eq!(FreshnessCategory::from(99), FreshnessCategory::Slow);
    }

    #[test]
    fn timeless_fact_never_stale() {
        let mut neuron = neuron();
        neuron.freshness_category = FreshnessCategory::Timeless as u8;
        neuron.born_at = 0;

        assert!(!is_stale(&neuron, NOW));
    }

    #[test]
    fn slow_fact_stale_after_365_days() {
        let mut neuron = neuron();
        neuron.freshness_category = FreshnessCategory::Slow as u8;
        neuron.born_at = NOW - 365 * DAY;
        assert!(!is_stale(&neuron, NOW));

        neuron.born_at = NOW - 366 * DAY;
        assert!(is_stale(&neuron, NOW));
    }

    #[test]
    fn fast_fact_stale_after_30_days() {
        let mut neuron = neuron();
        neuron.freshness_category = FreshnessCategory::Fast as u8;
        neuron.born_at = NOW - 30 * DAY;
        assert!(!is_stale(&neuron, NOW));

        neuron.born_at = NOW - 31 * DAY;
        assert!(is_stale(&neuron, NOW));
    }

    #[test]
    fn realtime_fact_stale_after_1_day() {
        let mut neuron = neuron();
        neuron.freshness_category = FreshnessCategory::Realtime as u8;
        neuron.born_at = NOW - DAY;
        assert!(!is_stale(&neuron, NOW));

        neuron.born_at = NOW - 2 * DAY;
        assert!(is_stale(&neuron, NOW));
    }

    #[test]
    fn staleness_warning_includes_category_and_age() {
        let mut neuron = neuron();
        neuron.freshness_category = FreshnessCategory::Fast as u8;
        neuron.born_at = NOW - 47 * DAY;

        assert_eq!(
            staleness_warning(&neuron, NOW),
            Some(FreshnessWarning {
                category: FreshnessCategory::Fast,
                age_days: 47,
            })
        );
    }

    fn neuron() -> manas_core::Neuron {
        Network::new(32, 1, 32).layers[0].neurons[0].clone()
    }
}
