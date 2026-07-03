use manas_core::{Neuron, ProtectionLevel};

pub const OPEN_TO_GUARDED_IMPORTANCE: f32 = 0.50;
pub const GUARDED_TO_FROZEN_IMPORTANCE: f32 = 0.85;

const SECONDS_PER_DAY: f32 = 86_400.0;
const FREQUENCY_NORMALIZER: f32 = 10_000.0;
const MAGNITUDE_NORMALIZER: f32 = 10.0;
const AGE_GRACE_DAYS: f32 = 7.0;

pub fn compute_importance(neuron: &Neuron, now_secs: u64) -> f32 {
    let freq = (neuron.activation_count as f32 / FREQUENCY_NORMALIZER).clamp(0.0, 1.0);
    let recency = recency_score(neuron.last_activated, now_secs);
    let magnitude = (l2_norm(&neuron.weights) / MAGNITUDE_NORMALIZER).clamp(0.0, 1.0);
    let age_grace = age_grace_score(neuron.born_at, now_secs);

    (0.40 * freq + 0.30 * recency + 0.20 * magnitude + 0.10 * age_grace).clamp(0.0, 1.0)
}

pub fn promote_if_needed(neuron: &mut Neuron, now_secs: u64) -> bool {
    neuron.importance_score = compute_importance(neuron, now_secs);
    let before = neuron.protection_level;

    match neuron.protection_level {
        ProtectionLevel::Open if neuron.importance_score >= OPEN_TO_GUARDED_IMPORTANCE => {
            neuron.guard_all();
        }
        ProtectionLevel::Guarded if neuron.importance_score >= GUARDED_TO_FROZEN_IMPORTANCE => {
            neuron.freeze_all();
        }
        ProtectionLevel::Open | ProtectionLevel::Guarded | ProtectionLevel::Frozen => {}
    }

    neuron.protection_level != before
}

fn recency_score(last_activated: u64, now_secs: u64) -> f32 {
    if last_activated == 0 {
        return 0.0;
    }

    let days_idle = now_secs.saturating_sub(last_activated) as f32 / SECONDS_PER_DAY;
    (-0.1 * days_idle).exp()
}

fn age_grace_score(born_at: u64, now_secs: u64) -> f32 {
    if born_at == 0 {
        return 0.0;
    }

    let age_days = now_secs.saturating_sub(born_at) as f32 / SECONDS_PER_DAY;
    (-age_days / AGE_GRACE_DAYS).exp()
}

fn l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|value| value * value).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use manas_core::{Activation, Source};

    const NOW: u64 = 1_800_000_000;
    const DAY: u64 = 86_400;

    #[test]
    fn importance_frequently_used_neuron_gets_high_score() {
        let mut neuron = test_neuron(vec![0.0; 32]);
        neuron.activation_count = 8_000;
        neuron.last_activated = NOW;
        neuron.born_at = NOW - DAY;

        let score = compute_importance(&neuron, NOW);

        assert!(
            score > 0.40,
            "frequently used neuron should score > 0.40, got {score:.4}"
        );
    }

    #[test]
    fn importance_idle_neuron_gets_low_recency() {
        let mut neuron = test_neuron(vec![0.0; 32]);
        neuron.activation_count = 100;
        neuron.last_activated = NOW - 30 * DAY;

        let score = compute_importance(&neuron, NOW);

        assert!(
            score < 0.05,
            "idle low-frequency neuron should stay low, got {score:.4}"
        );
    }

    #[test]
    fn importance_age_grace_decays_smoothly_after_first_week() {
        let mut young = test_neuron(vec![0.0; 32]);
        young.born_at = NOW - 3 * DAY;
        let mut old = test_neuron(vec![0.0; 32]);
        old.born_at = NOW - 8 * DAY;

        let young_score = compute_importance(&young, NOW);
        let old_score = compute_importance(&old, NOW);

        assert!(young_score > old_score);
        assert!(
            old_score > 0.0,
            "age grace should decay smoothly, not fall off a cliff"
        );
    }

    #[test]
    fn importance_weight_magnitude_is_clamped() {
        let low = compute_importance(&test_neuron(vec![0.0; 4]), NOW);
        let high = compute_importance(&test_neuron(vec![10.0; 4]), NOW);

        assert_eq!(low, 0.0);
        assert!(
            (high - 0.20).abs() < 1.0e-6,
            "clamped magnitude component should be 0.20, got {high:.4}"
        );
    }

    #[test]
    fn importance_unknown_timestamps_add_no_fake_score() {
        let neuron = test_neuron(vec![0.0; 32]);

        assert_eq!(compute_importance(&neuron, NOW), 0.0);
    }

    #[test]
    fn importance_promotes_open_to_guarded_one_level_per_pass() {
        let mut neuron = high_importance_neuron(ProtectionLevel::Open);

        assert!(promote_if_needed(&mut neuron, NOW));
        assert_eq!(neuron.protection_level, ProtectionLevel::Guarded);
        assert!(neuron.importance_score >= GUARDED_TO_FROZEN_IMPORTANCE);
    }

    #[test]
    fn importance_promotes_guarded_to_frozen() {
        let mut neuron = high_importance_neuron(ProtectionLevel::Guarded);

        assert!(promote_if_needed(&mut neuron, NOW));
        assert_eq!(neuron.protection_level, ProtectionLevel::Frozen);
    }

    #[test]
    fn importance_never_downgrades_frozen_neuron() {
        let mut neuron = test_neuron(vec![0.0; 32]);
        neuron.protection_level = ProtectionLevel::Frozen;

        assert!(!promote_if_needed(&mut neuron, NOW));
        assert_eq!(neuron.protection_level, ProtectionLevel::Frozen);
        assert_eq!(neuron.importance_score, 0.0);
    }

    fn high_importance_neuron(protection_level: ProtectionLevel) -> Neuron {
        let mut neuron = test_neuron(vec![10.0; 32]);
        neuron.activation_count = 10_000;
        neuron.last_activated = NOW;
        neuron.born_at = NOW;
        neuron.protection_level = protection_level;
        neuron
    }

    fn test_neuron(weights: Vec<f32>) -> Neuron {
        Neuron {
            id: 0,
            weight_protection: vec![ProtectionLevel::Open; weights.len()],
            weights,
            bias: 0.0,
            activation: Activation::Tanh,
            importance_score: 0.0,
            protection_level: ProtectionLevel::Open,
            bias_protection: ProtectionLevel::Open,
            born_at: 0,
            last_activated: 0,
            activation_count: 0,
            source: Source::Unknown,
            freshness_category: 1,
        }
    }
}
