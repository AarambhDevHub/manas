use manas_core::Network;
use manas_learn::fixtures::{
    ANCHOR_FACTS, ANCHOR_NEURONS_PER_FACT, ANCHOR_SURVIVAL_THRESHOLD, ANCHOR_TRAIN_EPOCHS,
    EMBED_DIM, HIDDEN_DIM, LEARNING_RATE, MAX_FORGETTING_DELTA, NEW_FACT_THRESHOLD, NOISE_FACTS,
    NOISE_TRAIN_EPOCHS, OUTPUT_DIM, SEEDS,
};
use manas_learn::{EncodedFact, Trainer};

#[derive(Debug)]
struct SeedReport {
    seed: u64,
    anchor_before: Vec<f32>,
    anchor_after: Vec<f32>,
    new_scores: Vec<f32>,
    frozen_hidden: usize,
    frozen_edges: usize,
}

#[test]
fn five_anchor_facts_survive_fifty_new_facts() {
    let report = run_stage3_proof(42);
    assert_report_passes(&report);
}

#[test]
fn anti_forgetting_passes_for_five_fixed_seeds() {
    for seed in SEEDS {
        let report = run_stage3_proof(seed);
        assert_report_passes(&report);
    }
}

#[test]
fn new_facts_are_fit_after_anchor_consolidation() {
    let report = run_stage3_proof(2026);
    assert!(
        report
            .new_scores
            .iter()
            .all(|score| *score >= NEW_FACT_THRESHOLD),
        "new fact similarity below threshold for seed {}: {:?}",
        report.seed,
        report.new_scores
    );
}

fn run_stage3_proof(seed: u64) -> SeedReport {
    let mut trainer = Trainer::with_seed(seed ^ 0x6a09_e667_f3bc_c909, EMBED_DIM, LEARNING_RATE);
    let anchors = trainer.encode_facts(&ANCHOR_FACTS);
    let noise = trainer.encode_facts(&NOISE_FACTS);
    let mut network = Network::with_seed(
        seed ^ 0xbb67_ae85_84ca_a73b,
        EMBED_DIM,
        HIDDEN_DIM,
        OUTPUT_DIM,
    );

    trainer
        .train_facts(&mut network, &anchors, ANCHOR_TRAIN_EPOCHS)
        .expect("anchor training should succeed");

    trainer
        .consolidate_anchors(&mut network, &anchors, ANCHOR_NEURONS_PER_FACT)
        .expect("anchor consolidation should succeed");

    let anchor_before = score_facts(&trainer, &network, &anchors);
    let frozen_hidden = network.frozen_hidden_neuron_count();
    let frozen_edges = network.frozen_output_edge_count();

    trainer
        .train_facts(&mut network, &noise, NOISE_TRAIN_EPOCHS)
        .expect("noise training should succeed");
    trainer
        .fit_new_facts(&mut network, &noise, &anchors)
        .expect("new fact fitting should succeed");

    SeedReport {
        seed,
        anchor_before,
        anchor_after: score_facts(&trainer, &network, &anchors),
        new_scores: score_facts(&trainer, &network, &noise),
        frozen_hidden,
        frozen_edges,
    }
}

fn score_facts(trainer: &Trainer, network: &Network, facts: &[EncodedFact]) -> Vec<f32> {
    facts
        .iter()
        .map(|fact| trainer.similarity_for_fact(network, fact))
        .collect()
}

fn assert_report_passes(report: &SeedReport) {
    assert_eq!(
        report.frozen_hidden,
        ANCHOR_FACTS.len() * ANCHOR_NEURONS_PER_FACT,
        "unexpected frozen hidden count for seed {}",
        report.seed
    );
    assert_eq!(
        report.frozen_edges,
        OUTPUT_DIM * ANCHOR_FACTS.len() * ANCHOR_NEURONS_PER_FACT,
        "unexpected frozen output edge count for seed {}",
        report.seed
    );

    for (index, (before, after)) in report
        .anchor_before
        .iter()
        .zip(report.anchor_after.iter())
        .enumerate()
    {
        assert!(
            *after >= ANCHOR_SURVIVAL_THRESHOLD,
            "anchor {} below survival threshold for seed {}: before {:.4}, after {:.4}",
            index,
            report.seed,
            before,
            after
        );
        assert!(
            before - after <= MAX_FORGETTING_DELTA,
            "anchor {} forgot too much for seed {}: before {:.4}, after {:.4}",
            index,
            report.seed,
            before,
            after
        );
    }

    assert!(
        report
            .new_scores
            .iter()
            .all(|score| *score >= NEW_FACT_THRESHOLD),
        "new fact similarity below threshold for seed {}: {:?}",
        report.seed,
        report.new_scores
    );
}
