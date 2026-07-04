use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

use manas_core::{Network, NeuronGradients, ProtectionLevel, Source};
use manas_ingest::{IngestSource, ingest};
use manas_learn::fixtures::{
    ANCHOR_FACTS, ANCHOR_NEURONS_PER_FACT, ANCHOR_SURVIVAL_THRESHOLD, ANCHOR_TRAIN_EPOCHS,
    EMBED_DIM, HIDDEN_DIM, LEARNING_RATE, MAX_FORGETTING_DELTA, NEW_FACT_THRESHOLD, NOISE_FACTS,
    NOISE_TRAIN_EPOCHS, OUTPUT_DIM,
};
use manas_learn::{EncodedFact, FreshnessCategory, Trainer};
use manas_store::{BrainState, ManasBrain, VocabEntry};

const DEMO_FACTS: &[&str] = &[
    "A cat is a small domesticated animal with fur and whiskers.",
    "The Eiffel Tower is located in Paris France and was built in 1889.",
    "The Amazon River is the largest river by discharge in the world.",
    "Photosynthesis is the process by which plants convert sunlight into energy.",
    "Hydrogen is the lightest and most abundant element in the universe.",
    "The human brain contains approximately 86 billion neurons.",
    "Mount Everest is the highest mountain on Earth at 8849 meters.",
    "Shakespeare wrote 37 plays and 154 sonnets during his lifetime.",
    "The speed of light in vacuum is approximately 299792458 meters per second.",
    "DNA is a double helix structure that carries genetic information.",
    "The Roman Empire fell in 476 AD when Romulus Augustulus was deposed.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "The Python programming language was created by Guido van Rossum in 1991.",
    "Jupiter is the largest planet in our solar system with 95 known moons.",
    "The Mona Lisa was painted by Leonardo da Vinci in the early 16th century.",
    "Rust programming language was first released by Mozilla Research in 2010.",
    "The mitochondria is the powerhouse of the cell in biology.",
    "Albert Einstein developed the theory of relativity in the early 20th century.",
    "The Pacific Ocean is the largest and deepest ocean on Earth.",
    "Bitcoin was created by Satoshi Nakamoto and launched in January 2009.",
    "The nitrogen cycle describes how nitrogen moves through ecosystems.",
    "Gravity pulls objects toward each other with a force proportional to mass.",
];

#[test]
fn stage16_it_1_demo_answers_from_neural_weights_only() {
    let dir = temp_dir("it1-demo");

    assert_success(&run(&dir, &["reset"]));
    for fact in DEMO_FACTS {
        assert_success(&run(&dir, &["teach", fact]));
    }
    remove_sidecars(&dir);

    for (question, expected_words) in [
        (
            "What is a cat?",
            &["small", "domesticated", "animal", "fur", "whiskers"][..],
        ),
        (
            "Where is the Eiffel Tower?",
            &["located", "paris", "france", "built", "1889"][..],
        ),
        (
            "What did Einstein develop?",
            &["theory", "relativity", "early", "20th", "century"][..],
        ),
    ] {
        let ask = run(&dir, &["ask", question]);
        assert_success(&ask);
        let output = stdout(&ask);
        assert!(
            output.contains("Answered from\n  neural weights"),
            "{output}"
        );
        for word in expected_words {
            assert!(output.to_lowercase().contains(word), "{output}");
        }
    }

    cleanup_dir(dir);
}

#[test]
fn stage16_it_2_anti_forgetting_anchor_survival() {
    let mut trainer = Trainer::with_seed(42 ^ 0x6a09_e667_f3bc_c909, EMBED_DIM, LEARNING_RATE);
    let anchors = trainer.encode_facts(&ANCHOR_FACTS);
    let noise = trainer.encode_facts(&NOISE_FACTS);
    let mut network = Network::with_seed(
        42 ^ 0xbb67_ae85_84ca_a73b,
        EMBED_DIM,
        HIDDEN_DIM,
        OUTPUT_DIM,
    );

    trainer
        .train_facts(&mut network, &anchors, ANCHOR_TRAIN_EPOCHS)
        .unwrap();
    trainer
        .consolidate_anchors(&mut network, &anchors, ANCHOR_NEURONS_PER_FACT)
        .unwrap();
    let before = score_facts(&trainer, &network, &anchors);
    trainer
        .train_facts(&mut network, &noise, NOISE_TRAIN_EPOCHS)
        .unwrap();
    trainer
        .fit_new_facts(&mut network, &noise, &anchors)
        .unwrap();

    let after = score_facts(&trainer, &network, &anchors);
    let new_scores = score_facts(&trainer, &network, &noise);
    for (index, (before_score, after_score)) in before.iter().zip(after.iter()).enumerate() {
        assert!(
            *after_score >= ANCHOR_SURVIVAL_THRESHOLD,
            "anchor {index} below threshold: before {before_score:.4}, after {after_score:.4}"
        );
        assert!(
            before_score - after_score <= MAX_FORGETTING_DELTA,
            "anchor {index} forgot too much: before {before_score:.4}, after {after_score:.4}"
        );
    }
    assert!(new_scores.iter().all(|score| *score >= NEW_FACT_THRESHOLD));
}

#[test]
fn stage16_it_3_persistence_survives_save_load_save_load_cycle() {
    let dir = temp_dir("it3-persistence");
    let path = dir.join("brain.manas");
    let mut network = Network::new_empty(EMBED_DIM);
    let mut trainer = Trainer::with_seed(42, EMBED_DIM, LEARNING_RATE);

    trainer
        .learn(&mut network, "cat", "small domesticated animal with fur")
        .unwrap();
    let state = BrainState::new(network, store_vocab(&trainer));
    let brain = ManasBrain::new(path);
    brain.save_state(&state).unwrap();

    let first = brain.load_state().unwrap();
    brain.save_state(&first).unwrap();
    let second = brain.load_state().unwrap();

    assert_eq!(second.vocab_entries.len(), first.vocab_entries.len());
    assert_eq!(second.network.neuron_count(), first.network.neuron_count());
    assert_eq!(second.network.input_dim, first.network.input_dim);
    cleanup_dir(dir);
}

#[test]
fn stage16_it_4_growth_handles_novel_and_repeated_input() {
    let mut network = Network::new_empty(EMBED_DIM);
    let mut trainer = Trainer::with_seed(42, EMBED_DIM, LEARNING_RATE);

    trainer.learn(&mut network, "cat", "animal").unwrap();
    assert!(network.neuron_count() > 0);
    let after_first = network.neuron_count();

    for _ in 0..10 {
        trainer.learn(&mut network, "cat", "animal").unwrap();
    }
    let after_repeated = network.neuron_count();
    assert_eq!(after_first, after_repeated);

    trainer
        .learn(&mut network, "eiffel tower", "located paris france")
        .unwrap();
    assert!(network.neuron_count() >= after_repeated);
}

#[test]
fn stage16_it_5_frozen_components_survive_stress_updates() {
    let mut network = Network::new(EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM);
    network.layers[0].neurons[0].freeze_all();
    network.layers[1].neurons[0].weight_protection[0] = ProtectionLevel::Frozen;
    let frozen_hidden = network.layers[0].neurons[0].weights.clone();
    let frozen_edge = network.layers[1].neurons[0].weights[0];

    let gradients = vec![
        (
            network.layers[0].neurons[0].id,
            NeuronGradients {
                weight_gradients: vec![10.0; EMBED_DIM],
                bias_gradient: 10.0,
            },
        ),
        (
            network.layers[1].neurons[0].id,
            NeuronGradients {
                weight_gradients: vec![10.0; HIDDEN_DIM],
                bias_gradient: 10.0,
            },
        ),
    ];

    for _ in 0..1000 {
        network.apply_gradients(&gradients, 1.0).unwrap();
    }

    assert_eq!(network.layers[0].neurons[0].weights, frozen_hidden);
    assert_eq!(network.layers[1].neurons[0].weights[0], frozen_edge);
}

#[test]
fn stage16_it_6_forget_shrinks_brain_file() {
    let dir = temp_dir("it6-forget");
    write_compressible_brain(&dir);
    let path = dir.join("brain.manas");
    let size_before = fs::metadata(&path).unwrap().len();

    let forget = run(&dir, &["forget", "--threshold", "0.20"]);
    assert_success(&forget);
    let output = stdout(&forget);
    assert!(output.contains("neurons removed       : 1"), "{output}");

    let size_after = fs::metadata(&path).unwrap().len();
    assert!(size_after < size_before, "{size_before} -> {size_after}");
    cleanup_dir(dir);
}

#[test]
fn stage16_it_7_stale_fast_fact_warns_on_ask() {
    let dir = temp_dir("it7-freshness");
    let brain = ManasBrain::new(dir.join("brain.manas"));
    let mut network = Network::new_empty(EMBED_DIM);
    let mut trainer = Trainer::with_seed(42, EMBED_DIM, LEARNING_RATE);

    trainer
        .learn_with_source_and_freshness(
            &mut network,
            "rust version",
            "released last month",
            Source::RawText,
            FreshnessCategory::Fast,
        )
        .unwrap();
    let stale_born_at = unix_now().saturating_sub(31 * 86_400);
    network.layers[0].neurons[0].born_at = stale_born_at;
    network.layers[0].neurons[0].last_activated = stale_born_at;
    brain
        .save_state(&BrainState::new(network, store_vocab(&trainer)))
        .unwrap();

    let ask = run(&dir, &["ask", "What is rust version?"]);
    assert_success(&ask);
    let output = stdout(&ask);
    assert!(
        output.contains("This knowledge may be outdated"),
        "{output}"
    );
    cleanup_dir(dir);
}

#[test]
fn stage16_it_8_ingestion_teaches_txt_md_and_rs_sources() {
    let dir = temp_dir("it8-ingest");
    let docs = dir.join("docs");
    fs::create_dir_all(&docs).unwrap();
    fs::write(docs.join("cat.txt"), "A cat is a small animal with fur.").unwrap();
    fs::write(
        docs.join("paris.md"),
        "# Paris\n\nParis is a city in France.",
    )
    .unwrap();
    fs::write(
        docs.join("code.rs"),
        "/// Rust is a systems language.\nfn main() {}",
    )
    .unwrap();

    let chunks = ingest(IngestSource::Folder(docs.clone())).unwrap();
    assert!(chunks.iter().any(|chunk| matches!(
        &chunk.source,
        Source::LocalFile { path } if path.ends_with("cat.txt")
    )));
    assert!(chunks.iter().any(|chunk| matches!(
        &chunk.source,
        Source::LocalFile { path } if path.ends_with("paris.md")
    )));
    assert!(chunks.iter().any(|chunk| matches!(
        &chunk.source,
        Source::LocalFile { path } if path.ends_with("code.rs")
    )));

    assert_success(&run(&dir, &["teach", "docs"]));
    let ask = run(&dir, &["ask", "Where is Paris?"]);
    assert_success(&ask);
    assert!(stdout(&ask).contains("neural weights"));
    cleanup_dir(dir);
}

fn score_facts(trainer: &Trainer, network: &Network, facts: &[EncodedFact]) -> Vec<f32> {
    facts
        .iter()
        .map(|fact| trainer.similarity_for_fact(network, fact))
        .collect()
}

fn store_vocab(trainer: &Trainer) -> Vec<VocabEntry> {
    trainer
        .encoder
        .export_vocab()
        .into_iter()
        .map(|entry| VocabEntry {
            token: entry.token,
            id: entry.id,
            embedding: entry.embedding,
        })
        .collect()
}

fn write_compressible_brain(dir: &Path) {
    let mut network = Network::new_empty(4);
    for _ in 0..3 {
        network.grow_neuron(0, 4).unwrap();
    }

    let now = unix_now();
    let day = 86_400;
    network.layers[0].neurons[0].weights = vec![1.0, 0.0, 0.0, 0.0];
    network.layers[0].neurons[0].guard_all();
    network.layers[0].neurons[0].importance_score = 0.75;
    network.layers[0].neurons[0].last_activated = now;
    network.layers[0].neurons[0].born_at = now - 2 * day;

    network.layers[0].neurons[1].weights = vec![1.0, 0.0, 0.0, 0.0];
    network.layers[0].neurons[1].importance_score = 0.01;
    network.layers[0].neurons[1].last_activated = now - 31 * day;
    network.layers[0].neurons[1].born_at = now - 60 * day;

    network.layers[0].neurons[2].weights = vec![0.0, 1.0, 0.0, 0.0];
    network.layers[0].neurons[2].importance_score = 0.80;
    network.layers[0].neurons[2].last_activated = now;
    network.layers[0].neurons[2].born_at = now - 2 * day;

    for output_neuron in &mut network.layers[1].neurons {
        output_neuron.weights = vec![0.25, 0.05, 0.10];
        output_neuron.weight_protection = vec![ProtectionLevel::Open; 3];
    }

    ManasBrain::new(dir.join("brain.manas"))
        .save_state(&BrainState::new(network, Vec::new()))
        .unwrap();
}

fn remove_sidecars(dir: &Path) {
    for sidecar in [
        "brain.manas.sources",
        "brain.manas.sourceindex",
        "brain.manas.seq",
        "brain.manas.transformer",
        "brain.manas.langmeta",
    ] {
        let _ = fs::remove_file(dir.join(sidecar));
    }
}

fn run(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_manas"))
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap()
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "status: {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        stdout(output),
        stderr(output)
    );
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn temp_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "manas-stage16-{name}-{}-{nanos}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn cleanup_dir(path: PathBuf) {
    let _ = fs::remove_dir_all(path);
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}
