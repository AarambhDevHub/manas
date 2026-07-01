use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use manas_core::{ManasError, Network, ProtectionLevel, Source};
use manas_learn::fixtures::{
    ANCHOR_FACTS, ANCHOR_NEURONS_PER_FACT, ANCHOR_TRAIN_EPOCHS, EMBED_DIM, HIDDEN_DIM,
    LEARNING_RATE, OUTPUT_DIM,
};
use manas_learn::{EncodedFact, Trainer};
use manas_store::{BrainState, ManasBrain, VocabEntry};

const CRC32_POLYNOMIAL: u32 = 0xEDB8_8320;

#[test]
fn brain_survives_save_and_load() {
    let path = temp_path("survives");
    let mut network = Network::new(32, 64, 32);
    let mut trainer = Trainer::new(0.01);

    for _ in 0..300 {
        trainer
            .learn_raw(&mut network, "cat", "small animal with fur")
            .unwrap();
    }

    let brain = ManasBrain::new(&path);
    assert!(!brain.exists());
    assert_eq!(brain.size_bytes(), 0);
    brain.save(&network).unwrap();
    assert!(brain.exists());
    assert!(brain.size_bytes() > 0);

    let loaded = brain.load().unwrap();
    let original_sim = trainer.similarity_to_target(&network, "cat", "small animal with fur");
    let loaded_sim = trainer.similarity_to_target(&loaded, "cat", "small animal with fur");

    assert!(
        (original_sim - loaded_sim).abs() < 0.01,
        "loaded network changed answer: original {:.4}, loaded {:.4}",
        original_sim,
        loaded_sim
    );

    cleanup(&path);
}

#[test]
fn checksum_catches_corruption() {
    let path = temp_path("checksum");
    let network = Network::new(32, 64, 32);
    let brain = ManasBrain::new(&path);
    brain.save(&network).unwrap();

    let mut bytes = fs::read(&path).unwrap();
    let middle = bytes.len() / 2;
    bytes[middle] ^= 0xFF;
    fs::write(&path, bytes).unwrap();

    let result = brain.load();
    assert!(
        matches!(result, Err(ManasError::ChecksumMismatch { .. })),
        "expected checksum mismatch, got {result:?}"
    );

    cleanup(&path);
}

#[test]
fn protection_levels_survive_save_load() {
    let path = temp_path("protection");
    let mut network = Network::new(32, 64, 32);
    network.layers[0].neurons[0].freeze_all();
    network.layers[0].neurons[0].source = Source::RawText;
    network.layers[0].neurons[1].guard_all();
    network.layers[0].neurons[1].source = Source::LocalFile {
        path: "notes/facts.md".to_string(),
    };
    network.layers[1].neurons[0].weight_protection[0] = ProtectionLevel::Frozen;
    network.layers[1].neurons[0].bias_protection = ProtectionLevel::Guarded;

    let brain = ManasBrain::new(&path);
    brain.save(&network).unwrap();
    let loaded = brain.load().unwrap();

    assert_eq!(
        loaded.layers[0].neurons[0].protection_level,
        ProtectionLevel::Frozen
    );
    assert_eq!(
        loaded.layers[0].neurons[0].weight_protection[0],
        ProtectionLevel::Frozen
    );
    assert_eq!(
        loaded.layers[0].neurons[1].protection_level,
        ProtectionLevel::Guarded
    );
    assert_eq!(
        loaded.layers[1].neurons[0].weight_protection[0],
        ProtectionLevel::Frozen
    );
    assert_eq!(
        loaded.layers[1].neurons[0].bias_protection,
        ProtectionLevel::Guarded
    );
    assert_eq!(loaded.layers[0].neurons[0].source, Source::RawText);
    assert_eq!(
        loaded.layers[0].neurons[1].source,
        Source::LocalFile {
            path: "notes/facts.md".to_string()
        }
    );

    cleanup(&path);
}

#[test]
fn vocab_entries_survive_save_load() {
    let path = temp_path("vocab");
    let network = Network::new_empty(32);
    let vocab_entries = vec![
        VocabEntry {
            token: "cat".to_string(),
            id: 0,
            embedding: vec![0.1; 32],
        },
        VocabEntry {
            token: "#cat".to_string(),
            id: 1,
            embedding: vec![0.2; 32],
        },
    ];
    let state = BrainState {
        network,
        vocab_entries: vocab_entries.clone(),
    };

    let brain = ManasBrain::new(&path);
    brain.save_state(&state).unwrap();
    let loaded = brain.load_state().unwrap();

    assert_eq!(loaded.vocab_entries, vocab_entries);
    assert_eq!(loaded.network.input_dim, 32);

    cleanup(&path);
}

#[test]
fn magic_bytes_and_version_are_validated() {
    let path = temp_path("magic");
    let network = Network::new(32, 64, 32);
    let brain = ManasBrain::new(&path);
    brain.save(&network).unwrap();

    let mut bytes = fs::read(&path).unwrap();
    bytes[0] = b'X';
    refresh_checksum(&mut bytes);
    fs::write(&path, bytes).unwrap();
    let result = brain.load();
    assert!(
        matches!(result, Err(ManasError::CorruptBrain { .. })),
        "expected corrupt brain for bad magic, got {result:?}"
    );

    brain.save(&network).unwrap();
    let mut bytes = fs::read(&path).unwrap();
    bytes[4] = 99;
    refresh_checksum(&mut bytes);
    fs::write(&path, bytes).unwrap();
    let result = brain.load();
    assert!(
        matches!(result, Err(ManasError::CorruptBrain { .. })),
        "expected corrupt brain for bad version, got {result:?}"
    );

    cleanup(&path);
}

#[test]
fn file_size_tracks_network_shape() {
    let small_a = temp_path("small-a");
    let small_b = temp_path("small-b");
    let large = temp_path("large");

    let small_network = Network::new(32, 8, 32);
    let large_network = Network::new(32, 16, 32);
    let small_brain_a = ManasBrain::new(&small_a);
    let small_brain_b = ManasBrain::new(&small_b);
    let large_brain = ManasBrain::new(&large);

    small_brain_a.save(&small_network).unwrap();
    small_brain_b.save(&small_network).unwrap();
    large_brain.save(&large_network).unwrap();

    assert_eq!(small_brain_a.size_bytes(), small_brain_b.size_bytes());
    assert!(
        large_brain.size_bytes() > small_brain_a.size_bytes(),
        "larger hidden layer should produce a larger .manas file"
    );

    cleanup(&small_a);
    cleanup(&small_b);
    cleanup(&large);
}

#[test]
fn empty_network_survives_save_and_load() {
    let path = temp_path("empty");
    let network = Network::new_empty(32);
    let brain = ManasBrain::new(&path);

    brain.save(&network).unwrap();
    let loaded = brain.load().unwrap();

    assert_eq!(loaded.neuron_count(), 0);
    assert_eq!(loaded.layer_count(), 2);
    assert_eq!(loaded.input_dim, 32);
    assert_eq!(loaded.output_dim, 32);
    assert_eq!(loaded.forward(&[0.1; 32]), vec![0.0; 32]);

    cleanup(&path);
}

#[test]
fn grown_empty_network_survives_save_and_load() {
    let path = temp_path("grown-empty");
    let mut network = Network::new_empty(32);
    network.grow_neuron(0, 32).unwrap();
    network.grow_neuron(0, 32).unwrap();
    let expected_output = network.forward(&[0.1; 32]);
    let brain = ManasBrain::new(&path);

    brain.save(&network).unwrap();
    let loaded = brain.load().unwrap();

    assert_eq!(loaded.neuron_count(), network.neuron_count());
    assert_eq!(loaded.layers[0].neurons.len(), 2);
    assert_eq!(loaded.layers[1].neurons.len(), 32);
    assert_eq!(loaded.layers[1].neurons[0].weights.len(), 2);
    assert_eq!(loaded.forward(&[0.1; 32]), expected_output);

    cleanup(&path);
}

#[test]
fn anchor_consolidated_network_survives_save_load() {
    let path = temp_path("anchors");
    let mut trainer = Trainer::with_seed(42, EMBED_DIM, LEARNING_RATE);
    let anchors = trainer.encode_facts(&ANCHOR_FACTS);
    let mut network = Network::with_seed(2026, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM);

    trainer
        .train_facts(&mut network, &anchors, ANCHOR_TRAIN_EPOCHS)
        .unwrap();
    trainer
        .consolidate_anchors(&mut network, &anchors, ANCHOR_NEURONS_PER_FACT)
        .unwrap();

    let before = score_facts(&trainer, &network, &anchors);
    let frozen_hidden = network.frozen_hidden_neuron_count();
    let frozen_edges = network.frozen_output_edge_count();

    let brain = ManasBrain::new(&path);
    brain.save(&network).unwrap();
    let loaded = brain.load().unwrap();
    let after = score_facts(&trainer, &loaded, &anchors);

    assert_eq!(loaded.frozen_hidden_neuron_count(), frozen_hidden);
    assert_eq!(loaded.frozen_output_edge_count(), frozen_edges);
    for (index, (before_score, after_score)) in before.iter().zip(after.iter()).enumerate() {
        assert!(
            (before_score - after_score).abs() < 1.0e-6,
            "anchor {index} changed after save/load: before {before_score:.6}, after {after_score:.6}"
        );
    }

    cleanup(&path);
}

fn score_facts(trainer: &Trainer, network: &Network, facts: &[EncodedFact]) -> Vec<f32> {
    facts
        .iter()
        .map(|fact| trainer.similarity_for_fact(network, fact))
        .collect()
}

fn temp_path(label: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "manas-stage4-{label}-{}-{nanos}.manas",
        std::process::id()
    ))
}

fn cleanup(path: &Path) {
    let _ = fs::remove_file(path);
}

fn refresh_checksum(bytes: &mut [u8]) {
    let checksum_offset = bytes.len() - 4;
    let checksum = crc32(&bytes[..checksum_offset]);
    bytes[checksum_offset..].copy_from_slice(&checksum.to_le_bytes());
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFF;

    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            if crc & 1 == 1 {
                crc = (crc >> 1) ^ CRC32_POLYNOMIAL;
            } else {
                crc >>= 1;
            }
        }
    }

    !crc
}
