use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

use manas_core::Network;
use manas_learn::Trainer;
use manas_store::{BrainState, ManasBrain, VocabEntry};

const EMBED_DIM: usize = 32;
const LEARNING_RATE: f32 = 0.01;

#[test]
fn stage17_cli_inspects_and_queries_deep_brain() {
    let dir = temp_dir("cli-deep");
    let mut network = Network::new_empty(EMBED_DIM);
    let mut trainer = Trainer::with_seed(42, EMBED_DIM, LEARNING_RATE);

    trainer
        .learn(&mut network, "cat", "small animal with fur")
        .unwrap();
    trainer
        .learn(&mut network, "paris", "city in france")
        .unwrap();
    for neuron in &mut network.layers[0].neurons {
        neuron.guard_all();
    }
    let report = trainer
        .learn(&mut network, "rust", "systems programming language")
        .unwrap();
    assert_eq!(report.layers_grown, 1);
    assert_eq!(network.layer_count(), 3);

    ManasBrain::new(dir.join("brain.manas"))
        .save_state(&BrainState::new(network, store_vocab(&trainer)))
        .unwrap();

    let inspect = run(&dir, &["inspect"]);
    assert_success(&inspect);
    let inspect_stdout = stdout(&inspect);
    assert!(
        inspect_stdout.contains("total layers          : 3"),
        "{inspect_stdout}"
    );
    assert!(
        inspect_stdout.contains("layers grown          : 1"),
        "{inspect_stdout}"
    );
    assert!(
        inspect_stdout.contains("layer 2 id=2 activation=linear"),
        "{inspect_stdout}"
    );

    let ask = run(&dir, &["ask", "What is Rust?"]);
    assert_success(&ask);
    let ask_stdout = stdout(&ask);
    assert!(
        ask_stdout.contains("Answered from\n  neural weights"),
        "{ask_stdout}"
    );
    assert!(
        ask_stdout.contains("systems") || ask_stdout.contains("programming"),
        "{ask_stdout}"
    );

    cleanup_dir(dir);
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
        "manas-stage17-{name}-{}-{nanos}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn cleanup_dir(path: PathBuf) {
    let _ = fs::remove_dir_all(path);
}
