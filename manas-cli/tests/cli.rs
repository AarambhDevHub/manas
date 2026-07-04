use std::fs;
use std::path::Path;
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

use manas_core::{Network, ProtectionLevel, Source};
use manas_learn::FreshnessCategory;
use manas_store::{BrainState, ManasBrain};

#[test]
fn cli_teach_ask_inspect_and_reset_work_across_processes() {
    let dir = temp_dir("teach-ask");

    let teach = run(
        &dir,
        &[
            "teach",
            "A cat is a small domesticated animal with fur and whiskers.",
        ],
    );
    assert_success(&teach);
    let teach_stdout = stdout(&teach);
    assert!(teach_stdout.contains("Teaching complete"));
    assert!(dir.join("brain.manas").exists());

    fs::write(dir.join("brain.manas.sources"), "sidecar should not matter").unwrap();
    fs::remove_file(dir.join("brain.manas.sources")).unwrap();

    let ask = run(&dir, &["ask", "What is a cat?"]);
    assert_success(&ask);
    let ask_stdout = stdout(&ask);
    assert!(ask_stdout.contains("Answered from"));
    assert!(ask_stdout.contains("neural weights"), "{ask_stdout}");
    assert!(
        ask_stdout.contains("animal") || ask_stdout.contains("fur"),
        "{ask_stdout}"
    );

    let inspect = run(&dir, &["inspect"]);
    assert_success(&inspect);
    let inspect_stdout = stdout(&inspect);
    assert!(inspect_stdout.contains("total neurons"));
    assert!(inspect_stdout.contains("guarded neurons"));

    let reset = run(&dir, &["reset"]);
    assert_success(&reset);
    assert!(!dir.join("brain.manas").exists());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn cli_stage14_inspect_neurons_and_trace_work() {
    let dir = temp_dir("stage14-debug");

    let teach = run(
        &dir,
        &[
            "teach",
            "A cat is a small domesticated animal with fur and whiskers.",
        ],
    );
    assert_success(&teach);

    let inspect = run(&dir, &["inspect"]);
    assert_success(&inspect);
    let inspect_stdout = stdout(&inspect);
    for expected in [
        "Brain",
        "format version",
        "created",
        "last modified",
        "Network",
        "Learning",
        "facts taught",
        "Freshness",
        "Sources",
        "Layers",
    ] {
        assert!(inspect_stdout.contains(expected), "{inspect_stdout}");
    }

    let neurons = run(&dir, &["neurons", "--protection", "open"]);
    assert_success(&neurons);
    let neurons_stdout = stdout(&neurons);
    assert!(neurons_stdout.contains("Neurons"), "{neurons_stdout}");
    assert!(
        neurons_stdout.contains("protection filter     : open"),
        "{neurons_stdout}"
    );
    assert!(neurons_stdout.contains("importance"), "{neurons_stdout}");
    assert!(neurons_stdout.contains("raw text"), "{neurons_stdout}");

    let trace = run(&dir, &["trace", "What is a cat?", "--limit", "3"]);
    assert_success(&trace);
    let trace_stdout = stdout(&trace);
    for expected in [
        "Trace",
        "selected variant",
        "Variants",
        "Top hidden activations",
        "Top output values",
        "Answered from\n  neural weights",
    ] {
        assert!(trace_stdout.contains(expected), "{trace_stdout}");
    }
    assert!(
        trace_stdout.contains("animal") || trace_stdout.contains("fur"),
        "{trace_stdout}"
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn cli_forget_without_brain_is_noop() {
    let dir = temp_dir("forget-empty");

    let forget = run(&dir, &["forget"]);
    assert_success(&forget);
    let forget_stdout = stdout(&forget);

    assert!(forget_stdout.contains("Forget"), "{forget_stdout}");
    assert!(
        forget_stdout.contains("candidates            : 0"),
        "{forget_stdout}"
    );
    assert!(
        forget_stdout.contains("neurons removed       : 0"),
        "{forget_stdout}"
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn cli_forget_dry_run_does_not_modify_brain() {
    let dir = temp_dir("forget-dry-run");
    write_compressible_brain(&dir);
    let path = dir.join("brain.manas");
    let size_before = fs::metadata(&path).unwrap().len();
    let count_before = ManasBrain::new(path.clone())
        .load_state()
        .unwrap()
        .network
        .neuron_count();

    let forget = run(&dir, &["forget", "--dry-run"]);
    assert_success(&forget);
    let forget_stdout = stdout(&forget);

    assert!(
        forget_stdout.contains("dry run               : yes"),
        "{forget_stdout}"
    );
    assert!(
        forget_stdout.contains("candidates            : 1"),
        "{forget_stdout}"
    );
    assert_eq!(fs::metadata(&path).unwrap().len(), size_before);
    assert_eq!(
        ManasBrain::new(path.clone())
            .load_state()
            .unwrap()
            .network
            .neuron_count(),
        count_before
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn cli_forget_compresses_and_shrinks_brain_file() {
    let dir = temp_dir("forget-shrink");
    write_compressible_brain(&dir);
    let path = dir.join("brain.manas");
    let size_before = fs::metadata(&path).unwrap().len();
    let count_before = ManasBrain::new(path.clone())
        .load_state()
        .unwrap()
        .network
        .neuron_count();

    let forget = run(&dir, &["forget", "--threshold", "0.20"]);
    assert_success(&forget);
    let forget_stdout = stdout(&forget);

    assert!(
        forget_stdout.contains("dry run               : no"),
        "{forget_stdout}"
    );
    assert!(
        forget_stdout.contains("neurons removed       : 1"),
        "{forget_stdout}"
    );
    let size_after = fs::metadata(&path).unwrap().len();
    let count_after = ManasBrain::new(path.clone())
        .load_state()
        .unwrap()
        .network
        .neuron_count();
    assert!(size_after < size_before, "{size_before} -> {size_after}");
    assert!(
        count_after < count_before,
        "{count_before} -> {count_after}"
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn cli_neurons_filters_by_protection_and_source() {
    let dir = temp_dir("stage14-filters");
    let notes = dir.join("notes.txt");
    fs::write(
        &notes,
        "A cat is a small domesticated animal with fur and whiskers.",
    )
    .unwrap();
    let notes_arg = notes.to_string_lossy().into_owned();

    let teach = run(&dir, &["teach", &notes_arg]);
    assert_success(&teach);

    let brain = ManasBrain::new(dir.join("brain.manas"));
    let mut state = brain.load_state().unwrap();
    state.network.layers[0].neurons[0].freeze_all();
    brain
        .save_state(&BrainState::new(state.network, state.vocab_entries))
        .unwrap();

    let frozen = run(&dir, &["neurons", "--protection", "frozen"]);
    assert_success(&frozen);
    let frozen_stdout = stdout(&frozen);
    assert!(
        frozen_stdout.contains("protection filter     : frozen"),
        "{frozen_stdout}"
    );
    assert!(frozen_stdout.contains("frozen"), "{frozen_stdout}");
    assert!(frozen_stdout.contains("notes.txt"), "{frozen_stdout}");

    let source = run(&dir, &["neurons", "--source", "notes.txt"]);
    assert_success(&source);
    let source_stdout = stdout(&source);
    assert!(
        source_stdout.contains("source filter         : notes.txt"),
        "{source_stdout}"
    );
    assert!(source_stdout.contains("notes.txt"), "{source_stdout}");

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn cli_teach_file_preserves_source_and_answers() {
    let dir = temp_dir("teach-file");
    let notes = dir.join("notes.txt");
    fs::write(
        &notes,
        "A cat is a small domesticated animal with fur and whiskers.",
    )
    .unwrap();
    let notes_arg = notes.to_string_lossy().into_owned();

    let teach = run(&dir, &["teach", &notes_arg]);
    assert_success(&teach);
    let teach_stdout = stdout(&teach);
    assert!(teach_stdout.contains("Teaching complete"));
    assert!(teach_stdout.contains("mode                  : file"));

    let state = ManasBrain::new(dir.join("brain.manas"))
        .load_state()
        .unwrap();
    let has_file_source = state
        .network
        .layers
        .first()
        .map(|layer| {
            layer.neurons.iter().any(|neuron| {
                matches!(&neuron.source, Source::LocalFile { path } if path.contains("notes.txt"))
            })
        })
        .unwrap_or(false);
    assert!(has_file_source, "expected notes.txt source metadata");

    let ask = run(&dir, &["ask", "What is a cat?"]);
    assert_success(&ask);
    let ask_stdout = stdout(&ask);
    assert!(ask_stdout.contains("neural weights"), "{ask_stdout}");
    assert!(
        ask_stdout.contains("animal") || ask_stdout.contains("fur"),
        "{ask_stdout}"
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn cli_teach_folder_walks_supported_files_recursively() {
    let dir = temp_dir("teach-folder");
    let docs = dir.join("docs");
    let nested = docs.join("nested");
    fs::create_dir_all(&nested).unwrap();
    fs::write(docs.join("cat.txt"), "A cat is a small animal with fur.").unwrap();
    fs::write(
        nested.join("paris.md"),
        "# Paris\n\nParis is a city in France.",
    )
    .unwrap();
    fs::write(docs.join("skip.exe"), "ignored").unwrap();

    let teach = run(&dir, &["teach", "docs", "--recursive"]);
    assert_success(&teach);
    let teach_stdout = stdout(&teach);
    assert!(teach_stdout.contains("Teaching complete"));
    assert!(teach_stdout.contains("mode                  : folder"));
    assert!(teach_stdout.contains("chunks processed"));
    assert!(teach_stdout.contains("facts learned"));

    let ask = run(&dir, &["ask", "Where is Paris?"]);
    assert_success(&ask);
    let ask_stdout = stdout(&ask);
    assert!(ask_stdout.contains("neural weights"), "{ask_stdout}");

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn cli_teach_stamps_realtime_freshness_metadata() {
    let dir = temp_dir("teach-freshness");

    let teach = run(
        &dir,
        &["teach", "Breaking news: the stock market fell today."],
    );
    assert_success(&teach);

    let state = ManasBrain::new(dir.join("brain.manas"))
        .load_state()
        .unwrap();
    let has_realtime_freshness = state
        .network
        .layers
        .first()
        .map(|layer| {
            layer
                .neurons
                .iter()
                .any(|neuron| neuron.freshness_category == FreshnessCategory::Realtime as u8)
        })
        .unwrap_or(false);
    assert!(
        has_realtime_freshness,
        "expected realtime freshness metadata"
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn cli_ask_without_brain_returns_not_enough() {
    let dir = temp_dir("empty-ask");

    let ask = run(&dir, &["ask", "What is a cat?"]);
    assert_success(&ask);
    let ask_stdout = stdout(&ask);

    assert!(ask_stdout.contains("Not enough knowledge yet."));
    assert!(ask_stdout.contains("not enough"));

    fs::remove_dir_all(dir).unwrap();
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

fn temp_dir(name: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("manas-cli-{name}-{}-{nanos}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_compressible_brain(dir: &Path) {
    let mut network = Network::new_empty(4);
    for _ in 0..3 {
        network.grow_neuron(0, 4).unwrap();
    }

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
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
