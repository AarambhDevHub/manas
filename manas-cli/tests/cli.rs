use std::fs;
use std::path::Path;
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

use manas_core::Source;
use manas_learn::FreshnessCategory;
use manas_store::ManasBrain;

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
