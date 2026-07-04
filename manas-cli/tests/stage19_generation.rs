use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

const FACTS: &[&str] = &[
    "A cat is a small domesticated animal with fur and whiskers.",
    "The Eiffel Tower is located in Paris France and was built in 1889.",
    "Albert Einstein developed the theory of relativity in the early 20th century.",
    "Bitcoin was created by Satoshi Nakamoto and launched in January 2009.",
];

#[test]
fn stage19_generate_answers_from_neural_weights_only() {
    let dir = temp_dir("stage19-generate");
    teach_facts(&dir);
    remove_sidecars(&dir);

    let output = run(&dir, &["generate", "What is a cat?"]);
    assert_success(&output);
    let stdout = stdout(&output);

    assert!(stdout.contains("Generated\n"), "{stdout}");
    assert!(
        stdout.contains("Generated from\n  neural weights"),
        "{stdout}"
    );
    for word in ["cat", "small", "domesticated", "animal", "fur", "whiskers"] {
        assert!(stdout.to_lowercase().contains(word), "{stdout}");
    }

    assert_no_sidecars(&dir);
    cleanup_dir(dir);
}

#[test]
fn stage19_ask_fluent_keeps_plain_ask_unchanged() {
    let dir = temp_dir("stage19-ask-fluent");
    teach_facts(&dir);
    remove_sidecars(&dir);

    let fluent = run(&dir, &["ask", "--fluent", "What did Einstein develop?"]);
    assert_success(&fluent);
    let fluent_stdout = stdout(&fluent);
    assert!(fluent_stdout.contains("Generated\n"), "{fluent_stdout}");
    assert!(
        fluent_stdout.contains("Generated from\n  neural weights"),
        "{fluent_stdout}"
    );
    for word in ["einstein", "developed", "theory", "relativity"] {
        assert!(
            fluent_stdout.to_lowercase().contains(word),
            "{fluent_stdout}"
        );
    }

    let plain = run(&dir, &["ask", "What did Einstein develop?"]);
    assert_success(&plain);
    let plain_stdout = stdout(&plain);
    assert!(plain_stdout.contains("Answer\n"), "{plain_stdout}");
    assert!(
        plain_stdout.contains("Answered from\n  neural weights"),
        "{plain_stdout}"
    );
    assert!(!plain_stdout.contains("Generated\n"), "{plain_stdout}");

    assert_no_sidecars(&dir);
    cleanup_dir(dir);
}

#[test]
fn stage19_generate_respects_max_words() {
    let dir = temp_dir("stage19-max-words");
    teach_facts(&dir);

    let output = run(
        &dir,
        &["generate", "When was Bitcoin created?", "--max-words", "8"],
    );
    assert_success(&output);
    let generated = generated_text(&stdout(&output));

    assert!(generated.split_whitespace().count() <= 8, "{generated}");
    assert!(generated.ends_with('.'), "{generated}");

    cleanup_dir(dir);
}

fn teach_facts(dir: &Path) {
    assert_success(&run(dir, &["reset"]));
    for fact in FACTS {
        assert_success(&run(dir, &["teach", fact]));
    }
}

fn generated_text(output: &str) -> String {
    let mut lines = output.lines();
    while let Some(line) = lines.next() {
        if line.trim() == "Generated" {
            return lines.next().unwrap_or_default().trim().to_string();
        }
    }
    String::new()
}

fn remove_sidecars(dir: &Path) {
    for sidecar in sidecar_paths(dir) {
        let _ = fs::remove_file(sidecar);
    }
}

fn assert_no_sidecars(dir: &Path) {
    for sidecar in sidecar_paths(dir) {
        assert!(!sidecar.exists(), "sidecar exists: {}", sidecar.display());
    }
}

fn sidecar_paths(dir: &Path) -> Vec<PathBuf> {
    [
        "brain.manas.sources",
        "brain.manas.sourceindex",
        "brain.manas.seq",
        "brain.manas.transformer",
        "brain.manas.langmeta",
    ]
    .into_iter()
    .map(|name| dir.join(name))
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
    let dir = std::env::temp_dir().join(format!("manas-{name}-{nanos}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn cleanup_dir(dir: PathBuf) {
    fs::remove_dir_all(dir).unwrap();
}
