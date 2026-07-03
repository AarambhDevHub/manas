use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

const MAX_STAGE13_BRAIN_BYTES: u64 = 500 * 1024;

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

struct DemoQuestion {
    question: &'static str,
    any_groups: &'static [&'static [&'static str]],
    all_words: &'static [&'static str],
}

const DEMO_QUESTIONS: &[DemoQuestion] = &[
    DemoQuestion {
        question: "What is a cat?",
        any_groups: &[&["small", "domesticated", "animal", "fur", "whiskers"]],
        all_words: &[],
    },
    DemoQuestion {
        question: "Where is the Eiffel Tower?",
        any_groups: &[&["paris", "france", "1889"]],
        all_words: &[],
    },
    DemoQuestion {
        question: "What did Einstein develop?",
        any_groups: &[],
        all_words: &["theory", "relativity"],
    },
    DemoQuestion {
        question: "What is the mitochondria?",
        any_groups: &[],
        all_words: &["powerhouse", "cell"],
    },
    DemoQuestion {
        question: "When was Bitcoin created?",
        any_groups: &[&["satoshi", "nakamoto", "2009"]],
        all_words: &[],
    },
];

#[test]
fn stage13_real_demo_answers_from_neural_weights_only() {
    let dir = temp_dir("stage13-demo");

    let reset = run(&dir, &["reset"]);
    assert_success(&reset);

    for fact in DEMO_FACTS {
        let teach = run(&dir, &["teach", fact]);
        assert_success(&teach);
        assert!(stdout(&teach).contains("Teaching complete"));
    }

    remove_sidecars(&dir);
    assert_no_sidecars(&dir);

    for demo_question in DEMO_QUESTIONS {
        let ask = run(&dir, &["ask", demo_question.question]);
        assert_success(&ask);
        let ask_stdout = stdout(&ask);
        assert!(
            ask_stdout.contains("Answered from\n  neural weights"),
            "{ask_stdout}"
        );
        assert!(
            !ask_stdout.contains("Not enough knowledge yet."),
            "{ask_stdout}"
        );
        assert_keywords(&ask_stdout, demo_question);
    }

    assert_no_sidecars(&dir);
    let brain_size = fs::metadata(dir.join("brain.manas")).unwrap().len();
    assert!(
        brain_size < MAX_STAGE13_BRAIN_BYTES,
        "brain.manas was {brain_size} bytes"
    );

    let inspect = run(&dir, &["inspect"]);
    assert_success(&inspect);
    let inspect_stdout = stdout(&inspect);
    for expected in [
        "total neurons",
        "total layers",
        "open neurons",
        "guarded neurons",
        "frozen neurons",
    ] {
        assert!(inspect_stdout.contains(expected), "{inspect_stdout}");
    }

    fs::remove_dir_all(dir).unwrap();
}

fn assert_keywords(output: &str, question: &DemoQuestion) {
    let normalized = output.to_lowercase();
    for word in question.all_words {
        assert!(
            normalized.contains(word),
            "answer to '{}' missed '{word}':\n{output}",
            question.question
        );
    }

    for group in question.any_groups {
        let matches = group
            .iter()
            .filter(|word| normalized.contains(**word))
            .count();
        assert!(
            matches >= 2,
            "answer to '{}' matched only {matches} keywords from {group:?}:\n{output}",
            question.question
        );
    }
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
    let dir = std::env::temp_dir().join(format!("manas-cli-{name}-{}-{nanos}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    dir
}
