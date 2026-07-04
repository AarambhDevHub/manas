use std::env;
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use manas_core::{Network, ProtectionLevel};
use manas_learn::fixtures::{
    ANCHOR_FACTS, ANCHOR_NEURONS_PER_FACT, ANCHOR_TRAIN_EPOCHS, EMBED_DIM, HIDDEN_DIM,
    LEARNING_RATE, NOISE_FACTS, NOISE_TRAIN_EPOCHS, OUTPUT_DIM,
};
use manas_learn::{Tokenizer, Trainer};
use manas_store::{BrainState, ManasBrain};

const KB: f64 = 1024.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Quick,
    Full,
}

#[derive(Clone, Debug)]
struct BenchOptions {
    mode: Mode,
    write_markdown: Option<PathBuf>,
}

#[derive(Clone, Debug)]
struct BenchResult {
    id: &'static str,
    name: &'static str,
    value: f64,
    unit: &'static str,
    detail: String,
}

fn main() {
    let options = parse_options();
    let results = run_benchmarks(options.mode);
    let markdown = render_markdown(options.mode, &results);
    println!("{markdown}");

    if let Some(path) = options.write_markdown {
        let report_path = resolve_report_path(&path);
        fs::write(&report_path, markdown).unwrap_or_else(|error| {
            panic!("failed to write {}: {error}", report_path.display());
        });
    }
}

fn run_benchmarks(mode: Mode) -> Vec<BenchResult> {
    vec![
        bench_single_teach(mode),
        bench_single_ask(mode),
        bench_save(mode),
        bench_load(mode),
        bench_tokenizer_1000_words(mode),
        bench_anti_forgetting(),
        bench_memory_1000_neurons(),
        bench_file_growth_per_fact(mode),
    ]
}

fn bench_single_teach(mode: Mode) -> BenchResult {
    let iterations = iterations(mode, 5, 25);
    let elapsed = repeat(iterations, || {
        let mut network = Network::new_empty(EMBED_DIM);
        let mut trainer = Trainer::with_seed(42, EMBED_DIM, LEARNING_RATE);
        trainer
            .learn(
                &mut network,
                "cat",
                "small domesticated animal with fur and whiskers",
            )
            .expect("teach benchmark should learn");
        black_box(network.neuron_count());
    });

    BenchResult {
        id: "B1",
        name: "single teach",
        value: millis_per_iter(elapsed, iterations),
        unit: "ms/op",
        detail: format!("{iterations} iterations"),
    }
}

fn bench_single_ask(mode: Mode) -> BenchResult {
    let iterations = iterations(mode, 25, 200);
    let mut network = Network::new_empty(EMBED_DIM);
    let mut trainer = Trainer::with_seed(42, EMBED_DIM, LEARNING_RATE);
    trainer
        .learn(
            &mut network,
            "cat",
            "small domesticated animal with fur and whiskers",
        )
        .expect("ask benchmark setup should learn");

    let elapsed = repeat(iterations, || {
        let result = trainer
            .query(&network, "What is a cat?")
            .expect("ask benchmark should query");
        black_box(result.confidence);
    });

    BenchResult {
        id: "B2",
        name: "single ask",
        value: millis_per_iter(elapsed, iterations),
        unit: "ms/op",
        detail: format!("{iterations} iterations"),
    }
}

fn bench_save(mode: Mode) -> BenchResult {
    let iterations = iterations(mode, 5, 25);
    let (state, _trainer) = trained_state(12);
    let dir = temp_dir("bench-save");
    let brain = ManasBrain::new(dir.join("brain.manas"));

    let elapsed = repeat(iterations, || {
        brain
            .save_state(&state)
            .expect("save benchmark should write state");
        black_box(brain.size_bytes());
    });

    cleanup_dir(&dir);
    BenchResult {
        id: "B3",
        name: ".manas save",
        value: millis_per_iter(elapsed, iterations),
        unit: "ms/op",
        detail: format!("{iterations} iterations"),
    }
}

fn bench_load(mode: Mode) -> BenchResult {
    let iterations = iterations(mode, 10, 50);
    let (state, _trainer) = trained_state(12);
    let dir = temp_dir("bench-load");
    let brain = ManasBrain::new(dir.join("brain.manas"));
    brain
        .save_state(&state)
        .expect("load benchmark setup should save state");

    let elapsed = repeat(iterations, || {
        let loaded = brain
            .load_state()
            .expect("load benchmark should load state");
        black_box(loaded.network.neuron_count());
    });

    cleanup_dir(&dir);
    BenchResult {
        id: "B4",
        name: ".manas load",
        value: millis_per_iter(elapsed, iterations),
        unit: "ms/op",
        detail: format!("{iterations} iterations"),
    }
}

fn bench_tokenizer_1000_words(mode: Mode) -> BenchResult {
    let iterations = iterations(mode, 10, 100);
    let text = thousand_word_text();
    let elapsed = repeat(iterations, || {
        let mut tokenizer = Tokenizer::new(4);
        let ids = tokenizer.encode(&text);
        black_box(ids.len());
    });

    BenchResult {
        id: "B5",
        name: "tokenizer 1000 words",
        value: millis_per_iter(elapsed, iterations),
        unit: "ms/op",
        detail: format!("{iterations} iterations"),
    }
}

fn bench_anti_forgetting() -> BenchResult {
    let elapsed = measure(|| {
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
            .expect("anchor training should succeed");
        trainer
            .consolidate_anchors(&mut network, &anchors, ANCHOR_NEURONS_PER_FACT)
            .expect("anchor consolidation should succeed");
        trainer
            .train_facts(&mut network, &noise, NOISE_TRAIN_EPOCHS)
            .expect("noise training should succeed");
        trainer
            .fit_new_facts(&mut network, &noise, &anchors)
            .expect("new fact fitting should succeed");
        black_box(network.neuron_count());
    });

    BenchResult {
        id: "B6",
        name: "anti-forgetting proof",
        value: elapsed.as_secs_f64(),
        unit: "s",
        detail: "single fixed seed".to_string(),
    }
}

fn bench_memory_1000_neurons() -> BenchResult {
    let network = Network::new(32, 968, 32);
    let bytes = estimate_network_bytes(&network);

    BenchResult {
        id: "B7",
        name: "1000-neuron footprint",
        value: bytes as f64 / KB,
        unit: "KiB",
        detail: format!("estimated heap footprint, total={}", network.neuron_count()),
    }
}

fn bench_file_growth_per_fact(mode: Mode) -> BenchResult {
    let fact_count = match mode {
        Mode::Quick => 8,
        Mode::Full => 32,
    };
    let dir = temp_dir("bench-growth");
    let brain = ManasBrain::new(dir.join("brain.manas"));
    let mut network = Network::new_empty(EMBED_DIM);
    let mut trainer = Trainer::with_seed(42, EMBED_DIM, LEARNING_RATE);
    let mut previous_size = 0_u64;
    let mut deltas = Vec::with_capacity(fact_count);

    for index in 0..fact_count {
        let input = format!("bench fact {index}");
        let target = format!("bench value {index}");
        trainer
            .learn(&mut network, &input, &target)
            .expect("file growth benchmark should learn");
        let state = BrainState::new(network.clone(), store_vocab(&trainer));
        brain
            .save_state(&state)
            .expect("file growth benchmark should save");
        let size = brain.size_bytes();
        deltas.push(size.saturating_sub(previous_size));
        previous_size = size;
    }

    cleanup_dir(&dir);
    let average = deltas.iter().sum::<u64>() as f64 / deltas.len() as f64;
    let min = deltas.iter().min().copied().unwrap_or(0);
    let max = deltas.iter().max().copied().unwrap_or(0);

    BenchResult {
        id: "B8",
        name: "brain growth per fact",
        value: average,
        unit: "bytes/fact",
        detail: format!("n={fact_count}, min={min}, max={max}"),
    }
}

fn trained_state(fact_count: usize) -> (BrainState, Trainer) {
    let mut network = Network::new_empty(EMBED_DIM);
    let mut trainer = Trainer::with_seed(42, EMBED_DIM, LEARNING_RATE);

    for index in 0..fact_count {
        let input = format!("benchmark fact {index}");
        let target = format!("benchmark value {index}");
        trainer
            .learn(&mut network, &input, &target)
            .expect("benchmark setup should learn");
    }

    (BrainState::new(network, store_vocab(&trainer)), trainer)
}

fn store_vocab(trainer: &Trainer) -> Vec<manas_store::VocabEntry> {
    trainer
        .encoder
        .export_vocab()
        .into_iter()
        .map(|entry| manas_store::VocabEntry {
            token: entry.token,
            id: entry.id,
            embedding: entry.embedding,
        })
        .collect()
}

fn estimate_network_bytes(network: &Network) -> usize {
    network
        .layers
        .iter()
        .map(|layer| {
            layer.neurons.capacity() * std::mem::size_of::<manas_core::Neuron>()
                + layer
                    .neurons
                    .iter()
                    .map(estimate_neuron_owned_bytes)
                    .sum::<usize>()
        })
        .sum()
}

fn estimate_neuron_owned_bytes(neuron: &manas_core::Neuron) -> usize {
    neuron.weights.capacity() * std::mem::size_of::<f32>()
        + neuron.weight_protection.capacity() * std::mem::size_of::<ProtectionLevel>()
}

fn thousand_word_text() -> String {
    let words = [
        "cat", "paris", "rust", "everest", "dna", "amazon", "bitcoin", "jupiter", "compiler",
        "network",
    ];
    (0..1000)
        .map(|index| words[index % words.len()])
        .collect::<Vec<_>>()
        .join(" ")
}

fn repeat(iterations: usize, mut body: impl FnMut()) -> Duration {
    let start = Instant::now();
    for _ in 0..iterations {
        body();
    }
    start.elapsed()
}

fn measure(body: impl FnOnce()) -> Duration {
    let start = Instant::now();
    body();
    start.elapsed()
}

fn millis_per_iter(elapsed: Duration, iterations: usize) -> f64 {
    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

fn iterations(mode: Mode, quick: usize, full: usize) -> usize {
    match mode {
        Mode::Quick => quick,
        Mode::Full => full,
    }
}

fn render_markdown(mode: Mode, results: &[BenchResult]) -> String {
    let mode_label = match mode {
        Mode::Quick => "quick",
        Mode::Full => "full",
    };
    let mut output = String::new();
    output.push_str("# Benchmarks\n\n");
    output.push_str(
        "Generated by `cargo bench -p manas-benches -- --write-markdown BENCHMARKS.md`.\n\n",
    );
    output.push_str(&format!("Mode: `{mode_label}`\n\n"));
    output.push_str("| ID | Benchmark | Result | Unit | Detail |\n");
    output.push_str("|---|---|---:|---|---|\n");
    for result in results {
        output.push_str(&format!(
            "| {} | {} | {:.4} | {} | {} |\n",
            result.id, result.name, result.value, result.unit, result.detail
        ));
    }
    output.push('\n');
    output.push_str("B7 reports an internal heap-footprint estimate for network-owned buffers and neuron storage, not process RSS.\n\n");
    output.push_str("Run full benchmarks with `cargo bench -p manas-benches`.\n");
    output.push_str("Run CI smoke benchmarks with `cargo bench -p manas-benches -- --quick`.\n");
    output
}

fn parse_options() -> BenchOptions {
    let mut mode = Mode::Full;
    let mut write_markdown = None;
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bench" => {}
            "--quick" => mode = Mode::Quick,
            "--write-markdown" => {
                let path = args
                    .next()
                    .unwrap_or_else(|| panic!("--write-markdown requires a path"));
                write_markdown = Some(PathBuf::from(path));
            }
            "--help" | "-h" => {
                println!(
                    "Usage: cargo bench -p manas-benches -- [--quick] [--write-markdown PATH]"
                );
                std::process::exit(0);
            }
            other => panic!("unknown benchmark option '{other}'"),
        }
    }

    BenchOptions {
        mode,
        write_markdown,
    }
}

fn resolve_report_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }

    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("bench crate should live under workspace root")
        .join(path)
}

fn temp_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after unix epoch")
        .as_nanos();
    let dir = env::temp_dir().join(format!("manas-{name}-{}-{nanos}", std::process::id()));
    fs::create_dir_all(&dir).expect("benchmark temp dir should be created");
    dir
}

fn cleanup_dir(path: &Path) {
    let _ = fs::remove_dir_all(path);
}
