use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

use manas_core::{Activation, Network, ProtectionLevel};
use manas_ingest::{IngestSource, ingest};
use manas_learn::{
    AnswerSource, BrainDiagnostics, CompressionConfig, CompressionPlan, CompressionReport,
    DEFAULT_COMPRESSION_THRESHOLD, EncoderVocabEntry, FreshnessWarning, LearnReport,
    NeuronDiagnostics, NeuronFilter, QueryTrace, Trainer, compress, detect_freshness,
    filtered_neuron_diagnostics, plan_compression, trace_query,
};
use manas_store::{BrainState, ManasBrain, VocabEntry};

const DEFAULT_BRAIN_PATH: &str = "brain.manas";
const DEFAULT_LEARNING_RATE: f32 = 0.01;
const DEFAULT_EMBED_DIM: usize = 32;
const DEFAULT_SEED: u64 = 42;
const DEFAULT_TRACE_LIMIT: usize = 8;
const MAX_TRACE_LIMIT: usize = 100;

fn main() {
    if let Err(error) = run_cli(env::args().skip(1), Path::new(DEFAULT_BRAIN_PATH)) {
        eprintln!("Error: {error}");
        process::exit(1);
    }
}

fn run_cli<I, S>(args: I, brain_path: &Path) -> Result<(), String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let args = args.into_iter().map(Into::into).collect::<Vec<_>>();
    let Some(command) = args.first().map(String::as_str) else {
        print_help();
        return Ok(());
    };

    match command {
        "teach" => teach(brain_path, &args[1..]),
        "ask" => ask(brain_path, &args[1..]),
        "inspect" => inspect(brain_path),
        "neurons" => neurons(brain_path, &args[1..]),
        "trace" => trace(brain_path, &args[1..]),
        "forget" => forget(brain_path, &args[1..]),
        "reset" => reset(brain_path),
        "help" | "--help" | "-h" => {
            print_help();
            Ok(())
        }
        other => Err(format!("unknown command '{other}'")),
    }
}

fn teach(brain_path: &Path, args: &[String]) -> Result<(), String> {
    let request = teach_request(args)?;
    let chunks = ingest(request.source).map_err(|error| error.to_string())?;
    let (mut network, mut trainer) = load_or_create_runtime(brain_path)?;
    let mut summary = TeachSummary::new(request.mode, chunks.len());

    for chunk in &chunks {
        for unit in teachable_units(&chunk.text) {
            let (input, target) = extract_association(&unit)?;
            let freshness = detect_freshness(&unit);
            let report = trainer
                .learn_with_source_and_freshness(
                    &mut network,
                    &input,
                    &target,
                    chunk.source.clone(),
                    freshness,
                )
                .map_err(|error| error.to_string())?;
            summary.record(&input, &target, &report);
        }
    }

    if summary.examples_learned == 0 {
        return Err("input text is empty".to_string());
    }

    save_runtime(brain_path, network, &trainer)?;

    print_teach_report(&summary);
    Ok(())
}

fn ask(brain_path: &Path, args: &[String]) -> Result<(), String> {
    let question = joined_text(args)?;
    let brain = ManasBrain::new(brain_path);

    if !brain.exists() {
        print_answer(
            "Not enough knowledge yet.",
            0.0,
            AnswerSource::NotEnough,
            None,
        );
        return Ok(());
    }

    let state = brain.load_state().map_err(|error| error.to_string())?;
    let mut trainer = Trainer::with_seed(
        DEFAULT_SEED,
        state.network.input_dim.max(1),
        DEFAULT_LEARNING_RATE,
    );
    trainer
        .encoder
        .import_vocab(&to_encoder_entries(&state.vocab_entries))
        .map_err(|error| error.to_string())?;

    let result = trainer
        .query(&state.network, &question)
        .map_err(|error| error.to_string())?;
    print_answer(
        &result.answer,
        result.confidence,
        result.answered_from,
        result.freshness_warning.as_ref(),
    );
    Ok(())
}

fn inspect(brain_path: &Path) -> Result<(), String> {
    let brain = ManasBrain::new(brain_path);

    if !brain.exists() {
        print_empty_inspect(brain_path);
        return Ok(());
    }

    let state = brain.load_state().map_err(|error| error.to_string())?;
    let diagnostics = BrainDiagnostics::from_network(&state.network, unix_now_secs());

    println!("Brain");
    println!("  file                  : {}", brain_path.display());
    println!("  exists                : yes");
    println!("  size bytes            : {}", brain.size_bytes());
    println!(
        "  size                  : {}",
        human_size(brain.size_bytes())
    );
    println!(
        "  format version        : {}",
        state.metadata.format_version
    );
    println!(
        "  created               : {}",
        format_unix_date(state.metadata.created_at)
    );
    println!(
        "  last modified         : {}",
        format_unix_date(state.metadata.modified_at)
    );
    println!("  vocab entries         : {}", state.vocab_entries.len());
    println!();
    println!("Network");
    println!(
        "  total neurons         : {}",
        diagnostics.network.total_neurons
    );
    println!(
        "  total layers          : {}",
        diagnostics.network.total_layers
    );
    println!(
        "  open neurons          : {}",
        diagnostics.network.open_neurons
    );
    println!(
        "  guarded neurons       : {}",
        diagnostics.network.guarded_neurons
    );
    println!(
        "  frozen neurons        : {}",
        diagnostics.network.frozen_neurons
    );
    println!("  input dim             : {}", state.network.input_dim);
    println!("  output dim            : {}", state.network.output_dim);
    println!();
    println!("Learning");
    println!(
        "  facts taught          : {}",
        diagnostics.learning.facts_taught
    );
    println!(
        "  total learn calls     : {}",
        diagnostics.learning.total_learn_calls
    );
    println!(
        "  neurons grown         : {}",
        diagnostics.learning.neurons_grown
    );
    println!(
        "  layers grown          : {}",
        diagnostics.learning.layers_grown
    );
    println!();
    println!("Freshness");
    println!(
        "  timeless neurons      : {}",
        diagnostics.freshness.timeless_neurons
    );
    println!(
        "  slow neurons          : {}",
        diagnostics.freshness.slow_neurons
    );
    println!(
        "  fast neurons          : {}",
        diagnostics.freshness.fast_neurons
    );
    println!(
        "  realtime neurons      : {}",
        diagnostics.freshness.realtime_neurons
    );
    println!(
        "  stale neurons         : {}",
        diagnostics.freshness.stale_neurons
    );
    println!();
    println!("Sources");
    println!(
        "  raw text neurons      : {}",
        diagnostics.sources.raw_text_neurons
    );
    println!(
        "  local file neurons    : {}",
        diagnostics.sources.local_file_neurons
    );
    println!(
        "  unknown neurons       : {}",
        diagnostics.sources.unknown_neurons
    );
    println!();
    println!("Layers");
    for layer in diagnostics.layers {
        println!(
            "  layer {} id={} activation={} neurons={} open={} guarded={} frozen={}",
            layer.layer_index,
            layer.layer_id,
            activation_label(layer.activation),
            layer.neurons,
            layer.open_neurons,
            layer.guarded_neurons,
            layer.frozen_neurons
        );
    }
    Ok(())
}

fn neurons(brain_path: &Path, args: &[String]) -> Result<(), String> {
    let filter = neuron_filter(args)?;
    let brain = ManasBrain::new(brain_path);

    if !brain.exists() {
        println!("Neurons");
        println!("  brain                 : missing");
        println!("  count                 : 0");
        return Ok(());
    }

    let state = brain.load_state().map_err(|error| error.to_string())?;
    let rows = filtered_neuron_diagnostics(&state.network, unix_now_secs(), &filter);
    print_neurons(&rows, &filter);
    Ok(())
}

fn trace(brain_path: &Path, args: &[String]) -> Result<(), String> {
    let request = trace_request(args)?;
    let brain = ManasBrain::new(brain_path);

    if !brain.exists() {
        let trace = QueryTrace {
            question: request.question,
            answer: "Not enough knowledge yet.".to_string(),
            confidence: 0.0,
            answered_from: AnswerSource::NotEnough,
            selected_variant: None,
            variants: Vec::new(),
            top_hidden_activations: Vec::new(),
            top_output_values: Vec::new(),
        };
        print_trace(&trace);
        return Ok(());
    }

    let state = brain.load_state().map_err(|error| error.to_string())?;
    let mut trainer = Trainer::with_seed(
        DEFAULT_SEED,
        state.network.input_dim.max(1),
        DEFAULT_LEARNING_RATE,
    );
    trainer
        .encoder
        .import_vocab(&to_encoder_entries(&state.vocab_entries))
        .map_err(|error| error.to_string())?;

    let trace = trace_query(&trainer, &state.network, &request.question, request.limit);
    print_trace(&trace);
    Ok(())
}

fn forget(brain_path: &Path, args: &[String]) -> Result<(), String> {
    let request = forget_request(args)?;
    let brain = ManasBrain::new(brain_path);
    let config = CompressionConfig::with_threshold(request.threshold);

    if !brain.exists() {
        let plan = CompressionPlan {
            threshold: config.threshold,
            min_idle_days: config.min_idle_days,
            min_merge_similarity: config.min_merge_similarity,
            candidates: Vec::new(),
            skipped: Default::default(),
        };
        print_forget_report(request.dry_run, 0, 0, 0, &plan, None);
        return Ok(());
    }

    let mut state = brain.load_state().map_err(|error| error.to_string())?;
    let size_before = brain.size_bytes();

    if request.dry_run {
        let plan = plan_compression(&state.network, &config).map_err(|error| error.to_string())?;
        print_forget_report(
            true,
            size_before,
            size_before,
            state.network.neuron_count(),
            &plan,
            None,
        );
        return Ok(());
    }

    let report = compress(&mut state.network, &config).map_err(|error| error.to_string())?;
    if report.neurons_removed > 0 {
        brain
            .save_state(&BrainState::new(state.network, state.vocab_entries))
            .map_err(|error| error.to_string())?;
    }
    let size_after = brain.size_bytes();
    print_forget_report(
        false,
        size_before,
        size_after,
        report.neurons_before,
        &report.plan,
        Some(&report),
    );
    Ok(())
}

fn reset(brain_path: &Path) -> Result<(), String> {
    remove_if_exists(brain_path)?;
    for sidecar in known_sidecar_paths(brain_path) {
        remove_if_exists(&sidecar)?;
    }
    println!("Brain reset");
    Ok(())
}

fn load_or_create_runtime(brain_path: &Path) -> Result<(Network, Trainer), String> {
    let brain = ManasBrain::new(brain_path);
    if !brain.exists() {
        return Ok((
            Network::new_empty(DEFAULT_EMBED_DIM),
            Trainer::with_seed(DEFAULT_SEED, DEFAULT_EMBED_DIM, DEFAULT_LEARNING_RATE),
        ));
    }

    let state = brain.load_state().map_err(|error| error.to_string())?;
    let mut trainer = Trainer::with_seed(
        DEFAULT_SEED,
        state.network.input_dim.max(1),
        DEFAULT_LEARNING_RATE,
    );
    trainer
        .encoder
        .import_vocab(&to_encoder_entries(&state.vocab_entries))
        .map_err(|error| error.to_string())?;
    Ok((state.network, trainer))
}

fn save_runtime(brain_path: &Path, network: Network, trainer: &Trainer) -> Result<(), String> {
    let state = BrainState::new(network, to_store_entries(trainer.encoder.export_vocab()));
    ManasBrain::new(brain_path)
        .save_state(&state)
        .map_err(|error| error.to_string())
}

fn print_teach_report(summary: &TeachSummary) {
    println!("Teaching complete");
    println!();
    println!("Input");
    println!("  mode                  : {}", summary.mode.label());
    println!("  chunks processed      : {}", summary.chunks_processed);
    println!("  facts learned         : {}", summary.examples_learned);
    if summary.examples_learned == 1 {
        println!("  learned input         : {}", summary.last_input);
        println!("  learned target        : {}", summary.last_target);
    } else {
        println!("  last learned input    : {}", summary.last_input);
        println!("  last learned target   : {}", summary.last_target);
    }
    println!();
    println!("Network");
    println!("  neurons grown         : {}", summary.neurons_grown);
    println!("  layers grown          : {}", summary.layers_grown);
    println!("  neurons promoted      : {}", summary.neurons_promoted);
    println!("  neurons frozen        : {}", summary.neurons_frozen);
    println!("  total neurons         : {}", summary.total_neurons);
    println!();
    println!("Learning");
    println!(
        "  loss before           : {:.4}",
        summary.first_loss_before.unwrap_or(0.0)
    );
    println!(
        "  loss after            : {:.4}",
        summary.last_loss_after.unwrap_or(0.0)
    );
    println!(
        "  update applied        : {}",
        if summary.update_applied { "yes" } else { "no" }
    );
}

fn print_answer(
    answer: &str,
    confidence: f32,
    source: AnswerSource,
    freshness_warning: Option<&FreshnessWarning>,
) {
    print!(
        "{}",
        render_answer(answer, confidence, source, freshness_warning)
    );
}

fn render_answer(
    answer: &str,
    confidence: f32,
    source: AnswerSource,
    freshness_warning: Option<&FreshnessWarning>,
) -> String {
    use std::fmt::Write as _;

    let mut output = String::new();
    writeln!(&mut output, "Answer").expect("writing to String should not fail");
    writeln!(&mut output, "  {answer}").expect("writing to String should not fail");
    writeln!(&mut output).expect("writing to String should not fail");
    writeln!(&mut output, "Confidence").expect("writing to String should not fail");
    writeln!(&mut output, "  {:.2}", confidence).expect("writing to String should not fail");
    writeln!(&mut output).expect("writing to String should not fail");
    writeln!(&mut output, "Answered from").expect("writing to String should not fail");
    writeln!(&mut output, "  {}", answer_source_label(source))
        .expect("writing to String should not fail");

    if let Some(warning) = freshness_warning {
        writeln!(&mut output).expect("writing to String should not fail");
        writeln!(&mut output, "Note").expect("writing to String should not fail");
        writeln!(
            &mut output,
            "  This knowledge may be outdated ({} freshness, learned {} days ago).",
            warning.category.label(),
            warning.age_days
        )
        .expect("writing to String should not fail");
    }

    output
}

fn print_help() {
    println!("Manas");
    println!();
    println!("Usage:");
    println!("  manas teach <text|file|folder> [--recursive]");
    println!("  manas ask <question>");
    println!("  manas inspect");
    println!("  manas neurons [--protection open|guarded|frozen] [--source <text>]");
    println!("  manas trace <question> [--limit N]");
    println!("  manas forget [--dry-run] [--threshold N]");
    println!("  manas reset");
}

fn print_forget_report(
    dry_run: bool,
    size_before: u64,
    size_after: u64,
    neurons_before: u64,
    plan: &CompressionPlan,
    report: Option<&CompressionReport>,
) {
    let projected_removed = plan.projected_removed();
    let neurons_removed = report
        .map(|report| report.neurons_removed)
        .unwrap_or(projected_removed);
    let neurons_after = report
        .map(|report| report.neurons_after)
        .unwrap_or_else(|| neurons_before.saturating_sub(projected_removed as u64));

    println!("Forget");
    println!("  dry run               : {}", yes_no(dry_run));
    println!("  threshold             : {:.4}", plan.threshold);
    println!("  min idle days         : {}", plan.min_idle_days);
    println!("  min merge similarity  : {:.4}", plan.min_merge_similarity);
    println!("  candidates            : {}", plan.candidates.len());
    println!("  neurons before        : {}", neurons_before);
    println!("  neurons after         : {}", neurons_after);
    println!("  neurons removed       : {}", neurons_removed);
    println!("  size before bytes     : {}", size_before);
    println!("  size after bytes      : {}", size_after);
    println!(
        "  size delta bytes      : {}",
        size_after as i128 - size_before as i128
    );
    println!();
    println!("Skipped");
    println!("  protected             : {}", plan.skipped.protected);
    println!("  high importance       : {}", plan.skipped.high_importance);
    println!("  recent                : {}", plan.skipped.recent);
    println!("  no merge target       : {}", plan.skipped.no_merge_target);
    println!(
        "  unsupported shape     : {}",
        plan.skipped.unsupported_shape
    );
    println!();
    println!("Candidates");
    if plan.candidates.is_empty() {
        println!("  none");
        return;
    }

    println!(
        "  {:<5} {:<6} {:<8} {:<6} {:<8} {:>10} {:>8} {:>10}",
        "index", "id", "target", "id", "idle", "importance", "days", "similarity"
    );
    for candidate in &plan.candidates {
        println!(
            "  {:<5} {:<6} {:<8} {:<6} {:<8} {:>10.4} {:>8} {:>10.4}",
            candidate.source_index,
            candidate.source_neuron_id,
            candidate.target_index,
            candidate.target_neuron_id,
            "merge",
            candidate.importance_score,
            candidate.idle_days,
            candidate.merge_similarity
        );
    }
}

fn print_empty_inspect(brain_path: &Path) {
    println!("Brain");
    println!("  file                  : {}", brain_path.display());
    println!("  exists                : no");
    println!("  size bytes            : 0");
    println!("  size                  : 0 B");
    println!("  format version        : 0");
    println!("  created               : unknown");
    println!("  last modified         : unknown");
    println!("  vocab entries         : 0");
    println!();
    println!("Network");
    println!("  total neurons         : 0");
    println!("  total layers          : 0");
    println!("  open neurons          : 0");
    println!("  guarded neurons       : 0");
    println!("  frozen neurons        : 0");
    println!("  input dim             : 0");
    println!("  output dim            : 0");
    println!();
    println!("Learning");
    println!("  facts taught          : 0");
    println!("  total learn calls     : 0");
    println!("  neurons grown         : 0");
    println!("  layers grown          : 0");
    println!();
    println!("Freshness");
    println!("  timeless neurons      : 0");
    println!("  slow neurons          : 0");
    println!("  fast neurons          : 0");
    println!("  realtime neurons      : 0");
    println!("  stale neurons         : 0");
    println!();
    println!("Sources");
    println!("  raw text neurons      : 0");
    println!("  local file neurons    : 0");
    println!("  unknown neurons       : 0");
    println!();
    println!("Layers");
    println!("  none");
}

fn print_neurons(rows: &[NeuronDiagnostics], filter: &NeuronFilter) {
    println!("Neurons");
    println!("  count                 : {}", rows.len());
    println!(
        "  protection filter     : {}",
        filter.protection.map(protection_label).unwrap_or("all")
    );
    println!(
        "  source filter         : {}",
        filter.source_contains.as_deref().unwrap_or("all")
    );
    println!();

    if rows.is_empty() {
        println!("  none");
        return;
    }

    println!(
        "  {:<5} {:<5} {:<6} {:<10} {:<10} {:>10} {:>11} {:<10} {:<5} source",
        "layer",
        "index",
        "id",
        "activation",
        "protection",
        "importance",
        "activations",
        "freshness",
        "stale"
    );
    for row in rows {
        println!(
            "  {:<5} {:<5} {:<6} {:<10} {:<10} {:>10.4} {:>11} {:<10} {:<5} {}",
            row.layer_index,
            row.neuron_index,
            row.neuron_id,
            activation_label(row.activation),
            protection_label(row.protection),
            row.importance_score,
            row.activation_count,
            row.freshness.label(),
            yes_no(row.stale),
            row.source_label
        );
    }
}

fn print_trace(trace: &QueryTrace) {
    println!("Trace");
    println!("  question              : {}", trace.question);
    println!(
        "  selected variant      : {}",
        trace.selected_variant.as_deref().unwrap_or("none")
    );
    println!();
    println!("Variants");
    if trace.variants.is_empty() {
        println!("  none");
    } else {
        println!(
            "  {:<9} {:<7} {:<8} {:<8} {:>10} {:>10} answer",
            "selected", "encoded", "hidden", "id", "activation", "score"
        );
        for variant in &trace.variants {
            println!(
                "  {:<9} {:<7} {:<8} {:<8} {:>10.4} {:>10.4} {}",
                yes_no(variant.selected),
                yes_no(variant.encoded),
                optional_usize(variant.hidden_index),
                optional_u64(variant.hidden_neuron_id),
                variant.hidden_activation,
                variant.score,
                variant.decoded_answer.as_deref().unwrap_or("")
            );
            println!("    variant             : {}", variant.text);
        }
    }
    println!();
    println!("Top hidden activations");
    if trace.top_hidden_activations.is_empty() {
        println!("  none");
    } else {
        println!(
            "  {:<5} {:<5} {:<6} {:<10} {:>10} source",
            "layer", "index", "id", "protect", "activation"
        );
        for row in &trace.top_hidden_activations {
            println!(
                "  {:<5} {:<5} {:<6} {:<10} {:>10.4} {}",
                row.layer_index,
                row.neuron_index,
                row.neuron_id,
                protection_label(row.protection),
                row.activation,
                row.source_label
            );
        }
    }
    println!();
    println!("Top output values");
    if trace.top_output_values.is_empty() {
        println!("  none");
    } else {
        println!("  {:<6} {:<8} {:>10}", "index", "id", "value");
        for row in &trace.top_output_values {
            println!(
                "  {:<6} {:<8} {:>10.4}",
                row.output_index,
                optional_u64(row.neuron_id),
                row.value
            );
        }
    }
    println!();
    print!(
        "{}",
        render_answer(&trace.answer, trace.confidence, trace.answered_from, None)
    );
}

fn neuron_filter(args: &[String]) -> Result<NeuronFilter, String> {
    let mut filter = NeuronFilter::default();
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--protection" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--protection requires a value".to_string())?;
                filter.protection = Some(parse_protection(value)?);
            }
            "--source" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--source requires a value".to_string())?;
                if value.trim().is_empty() {
                    return Err("--source cannot be empty".to_string());
                }
                filter.source_contains = Some(value.trim().to_string());
            }
            option if option.starts_with("--") => {
                return Err(format!("unknown neurons option '{option}'"));
            }
            value => return Err(format!("unexpected neurons argument '{value}'")),
        }
        index += 1;
    }

    Ok(filter)
}

struct TraceRequest {
    question: String,
    limit: usize,
}

fn trace_request(args: &[String]) -> Result<TraceRequest, String> {
    let mut limit = DEFAULT_TRACE_LIMIT;
    let mut question_parts = Vec::new();
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--limit" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--limit requires a value".to_string())?;
                limit = value
                    .parse::<usize>()
                    .map_err(|_| format!("invalid --limit value '{value}'"))?;
                if !(1..=MAX_TRACE_LIMIT).contains(&limit) {
                    return Err(format!("--limit must be between 1 and {MAX_TRACE_LIMIT}"));
                }
            }
            option if option.starts_with("--") => {
                return Err(format!("unknown trace option '{option}'"));
            }
            value => question_parts.push(value.to_string()),
        }
        index += 1;
    }

    Ok(TraceRequest {
        question: joined_text(&question_parts)?,
        limit,
    })
}

struct ForgetRequest {
    dry_run: bool,
    threshold: f32,
}

fn forget_request(args: &[String]) -> Result<ForgetRequest, String> {
    let mut dry_run = false;
    let mut threshold = DEFAULT_COMPRESSION_THRESHOLD;
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--dry-run" => dry_run = true,
            "--threshold" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--threshold requires a value".to_string())?;
                threshold = value
                    .parse::<f32>()
                    .map_err(|_| format!("invalid --threshold value '{value}'"))?;
                if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
                    return Err("--threshold must be between 0.0 and 1.0".to_string());
                }
            }
            option if option.starts_with("--") => {
                return Err(format!("unknown forget option '{option}'"));
            }
            value => return Err(format!("unexpected forget argument '{value}'")),
        }
        index += 1;
    }

    Ok(ForgetRequest { dry_run, threshold })
}

fn parse_protection(value: &str) -> Result<ProtectionLevel, String> {
    match value.to_ascii_lowercase().as_str() {
        "open" => Ok(ProtectionLevel::Open),
        "guarded" => Ok(ProtectionLevel::Guarded),
        "frozen" => Ok(ProtectionLevel::Frozen),
        _ => Err(format!(
            "invalid protection '{value}', expected open, guarded, or frozen"
        )),
    }
}

fn activation_label(activation: Activation) -> &'static str {
    match activation {
        Activation::ReLU => "relu",
        Activation::Sigmoid => "sigmoid",
        Activation::Tanh => "tanh",
        Activation::Linear => "linear",
        Activation::Keyed => "keyed",
    }
}

fn protection_label(protection: ProtectionLevel) -> &'static str {
    match protection {
        ProtectionLevel::Open => "open",
        ProtectionLevel::Guarded => "guarded",
        ProtectionLevel::Frozen => "frozen",
    }
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

fn optional_usize(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".to_string())
}

fn optional_u64(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".to_string())
}

fn human_size(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * 1024.0;

    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / KB)
    } else {
        format!("{:.1} MB", bytes as f64 / MB)
    }
}

fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn format_unix_date(timestamp: u64) -> String {
    if timestamp == 0 {
        return "unknown".to_string();
    }

    let days = (timestamp / 86_400) as i64;
    let (year, month, day) = civil_from_days(days);
    format!("{year:04}-{month:02}-{day:02}")
}

fn civil_from_days(days_since_epoch: i64) -> (i64, i64, i64) {
    let z = days_since_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_part = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_part + 2) / 5 + 1;
    let month = month_part + if month_part < 10 { 3 } else { -9 };
    if month <= 2 {
        year += 1;
    }

    (year, month, day)
}

fn answer_source_label(source: AnswerSource) -> &'static str {
    match source {
        AnswerSource::NeuralWeights => "neural weights",
        AnswerSource::NotEnough => "not enough",
    }
}

fn joined_text(args: &[String]) -> Result<String, String> {
    let text = args.join(" ");
    let trimmed = text.trim();
    if trimmed.is_empty() {
        Err("input text is empty".to_string())
    } else {
        Ok(trimmed.to_string())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TeachMode {
    Text,
    File,
    Folder,
}

impl TeachMode {
    fn label(self) -> &'static str {
        match self {
            TeachMode::Text => "text",
            TeachMode::File => "file",
            TeachMode::Folder => "folder",
        }
    }
}

struct TeachRequest {
    mode: TeachMode,
    source: IngestSource,
}

struct TeachSummary {
    mode: TeachMode,
    chunks_processed: usize,
    examples_learned: usize,
    neurons_grown: u32,
    layers_grown: u32,
    neurons_promoted: u32,
    neurons_frozen: u32,
    total_neurons: u64,
    first_loss_before: Option<f32>,
    last_loss_after: Option<f32>,
    update_applied: bool,
    last_input: String,
    last_target: String,
}

impl TeachSummary {
    fn new(mode: TeachMode, chunks_processed: usize) -> Self {
        Self {
            mode,
            chunks_processed,
            examples_learned: 0,
            neurons_grown: 0,
            layers_grown: 0,
            neurons_promoted: 0,
            neurons_frozen: 0,
            total_neurons: 0,
            first_loss_before: None,
            last_loss_after: None,
            update_applied: false,
            last_input: String::new(),
            last_target: String::new(),
        }
    }

    fn record(&mut self, input: &str, target: &str, report: &LearnReport) {
        if self.first_loss_before.is_none() {
            self.first_loss_before = Some(report.loss_before);
        }

        self.examples_learned += 1;
        self.neurons_grown = self.neurons_grown.saturating_add(report.neurons_grown);
        self.layers_grown = self.layers_grown.saturating_add(report.layers_grown);
        self.neurons_promoted = self
            .neurons_promoted
            .saturating_add(report.neurons_promoted);
        self.neurons_frozen = self.neurons_frozen.saturating_add(report.neurons_frozen);
        self.total_neurons = report.total_neurons;
        self.last_loss_after = Some(report.loss_after);
        self.update_applied |= report.update_applied;
        self.last_input = input.to_string();
        self.last_target = target.to_string();
    }
}

fn teach_request(args: &[String]) -> Result<TeachRequest, String> {
    let mut recursive = false;
    let mut values = Vec::new();

    for arg in args {
        match arg.as_str() {
            "--recursive" => recursive = true,
            option if option.starts_with("--") => {
                return Err(format!("unknown teach option '{option}'"));
            }
            value => values.push(value.to_string()),
        }
    }

    if values.is_empty() {
        return Err("input text is empty".to_string());
    }

    if values.len() == 1 {
        let path = PathBuf::from(&values[0]);
        if path.exists() {
            if path.is_file() {
                return Ok(TeachRequest {
                    mode: TeachMode::File,
                    source: IngestSource::File(path),
                });
            }
            if path.is_dir() {
                return Ok(TeachRequest {
                    mode: TeachMode::Folder,
                    source: IngestSource::Folder(path),
                });
            }
            return Err(format!(
                "input path is not a file or folder: {}",
                path.display()
            ));
        }

        if looks_like_path(&values[0]) {
            return Err(format!("input path not found: {}", path.display()));
        }
    }

    if recursive {
        return Err("--recursive can only be used with folder input".to_string());
    }

    Ok(TeachRequest {
        mode: TeachMode::Text,
        source: IngestSource::RawText(values.join(" ")),
    })
}

fn looks_like_path(value: &str) -> bool {
    let has_whitespace = value.chars().any(char::is_whitespace);
    value.contains('/')
        || value.contains('\\')
        || (!has_whitespace && Path::new(value).extension().is_some())
        || value == "."
        || value == ".."
}

fn teachable_units(text: &str) -> Vec<String> {
    let mut units = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?' | '\n') {
            push_teachable_unit(&mut units, &mut current);
        }
    }
    push_teachable_unit(&mut units, &mut current);

    if units.is_empty() {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            units.push(trimmed.to_string());
        }
    }

    units
}

fn push_teachable_unit(units: &mut Vec<String>, current: &mut String) {
    let unit = current.trim();
    if !unit.is_empty() {
        units.push(unit.to_string());
    }
    current.clear();
}

fn extract_association(text: &str) -> Result<(String, String), String> {
    let cleaned = trim_sentence(text);
    if cleaned.is_empty() {
        return Err("input text is empty".to_string());
    }

    let lower = cleaned.to_lowercase();
    let markers = [
        " refers to ",
        " means ",
        " contains ",
        " describes ",
        " developed ",
        " pulls ",
        " boils ",
        " wrote ",
        " fell ",
        " were ",
        " was ",
        " are ",
        " is ",
    ];

    if let Some((index, marker)) = markers
        .iter()
        .filter_map(|marker| lower.find(marker).map(|index| (index, *marker)))
        .min_by_key(|(index, _)| *index)
    {
        let input = normalize_subject(&cleaned[..index], marker);
        let target = strip_leading_article(&cleaned[index + marker.len()..]);
        if !input.is_empty() && !target.is_empty() {
            return Ok((input, target));
        }
    }

    let input = normalized_words(&cleaned)
        .into_iter()
        .find(|word| !is_stopword(word))
        .ok_or_else(|| "could not find a teachable subject".to_string())?;
    Ok((input, cleaned))
}

fn normalize_subject(text: &str, marker: &str) -> String {
    let subject = strip_leading_article(text);
    if marker == " developed " {
        let words = subject.split_whitespace().collect::<Vec<_>>();
        if words.len() == 2 {
            return words[1].to_string();
        }
    }
    subject
}

fn trim_sentence(text: &str) -> String {
    text.trim()
        .trim_matches(|ch: char| matches!(ch, '.' | ',' | ';' | ':' | '!' | '?' | '"' | '\''))
        .trim()
        .to_string()
}

fn strip_leading_article(text: &str) -> String {
    let mut words = text.split_whitespace().collect::<Vec<_>>();
    if words
        .first()
        .map(|word| {
            let normalized = normalize_word(word);
            matches!(normalized.as_str(), "a" | "an" | "the")
        })
        .unwrap_or(false)
    {
        words.remove(0);
    }
    trim_sentence(&words.join(" "))
}

fn normalized_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter_map(|raw| {
            let cleaned = normalize_word(raw);
            if cleaned.is_empty() {
                None
            } else {
                Some(cleaned)
            }
        })
        .collect()
}

fn normalize_word(word: &str) -> String {
    word.chars()
        .filter(|ch| ch.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn is_stopword(word: &str) -> bool {
    matches!(
        word,
        "a" | "an" | "and" | "are" | "is" | "of" | "the" | "to" | "was" | "were"
    )
}

fn to_store_entries(entries: Vec<EncoderVocabEntry>) -> Vec<VocabEntry> {
    entries
        .into_iter()
        .map(|entry| VocabEntry {
            token: entry.token,
            id: entry.id,
            embedding: entry.embedding,
        })
        .collect()
}

fn to_encoder_entries(entries: &[VocabEntry]) -> Vec<EncoderVocabEntry> {
    entries
        .iter()
        .map(|entry| EncoderVocabEntry {
            token: entry.token.clone(),
            id: entry.id,
            embedding: entry.embedding.clone(),
        })
        .collect()
}

fn remove_if_exists(path: &Path) -> Result<(), String> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("failed to remove {}: {error}", path.display())),
    }
}

fn known_sidecar_paths(brain_path: &Path) -> Vec<PathBuf> {
    let base = brain_path.to_string_lossy();
    ["sources", "sourceindex", "seq", "transformer", "langmeta"]
        .into_iter()
        .map(|suffix| PathBuf::from(format!("{base}.{suffix}")))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use manas_learn::FreshnessCategory;

    #[test]
    fn extracts_simple_is_association() {
        let (input, target) =
            extract_association("A cat is a small domesticated animal with fur.").unwrap();

        assert_eq!(input, "cat");
        assert_eq!(target, "small domesticated animal with fur");
    }

    #[test]
    fn extracts_multi_word_subject() {
        let (input, target) =
            extract_association("The Eiffel Tower is located in Paris France.").unwrap();

        assert_eq!(input, "Eiffel Tower");
        assert_eq!(target, "located in Paris France");
    }

    #[test]
    fn extracts_earliest_relation_in_sentence() {
        let (input, target) = extract_association(
            "The Eiffel Tower is located in Paris France and was built in 1889.",
        )
        .unwrap();

        assert_eq!(input, "Eiffel Tower");
        assert_eq!(target, "located in Paris France and was built in 1889");
    }

    #[test]
    fn extracts_developed_relation_with_last_name_subject() {
        let (input, target) =
            extract_association("Albert Einstein developed the theory of relativity.").unwrap();

        assert_eq!(input, "Einstein");
        assert_eq!(target, "theory of relativity");
    }

    #[test]
    fn render_answer_omits_note_without_freshness_warning() {
        let output = render_answer("small animal", 0.91, AnswerSource::NeuralWeights, None);

        assert!(output.contains("Answer\n  small animal"));
        assert!(output.contains("Answered from\n  neural weights"));
        assert!(!output.contains("Note"));
    }

    #[test]
    fn render_answer_appends_stale_freshness_note() {
        let warning = FreshnessWarning {
            category: FreshnessCategory::Fast,
            age_days: 47,
        };

        let output = render_answer(
            "Rust 2.0 was released last month",
            0.88,
            AnswerSource::NeuralWeights,
            Some(&warning),
        );

        assert!(output.contains("Note\n"));
        assert!(
            output.contains(
                "  This knowledge may be outdated (Fast freshness, learned 47 days ago)."
            )
        );
    }
}
