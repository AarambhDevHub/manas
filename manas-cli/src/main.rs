use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use manas_core::Network;
use manas_ingest::{IngestSource, ingest};
use manas_learn::{
    AnswerSource, EncoderVocabEntry, FreshnessWarning, LearnReport, Trainer, detect_freshness,
};
use manas_store::{BrainState, ManasBrain, VocabEntry};

const DEFAULT_BRAIN_PATH: &str = "brain.manas";
const DEFAULT_LEARNING_RATE: f32 = 0.01;
const DEFAULT_EMBED_DIM: usize = 32;
const DEFAULT_SEED: u64 = 42;

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
        println!("Brain");
        println!("  file                  : {}", brain_path.display());
        println!("  exists                : no");
        println!("  size bytes            : 0");
        println!();
        println!("Network");
        println!("  total neurons         : 0");
        println!("  total layers          : 0");
        println!("  open neurons          : 0");
        println!("  guarded neurons       : 0");
        println!("  frozen neurons        : 0");
        return Ok(());
    }

    let state = brain.load_state().map_err(|error| error.to_string())?;
    println!("Brain");
    println!("  file                  : {}", brain_path.display());
    println!("  exists                : yes");
    println!("  size bytes            : {}", brain.size_bytes());
    println!("  vocab entries         : {}", state.vocab_entries.len());
    println!();
    println!("Network");
    println!("  total neurons         : {}", state.network.neuron_count());
    println!("  total layers          : {}", state.network.layer_count());
    println!(
        "  open neurons          : {}",
        state.network.open_neuron_count()
    );
    println!(
        "  guarded neurons       : {}",
        state.network.guarded_neuron_count()
    );
    println!(
        "  frozen neurons        : {}",
        state.network.frozen_neuron_count()
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
    let state = BrainState {
        network,
        vocab_entries: to_store_entries(trainer.encoder.export_vocab()),
    };
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
    println!("  manas reset");
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
