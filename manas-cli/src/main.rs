use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use manas_core::Network;
use manas_learn::{AnswerSource, EncoderVocabEntry, LearnReport, Trainer};
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
    let text = joined_text(args)?;
    let (input, target) = extract_association(&text)?;
    let (mut network, mut trainer) = load_or_create_runtime(brain_path)?;

    let report = trainer
        .learn(&mut network, &input, &target)
        .map_err(|error| error.to_string())?;
    save_runtime(brain_path, network, &trainer)?;

    print_teach_report(&input, &target, &report);
    Ok(())
}

fn ask(brain_path: &Path, args: &[String]) -> Result<(), String> {
    let question = joined_text(args)?;
    let brain = ManasBrain::new(brain_path);

    if !brain.exists() {
        print_answer("Not enough knowledge yet.", 0.0, AnswerSource::NotEnough);
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
    print_answer(&result.answer, result.confidence, result.answered_from);
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

fn print_teach_report(input: &str, target: &str, report: &LearnReport) {
    println!("Teaching complete");
    println!();
    println!("Input");
    println!("  mode                  : text");
    println!("  chunks processed      : 1");
    println!("  learned input         : {input}");
    println!("  learned target        : {target}");
    println!();
    println!("Network");
    println!("  neurons grown         : {}", report.neurons_grown);
    println!("  layers grown          : {}", report.layers_grown);
    println!("  neurons promoted      : {}", report.neurons_promoted);
    println!("  neurons frozen        : {}", report.neurons_frozen);
    println!("  total neurons         : {}", report.total_neurons);
    println!();
    println!("Learning");
    println!("  loss before           : {:.4}", report.loss_before);
    println!("  loss after            : {:.4}", report.loss_after);
    println!(
        "  update applied        : {}",
        if report.update_applied { "yes" } else { "no" }
    );
}

fn print_answer(answer: &str, confidence: f32, source: AnswerSource) {
    println!("Answer");
    println!("  {answer}");
    println!();
    println!("Confidence");
    println!("  {:.2}", confidence);
    println!();
    println!("Answered from");
    println!("  {}", answer_source_label(source));
}

fn print_help() {
    println!("Manas");
    println!();
    println!("Usage:");
    println!("  manas teach <text>");
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

fn extract_association(text: &str) -> Result<(String, String), String> {
    let cleaned = trim_sentence(text);
    if cleaned.is_empty() {
        return Err("input text is empty".to_string());
    }

    let lower = cleaned.to_lowercase();
    for marker in [" refers to ", " means ", " were ", " was ", " are ", " is "] {
        if let Some(index) = lower.find(marker) {
            let input = strip_leading_article(&cleaned[..index]);
            let target = strip_leading_article(&cleaned[index + marker.len()..]);
            if !input.is_empty() && !target.is_empty() {
                return Ok((input, target));
            }
        }
    }

    let input = normalized_words(&cleaned)
        .into_iter()
        .find(|word| !is_stopword(word))
        .ok_or_else(|| "could not find a teachable subject".to_string())?;
    Ok((input, cleaned))
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
}
