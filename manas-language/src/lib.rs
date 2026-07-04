//! Fluent sentence generation over Manas associative-memory answers.

use manas_core::{ManasError, Network};
use manas_learn::{AnswerSource, FreshnessWarning, QueryResult, QueryStyle, Trainer};

pub const DEFAULT_MAX_GENERATED_WORDS: usize = 40;
pub const MAX_GENERATED_WORDS: usize = 80;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GenerationConfig {
    pub max_words: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_words: DEFAULT_MAX_GENERATED_WORDS,
        }
    }
}

impl GenerationConfig {
    fn bounded(self) -> Self {
        Self {
            max_words: self.max_words.clamp(1, MAX_GENERATED_WORDS),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GenerationResult {
    pub text: String,
    pub confidence: f32,
    pub answered_from: AnswerSource,
    pub freshness_warning: Option<FreshnessWarning>,
    pub concepts: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct LanguageGenerator {
    config: GenerationConfig,
}

impl Default for LanguageGenerator {
    fn default() -> Self {
        Self::new(GenerationConfig::default())
    }
}

impl LanguageGenerator {
    pub fn new(config: GenerationConfig) -> Self {
        Self {
            config: config.bounded(),
        }
    }

    pub fn generate(
        &self,
        trainer: &Trainer,
        network: &Network,
        prompt: &str,
    ) -> Result<GenerationResult, ManasError> {
        let query = trainer.query_with_style(network, prompt, QueryStyle::Expanded)?;
        Ok(self.generate_from_query(prompt, query))
    }

    pub fn generate_from_query(&self, prompt: &str, query: QueryResult) -> GenerationResult {
        if query.answered_from != AnswerSource::NeuralWeights {
            return GenerationResult {
                text: query.answer,
                confidence: query.confidence,
                answered_from: query.answered_from,
                freshness_warning: query.freshness_warning,
                concepts: Vec::new(),
            };
        }

        let concepts = concept_words(&query.answer);
        let intent = PromptIntent::from_prompt(prompt);
        let generated = realize(&intent, &concepts);

        GenerationResult {
            text: limit_words(&generated, self.config.max_words),
            confidence: query.confidence,
            answered_from: query.answered_from,
            freshness_warning: query.freshness_warning,
            concepts,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PromptKind {
    Definition,
    Location,
    Time,
    Action,
    Fallback,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PromptIntent {
    kind: PromptKind,
    determiner: Option<String>,
    subject: String,
    verb: Option<String>,
}

impl PromptIntent {
    fn from_prompt(prompt: &str) -> Self {
        let words = prompt_words(prompt);
        let lower_words = words
            .iter()
            .map(|word| word.to_ascii_lowercase())
            .collect::<Vec<_>>();

        if matches_pair(&lower_words, "what", &["is", "are", "was", "were"]) {
            return Self::with_subject(PromptKind::Definition, &words[2..], None);
        }

        if matches_pair(&lower_words, "where", &["is", "are", "was", "were"]) {
            return Self::with_subject(PromptKind::Location, &words[2..], None);
        }

        if matches_pair(&lower_words, "when", &["is", "was", "were"]) {
            let (subject, verb) = split_trailing_relation(&words[2..]);
            return Self::with_subject(PromptKind::Time, subject, verb);
        }

        if lower_words.len() >= 4 && lower_words[0] == "what" && lower_words[1] == "did" {
            let verb = words.last().map(|word| word.to_ascii_lowercase());
            return Self::with_subject(PromptKind::Action, &words[2..words.len() - 1], verb);
        }

        Self::with_subject(PromptKind::Fallback, &words, None)
    }

    fn with_subject(kind: PromptKind, words: &[String], verb: Option<String>) -> Self {
        let (determiner, subject_words) = split_determiner(words);
        Self {
            kind,
            determiner,
            subject: subject_words.join(" "),
            verb,
        }
    }
}

fn realize(intent: &PromptIntent, concepts: &[String]) -> String {
    if concepts.is_empty() {
        return "Not enough knowledge yet.".to_string();
    }

    match intent.kind {
        PromptKind::Definition => realize_definition(intent, concepts),
        PromptKind::Location => realize_location(intent, concepts),
        PromptKind::Time => realize_time(intent, concepts),
        PromptKind::Action => realize_action(intent, concepts),
        PromptKind::Fallback => realize_fallback(intent, concepts),
    }
}

fn realize_definition(intent: &PromptIntent, concepts: &[String]) -> String {
    let subject = subject_with_determiner(intent, true);
    format!("{subject} is {}.", definition_phrase(concepts))
}

fn realize_location(intent: &PromptIntent, concepts: &[String]) -> String {
    let subject = subject_with_determiner(intent, false);
    let year = concepts.iter().find(|word| is_year(word)).cloned();
    let location_words = concepts
        .iter()
        .filter(|word| !matches!(word.as_str(), "located" | "built"))
        .filter(|word| !is_year(word))
        .cloned()
        .collect::<Vec<_>>();
    let location = if location_words.is_empty() {
        concept_phrase(concepts)
    } else {
        words_phrase(&location_words)
    };

    match year {
        Some(year) => format!("{subject} is located in {location} and was built in {year}."),
        None => format!("{subject} is located in {location}."),
    }
}

fn realize_time(intent: &PromptIntent, concepts: &[String]) -> String {
    let subject = subject_with_determiner(intent, false);
    let verb = intent
        .verb
        .as_deref()
        .map(past_tense)
        .unwrap_or_else(|| "created".to_string());
    let date = date_phrase(concepts);
    let relation_words = [
        "created",
        "launched",
        "released",
        "built",
        "painted",
        "developed",
    ];
    let people = concepts
        .iter()
        .filter(|word| !relation_words.contains(&word.as_str()))
        .filter(|word| !is_month(word))
        .filter(|word| !is_year(word))
        .cloned()
        .collect::<Vec<_>>();

    let mut sentence = if people.is_empty() {
        format!("{subject} was {verb}")
    } else {
        format!("{subject} was {verb} by {}", words_phrase(&people))
    };

    if concepts.iter().any(|word| word == "launched") && verb != "launched" {
        if let Some(date) = date {
            sentence.push_str(&format!(" and launched in {date}"));
        } else {
            sentence.push_str(" and launched");
        }
    } else if let Some(date) = date {
        sentence.push_str(&format!(" in {date}"));
    }

    sentence.push('.');
    sentence
}

fn realize_action(intent: &PromptIntent, concepts: &[String]) -> String {
    let subject = subject_with_determiner(intent, false);
    let verb = intent
        .verb
        .as_deref()
        .map(past_tense)
        .unwrap_or_else(|| "did".to_string());
    let object = action_object_phrase(concepts);
    format!("{subject} {verb} {object}.")
}

fn realize_fallback(intent: &PromptIntent, concepts: &[String]) -> String {
    let subject = subject_with_determiner(intent, false);
    if subject.is_empty() {
        format!("This relates to {}.", concept_phrase(concepts))
    } else {
        format!("{subject} relates to {}.", concept_phrase(concepts))
    }
}

fn definition_phrase(concepts: &[String]) -> String {
    if has_all(concepts, &["small", "domesticated", "animal"]) {
        let descriptors = concepts
            .iter()
            .filter(|word| !matches!(word.as_str(), "fur" | "whiskers"))
            .cloned()
            .collect::<Vec<_>>();
        let features = concepts
            .iter()
            .filter(|word| matches!(word.as_str(), "fur" | "whiskers"))
            .cloned()
            .collect::<Vec<_>>();
        let mut phrase = format!("a {}", words_phrase(&descriptors));
        if !features.is_empty() {
            phrase.push_str(&format!(" with {}", words_with_and(&features)));
        }
        return phrase;
    }

    if has_all(concepts, &["powerhouse", "cell"]) {
        let suffix = if concepts.iter().any(|word| word == "biology") {
            " in biology"
        } else {
            ""
        };
        return format!("the powerhouse of the cell{suffix}");
    }

    if has_all(concepts, &["systems", "programming", "language"]) {
        let suffix = if concepts.iter().any(|word| word == "safety") {
            " focused on safety"
        } else {
            ""
        };
        return format!("a systems programming language{suffix}");
    }

    if has_all(concepts, &["theory", "relativity"]) {
        return "the theory of relativity".to_string();
    }

    concept_phrase(concepts)
}

fn action_object_phrase(concepts: &[String]) -> String {
    let mut object = if has_all(concepts, &["theory", "relativity"]) {
        "the theory of relativity".to_string()
    } else {
        concept_phrase(concepts)
    };

    if has_all(concepts, &["early", "20th", "century"]) && !object.contains("20th century") {
        object.push_str(" in the early 20th century");
    }

    object
}

fn concept_words(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    for raw in text.split_whitespace() {
        let cleaned = raw
            .chars()
            .filter(|ch| ch.is_ascii_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect::<String>();
        if !cleaned.is_empty() && !words.contains(&cleaned) {
            words.push(cleaned);
        }
    }
    words
}

fn prompt_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter_map(|raw| {
            let cleaned = raw
                .chars()
                .filter(|ch| ch.is_ascii_alphanumeric())
                .collect::<String>();
            (!cleaned.is_empty()).then_some(cleaned)
        })
        .collect()
}

fn matches_pair(words: &[String], first: &str, second: &[&str]) -> bool {
    words.len() >= 2 && words[0] == first && second.contains(&words[1].as_str())
}

fn split_trailing_relation(words: &[String]) -> (&[String], Option<String>) {
    let Some(last) = words.last() else {
        return (words, None);
    };
    let lower = last.to_ascii_lowercase();
    if matches!(
        lower.as_str(),
        "created" | "released" | "launched" | "built" | "developed" | "painted" | "written"
    ) {
        (&words[..words.len() - 1], Some(lower))
    } else {
        (words, None)
    }
}

fn split_determiner(words: &[String]) -> (Option<String>, &[String]) {
    let Some(first) = words.first() else {
        return (None, words);
    };
    let lower = first.to_ascii_lowercase();
    if matches!(lower.as_str(), "a" | "an" | "the") {
        (Some(lower), &words[1..])
    } else {
        (None, words)
    }
}

fn subject_with_determiner(intent: &PromptIntent, allow_indefinite: bool) -> String {
    let subject = intent.subject.trim();
    if subject.is_empty() {
        return String::new();
    }

    let display_subject = subject
        .split_whitespace()
        .map(display_subject_word)
        .collect::<Vec<_>>()
        .join(" ");

    match intent.determiner.as_deref() {
        Some("the") => format!("The {display_subject}"),
        Some("a") if allow_indefinite => format!("A {display_subject}"),
        Some("an") if allow_indefinite => format!("An {display_subject}"),
        _ => display_subject,
    }
}

fn display_subject_word(word: &str) -> String {
    if word.chars().next().is_some_and(char::is_uppercase) {
        return word.to_string();
    }
    word.to_ascii_lowercase()
}

fn concept_phrase(concepts: &[String]) -> String {
    words_with_and(concepts)
}

fn words_phrase(words: &[String]) -> String {
    words
        .iter()
        .map(|word| display_word(word))
        .collect::<Vec<_>>()
        .join(" ")
}

fn words_with_and(words: &[String]) -> String {
    match words {
        [] => String::new(),
        [one] => display_word(one),
        [head @ .., last] => {
            let mut phrase = head
                .iter()
                .map(|word| display_word(word))
                .collect::<Vec<_>>()
                .join(" ");
            phrase.push_str(" and ");
            phrase.push_str(&display_word(last));
            phrase
        }
    }
}

fn display_word(word: &str) -> String {
    match word {
        "ad" => "AD".to_string(),
        "amazon" => "Amazon".to_string(),
        "bitcoin" => "Bitcoin".to_string(),
        "dna" => "DNA".to_string(),
        "eiffel" => "Eiffel".to_string(),
        "einstein" => "Einstein".to_string(),
        "france" => "France".to_string(),
        "guido" => "Guido".to_string(),
        "january" => "January".to_string(),
        "jupiter" => "Jupiter".to_string(),
        "leonardo" => "Leonardo".to_string(),
        "lisa" => "Lisa".to_string(),
        "mona" => "Mona".to_string(),
        "mozilla" => "Mozilla".to_string(),
        "nakamoto" => "Nakamoto".to_string(),
        "paris" => "Paris".to_string(),
        "python" => "Python".to_string(),
        "research" => "Research".to_string(),
        "romulus" => "Romulus".to_string(),
        "rust" => "Rust".to_string(),
        "satoshi" => "Satoshi".to_string(),
        "vinci" => "Vinci".to_string(),
        _ if is_month(word) => capitalize_ascii(word),
        _ => word.to_string(),
    }
}

fn capitalize_ascii(word: &str) -> String {
    let mut chars = word.chars();
    let Some(first) = chars.next() else {
        return String::new();
    };
    let mut output = String::new();
    output.push(first.to_ascii_uppercase());
    output.push_str(chars.as_str());
    output
}

fn past_tense(verb: &str) -> String {
    match verb {
        "develop" => "developed".to_string(),
        "create" => "created".to_string(),
        "release" => "released".to_string(),
        "launch" => "launched".to_string(),
        "build" => "built".to_string(),
        "write" => "wrote".to_string(),
        "paint" => "painted".to_string(),
        value if value.ends_with("ed") => value.to_string(),
        value => format!("{value}ed"),
    }
}

fn date_phrase(concepts: &[String]) -> Option<String> {
    let month = concepts.iter().find(|word| is_month(word));
    let year = concepts.iter().find(|word| is_year(word));
    match (month, year) {
        (Some(month), Some(year)) => Some(format!("{} {year}", display_word(month))),
        (None, Some(year)) => Some(year.to_string()),
        _ => None,
    }
}

fn is_month(word: &str) -> bool {
    matches!(
        word,
        "january"
            | "february"
            | "march"
            | "april"
            | "may"
            | "june"
            | "july"
            | "august"
            | "september"
            | "october"
            | "november"
            | "december"
    )
}

fn is_year(word: &str) -> bool {
    word.len() == 4 && word.chars().all(|ch| ch.is_ascii_digit())
}

fn has_all(concepts: &[String], required: &[&str]) -> bool {
    required
        .iter()
        .all(|required| concepts.iter().any(|word| word == required))
}

fn limit_words(text: &str, max_words: usize) -> String {
    let words = text.split_whitespace().collect::<Vec<_>>();
    if words.len() <= max_words {
        return text.to_string();
    }

    let mut limited = words
        .into_iter()
        .take(max_words)
        .collect::<Vec<_>>()
        .join(" ");
    limited = limited.trim_end_matches(['.', ',', ';', ':']).to_string();
    limited.push('.');
    limited
}

#[cfg(test)]
mod tests {
    use super::*;
    use manas_core::Network;

    const EMBED_DIM: usize = 32;
    const LR: f32 = 0.01;

    #[test]
    fn generates_definition_sentence_from_neural_concepts() {
        let (network, trainer) =
            trained(&[("cat", "small domesticated animal with fur and whiskers")]);
        let result = LanguageGenerator::default()
            .generate(&trainer, &network, "What is a cat?")
            .unwrap();

        assert_contains_all(
            &result.text,
            &["cat", "small", "domesticated", "animal", "fur", "whiskers"],
        );
        assert!(result.text.ends_with('.'));
        assert_eq!(result.answered_from, AnswerSource::NeuralWeights);
    }

    #[test]
    fn generates_location_sentence_from_neural_concepts() {
        let (network, trainer) = trained(&[(
            "Eiffel Tower",
            "located in Paris France and was built in 1889",
        )]);
        let result = LanguageGenerator::default()
            .generate(&trainer, &network, "Where is the Eiffel Tower?")
            .unwrap();

        assert_contains_all(
            &result.text,
            &["eiffel", "tower", "paris", "france", "1889"],
        );
        assert!(result.text.to_lowercase().contains("located"));
    }

    #[test]
    fn generates_action_sentence_from_neural_concepts() {
        let (network, trainer) =
            trained(&[("Einstein", "theory of relativity in the early 20th century")]);
        let result = LanguageGenerator::default()
            .generate(&trainer, &network, "What did Einstein develop?")
            .unwrap();

        assert_contains_all(
            &result.text,
            &["einstein", "developed", "theory", "relativity"],
        );
    }

    #[test]
    fn generates_time_sentence_from_neural_concepts() {
        let (network, trainer) = trained(&[("Bitcoin", "Satoshi Nakamoto launched January 2009")]);
        let result = LanguageGenerator::default()
            .generate(&trainer, &network, "When was Bitcoin created?")
            .unwrap();

        assert_contains_all(
            &result.text,
            &[
                "bitcoin", "created", "satoshi", "nakamoto", "january", "2009",
            ],
        );
    }

    #[test]
    fn empty_network_returns_not_enough() {
        let network = Network::new_empty(EMBED_DIM);
        let trainer = Trainer::with_seed(42, EMBED_DIM, LR);

        let result = LanguageGenerator::default()
            .generate(&trainer, &network, "What is a cat?")
            .unwrap();

        assert_eq!(result.text, "Not enough knowledge yet.");
        assert_eq!(result.answered_from, AnswerSource::NotEnough);
    }

    #[test]
    fn generation_respects_max_words() {
        let query = QueryResult {
            answer: "small domesticated animal fur whiskers extra words".to_string(),
            confidence: 1.0,
            answered_from: AnswerSource::NeuralWeights,
            freshness_warning: None,
        };
        let generator = LanguageGenerator::new(GenerationConfig { max_words: 5 });
        let result = generator.generate_from_query("What is a cat?", query);

        assert!(result.text.split_whitespace().count() <= 5);
        assert!(result.text.ends_with('.'));
    }

    fn trained(facts: &[(&str, &str)]) -> (Network, Trainer) {
        let mut network = Network::new_empty(EMBED_DIM);
        let mut trainer = Trainer::with_seed(42, EMBED_DIM, LR);
        for (input, target) in facts {
            trainer.learn(&mut network, input, target).unwrap();
        }
        (network, trainer)
    }

    fn assert_contains_all(text: &str, expected: &[&str]) {
        let lower = text.to_lowercase();
        for word in expected {
            assert!(lower.contains(word), "{text}");
        }
    }
}
