use std::collections::HashSet;
use std::fs;
use std::path::Path;

use manas_core::{ManasError, Network, Source};
use manas_learn::{
    FreshnessCategory, LearnReport, Trainer, detect_freshness, freshness_age_days, is_stale,
};
use serde_json::Value;

const DUCKDUCKGO_ENDPOINT: &str = "https://api.duckduckgo.com/";
const DEFAULT_LIMIT: usize = 25;
const MAX_LIMIT: usize = 100;
const MAX_REFRESH_WORDS: usize = 48;

#[derive(Debug)]
pub enum AgentError {
    Search(String),
    Io(String),
    Json(String),
    Learn(ManasError),
}

impl std::fmt::Display for AgentError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Search(reason) => write!(formatter, "search failed: {reason}"),
            Self::Io(reason) => write!(formatter, "I/O failed: {reason}"),
            Self::Json(reason) => write!(formatter, "JSON parse failed: {reason}"),
            Self::Learn(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for AgentError {}

impl From<ManasError> for AgentError {
    fn from(error: ManasError) -> Self {
        Self::Learn(error)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchResponse {
    pub title: String,
    pub url: String,
    pub text: String,
}

pub trait SearchClient {
    fn search(&self, query: &str) -> Result<SearchResponse, AgentError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DuckDuckGoClient;

impl SearchClient for DuckDuckGoClient {
    fn search(&self, query: &str) -> Result<SearchResponse, AgentError> {
        let url = duckduckgo_url(query);
        let mut response = ureq::get(&url)
            .header("User-Agent", "manas/0.1")
            .call()
            .map_err(|error| AgentError::Search(error.to_string()))?;
        let body = response
            .body_mut()
            .read_to_string()
            .map_err(|error| AgentError::Io(error.to_string()))?;
        parse_duckduckgo_response(&body)
    }
}

#[derive(Clone, Debug, Default)]
pub struct FixtureSearchClient {
    responses: Vec<(String, SearchResponse)>,
}

impl FixtureSearchClient {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, AgentError> {
        let text = fs::read_to_string(path.as_ref()).map_err(|error| {
            AgentError::Io(format!(
                "failed to read {}: {error}",
                path.as_ref().display()
            ))
        })?;
        let mut client = Self::default();
        for (line_index, line) in text.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let parts = trimmed.split('\t').collect::<Vec<_>>();
            if !(2..=3).contains(&parts.len()) {
                return Err(AgentError::Io(format!(
                    "fixture line {} must be query<TAB>text[<TAB>url]",
                    line_index + 1
                )));
            }
            client.responses.push((
                normalize_query_key(parts[0]),
                SearchResponse {
                    title: "fixture".to_string(),
                    text: parts[1].to_string(),
                    url: parts
                        .get(2)
                        .copied()
                        .unwrap_or("https://duckduckgo.com/")
                        .to_string(),
                },
            ));
        }
        Ok(client)
    }

    pub fn with_response(query: &str, text: &str, url: &str) -> Self {
        Self {
            responses: vec![(
                normalize_query_key(query),
                SearchResponse {
                    title: "fixture".to_string(),
                    url: url.to_string(),
                    text: text.to_string(),
                },
            )],
        }
    }
}

impl SearchClient for FixtureSearchClient {
    fn search(&self, query: &str) -> Result<SearchResponse, AgentError> {
        let key = normalize_query_key(query);
        self.responses
            .iter()
            .find(|(candidate, _)| candidate == &key)
            .map(|(_, response)| response.clone())
            .ok_or_else(|| AgentError::Search(format!("no fixture response for '{query}'")))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RefreshConfig {
    pub include_fast: bool,
    pub include_realtime: bool,
    pub dry_run: bool,
    pub limit: usize,
}

impl Default for RefreshConfig {
    fn default() -> Self {
        Self {
            include_fast: false,
            include_realtime: true,
            dry_run: false,
            limit: DEFAULT_LIMIT,
        }
    }
}

impl RefreshConfig {
    pub fn normalized(mut self) -> Self {
        self.limit = self.limit.clamp(1, MAX_LIMIT);
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RefreshCandidate {
    pub neuron_id: u64,
    pub memory_input: String,
    pub memory_target: String,
    pub freshness: FreshnessCategory,
    pub age_days: u64,
    pub query: String,
}

#[derive(Clone, Debug, Default)]
pub struct RefreshReport {
    pub candidates: usize,
    pub fetched: usize,
    pub refreshed: usize,
    pub skipped: usize,
    pub failed: usize,
    pub neurons_grown: u32,
    pub layers_grown: u32,
    pub neurons_promoted: u32,
    pub neurons_frozen: u32,
    pub candidates_list: Vec<RefreshCandidate>,
}

pub fn plan_refresh(
    network: &Network,
    config: &RefreshConfig,
    now_secs: u64,
) -> Vec<RefreshCandidate> {
    let config = config.clone().normalized();
    let mut seen_queries = HashSet::new();
    let mut candidates = Vec::new();

    for layer in network
        .layers
        .iter()
        .take(network.layer_count().saturating_sub(1))
    {
        for neuron in &layer.neurons {
            let freshness = FreshnessCategory::from(neuron.freshness_category);
            if !category_included(freshness, &config) || !is_stale(neuron, now_secs) {
                continue;
            }

            let input = collapse_whitespace(&neuron.memory_input);
            let target = collapse_whitespace(&neuron.memory_target);
            if input.is_empty() || target.is_empty() {
                continue;
            }

            let query = collapse_whitespace(&format!("{input} {target}"));
            if !seen_queries.insert(normalize_query_key(&query)) {
                continue;
            }

            candidates.push(RefreshCandidate {
                neuron_id: neuron.id,
                memory_input: input,
                memory_target: target,
                freshness,
                age_days: freshness_age_days(neuron, now_secs),
                query,
            });
            if candidates.len() >= config.limit {
                return candidates;
            }
        }
    }

    candidates
}

pub fn refresh_network<C: SearchClient>(
    network: &mut Network,
    trainer: &mut Trainer,
    client: &C,
    config: &RefreshConfig,
    now_secs: u64,
) -> Result<RefreshReport, AgentError> {
    let candidates = plan_refresh(network, config, now_secs);
    let mut report = RefreshReport {
        candidates: candidates.len(),
        candidates_list: candidates.clone(),
        ..RefreshReport::default()
    };

    if config.dry_run {
        return Ok(report);
    }

    for candidate in candidates {
        let response = match client.search(&candidate.query) {
            Ok(response) => response,
            Err(_) => {
                report.failed += 1;
                continue;
            }
        };
        report.fetched += 1;

        let Some(target) = normalize_refresh_text(&response.text) else {
            report.skipped += 1;
            continue;
        };
        let freshness = detect_freshness(&format!("{} {}", candidate.memory_input, target));
        let source = Source::Internet {
            url: if response.url.trim().is_empty() {
                "https://duckduckgo.com/".to_string()
            } else {
                response.url
            },
        };
        let learn = trainer.refresh_memory_at(
            network,
            &candidate.memory_input,
            &target,
            source,
            freshness,
            now_secs,
        )?;
        report.record_learn(&learn);
        report.refreshed += 1;
    }

    Ok(report)
}

impl RefreshReport {
    fn record_learn(&mut self, report: &LearnReport) {
        self.neurons_grown = self.neurons_grown.saturating_add(report.neurons_grown);
        self.layers_grown = self.layers_grown.saturating_add(report.layers_grown);
        self.neurons_promoted = self
            .neurons_promoted
            .saturating_add(report.neurons_promoted);
        self.neurons_frozen = self.neurons_frozen.saturating_add(report.neurons_frozen);
    }
}

pub fn duckduckgo_url(query: &str) -> String {
    format!(
        "{DUCKDUCKGO_ENDPOINT}?q={}&format=json&no_redirect=1&no_html=1&skip_disambig=1",
        percent_encode_query(query)
    )
}

pub fn parse_duckduckgo_response(body: &str) -> Result<SearchResponse, AgentError> {
    let value = serde_json::from_str::<Value>(body).map_err(|error| {
        AgentError::Json(format!("DuckDuckGo response was not valid JSON: {error}"))
    })?;

    let title = string_field(&value, "Heading")
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| "DuckDuckGo".to_string());
    let url = string_field(&value, "AbstractURL")
        .or_else(|| first_related_field(&value, "FirstURL"))
        .unwrap_or_else(|| "https://duckduckgo.com/".to_string());
    let text = string_field(&value, "AbstractText")
        .or_else(|| string_field(&value, "Answer"))
        .or_else(|| first_related_field(&value, "Text"))
        .unwrap_or_default();

    Ok(SearchResponse { title, url, text })
}

fn category_included(category: FreshnessCategory, config: &RefreshConfig) -> bool {
    match category {
        FreshnessCategory::Realtime => config.include_realtime,
        FreshnessCategory::Fast => config.include_fast,
        FreshnessCategory::Timeless | FreshnessCategory::Slow => false,
    }
}

fn string_field(value: &Value, field: &str) -> Option<String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .map(collapse_whitespace)
        .filter(|text| !text.is_empty())
}

fn first_related_field(value: &Value, field: &str) -> Option<String> {
    let topics = value.get("RelatedTopics")?.as_array()?;
    first_related_field_in_array(topics, field)
}

fn first_related_field_in_array(topics: &[Value], field: &str) -> Option<String> {
    for topic in topics {
        if let Some(text) = string_field(topic, field) {
            return Some(text);
        }
        if let Some(nested) = topic.get("Topics").and_then(Value::as_array)
            && let Some(text) = first_related_field_in_array(nested, field)
        {
            return Some(text);
        }
    }
    None
}

fn normalize_refresh_text(text: &str) -> Option<String> {
    let collapsed = collapse_whitespace(text);
    if collapsed.is_empty() {
        return None;
    }
    let words = collapsed.split_whitespace().collect::<Vec<_>>();
    if words.len() <= MAX_REFRESH_WORDS {
        return Some(collapsed);
    }
    Some(words[..MAX_REFRESH_WORDS].join(" "))
}

fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn normalize_query_key(query: &str) -> String {
    collapse_whitespace(query).to_ascii_lowercase()
}

fn percent_encode_query(query: &str) -> String {
    let mut encoded = String::new();
    for byte in query.as_bytes() {
        match *byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(char::from(*byte));
            }
            b' ' => encoded.push('+'),
            other => encoded.push_str(&format!("%{other:02X}")),
        }
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;
    use manas_learn::FreshnessCategory;

    const NOW: u64 = 1_800_000_000;
    const DAY: u64 = 86_400;

    #[test]
    fn refresh_plan_selects_stale_realtime_memories() {
        let (network, _) = stale_network(FreshnessCategory::Realtime, 2);
        let candidates = plan_refresh(&network, &RefreshConfig::default(), NOW);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].memory_input, "stock price");
        assert!(candidates[0].query.contains("stock price"));
    }

    #[test]
    fn refresh_plan_needs_fast_flag_for_fast_memories() {
        let (network, _) = stale_network(FreshnessCategory::Fast, 31);
        assert!(plan_refresh(&network, &RefreshConfig::default(), NOW).is_empty());

        let config = RefreshConfig {
            include_fast: true,
            ..RefreshConfig::default()
        };
        assert_eq!(plan_refresh(&network, &config, NOW).len(), 1);
    }

    #[test]
    fn dry_run_does_not_call_client_or_mutate_network() {
        let (mut network, mut trainer) = stale_network(FreshnessCategory::Realtime, 2);
        let before = network.neuron_count();
        let client = FixtureSearchClient::default();
        let config = RefreshConfig {
            dry_run: true,
            ..RefreshConfig::default()
        };

        let report = refresh_network(&mut network, &mut trainer, &client, &config, NOW).unwrap();

        assert_eq!(report.candidates, 1);
        assert_eq!(report.fetched, 0);
        assert_eq!(network.neuron_count(), before);
    }

    #[test]
    fn refresh_adds_fresh_neural_memory_for_protected_stale_fact() {
        let (mut network, mut trainer) = stale_network(FreshnessCategory::Realtime, 2);
        for layer in network.layers.iter_mut().take(1) {
            for neuron in &mut layer.neurons {
                neuron.freeze_all();
            }
        }
        let client = FixtureSearchClient::with_response(
            "stock price 10 today",
            "stock price is 20 today",
            "https://example.test/stock",
        );

        let report = refresh_network(
            &mut network,
            &mut trainer,
            &client,
            &RefreshConfig::default(),
            NOW,
        )
        .unwrap();

        assert_eq!(report.refreshed, 1);
        assert_eq!(report.neurons_grown, 1);
        let result = trainer.query(&network, "What is stock price?").unwrap();
        assert!(result.answer.contains("20"), "{}", result.answer);
        assert_eq!(result.freshness_warning, None);
    }

    #[test]
    fn duckduckgo_url_percent_encodes_query() {
        let url = duckduckgo_url("rust 1.90?");

        assert!(url.contains("q=rust+1.90%3F"), "{url}");
        assert!(url.contains("format=json"), "{url}");
    }

    #[test]
    fn parse_duckduckgo_prefers_abstract_text() {
        let response = parse_duckduckgo_response(
            r#"{
                "Heading": "Rust",
                "AbstractURL": "https://example.test/rust",
                "AbstractText": "Rust is a programming language.",
                "Answer": "fallback"
            }"#,
        )
        .unwrap();

        assert_eq!(response.title, "Rust");
        assert_eq!(response.url, "https://example.test/rust");
        assert_eq!(response.text, "Rust is a programming language.");
    }

    #[test]
    fn parse_duckduckgo_uses_related_topics_when_needed() {
        let response = parse_duckduckgo_response(
            r#"{
                "RelatedTopics": [
                    {"Topics": [
                        {"Text": "Nested answer text.", "FirstURL": "https://example.test/nested"}
                    ]}
                ]
            }"#,
        )
        .unwrap();

        assert_eq!(response.text, "Nested answer text.");
        assert_eq!(response.url, "https://example.test/nested");
    }

    fn stale_network(category: FreshnessCategory, age_days: u64) -> (Network, Trainer) {
        let mut network = Network::new_empty(32);
        let mut trainer = Trainer::new(0.01);
        trainer
            .learn_with_source_and_freshness(
                &mut network,
                "stock price",
                "10 today",
                Source::RawText,
                category,
            )
            .unwrap();
        let hidden_layers = network.layer_count().saturating_sub(1);
        for layer in network.layers.iter_mut().take(hidden_layers) {
            for neuron in &mut layer.neurons {
                if !neuron.memory_input.is_empty() {
                    neuron.born_at = NOW - age_days * DAY;
                    neuron.last_activated = neuron.born_at;
                    neuron.freshness_category = category as u8;
                }
            }
        }
        (network, trainer)
    }
}
