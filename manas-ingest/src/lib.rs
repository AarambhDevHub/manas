//! Input ingestion for Manas.
//!
//! This crate turns raw text, local files, and folders into deterministic text
//! chunks that can be passed to the learning engine.

use std::fs;
use std::path::{Path, PathBuf};

use manas_core::{ManasError, Source};

pub const CHUNK_SIZE: usize = 512;
pub const CHUNK_OVERLAP: usize = 64;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IngestSource {
    RawText(String),
    File(PathBuf),
    Folder(PathBuf),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextChunk {
    pub text: String,
    pub source: Source,
    pub chunk_id: u64,
}

pub fn ingest(source: IngestSource) -> Result<Vec<TextChunk>, ManasError> {
    let chunks = match source {
        IngestSource::RawText(text) => chunks_for_text(&parse_markdown(&text), Source::RawText, 0),
        IngestSource::File(path) => ingest_file(&path, 0)?,
        IngestSource::Folder(path) => ingest_folder(&path)?,
    };

    if chunks.is_empty() {
        Err(ManasError::EmptyInput)
    } else {
        Ok(chunks)
    }
}

pub fn normalize(text: &str) -> String {
    let mut cleaned = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '\r' => {
                if chars.peek() == Some(&'\n') {
                    continue;
                }
                cleaned.push('\n');
            }
            '\n' => cleaned.push('\n'),
            '\t' => cleaned.push(' '),
            ch if ch.is_control() => {}
            ch => cleaned.push(ch),
        }
    }

    let mut lines = Vec::new();
    for line in cleaned.lines() {
        let collapsed = collapse_spaces(line);
        if collapsed.is_empty() {
            if !lines
                .last()
                .map(|line: &String| line.is_empty())
                .unwrap_or(true)
            {
                lines.push(String::new());
            }
        } else {
            lines.push(collapsed);
        }
    }

    while lines.last().map(|line| line.is_empty()).unwrap_or(false) {
        lines.pop();
    }

    lines.join("\n")
}

pub fn chunk_text(text: &str) -> Vec<String> {
    let normalized = normalize(text);
    if normalized.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut current = String::new();

    for unit in text_units(&normalized) {
        if char_count(&unit) > CHUNK_SIZE {
            push_chunk(&mut chunks, &mut current);
            chunks.extend(split_long_unit(&unit));
            continue;
        }

        let separator = if current.is_empty() { 0 } else { 1 };
        if char_count(&current) + separator + char_count(&unit) <= CHUNK_SIZE {
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(&unit);
        } else {
            push_chunk(&mut chunks, &mut current);
            current = unit;
        }
    }

    push_chunk(&mut chunks, &mut current);
    chunks
}

fn ingest_file(path: &Path, start_id: u64) -> Result<Vec<TextChunk>, ManasError> {
    let extension = supported_extension(path).ok_or_else(|| unsupported_file(path))?;
    let text = fs::read_to_string(path).map_err(|source| ManasError::FileReadError {
        path: path.to_path_buf(),
        source,
    })?;
    let parsed = parse_by_extension(extension, &text)?;
    Ok(chunks_for_text(
        &parsed,
        Source::LocalFile {
            path: path.display().to_string(),
        },
        start_id,
    ))
}

fn ingest_folder(path: &Path) -> Result<Vec<TextChunk>, ManasError> {
    if !path.is_dir() {
        return Err(ManasError::EncodingError(format!(
            "folder input is not a directory: {}",
            path.display()
        )));
    }

    let mut files = Vec::new();
    collect_supported_files(path, &mut files)?;
    files.sort_by(|left, right| left.to_string_lossy().cmp(&right.to_string_lossy()));

    let mut chunks = Vec::new();
    for file in files {
        let next_id = chunks.len() as u64;
        chunks.extend(ingest_file(&file, next_id)?);
    }

    Ok(chunks)
}

fn collect_supported_files(path: &Path, files: &mut Vec<PathBuf>) -> Result<(), ManasError> {
    let entries = fs::read_dir(path).map_err(|source| ManasError::FileReadError {
        path: path.to_path_buf(),
        source,
    })?;

    let mut paths = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source| ManasError::FileReadError {
            path: path.to_path_buf(),
            source,
        })?;
        paths.push(entry.path());
    }
    paths.sort_by(|left, right| left.to_string_lossy().cmp(&right.to_string_lossy()));

    for path in paths {
        if path.is_dir() {
            collect_supported_files(&path, files)?;
        } else if supported_extension(&path).is_some() {
            files.push(path);
        }
    }

    Ok(())
}

fn chunks_for_text(text: &str, source: Source, start_id: u64) -> Vec<TextChunk> {
    chunk_text(text)
        .into_iter()
        .enumerate()
        .map(|(offset, text)| TextChunk {
            text,
            source: source.clone(),
            chunk_id: start_id + offset as u64,
        })
        .collect()
}

fn supported_extension(path: &Path) -> Option<&str> {
    let extension = path.extension()?.to_str()?.to_ascii_lowercase();
    match extension.as_str() {
        "txt" | "md" | "rs" | "toml" | "json" | "csv" => Some(match extension.as_str() {
            "txt" => "txt",
            "md" => "md",
            "rs" => "rs",
            "toml" => "toml",
            "json" => "json",
            "csv" => "csv",
            _ => unreachable!(),
        }),
        _ => None,
    }
}

fn unsupported_file(path: &Path) -> ManasError {
    ManasError::EncodingError(format!("unsupported file format: {}", path.display()))
}

fn parse_by_extension(extension: &str, text: &str) -> Result<String, ManasError> {
    match extension {
        "txt" => Ok(text.to_string()),
        "md" => Ok(parse_markdown(text)),
        "rs" => Ok(parse_rust_source(text)),
        "toml" => Ok(parse_toml(text)),
        "json" => parse_json(text),
        "csv" => Ok(parse_csv(text)),
        _ => Err(ManasError::EncodingError(format!(
            "unsupported file extension: {extension}"
        ))),
    }
}

fn parse_markdown(text: &str) -> String {
    let mut output = Vec::new();
    let mut in_code_fence = false;

    for raw_line in text.lines() {
        let trimmed = raw_line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_code_fence = !in_code_fence;
            continue;
        }
        if in_code_fence {
            continue;
        }

        let without_prefix = strip_markdown_prefix(trimmed);
        let stripped = strip_markdown_inline(&without_prefix);
        if !stripped.trim().is_empty() {
            output.push(stripped);
        } else {
            output.push(String::new());
        }
    }

    output.join("\n")
}

fn strip_markdown_prefix(line: &str) -> String {
    let mut text = line.trim_start();

    while let Some(stripped) = text.strip_prefix('>') {
        text = stripped.trim_start();
    }

    let heading_trimmed = text.trim_start_matches('#').trim_start();
    if heading_trimmed.len() != text.len() {
        text = heading_trimmed;
    }

    for marker in ["- ", "* ", "+ "] {
        if let Some(stripped) = text.strip_prefix(marker) {
            return stripped.trim_start().to_string();
        }
    }

    let mut digit_end = 0;
    for (index, ch) in text.char_indices() {
        if ch.is_ascii_digit() {
            digit_end = index + ch.len_utf8();
            continue;
        }
        if ch == '.' && digit_end > 0 {
            let rest = &text[index + ch.len_utf8()..];
            if rest.starts_with(' ') {
                return rest.trim_start().to_string();
            }
        }
        break;
    }

    text.to_string()
}

fn strip_markdown_inline(line: &str) -> String {
    let chars = line.chars().collect::<Vec<_>>();
    let mut output = String::with_capacity(line.len());
    let mut index = 0;

    while index < chars.len() {
        match chars[index] {
            '!' if chars.get(index + 1) == Some(&'[') => {
                index += 2;
                let mut alt = String::new();
                while index < chars.len() && chars[index] != ']' {
                    alt.push(chars[index]);
                    index += 1;
                }
                if !alt.is_empty() {
                    output.push_str(&alt);
                }
                if chars.get(index) == Some(&']') && chars.get(index + 1) == Some(&'(') {
                    index += 2;
                    while index < chars.len() && chars[index] != ')' {
                        index += 1;
                    }
                }
            }
            '[' => {
                index += 1;
                let mut label = String::new();
                while index < chars.len() && chars[index] != ']' {
                    label.push(chars[index]);
                    index += 1;
                }
                output.push_str(&label);
                if chars.get(index) == Some(&']') && chars.get(index + 1) == Some(&'(') {
                    index += 2;
                    while index < chars.len() && chars[index] != ')' {
                        index += 1;
                    }
                }
            }
            '`' | '*' | '_' | '~' => {}
            ch => output.push(ch),
        }
        index += 1;
    }

    output
}

fn parse_rust_source(text: &str) -> String {
    let mut output = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim_start();
        if let Some(doc) = trimmed.strip_prefix("///") {
            output.push(doc.trim().to_string());
        } else if let Some(doc) = trimmed.strip_prefix("//!") {
            output.push(doc.trim().to_string());
        } else if let Some(signature) = rust_signature(trimmed) {
            output.push(signature);
        }
    }

    output.join("\n")
}

fn rust_signature(line: &str) -> Option<String> {
    let starts_like_item = line.starts_with("fn ")
        || line.starts_with("pub fn ")
        || line.starts_with("pub(crate) fn ")
        || line.starts_with("async fn ")
        || line.starts_with("pub async fn ")
        || line.starts_with("struct ")
        || line.starts_with("pub struct ")
        || line.starts_with("enum ")
        || line.starts_with("pub enum ");

    if !starts_like_item {
        return None;
    }

    let end = line
        .find('{')
        .or_else(|| line.find(';'))
        .unwrap_or(line.len());
    let signature = line[..end].trim();
    if signature.is_empty() {
        None
    } else {
        Some(signature.to_string())
    }
}

fn parse_toml(text: &str) -> String {
    let mut section = String::new();
    let mut output = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            section = readable_key(trimmed.trim_matches(['[', ']']));
            continue;
        }

        let Some(index) = trimmed.find('=') else {
            continue;
        };
        let key = readable_key(&trimmed[..index]);
        let value = strip_data_value(&trimmed[index + 1..]);
        if key.is_empty() || value.is_empty() {
            continue;
        }

        if section.is_empty() {
            output.push(format!("{key} is {value}."));
        } else {
            output.push(format!("{section} {key} is {value}."));
        }
    }

    output.join("\n")
}

fn parse_json(text: &str) -> Result<String, ManasError> {
    let mut parser = JsonParser::new(text);
    let value = parser.parse()?;
    let mut output = Vec::new();
    flatten_json(&value, &mut Vec::new(), &mut output);
    Ok(output.join("\n"))
}

fn parse_csv(text: &str) -> String {
    let records = parse_csv_records(text);
    if records.is_empty() {
        return String::new();
    }

    let headers = records[0]
        .iter()
        .map(|header| readable_key(header))
        .collect::<Vec<_>>();
    let mut output = Vec::new();

    for row in records.iter().skip(1) {
        let mut parts = Vec::new();
        for (index, value) in row.iter().enumerate() {
            let value = strip_data_value(value);
            if value.is_empty() {
                continue;
            }
            let header = headers
                .get(index)
                .filter(|header| !header.is_empty())
                .cloned()
                .unwrap_or_else(|| format!("column {}", index + 1));
            parts.push(format!("{header} is {value}"));
        }
        if !parts.is_empty() {
            output.push(format!("{}.", parts.join(", ")));
        }
    }

    if output.is_empty() {
        records
            .into_iter()
            .map(|row| row.join(" "))
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        output.join("\n")
    }
}

fn parse_csv_records(text: &str) -> Vec<Vec<String>> {
    let mut records = Vec::new();
    let mut record = Vec::new();
    let mut field = String::new();
    let mut chars = text.chars().peekable();
    let mut in_quotes = false;

    while let Some(ch) = chars.next() {
        match ch {
            '"' if in_quotes && chars.peek() == Some(&'"') => {
                field.push('"');
                chars.next();
            }
            '"' => in_quotes = !in_quotes,
            ',' if !in_quotes => {
                record.push(field.trim().to_string());
                field.clear();
            }
            '\n' if !in_quotes => {
                record.push(field.trim().to_string());
                field.clear();
                if record.iter().any(|value| !value.is_empty()) {
                    records.push(record);
                }
                record = Vec::new();
            }
            '\r' => {}
            ch => field.push(ch),
        }
    }

    if !field.is_empty() || !record.is_empty() {
        record.push(field.trim().to_string());
        if record.iter().any(|value| !value.is_empty()) {
            records.push(record);
        }
    }

    records
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum JsonValue {
    Object(Vec<(String, JsonValue)>),
    Array(Vec<JsonValue>),
    String(String),
    Number(String),
    Bool(bool),
    Null,
}

struct JsonParser {
    chars: Vec<char>,
    index: usize,
}

impl JsonParser {
    fn new(text: &str) -> Self {
        Self {
            chars: text.chars().collect(),
            index: 0,
        }
    }

    fn parse(&mut self) -> Result<JsonValue, ManasError> {
        let value = self.parse_value()?;
        self.skip_ws();
        if self.index == self.chars.len() {
            Ok(value)
        } else {
            Err(self.json_error("trailing data"))
        }
    }

    fn parse_value(&mut self) -> Result<JsonValue, ManasError> {
        self.skip_ws();
        let Some(ch) = self.peek() else {
            return Err(self.json_error("unexpected end of input"));
        };

        match ch {
            '{' => self.parse_object(),
            '[' => self.parse_array(),
            '"' => self.parse_string().map(JsonValue::String),
            '-' | '0'..='9' => self.parse_number().map(JsonValue::Number),
            't' => {
                self.expect_literal("true")?;
                Ok(JsonValue::Bool(true))
            }
            'f' => {
                self.expect_literal("false")?;
                Ok(JsonValue::Bool(false))
            }
            'n' => {
                self.expect_literal("null")?;
                Ok(JsonValue::Null)
            }
            _ => Err(self.json_error("unexpected value")),
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue, ManasError> {
        self.expect('{')?;
        self.skip_ws();
        let mut pairs = Vec::new();

        if self.consume('}') {
            return Ok(JsonValue::Object(pairs));
        }

        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            self.expect(':')?;
            let value = self.parse_value()?;
            pairs.push((key, value));
            self.skip_ws();

            if self.consume('}') {
                break;
            }
            self.expect(',')?;
        }

        Ok(JsonValue::Object(pairs))
    }

    fn parse_array(&mut self) -> Result<JsonValue, ManasError> {
        self.expect('[')?;
        self.skip_ws();
        let mut values = Vec::new();

        if self.consume(']') {
            return Ok(JsonValue::Array(values));
        }

        loop {
            values.push(self.parse_value()?);
            self.skip_ws();

            if self.consume(']') {
                break;
            }
            self.expect(',')?;
        }

        Ok(JsonValue::Array(values))
    }

    fn parse_string(&mut self) -> Result<String, ManasError> {
        self.expect('"')?;
        let mut output = String::new();

        while let Some(ch) = self.next() {
            match ch {
                '"' => return Ok(output),
                '\\' => {
                    let Some(escaped) = self.next() else {
                        return Err(self.json_error("unterminated escape"));
                    };
                    match escaped {
                        '"' | '\\' | '/' => output.push(escaped),
                        'b' => output.push('\u{0008}'),
                        'f' => output.push('\u{000c}'),
                        'n' => output.push('\n'),
                        'r' => output.push('\r'),
                        't' => output.push('\t'),
                        'u' => output.push(self.parse_unicode_escape()?),
                        _ => return Err(self.json_error("invalid escape")),
                    }
                }
                ch => output.push(ch),
            }
        }

        Err(self.json_error("unterminated string"))
    }

    fn parse_unicode_escape(&mut self) -> Result<char, ManasError> {
        let mut value = 0u32;
        for _ in 0..4 {
            let Some(ch) = self.next() else {
                return Err(self.json_error("unterminated unicode escape"));
            };
            value = value
                .checked_mul(16)
                .and_then(|value| ch.to_digit(16).map(|digit| value + digit))
                .ok_or_else(|| self.json_error("invalid unicode escape"))?;
        }

        char::from_u32(value).ok_or_else(|| self.json_error("invalid unicode scalar"))
    }

    fn parse_number(&mut self) -> Result<String, ManasError> {
        let start = self.index;

        self.consume('-');
        while self.peek().map(|ch| ch.is_ascii_digit()).unwrap_or(false) {
            self.index += 1;
        }
        if self.consume('.') {
            while self.peek().map(|ch| ch.is_ascii_digit()).unwrap_or(false) {
                self.index += 1;
            }
        }
        if matches!(self.peek(), Some('e' | 'E')) {
            self.index += 1;
            if matches!(self.peek(), Some('+' | '-')) {
                self.index += 1;
            }
            while self.peek().map(|ch| ch.is_ascii_digit()).unwrap_or(false) {
                self.index += 1;
            }
        }

        if self.index == start {
            return Err(self.json_error("invalid number"));
        }

        Ok(self.chars[start..self.index].iter().collect())
    }

    fn expect_literal(&mut self, literal: &str) -> Result<(), ManasError> {
        for expected in literal.chars() {
            self.expect(expected)?;
        }
        Ok(())
    }

    fn expect(&mut self, expected: char) -> Result<(), ManasError> {
        match self.next() {
            Some(ch) if ch == expected => Ok(()),
            _ => Err(self.json_error(&format!("expected '{expected}'"))),
        }
    }

    fn consume(&mut self, expected: char) -> bool {
        if self.peek() == Some(expected) {
            self.index += 1;
            true
        } else {
            false
        }
    }

    fn skip_ws(&mut self) {
        while self.peek().map(|ch| ch.is_whitespace()).unwrap_or(false) {
            self.index += 1;
        }
    }

    fn next(&mut self) -> Option<char> {
        let ch = self.peek()?;
        self.index += 1;
        Some(ch)
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.index).copied()
    }

    fn json_error(&self, reason: &str) -> ManasError {
        ManasError::EncodingError(format!(
            "invalid json at character {}: {reason}",
            self.index
        ))
    }
}

fn flatten_json(value: &JsonValue, path: &mut Vec<String>, output: &mut Vec<String>) {
    match value {
        JsonValue::Object(pairs) => {
            for (key, value) in pairs {
                path.push(readable_key(key));
                flatten_json(value, path, output);
                path.pop();
            }
        }
        JsonValue::Array(values) => {
            for (index, value) in values.iter().enumerate() {
                path.push((index + 1).to_string());
                flatten_json(value, path, output);
                path.pop();
            }
        }
        JsonValue::String(value) if !value.trim().is_empty() => {
            push_json_sentence(path, &strip_data_value(value), output);
        }
        JsonValue::Number(value) => push_json_sentence(path, value, output),
        JsonValue::Bool(value) => {
            push_json_sentence(path, if *value { "true" } else { "false" }, output)
        }
        JsonValue::Null => {}
        JsonValue::String(_) => {}
    }
}

fn push_json_sentence(path: &[String], value: &str, output: &mut Vec<String>) {
    let key = collapse_spaces(&path.join(" "));
    let value = strip_data_value(value);
    if !key.is_empty() && !value.is_empty() {
        output.push(format!("{key} is {value}."));
    }
}

fn text_units(text: &str) -> Vec<String> {
    let mut units = Vec::new();

    for paragraph in text.split("\n\n") {
        let mut sentence = String::new();
        for ch in paragraph.chars() {
            sentence.push(ch);
            if matches!(ch, '.' | '!' | '?') {
                let unit = collapse_spaces(&sentence);
                if !unit.is_empty() {
                    units.push(unit);
                }
                sentence.clear();
            }
        }

        let unit = collapse_spaces(&sentence);
        if !unit.is_empty() {
            units.push(unit);
        }
    }

    if units.is_empty() {
        let unit = collapse_spaces(text);
        if !unit.is_empty() {
            units.push(unit);
        }
    }

    units
}

fn split_long_unit(unit: &str) -> Vec<String> {
    let chars = unit.chars().collect::<Vec<_>>();
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < chars.len() {
        let end = (start + CHUNK_SIZE).min(chars.len());
        let chunk = chars[start..end].iter().collect::<String>();
        let chunk = chunk.trim();
        if !chunk.is_empty() {
            chunks.push(chunk.to_string());
        }
        if end == chars.len() {
            break;
        }
        let next_start = end.saturating_sub(CHUNK_OVERLAP);
        start = if next_start > start { next_start } else { end };
    }

    chunks
}

fn push_chunk(chunks: &mut Vec<String>, current: &mut String) {
    let chunk = current.trim();
    if !chunk.is_empty() {
        chunks.push(chunk.to_string());
    }
    current.clear();
}

fn char_count(text: &str) -> usize {
    text.chars().count()
}

fn readable_key(text: &str) -> String {
    let replaced = text
        .trim()
        .chars()
        .map(|ch| {
            if matches!(ch, '_' | '-' | '.' | '/' | '\\' | '$') {
                ' '
            } else {
                ch
            }
        })
        .collect::<String>();
    collapse_spaces(&replaced)
}

fn strip_data_value(text: &str) -> String {
    let trimmed = text.trim().trim_end_matches(',').trim();
    let without_comment = trimmed
        .split_once(" #")
        .map(|(value, _)| value)
        .unwrap_or(trimmed)
        .trim();
    let stripped = without_comment
        .trim_matches('"')
        .trim_matches('\'')
        .trim_matches('[')
        .trim_matches(']')
        .trim_matches('{')
        .trim_matches('}')
        .trim();
    collapse_spaces(stripped)
}

fn collapse_spaces(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn txt_file_ingests_correctly() {
        let dir = temp_dir("txt");
        let path = dir.join("test.txt");
        write_file(
            &path,
            "The cat sat on the mat.\nRust is a programming language.",
        );

        let chunks = ingest(IngestSource::File(path)).unwrap();

        assert!(!chunks.is_empty());
        assert!(chunks[0].text.contains("cat") || chunks[0].text.contains("Rust"));
        assert!(matches!(chunks[0].source, Source::LocalFile { .. }));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn folder_walk_finds_all_supported_files() {
        let dir = temp_dir("folder");
        let nested = dir.join("nested");
        fs::create_dir_all(&nested).unwrap();
        write_file(&dir.join("a.txt"), "fact a is true");
        write_file(&dir.join("b.md"), "# fact b\n\nfact b is markdown");
        write_file(
            &nested.join("c.rs"),
            "/// fact c is documented\nfn main() {}",
        );
        write_file(&dir.join("skip.exe"), "ignored");

        let chunks = ingest(IngestSource::Folder(dir.clone())).unwrap();
        let sources = chunks
            .iter()
            .map(|chunk| chunk.source.to_string())
            .collect::<Vec<_>>();

        assert!(sources.iter().any(|source| source.contains("a.txt")));
        assert!(sources.iter().any(|source| source.contains("b.md")));
        assert!(sources.iter().any(|source| source.contains("c.rs")));
        assert!(!sources.iter().any(|source| source.contains("skip.exe")));
        assert_eq!(
            chunks
                .iter()
                .map(|chunk| chunk.chunk_id)
                .collect::<Vec<_>>(),
            (0..chunks.len() as u64).collect::<Vec<_>>()
        );

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn markdown_strips_syntax() {
        let md = "# Title\n\n**bold** text and `code` here.";
        let chunks = ingest(IngestSource::RawText(md.to_string())).unwrap();
        let text = &chunks[0].text;

        assert!(!text.contains('#'), "Markdown headers should be stripped");
        assert!(!text.contains("**"), "Markdown bold should be stripped");
        assert!(!text.contains('`'), "Markdown code should be stripped");
        assert!(text.contains("Title"));
        assert!(text.contains("bold text"));
    }

    #[test]
    fn rust_source_extracts_docs_and_signatures() {
        let parsed = parse_rust_source(
            "use std::fmt;\n/// Adds two values.\npub fn add(left: u32, right: u32) -> u32 {\n    left + right\n}\n",
        );

        assert!(parsed.contains("Adds two values"));
        assert!(parsed.contains("pub fn add(left: u32, right: u32) -> u32"));
        assert!(!parsed.contains("use std::fmt"));
    }

    #[test]
    fn toml_json_csv_parse_to_sentences() {
        let toml = parse_toml("[package]\nname = \"manas\"\nversion = \"0.1.0\"\n");
        assert!(toml.contains("package name is manas"));
        assert!(toml.contains("package version is 0.1.0"));

        let json = parse_json(r#"{"animal":{"name":"cat","small":true},"age":3}"#).unwrap();
        assert!(json.contains("animal name is cat"));
        assert!(json.contains("animal small is true"));
        assert!(json.contains("age is 3"));

        let csv = parse_csv("name,kind\ncat,animal\nparis,city\n");
        assert!(csv.contains("name is cat"));
        assert!(csv.contains("kind is animal"));
        assert!(csv.contains("name is paris"));
    }

    #[test]
    fn unsupported_explicit_file_errors() {
        let dir = temp_dir("unsupported");
        let path = dir.join("skip.exe");
        write_file(&path, "ignored");

        let error = ingest(IngestSource::File(path)).unwrap_err().to_string();

        assert!(error.contains("unsupported file format"));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn long_text_chunks_on_character_boundaries() {
        let text = "ज्ञान ".repeat(300);
        let chunks = chunk_text(&text);

        assert!(chunks.len() > 1);
        assert!(
            chunks
                .iter()
                .all(|chunk| chunk.chars().count() <= CHUNK_SIZE)
        );
    }

    fn write_file(path: &Path, content: &str) {
        fs::write(path, content).unwrap();
    }

    fn temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "manas-ingest-{name}-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }
}
