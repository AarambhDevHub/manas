#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$ROOT_DIR/target/release/manas"
MAX_BRAIN_BYTES=$((500 * 1024))

if [[ -n "${MANAS_DEMO_DIR:-}" ]]; then
  DEMO_DIR="$MANAS_DEMO_DIR"
  mkdir -p "$DEMO_DIR"
else
  DEMO_DIR="$(mktemp -d /tmp/manas-stage13-demo-XXXXXX)"
fi

facts=(
  "A cat is a small domesticated animal with fur and whiskers."
  "The Eiffel Tower is located in Paris France and was built in 1889."
  "The Amazon River is the largest river by discharge in the world."
  "Photosynthesis is the process by which plants convert sunlight into energy."
  "Hydrogen is the lightest and most abundant element in the universe."
  "The human brain contains approximately 86 billion neurons."
  "Mount Everest is the highest mountain on Earth at 8849 meters."
  "Shakespeare wrote 37 plays and 154 sonnets during his lifetime."
  "The speed of light in vacuum is approximately 299792458 meters per second."
  "DNA is a double helix structure that carries genetic information."
  "The Roman Empire fell in 476 AD when Romulus Augustulus was deposed."
  "Water boils at 100 degrees Celsius at standard atmospheric pressure."
  "The Python programming language was created by Guido van Rossum in 1991."
  "Jupiter is the largest planet in our solar system with 95 known moons."
  "The Mona Lisa was painted by Leonardo da Vinci in the early 16th century."
  "Rust programming language was first released by Mozilla Research in 2010."
  "The mitochondria is the powerhouse of the cell in biology."
  "Albert Einstein developed the theory of relativity in the early 20th century."
  "The Pacific Ocean is the largest and deepest ocean on Earth."
  "Bitcoin was created by Satoshi Nakamoto and launched in January 2009."
  "The nitrogen cycle describes how nitrogen moves through ecosystems."
  "Gravity pulls objects toward each other with a force proportional to mass."
)

sidecars=(
  "brain.manas.sources"
  "brain.manas.sourceindex"
  "brain.manas.seq"
  "brain.manas.transformer"
  "brain.manas.langmeta"
)

echo "=== Building release binary ==="
cargo build --workspace --release

cd "$DEMO_DIR"
echo "=== Demo directory ==="
echo "$DEMO_DIR"

echo ""
echo "=== Starting fresh ==="
rm -f brain.manas "${sidecars[@]}"
"$BIN" reset

echo ""
echo "=== Teaching 22 facts ==="
for fact in "${facts[@]}"; do
  "$BIN" teach "$fact" >/dev/null
done

echo ""
echo "=== Deleting all sidecars: neural weights only ==="
rm -f "${sidecars[@]}"
for sidecar in "${sidecars[@]}"; do
  if [[ -e "$sidecar" ]]; then
    echo "sidecar still exists: $sidecar" >&2
    exit 1
  fi
done

require_neural_answer() {
  local output="$1"
  if ! grep -q $'Answered from\n  neural weights' <<<"$output"; then
    echo "answer did not come from neural weights:" >&2
    echo "$output" >&2
    exit 1
  fi
  if grep -q "Not enough knowledge yet." <<<"$output"; then
    echo "answer reported not enough knowledge:" >&2
    echo "$output" >&2
    exit 1
  fi
}

require_all_words() {
  local output
  output="$(tr '[:upper:]' '[:lower:]' <<<"$1")"
  shift
  for word in "$@"; do
    if ! grep -q "$word" <<<"$output"; then
      echo "answer missed required word '$word':" >&2
      echo "$output" >&2
      exit 1
    fi
  done
}

require_two_words() {
  local output
  output="$(tr '[:upper:]' '[:lower:]' <<<"$1")"
  shift
  local count=0
  for word in "$@"; do
    if grep -q "$word" <<<"$output"; then
      count=$((count + 1))
    fi
  done
  if (( count < 2 )); then
    echo "answer matched only $count keywords from: $*" >&2
    echo "$output" >&2
    exit 1
  fi
}

ask_and_print() {
  local question="$1"
  echo ""
  echo "QUESTION: $question"
  "$BIN" ask "$question"
}

echo ""
echo "=== Asking: must answer from neural weights ==="
cat_answer="$(ask_and_print "What is a cat?")"
echo "$cat_answer"
require_neural_answer "$cat_answer"
require_two_words "$cat_answer" small domesticated animal fur whiskers

eiffel_answer="$(ask_and_print "Where is the Eiffel Tower?")"
echo "$eiffel_answer"
require_neural_answer "$eiffel_answer"
require_two_words "$eiffel_answer" paris france 1889

einstein_answer="$(ask_and_print "What did Einstein develop?")"
echo "$einstein_answer"
require_neural_answer "$einstein_answer"
require_all_words "$einstein_answer" theory relativity

mitochondria_answer="$(ask_and_print "What is the mitochondria?")"
echo "$mitochondria_answer"
require_neural_answer "$mitochondria_answer"
require_all_words "$mitochondria_answer" powerhouse cell

bitcoin_answer="$(ask_and_print "When was Bitcoin created?")"
echo "$bitcoin_answer"
require_neural_answer "$bitcoin_answer"
require_two_words "$bitcoin_answer" satoshi nakamoto 2009

brain_size="$(wc -c < brain.manas)"
if (( brain_size >= MAX_BRAIN_BYTES )); then
  echo "brain.manas is too large: $brain_size bytes" >&2
  exit 1
fi

echo ""
echo "=== Brain state ==="
"$BIN" inspect
echo ""
echo "Stage 13 demo passed: brain.manas is $brain_size bytes."
