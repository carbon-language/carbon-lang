// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include <algorithm>
#include <utility>

#include "absl/random/random.h"
#include "common/check.h"
#include "common/raw_string_ostream.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
#include "testing/base/source_gen.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/diagnostics/null_diagnostics.h"
#include "toolchain/lex/lex.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/lex/tokenized_buffer.h"

namespace Carbon::Lex {
namespace {

// A large value for measurement stability without making benchmarking too slow.
// Needs to be a multiple of 100 so we can easily divide it up into percentages,
// and 1% itself needs to not be too tiny. This makes 100,000 a great balance.
constexpr int NumTokens = 100'000;

// Compute a random sequence of just identifiers.
static auto RandomIdentifierSeq(int min_length, int max_length, bool uniform,
                                llvm::StringRef separator = " ")
    -> std::string {
  auto& gen = Testing::SourceGen::Global();
  llvm::SmallVector<llvm::StringRef> ids =
      gen.GetShuffledIdentifiers(NumTokens, min_length, max_length, uniform);
  return llvm::join(ids, separator);
}

auto GetSymbolTokenTable() -> llvm::ArrayRef<TokenKind> {
  // Build our own table of symbols so we can use repetitions to skew the
  // distribution.
  static auto symbol_token_table_storage = [] {
    llvm::SmallVector<TokenKind> table;
#define CARBON_SYMBOL_TOKEN(TokenName, Spelling) \
  table.push_back(TokenKind::TokenName);
#define CARBON_OPENING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, ClosingName)
#define CARBON_CLOSING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, OpeningName)
#include "toolchain/lex/token_kind.def"
    table.insert(table.end(), 32, TokenKind::Semi);
    table.insert(table.end(), 16, TokenKind::Comma);
    table.insert(table.end(), 12, TokenKind::Period);
    table.insert(table.end(), 8, TokenKind::Colon);
    table.insert(table.end(), 8, TokenKind::Equal);
    table.insert(table.end(), 4, TokenKind::Amp);
    table.insert(table.end(), 4, TokenKind::ColonExclaim);
    table.insert(table.end(), 4, TokenKind::EqualEqual);
    table.insert(table.end(), 4, TokenKind::ExclaimEqual);
    table.insert(table.end(), 4, TokenKind::MinusGreater);
    table.insert(table.end(), 4, TokenKind::Star);
    return table;
  }();
  return symbol_token_table_storage;
}

struct RandomSourceOptions {
  int symbol_percent = 0;
  int keyword_percent = 0;
  int numeric_literal_percent = 0;
  int string_literal_percent = 0;

  int tokens_per_line = NumTokens;

  int comment_line_percent = 0;
  int blank_line_percent = 0;

  auto Validate() -> void {
    auto is_percentage = [](int n) { return 0 <= n && n <= 100; };
    CARBON_CHECK(is_percentage(symbol_percent));
    CARBON_CHECK(is_percentage(keyword_percent));
    CARBON_CHECK(is_percentage(numeric_literal_percent));
    CARBON_CHECK(is_percentage(string_literal_percent));
    CARBON_CHECK(is_percentage(symbol_percent + keyword_percent +
                               numeric_literal_percent +
                               string_literal_percent));

    CARBON_CHECK(tokens_per_line <= NumTokens);
    CARBON_CHECK(
        NumTokens % tokens_per_line == 0,
        "Tokens per line of {0} does not divide the number of tokens {1}",
        tokens_per_line, NumTokens);

    CARBON_CHECK(is_percentage(comment_line_percent));
    CARBON_CHECK(is_percentage(blank_line_percent));

    // Ensure that comment and blank lines are less than 100% so we eventually
    // produce a token line.
    CARBON_CHECK(comment_line_percent + blank_line_percent < 100);
  }
};

// Based on measurements of LLVM's source code, a rough approximation of the
// distribution of these kinds of tokens.
constexpr RandomSourceOptions DefaultSourceDist = {
    .symbol_percent = 50,
    .keyword_percent = 7,
    .numeric_literal_percent = 17,
    .string_literal_percent = 1,

    // The median for LLVM is roughly 5.
    .tokens_per_line = 5,

    // Observed percentage of lines in LLVM.
    .comment_line_percent = 22,
    .blank_line_percent = 15,
};

// Compute random source code with a mixture of tokens and whitespace according
// to the options. The source isn't designed to be valid, or directly
// representative of real-world Carbon code. However, it tries to provide
// reasonable coverage of the different aspects of Carbon's lexer, such that for
// real world source code with distributions similar to the options provided the
// lexer performance will be roughly representative.
//
// TODO: Does not yet support generating numeric or string literals.
//
// TODO: The shape of lines is handled very arbitrarily and should vary more to
// avoid over-fitting to a specific shape (number of tokens, length of comment).
auto RandomSource(RandomSourceOptions options) -> std::string {
  options.Validate();
  static_assert((NumTokens % 100) == 0,
                "The number of tokens must be divisible by 100 so that we can "
                "easily scale integer percentages up to it.");

  // Get static pools of symbols, keywords, and identifiers.
  llvm::ArrayRef<TokenKind> symbols = GetSymbolTokenTable();
  llvm::ArrayRef<TokenKind> keywords = TokenKind::KeywordTokens;

  // Build a list of StringRefs from the different types with the desired
  // distribution, then shuffle that list.
  llvm::OwningArrayRef<llvm::StringRef> tokens(NumTokens);

  int num_symbols = (NumTokens / 100) * options.symbol_percent;
  int num_keywords = (NumTokens / 100) * options.keyword_percent;
  int num_identifiers = NumTokens - num_symbols - num_keywords;
  CARBON_CHECK(
      num_identifiers == 0 || num_identifiers > 500,
      "We require at least 500 identifiers as we need to collect a reasonable "
      "number of samples to end up with a reasonable distribution of lengths.");
  llvm::SmallVector<llvm::StringRef> ids =
      Testing::SourceGen::Global().GetIdentifiers(num_identifiers);

  for (int i : llvm::seq(num_symbols)) {
    tokens[i] = symbols[i % symbols.size()].fixed_spelling();
  }
  for (int i : llvm::seq(num_keywords)) {
    tokens[num_symbols + i] = keywords[i % keywords.size()].fixed_spelling();
  }
  for (int i : llvm::seq(num_identifiers)) {
    // We always have enough identifiers, so no need to mod here.
    tokens[num_symbols + num_keywords + i] = ids[i];
  }
  std::shuffle(tokens.begin(), tokens.end(), absl::BitGen());

  // Distribute the tokens across lines as well as horizontal whitespace. The
  // goal isn't to make any one line representative of anything, but to make the
  // rough density of different kinds of whitespace roughly representative.
  //
  // TODO: This is a really coarse approach that just picks a fixed number of
  // tokens per line rather than using some distribution with this as the median
  // or mean.
  llvm::SmallVector<std::string> lines;
  // First place tokens onto each line.
  for (auto i : llvm::seq(NumTokens / options.tokens_per_line)) {
    lines.push_back("");
    RawStringOstream os;
    // Arbitrarily indent each line by two spaces.
    os << "  ";
    llvm::ListSeparator sep(" ");
    for (int j : llvm::seq(options.tokens_per_line)) {
      os << sep << tokens[i * options.tokens_per_line + j];
    }
    lines.push_back(os.TakeStr());
  }

  // Next, synthesize blank and comment lines with the correct distribution.
  int token_line_percent =
      100 - options.blank_line_percent - options.comment_line_percent;
  CARBON_CHECK(token_line_percent > 0);
  int num_token_lines = lines.size();
  int num_lines = num_token_lines * 100 / token_line_percent;
  int num_blank_lines = num_lines * options.blank_line_percent / 100;
  int num_comment_lines = num_lines - num_blank_lines - num_token_lines;
  CARBON_CHECK(num_comment_lines >= 0);
  lines.resize(num_lines);
  for (auto& line :
       llvm::MutableArrayRef(lines).slice(num_lines - num_comment_lines)) {
    // TODO: We should vary the content and length, especially as the
    // distribution is weirdly shaped with just over half the comment lines
    // being blank and the median length of non-black comment lines being 64!
    // This is a *very* coarse approximation of the mean at 30 characters long.
    line = "  // abcdefghijklmnopqrstuvwxyz";
  }
  // Now shuffle the lines.
  std::shuffle(lines.begin(), lines.end(), absl::BitGen());
  // And join them into the source string.
  return llvm::join(lines, "\n");
}

class LexerBenchHelper {
 public:
  explicit LexerBenchHelper(llvm::StringRef text)
      : source_(MakeSourceBuffer(text)) {}

  auto Lex() -> TokenizedBuffer {
    Diagnostics::Consumer& consumer = Diagnostics::NullConsumer();
    return Lex::Lex(value_stores_, source_, consumer);
  }

  auto DiagnoseErrors() -> std::string {
    RawStringOstream result;
    Diagnostics::StreamConsumer consumer(&result);
    auto buffer = Lex::Lex(value_stores_, source_, consumer);
    consumer.Flush();
    CARBON_CHECK(buffer.has_errors(),
                 "Asked to diagnose errors but none found!");
    return result.TakeStr();
  }

  auto source_text() -> llvm::StringRef { return source_.text(); }

 private:
  auto MakeSourceBuffer(llvm::StringRef text) -> SourceBuffer {
    CARBON_CHECK(fs_.addFile(filename_, /*ModificationTime=*/0,
                             llvm::MemoryBuffer::getMemBuffer(text)));
    return std::move(*SourceBuffer::MakeFromFile(
        fs_, filename_, Diagnostics::ConsoleConsumer()));
  }

  SharedValueStores value_stores_;
  llvm::vfs::InMemoryFileSystem fs_;
  std::string filename_ = "test.carbon";
  SourceBuffer source_;
};

auto BM_ValidKeywords(benchmark::State& state) -> void {
  absl::BitGen gen;
  std::array<llvm::StringRef, NumTokens> tokens;
  for (int i : llvm::seq(NumTokens)) {
    tokens[i] = TokenKind::KeywordTokens[i % TokenKind::KeywordTokens.size()]
                    .fixed_spelling();
  }
  std::shuffle(tokens.begin(), tokens.end(), gen);
  std::string source = llvm::join(tokens, " ");

  LexerBenchHelper helper(source);
  for (auto _ : state) {
    TokenizedBuffer buffer = helper.Lex();
    CARBON_CHECK(!buffer.has_errors());
  }

  state.SetBytesProcessed(state.iterations() * source.size());
  state.counters["tokens_per_second"] = benchmark::Counter(
      NumTokens, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_ValidKeywords);

auto BM_ValidKeywordsAsRawIdentifiers(benchmark::State& state) -> void {
  absl::BitGen gen;
  std::array<llvm::StringRef, NumTokens> tokens;
  for (int i : llvm::seq(NumTokens)) {
    tokens[i] = TokenKind::KeywordTokens[i % TokenKind::KeywordTokens.size()]
                    .fixed_spelling();
  }
  std::shuffle(tokens.begin(), tokens.end(), gen);
  std::string source("r#");
  source.append(llvm::join(tokens, " r#"));

  LexerBenchHelper helper(source);
  for (auto _ : state) {
    TokenizedBuffer buffer = helper.Lex();
    CARBON_CHECK(!buffer.has_errors());
  }

  state.SetBytesProcessed(state.iterations() * source.size());
  state.counters["tokens_per_second"] = benchmark::Counter(
      NumTokens, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_ValidKeywordsAsRawIdentifiers);

// This benchmark does a 50-50 split of r-prefixed and r#-prefixed identifiers
// to directly compare raw and non-raw performance.
auto BM_RawIdentifierFocus(benchmark::State& state) -> void {
  llvm::SmallVector<llvm::StringRef> ids =
      Testing::SourceGen::Global().GetIdentifiers(NumTokens / 2);

  llvm::SmallVector<std::string> modified_ids;
  // As we resize, start with the in-use prefix. Note that `r#` uses the first
  // character of the original identifier.
  modified_ids.resize(NumTokens / 2, "r#");
  modified_ids.resize(NumTokens, "r");
  for (int i : llvm::seq(NumTokens / 2)) {
    // Use the same identifier both ways.
    modified_ids[i].append(ids[i]);
    modified_ids[i + NumTokens / 2].append(
        llvm::StringRef(ids[i]).drop_front());
  }

  absl::BitGen gen;
  std::array<llvm::StringRef, NumTokens> tokens;
  for (int i : llvm::seq(NumTokens)) {
    tokens[i] = modified_ids[i];
  }
  std::shuffle(tokens.begin(), tokens.end(), gen);
  std::string source = llvm::join(tokens, " ");

  LexerBenchHelper helper(source);
  for (auto _ : state) {
    TokenizedBuffer buffer = helper.Lex();
    CARBON_CHECK(!buffer.has_errors());
  }

  state.SetBytesProcessed(state.iterations() * source.size());
  state.counters["tokens_per_second"] = benchmark::Counter(
      NumTokens, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_RawIdentifierFocus);

template <int MinLength, int MaxLength, bool Uniform>
auto BM_ValidIdentifiers(benchmark::State& state) -> void {
  std::string source = RandomIdentifierSeq(MinLength, MaxLength, Uniform);

  LexerBenchHelper helper(source);
  for (auto _ : state) {
    TokenizedBuffer buffer = helper.Lex();
    CARBON_CHECK(!buffer.has_errors(), "{0}", helper.DiagnoseErrors());
  }

  state.SetBytesProcessed(state.iterations() * source.size());
  state.counters["tokens_per_second"] = benchmark::Counter(
      NumTokens, benchmark::Counter::kIsIterationInvariantRate);
}
// Benchmark the non-uniform distribution we observe in C++ code.
BENCHMARK(BM_ValidIdentifiers<1, 64, /*Uniform=*/false>);

// Also benchmark a few uniform distribution ranges of identifier widths to
// cover different patterns that emerge with small, medium, and longer
// identifiers.
BENCHMARK(BM_ValidIdentifiers<1, 1, /*Uniform=*/true>);
BENCHMARK(BM_ValidIdentifiers<3, 5, /*Uniform=*/true>);
BENCHMARK(BM_ValidIdentifiers<3, 16, /*Uniform=*/true>);
BENCHMARK(BM_ValidIdentifiers<12, 64, /*Uniform=*/true>);
BENCHMARK(BM_ValidIdentifiers<16, 16, /*Uniform=*/true>);
BENCHMARK(BM_ValidIdentifiers<24, 24, /*Uniform=*/true>);
BENCHMARK(BM_ValidIdentifiers<32, 32, /*Uniform=*/true>);
BENCHMARK(BM_ValidIdentifiers<48, 48, /*Uniform=*/true>);
BENCHMARK(BM_ValidIdentifiers<64, 64, /*Uniform=*/true>);
BENCHMARK(BM_ValidIdentifiers<80, 80, /*Uniform=*/true>);

// Benchmark to stress the lexing of horizontal whitespace. This sets up what is
// nearly a worst-case scenario of short-but-expensive-to-lex tokens with runs
// of horizontal whitespace between them.
auto BM_HorizontalWhitespace(benchmark::State& state) -> void {
  int num_spaces = state.range(0);
  std::string separator(num_spaces, ' ');
  std::string source = RandomIdentifierSeq(3, 5, /*uniform=*/true, separator);

  LexerBenchHelper helper(source);
  for (auto _ : state) {
    TokenizedBuffer buffer = helper.Lex();

    // Ensure that lexing actually occurs for benchmarking and that it doesn't
    // hit errors that would skew the benchmark results.
    CARBON_CHECK(!buffer.has_errors(), "{0}", helper.DiagnoseErrors());
  }

  state.SetBytesProcessed(state.iterations() * source.size());
  state.counters["tokens_per_second"] = benchmark::Counter(
      NumTokens, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_HorizontalWhitespace)->RangeMultiplier(4)->Range(1, 128);

auto BM_RandomSource(benchmark::State& state) -> void {
  std::string source = RandomSource(DefaultSourceDist);

  LexerBenchHelper helper(source);
  for (auto _ : state) {
    TokenizedBuffer buffer = helper.Lex();

    // Ensure that lexing actually occurs for benchmarking and that it doesn't
    // hit errors that would skew the benchmark results.
    CARBON_CHECK(!buffer.has_errors(), "{0}", helper.DiagnoseErrors());
  }

  state.SetBytesProcessed(state.iterations() * source.size());
  state.counters["tokens_per_second"] = benchmark::Counter(
      NumTokens, benchmark::Counter::kIsIterationInvariantRate);
  state.counters["lines_per_second"] =
      benchmark::Counter(llvm::StringRef(source).count('\n'),
                         benchmark::Counter::kIsIterationInvariantRate);
}
// The distributions between symbols, keywords, and identifiers here are
// guesses. Eventually, we should collect more data to help tune these, but
// hopefully the performance isn't too sensitive and we can just cover a wide
// range here.
BENCHMARK(BM_RandomSource);

// Benchmark to stress opening and closing grouped symbols.
auto BM_GroupingSymbols(benchmark::State& state) -> void {
  int curly_brace_depth = state.range(0);
  int paren_depth = state.range(1);
  int square_bracket_depth = state.range(2);

  // TODO: It might be interesting to have some random pattern of nesting, but
  // the obvious ways to do that result it really unstable total size of input
  // or unbalanced groups. For now, just use a simple strict nesting approach.
  // It should still let us look for specific pain points. We do include some
  // whitespace and keywords to make sure *some* other parts of the benchmark
  // are also active and have some reasonable icache pressure.
  llvm::SmallVector<llvm::StringRef> ids =
      Testing::SourceGen::Global().GetShuffledIdentifiers(NumTokens);
  RawStringOstream os;
  int num_tokens_per_nest =
      curly_brace_depth * 2 + paren_depth * 2 + square_bracket_depth * 2 + 2;
  int num_nests = NumTokens / num_tokens_per_nest;
  for (int i : llvm::seq(num_nests)) {
    for (int j : llvm::seq(curly_brace_depth)) {
      os.indent(j * 2) << "{\n";
    }
    os.indent(curly_brace_depth * 2);
    for ([[maybe_unused]] int j : llvm::seq(paren_depth)) {
      os << "(";
    }
    for ([[maybe_unused]] int j : llvm::seq(square_bracket_depth)) {
      os << "[";
    }
    os << ids[(i * 2) % NumTokens];
    for ([[maybe_unused]] int j : llvm::seq(square_bracket_depth)) {
      os << "]";
    }
    for ([[maybe_unused]] int j : llvm::seq(paren_depth)) {
      os << ")";
    }
    for (int j : llvm::reverse(llvm::seq(curly_brace_depth))) {
      os << "\n";
      os.indent(j * 2) << "}";
    }
    os << ids[(i * 2 + 1) % NumTokens] << "\n";
  }

  std::string source = os.TakeStr();
  LexerBenchHelper helper(source);
  for (auto _ : state) {
    TokenizedBuffer buffer = helper.Lex();

    // Ensure that lexing actually occurs for benchmarking and that it doesn't
    // hit errors that would skew the benchmark results.
    CARBON_CHECK(!buffer.has_errors(), "{0}", helper.DiagnoseErrors());
  }

  state.SetBytesProcessed(state.iterations() * source.size());
  state.counters["tokens_per_second"] = benchmark::Counter(
      NumTokens, benchmark::Counter::kIsIterationInvariantRate);
  state.counters["lines_per_second"] =
      benchmark::Counter(llvm::StringRef(source).count('\n'),
                         benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_GroupingSymbols)
    ->ArgsProduct({
        {1, 2, 3, 4, 8, 16, 32},
        {0},
        {0},
    })
    ->ArgsProduct({
        {0},
        {1, 2, 3, 4, 8, 16, 32},
        {0},
    })
    ->ArgsProduct({
        {0},
        {0},
        {1, 2, 3, 4, 8, 16, 32},
    })
    ->ArgsProduct({
        {32},
        {1, 2, 3, 4, 8, 16, 32},
        {0},
    })
    ->ArgsProduct({
        {32},
        {32},
        {1, 2, 3, 4, 8, 16, 32},
    });

// Benchmark to stress the lexing of blank lines. This uses a simple, easy to
// lex token, but separates each one by varying numbers of blank lines.
auto BM_BlankLines(benchmark::State& state) -> void {
  int num_blank_lines = state.range(0);
  std::string separator(num_blank_lines, '\n');
  std::string source = RandomIdentifierSeq(3, 5, /*uniform=*/true, separator);

  LexerBenchHelper helper(source);
  for (auto _ : state) {
    TokenizedBuffer buffer = helper.Lex();

    // Ensure that lexing actually occurs for benchmarking and that it doesn't
    // hit errors that would skew the benchmark results.
    CARBON_CHECK(!buffer.has_errors(), "{0}", helper.DiagnoseErrors());
  }

  state.SetBytesProcessed(state.iterations() * source.size());
  state.counters["tokens_per_second"] = benchmark::Counter(
      NumTokens, benchmark::Counter::kIsIterationInvariantRate);
  state.counters["lines_per_second"] =
      benchmark::Counter(llvm::StringRef(source).count('\n'),
                         benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_BlankLines)->RangeMultiplier(4)->Range(1, 128);

// Benchmark to stress the lexing of comment lines. This uses a simple, easy to
// lex token, but separates each one by varying numbers of comment lines, with
// varying comment line length and indentation.
auto BM_CommentLines(benchmark::State& state) -> void {
  int num_comment_lines = state.range(0);
  int comment_length = state.range(1);
  int comment_indent = state.range(2);
  RawStringOstream os;
  os << "\n";
  for (int i : llvm::seq(num_comment_lines)) {
    static_cast<void>(i);
    os << std::string(comment_indent, ' ') << "//"
       << std::string(comment_length, ' ') << "\n";
  }
  std::string source =
      RandomIdentifierSeq(3, 5, /*uniform=*/true, os.TakeStr());

  LexerBenchHelper helper(source);
  for (auto _ : state) {
    TokenizedBuffer buffer = helper.Lex();

    // Ensure that lexing actually occurs for benchmarking and that it doesn't
    // hit errors that would skew the benchmark results.
    CARBON_CHECK(!buffer.has_errors(), "{0}", helper.DiagnoseErrors());
  }

  state.SetBytesProcessed(state.iterations() * source.size());
  state.counters["tokens_per_second"] = benchmark::Counter(
      NumTokens, benchmark::Counter::kIsIterationInvariantRate);
  state.counters["lines_per_second"] =
      benchmark::Counter(llvm::StringRef(source).count('\n'),
                         benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_CommentLines)
    ->ArgsProduct({
        // How many lines of comment. Focused on a couple of small and checking
        // how it scales up to large blocks.
        {1, 4, 128},
        // Comment lengths: the two extremes and a middling length.
        {0, 30, 70},
        // Comment indentations.
        {0, 2, 8},
    });

// This is a speed-of-light benchmark that should reflect memory bandwidth
// (ideally) of simply reading all the source code. For speed-of-light we use
// `strcpy` -- this both examines ever byte of the input looking for a null to
// end the copy, and also writes to a data structure of roughly the same size as
// the input. This routine is one we expect to be *very* well optimized and give
// a good approximation of the fastest possible lexer given the physical
// constraints of the machine. Note that which particular source we use as input
// here isn't especially interesting, so we just pick one and should update it
// to reflect whatever distribution is most realistic long-term. The
// bytes/second throughput is the important output of this routine.
auto BM_SpeedOfLightStrCpy(benchmark::State& state) -> void {
  std::string source = RandomSource(DefaultSourceDist);

  // A buffer to write the null-terminated contents of `source` into.
  llvm::OwningArrayRef<char> buffer(source.size() + 1);

  for (auto _ : state) {
    const char* text = source.data();
    benchmark::DoNotOptimize(text);
    strcpy(buffer.data(), text);
    benchmark::DoNotOptimize(buffer.data());
  }

  state.SetBytesProcessed(state.iterations() * source.size());
  state.counters["tokens_per_second"] = benchmark::Counter(
      NumTokens, benchmark::Counter::kIsIterationInvariantRate);
  state.counters["lines_per_second"] =
      benchmark::Counter(llvm::StringRef(source).count('\n'),
                         benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SpeedOfLightStrCpy);

// This is a speed-of-light benchmark that builds up a best-case byte-wise table
// dispatch using guaranteed tail recursion. The goal is both to ensure the
// general technique can reasonably hit the level of performance we need and to
// establish how far from this speed of light the actual lexer currently sits.
//
// A major impact on the observed performance of this technique is how many
// different functions are reached in this dispatch loop. This benchmark
// infrastructure tries to bracket the range of performance this technique
// affords with different numbers of dispatch target functions.
using DispatchPtrT = auto (*)(ssize_t& index, const char* text, char* buffer)
    -> void;
using DispatchTableT = std::array<DispatchPtrT, 256>;

template <const DispatchTableT& Table>
auto BasicDispatch(ssize_t& index, const char* text, char* buffer) -> void {
  *buffer = text[index];
  ++index;
  // NOLINTNEXTLINE(readability-avoid-return-with-void-value): For musttail.
  [[clang::musttail]] return Table[static_cast<unsigned char>(text[index])](
      index, text, buffer);
}

template <const DispatchTableT& Table, char C>
auto SpecializedDispatch(ssize_t& index, const char* text, char* buffer)
    -> void {
  CARBON_CHECK(C == text[index]);
  *buffer = C;
  ++index;
  // NOLINTNEXTLINE(readability-avoid-return-with-void-value): For musttail.
  [[clang::musttail]] return Table[static_cast<unsigned char>(text[index])](
      index, text, buffer);
}

// A sample of the symbol characters used in Carbon code. Doesn't need to be
// perfect, as we just need to have a reasonably large # of distinct dispatch
// functions.
constexpr char DispatchSpecializableSymbols[] = {
    '!', '%', '(', ')', '*', '+', ',', '-', '.', ':',
    ';', '<', '=', '>', '?', '[', ']', '{', '}', '~',
};

// Create an array of all the characters we can specialize dispatch over --
// [0-9A-Za-z] and the symbols above. Similar to the above symbols, doesn't need
// to be exhaustive.
constexpr std::array<char, 26 * 2 + 10 + sizeof(DispatchSpecializableSymbols)>
    DispatchSpecializableChars = []() {
      constexpr int Size = sizeof(DispatchSpecializableChars);
      std::array<char, Size> chars = {};
      int i = 0;
      for (char c = '0'; c <= '9'; ++c) {
        chars[i] = c;
        ++i;
      }
      for (char c = 'A'; c <= 'Z'; ++c) {
        chars[i] = c;
        ++i;
      }
      for (char c = 'a'; c <= 'z'; ++c) {
        chars[i] = c;
        ++i;
      }
      for (char c : DispatchSpecializableSymbols) {
        chars[i] = c;
        ++i;
      }
      CARBON_CHECK(i == Size);
      return chars;
    }();

// Instantiate a number of specialized dispatch functions for characters in the
// array above, and assign those function addresses to the character's entry in
// the provided table. The provided `tmp_table` is a temporary that will
// eventually initialize the provided `Table` constant, so the constant is what
// we propagate to the instantiated function and the temporary is the one we
// initialize.
template <const DispatchTableT& Table, size_t... Indices>
constexpr auto SpecializeDispatchTable(
    DispatchTableT& tmp_table, std::index_sequence<Indices...> /*indices*/)
    -> void {
  static_assert(sizeof...(Indices) <= sizeof(DispatchSpecializableChars));
  ((tmp_table[static_cast<unsigned char>(DispatchSpecializableChars[Indices])] =
        &SpecializedDispatch<Table, DispatchSpecializableChars[Indices]>),
   ...);
}

// The maximum number of dispatch targets is the size of the array + 1 (for the
// base case target).
constexpr int MaxDispatchTargets = sizeof(DispatchSpecializableChars) + 1;

// Dispatch tables with a provided number of distinct dispatch targets. There
// will always be one additional target for the null byte to end the loop.
template <int NumDispatchTargets>
constexpr DispatchTableT DispatchTable = []() {
  static_assert(NumDispatchTargets > 0, "Need at least one dispatch target.");
  static_assert(NumDispatchTargets <= MaxDispatchTargets,
                "Limited number of dispatch targets available.");

  DispatchTableT tmp_table = {};
  // Start with the basic dispatch target.
  for (int i = 0; i < 256; ++i) {
    tmp_table[i] = &BasicDispatch<DispatchTable<NumDispatchTargets>>;
  }
  // NOLINTNEXTLINE(readability-braces-around-statements): False positive.
  if constexpr (NumDispatchTargets > 1) {
    // Add additional dispatch targets from our specializable array.
    SpecializeDispatchTable<DispatchTable<NumDispatchTargets>>(
        tmp_table, std::make_index_sequence<NumDispatchTargets - 1>());
  }
  // Special case the null byte index to end the tail-dispatch.
  tmp_table[0] =
      +[](ssize_t& index, const char* text, char* /*buffer*/) -> void {
    CARBON_CHECK(text[index] == '\0');
    return;
  };
  return tmp_table;
}();

template <int NumDispatchTargets>
auto BM_SpeedOfLightDispatch(benchmark::State& state) -> void {
  std::string source = RandomSource(DefaultSourceDist);

  // A buffer to write to, simulating some minimal write traffic.
  llvm::OwningArrayRef<char> buffer(source.size());

  for (auto _ : state) {
    const char* text = source.data();
    benchmark::DoNotOptimize(text);

    // Use `ssize_t` to minimize indexing overhead.
    ssize_t i = 0;
    // The dispatch table tail-recurses through the entire string.
    DispatchTable<NumDispatchTargets>[static_cast<unsigned char>(text[i])](
        i, text, buffer.data());
    CARBON_CHECK(i == static_cast<ssize_t>(source.size()));

    benchmark::DoNotOptimize(buffer.data());
  }

  state.SetBytesProcessed(state.iterations() * source.size());
  state.counters["tokens_per_second"] = benchmark::Counter(
      NumTokens, benchmark::Counter::kIsIterationInvariantRate);
  state.counters["lines_per_second"] =
      benchmark::Counter(llvm::StringRef(source).count('\n'),
                         benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SpeedOfLightDispatch<1>);
BENCHMARK(BM_SpeedOfLightDispatch<2>);
BENCHMARK(BM_SpeedOfLightDispatch<4>);
BENCHMARK(BM_SpeedOfLightDispatch<8>);
BENCHMARK(BM_SpeedOfLightDispatch<16>);
BENCHMARK(BM_SpeedOfLightDispatch<32>);
BENCHMARK(BM_SpeedOfLightDispatch<MaxDispatchTargets>);

}  // namespace
}  // namespace Carbon::Lex
