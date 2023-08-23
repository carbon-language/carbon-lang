// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include "absl/random/random.h"
#include "common/check.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
#include "toolchain/diagnostics/null_diagnostics.h"
#include "toolchain/lexer/token_kind.h"
#include "toolchain/lexer/tokenized_buffer.h"

namespace Carbon::Testing {
namespace {

auto IdentifierStartChars() -> llvm::ArrayRef<char> {
  static llvm::SmallVector<char> chars = [] {
    llvm::SmallVector<char> chars;
    chars.push_back('_');
    for (char c : llvm::seq_inclusive('A', 'Z')) {
      chars.push_back(c);
    }
    for (char c : llvm::seq_inclusive('a', 'z')) {
      chars.push_back(c);
    }
    return chars;
  }();
  return chars;
}

auto IdentifierChars() -> llvm::ArrayRef<char> {
  static llvm::SmallVector<char> chars = [] {
    llvm::ArrayRef<char> start_chars = IdentifierStartChars();
    llvm::SmallVector<char> chars(start_chars.begin(), start_chars.end());
    for (char c : llvm::seq_inclusive('0', '9')) {
      chars.push_back(c);
    }
    return chars;
  }();
  return chars;
}

// Generates a random identifier string using the provided RNG BitGen.
//
// Optionally, can specify a min and max length for the generated identifier.
//
// Optionally, can request a uniform distribution of lengths. When this is false
// (the default) the routine tries to generate a distribution that roughly
// matches what we observe in C++ code.
auto GenerateRandomIdentifier(absl::BitGen& gen, int min_length = 1,
                              int max_length = 64, bool uniform_lengths = false)
    -> std::string {
  llvm::ArrayRef<char> start_chars = IdentifierStartChars();
  llvm::ArrayRef<char> chars = IdentifierChars();

  int length =
      uniform_lengths
          ? absl::Uniform<int>(gen, min_length, max_length)
          // None of the Abseil distributions are *great* fits for observed data
          // on identifier length, but log-uniform is vaguely close. A better
          // distribution would have two peaks -- one at 1 and the other at 4,
          // with a minor dip between and a fairly slow log-uniform falloff into
          // the long tail. Lacking more nuanced distribution functions, we work
          // with a basic log-uniform.
          : absl::LogUniform<int>(gen, min_length, max_length);

  std::string id_result;
  llvm::raw_string_ostream os(id_result);
  llvm::StringRef id;
  do {
    // Erase any prior attempts to find an identifier.
    id_result.clear();
    os << start_chars[absl::Uniform<int>(gen, 0, start_chars.size())];
    for (int j : llvm::seq(0, length)) {
      static_cast<void>(j);
      os << chars[absl::Uniform<int>(gen, 0, chars.size())];
    }
    // Check if we ended up forming an integer type literal or a keyword, and
    // try again.
    id = llvm::StringRef(id_result);
  } while (
      llvm::any_of(TokenKind::KeywordTokens,
                   [id](auto token) { return id == token.fixed_spelling(); }) ||
      ((id.consume_front("i") || id.consume_front("u") ||
        id.consume_front("f")) &&
       llvm::all_of(id, [](const char c) { return llvm::isDigit(c); })));
  return id_result;
}

// Build our own table of symbols so we can use repetitions to skew the
// distribution.
auto GetSymbolTokenTableImpl() -> llvm::SmallVector<TokenKind> {
  llvm::SmallVector<TokenKind> table;
#define CARBON_SYMBOL_TOKEN(TokenName, Spelling) \
  table.push_back(TokenKind::TokenName);
#define CARBON_OPENING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, ClosingName)
#define CARBON_CLOSING_GROUP_SYMBOL_TOKEN(TokenName, Spelling, OpeningName)
#include "toolchain/lexer/token_kind.def"
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
}

auto GetSymbolTokenTable() -> llvm::ArrayRef<TokenKind> {
  static auto symbol_token_table_storage = GetSymbolTokenTableImpl();
  return symbol_token_table_storage;
}

// Generate random symbols. This skews the distribution as best it can towards
// what we expect in real world source code, but doesn't include grouping
// symbols for simplicity.
auto GenerateRandomSymbol(absl::BitGen& gen) -> llvm::StringRef {
  llvm::ArrayRef<TokenKind> table = GetSymbolTokenTable();
  auto index = absl::Uniform<int>(gen, 0, table.size());
  return table[index].fixed_spelling();
}

class LexerBenchHelper {
 public:
  explicit LexerBenchHelper(llvm::StringRef text)
      : source_(MakeSourceBuffer(text)) {}

  auto Lex() -> TokenizedBuffer {
    DiagnosticConsumer& consumer = NullDiagnosticConsumer();
    return TokenizedBuffer::Lex(source_, consumer);
  }

  auto DiagnoseErrors() -> std::string {
    std::string result;
    llvm::raw_string_ostream out(result);
    StreamDiagnosticConsumer consumer(out);
    auto buffer = TokenizedBuffer::Lex(source_, consumer);
    consumer.Flush();
    CARBON_CHECK(buffer.has_errors())
        << "Asked to diagnose errors but none found!";
    return result;
  }

 private:
  auto MakeSourceBuffer(llvm::StringRef text) -> SourceBuffer {
    CARBON_CHECK(fs_.addFile(filename_, /*ModificationTime=*/0,
                             llvm::MemoryBuffer::getMemBuffer(text)));
    return std::move(*SourceBuffer::CreateFromFile(fs_, filename_));
  }

  llvm::vfs::InMemoryFileSystem fs_;
  std::string filename_ = "test.carbon";
  SourceBuffer source_;
};

// A large value for measurement stability without making benchmarking too slow.
constexpr int NumTokens = 100000;

void BM_ValidKeywords(benchmark::State& state) {
  absl::BitGen gen;
  std::string source;
  llvm::raw_string_ostream os(source);
  llvm::ListSeparator sep(" ");
  for (int i : llvm::seq(0, NumTokens)) {
    static_cast<void>(i);
    int token = absl::Uniform<int>(gen, 0, TokenKind::KeywordTokens.size());
    os << sep << TokenKind::KeywordTokens[token].fixed_spelling();
  }

  LexerBenchHelper helper(source);
  for (auto _ : state) {
    TokenizedBuffer buffer = helper.Lex();
    CARBON_CHECK(!buffer.has_errors());
  }

  state.counters["TokenRate"] = benchmark::Counter(
      NumTokens, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_ValidKeywords);

void BM_ValidIdentifiers(benchmark::State& state, bool uniform_lengths) {
  int min_length = state.range(0);
  int max_length = state.range(1);
  absl::BitGen gen;
  std::string source;
  llvm::raw_string_ostream os(source);
  llvm::ListSeparator sep(" ");
  for (int i = 0; i < NumTokens; ++i) {
    os << sep
       << GenerateRandomIdentifier(gen, min_length, max_length,
                                   uniform_lengths);
  }

  LexerBenchHelper helper(source);
  for (auto _ : state) {
    TokenizedBuffer buffer = helper.Lex();
    CARBON_CHECK(!buffer.has_errors()) << helper.DiagnoseErrors();
  }

  state.counters["TokenRate"] = benchmark::Counter(
      NumTokens, benchmark::Counter::kIsIterationInvariantRate);
}
// Benchmark the non-uniform distribution we observe in C++ code.
BENCHMARK_CAPTURE(BM_ValidIdentifiers, Representative,
                  /*uniform_lengths=*/false)
    ->Args({1, 64});

// Also benchmark a few uniform distribution ranges of identifier widths to
// cover different patterns that emerge with small, medium, and longer
// identifiers.
BENCHMARK_CAPTURE(BM_ValidIdentifiers, Uniform,
                  /*uniform_lengths=*/true)
    ->Args({3, 5})
    ->Args({3, 16})
    ->Args({12, 64});

void BM_ValidMix(benchmark::State& state) {
  int symbol_percent = state.range(0);
  int keyword_percent = state.range(1);
  absl::BitGen gen;
  std::string source;
  llvm::raw_string_ostream os(source);
  llvm::ListSeparator sep(" ");
  for (int i = 0; i < NumTokens; ++i) {
    os << sep;
    int percent_bucket = absl::Uniform<int>(gen, 0, 100);
    if (percent_bucket < symbol_percent) {
      os << GenerateRandomSymbol(gen);
    } else if (percent_bucket < symbol_percent + keyword_percent) {
      int index = absl::Uniform<int>(gen, 0, TokenKind::KeywordTokens.size());
      os << TokenKind::KeywordTokens[index].fixed_spelling();
    } else {
      os << GenerateRandomIdentifier(gen);
    }
  }

  LexerBenchHelper helper(source);
  for (auto _ : state) {
    TokenizedBuffer buffer = helper.Lex();

    // Ensure that lexing actually occurs for benchmarking and that it doesn't
    // hit errors that would skew the benchmark results.
    CARBON_CHECK(!buffer.has_errors()) << helper.DiagnoseErrors();
  }

  state.counters["TokenRate"] = benchmark::Counter(
      NumTokens, benchmark::Counter::kIsIterationInvariantRate);
}
// The distributions between symbols, keywords, and identifiers here are
// guesses. Eventually, we should collect more data to help tune these, but
// hopefully the performance isn't too sensitive and we can just cover a wide
// range here.
BENCHMARK(BM_ValidMix)
    ->Args({10, 40})
    ->Args({25, 30})
    ->Args({50, 20})
    ->Args({75, 10});

}  // namespace
}  // namespace Carbon::Testing
