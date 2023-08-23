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

// A large value for measurement stability without making benchmarking too slow.
// 2^17 is 128 * 1024, so a bit over 100k but a power of two.
constexpr int NumTokens = 1 << 17;

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

// Generates a random identifier string of the specified length using the
// provided RNG BitGen.
//
// Optionally, can specify a min and max length for the generated identifier.
//
// Optionally, can request a uniform distribution of lengths. When this is false
// (the default) the routine tries to generate a distribution that roughly
// matches what we observe in C++ code.
auto GenerateRandomIdentifier(absl::BitGen& gen, int length) -> std::string {
  llvm::ArrayRef<char> start_chars = IdentifierStartChars();
  llvm::ArrayRef<char> chars = IdentifierChars();

#if 0
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
#endif

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

// Get a static pool of random identifiers with the desired distribution.
template <int MinLength = 1, int MaxLength = 64, bool Uniform = false>
auto GetRandomIdentifiers() -> llvm::ArrayRef<std::string> {
  static_assert(MinLength < MaxLength);
  static_assert(
      Uniform || MaxLength <= 64,
      "Cannot produce a meaningful non-uniform distribution of lengths longer "
      "than 64 as those are exceedingly rare in our observed data sets.");

  static std::array<std::string, NumTokens> id_storage = [] {
    std::array<std::string, NumTokens> ids;
    absl::BitGen gen;

    std::array<int, 64> id_length_counts;
    // For non-uniform distribution, we simulate a distribution roughly based on
    // the observed histogram of identifier lengths in LLVM, but smoothed a bit
    // and reduced to small counts so that we cycle through all the lengths
    // reasonably quickly. We want the sampling even 10% of NumTokens from this
    // in a round-robin form to not be skewed overly much. This still inherently
    // compresses the long tail as we'd rather have coverage even though it
    // distorts the distribution a bit.
    //
    // 1 characters   [3976]  ███████████████████████████████▊
    id_length_counts[0] = 40;
    // 2 characters   [3724]  █████████████████████████████▊
    id_length_counts[1] = 40;
    // 3 characters   [4173]  █████████████████████████████████▍
    id_length_counts[2] = 40;
    // 4 characters   [5000]  ████████████████████████████████████████
    id_length_counts[3] = 50;
    // 5 characters   [1568]  ████████████▌
    id_length_counts[4] = 20;
    // 6 characters   [2226]  █████████████████▊
    id_length_counts[5] = 20;
    // 7 characters   [2380]  ███████████████████
    id_length_counts[6] = 20;
    // 8 characters   [1786]  ██████████████▎
    id_length_counts[7] = 18;
    // 9 characters   [1397]  ███████████▏
    id_length_counts[8] = 12;
    // 10 characters  [ 739]  █████▉
    id_length_counts[9] = 12;
    // 11 characters  [ 779]  ██████▎
    id_length_counts[10] = 12;
    // 12 characters  [1344]  ██████████▊
    id_length_counts[11] = 12;
    // 13 characters  [ 498]  ████
    id_length_counts[12] = 5;
    // 14 characters  [ 284]  ██▎
    id_length_counts[13] = 3;
    // 15 characters  [ 172]  █▍
    // 16 characters  [ 278]  ██▎
    // 17 characters  [ 191]  █▌
    // 18 characters  [ 207]  █▋
    for (int i : llvm::seq(14, 18)) {
      id_length_counts[i] = 2;
    }
    // 19 - 63 characters are all <100 in LLVM but non-zero., and we map them to
    // 1 for coverage despite slightly over weighting the tail.
    for (int i : llvm::seq(18, 64)) {
      id_length_counts[i] = 1;
    }

    // Used to track the different count buckets when in a non-uniform
    // distribution.
    int length_bucket_index = 0;
    int length_count = 0;

    for (int i : llvm::seq(0, NumTokens)) {
      if (Uniform) {
        // Rather than using randomness, for a uniform distribution rotate
        // lengths in round-robin to get a deterministic and exact size on every
        // run. We will then shuffle them at the end to produce a random
        // ordering.
        int length = MinLength + (i % (MaxLength - MinLength));
        ids[i] = GenerateRandomIdentifier(gen, length);
        continue;
      }

      // For non-uniform distribution, walk through each each length bucket
      // until our count matches the desired distribution, and then move to the
      // next.
      ids[i] = GenerateRandomIdentifier(gen, length_bucket_index + 1);

      if (length_count < id_length_counts[length_bucket_index]) {
        ++length_count;
      } else {
        length_bucket_index =
            (length_bucket_index + 1) % id_length_counts.size();
        length_count = 0;
      }
    }

    return ids;
  }();
  return id_storage;
}

// Compute a random sequence of just identifiers.
template <int MinLength = 1, int MaxLength = 64, bool Uniform = false>
auto RandomIdentifierSeq() -> std::string {
  std::string result;
  absl::BitGen gen;

  // Get a static pool of identifiers with the desired distribution.
  llvm::ArrayRef<std::string> ids =
      GetRandomIdentifiers<MinLength, MaxLength, Uniform>();

  // Shuffle indices so we get exactly one of each identifier but in a random
  // order.
  std::array<int, NumTokens> indices;
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), gen);

  llvm::raw_string_ostream os(result);
  llvm::ListSeparator sep(" ");
  for (int i : indices) {
    os << sep << ids[i];
  }

  return result;
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
  }();
  return symbol_token_table_storage;
}

// Compute a random sequence of mixed symbols, keywords, and identifiers, with
// percentages of each according to the parameters.
auto RandomMixedSeq(int symbol_percent, int keyword_percent) -> std::string {
  std::string result;
  absl::BitGen gen;

  // Get static pools of symbols, keywords, and identifiers.
  llvm::ArrayRef<TokenKind> symbols = GetSymbolTokenTable();
  llvm::ArrayRef<TokenKind> keywords = TokenKind::KeywordTokens;
  llvm::ArrayRef<std::string> ids = GetRandomIdentifiers();

  // Build a list of kind keys and indices into the relevant tables that have
  // the desired distribution, then shuffle that list.
  enum ElementKind {
    Symbol,
    Keyword,
    Identifier,
  };
  std::array<std::pair<ElementKind, int>, NumTokens> indices;
  for (int i : llvm::seq(0, NumTokens)) {
    int i_percent = i % 100;
    if (i_percent < symbol_percent) {
      indices[i] = {Symbol, i % symbols.size()};
    } else if (i_percent < keyword_percent) {
      indices[i] = {Keyword, i % keywords.size()};
    } else {
      indices[i] = {Identifier, i % ids.size()};
    }
  }
  std::shuffle(indices.begin(), indices.end(), gen);

  llvm::raw_string_ostream os(result);
  llvm::ListSeparator sep(" ");
  for (auto [kind, i] : indices) {
    os << sep
       << (kind == Symbol    ? symbols[i].fixed_spelling()
           : kind == Keyword ? keywords[i].fixed_spelling()
                             : ids[i]);
  }

  return result;
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

void BM_ValidKeywords(benchmark::State& state) {
  absl::BitGen gen;
  std::array<int, NumTokens> indices;
  for (int i : llvm::seq(0, NumTokens)) {
    indices[i] = i % TokenKind::KeywordTokens.size();
  }
  std::shuffle(indices.begin(), indices.end(), gen);

  std::string source;
  llvm::raw_string_ostream os(source);
  llvm::ListSeparator sep(" ");
  for (int i : llvm::seq(0, NumTokens)) {
    os << sep << TokenKind::KeywordTokens[indices[i]].fixed_spelling();
  }

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

template <int MinLength = 1, int MaxLength = 64, bool Uniform = false>
void BM_ValidIdentifiers(benchmark::State& state) {
  std::string source = RandomIdentifierSeq<MinLength, MaxLength, Uniform>();

  LexerBenchHelper helper(source);
  for (auto _ : state) {
    TokenizedBuffer buffer = helper.Lex();
    CARBON_CHECK(!buffer.has_errors()) << helper.DiagnoseErrors();
  }

  state.SetBytesProcessed(state.iterations() * source.size());
  state.counters["tokens_per_second"] = benchmark::Counter(
      NumTokens, benchmark::Counter::kIsIterationInvariantRate);
}
// Benchmark the non-uniform distribution we observe in C++ code.
BENCHMARK(BM_ValidIdentifiers<>);

// Also benchmark a few uniform distribution ranges of identifier widths to
// cover different patterns that emerge with small, medium, and longer
// identifiers.
BENCHMARK(BM_ValidIdentifiers<3, 5, /*Uniform=*/true>);
BENCHMARK(BM_ValidIdentifiers<3, 16, /*Uniform=*/true>);
BENCHMARK(BM_ValidIdentifiers<12, 64, /*Uniform=*/true>);

void BM_ValidMix(benchmark::State& state) {
  int symbol_percent = state.range(0);
  int keyword_percent = state.range(1);
  std::string source = RandomMixedSeq(symbol_percent, keyword_percent);

  LexerBenchHelper helper(source);
  for (auto _ : state) {
    TokenizedBuffer buffer = helper.Lex();

    // Ensure that lexing actually occurs for benchmarking and that it doesn't
    // hit errors that would skew the benchmark results.
    CARBON_CHECK(!buffer.has_errors()) << helper.DiagnoseErrors();
  }

  state.SetBytesProcessed(state.iterations() * source.size());
  state.counters["tokens_per_second"] = benchmark::Counter(
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
