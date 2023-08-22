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

constexpr int NumTokens = 100000;

void BM_ValidKeywords(benchmark::State& state) {
  absl::BitGen gen;
  std::string source;
  llvm::raw_string_ostream os(source);
  llvm::ListSeparator sep(" ");
  for (int i = 0; i < NumTokens; ++i) {
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

auto IdentifierStartChars() -> llvm::ArrayRef<char> {
  static llvm::SmallVector<char> chars = [] {
    llvm::SmallVector<char> chars;
    chars.push_back('_');
    for (char c : llvm::seq('A', 'Z')) {
      chars.push_back(c);
    }
    for (char c : llvm::seq('a', 'z')) {
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
    for (char c : llvm::seq('0', '9')) {
      chars.push_back(c);
    }
    return chars;
  }();
  return chars;
}

void BM_ValidIdentifiers(benchmark::State& state) {
  int min_length = state.range(0);
  int max_length = state.range(1);
  absl::BitGen gen;
  std::string source;
  llvm::raw_string_ostream os(source);
  llvm::ListSeparator sep(" ");
  for (int i = 0; i < NumTokens; ++i) {
    int length = absl::Uniform<int>(gen, min_length, max_length);
    os << sep;
    int id_start = source.size();
    llvm::StringRef id;
    do {
      // Erase any prior attempts to find an identifier.
      source.resize(id_start);
      llvm::ArrayRef<char> start_chars = IdentifierStartChars();
      os << start_chars[absl::Uniform<int>(gen, 0, start_chars.size())];
      llvm::ArrayRef<char> chars = IdentifierChars();
      for (int j = 0; j < length; ++j) {
        os << chars[absl::Uniform<int>(gen, 0, chars.size())];
      }
      // Check if we ended up forming an integer type literal or a keyword, and
      // try again.
      id = llvm::StringRef(source).substr(id_start);
    } while (
      llvm::any_of(TokenKind::KeywordTokens, [id](auto token) {
            return id == token.fixed_spelling();
          }) ||
      ((id.consume_front("i") || id.consume_front("u") ||
          id.consume_front("f")) &&
        llvm::all_of(id, [](const char c) {
      return llvm::isDigit(c); })));
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
BENCHMARK(BM_ValidIdentifiers)->Args({3, 5})->Args({3, 16})->Args({12, 64});

}  // namespace
}  // namespace Carbon::Testing
