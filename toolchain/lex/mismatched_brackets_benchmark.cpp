// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include "absl/random/random.h"
#include "common/check.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/benchmarking/source_gen.h"
#include "toolchain/diagnostics/emitter.h"
#include "toolchain/diagnostics/null_diagnostics.h"
#include "toolchain/lex/lex.h"
#include "toolchain/lex/mismatched_brackets.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon::Lex {
namespace {

using Kind = BracketTokenKind;

// How many columns each level of nesting indents by, matching the toolchain's
// own style.
constexpr int32_t IndentWidth = 2;

// Builds a token sequence to hand to `FixMismatchedBrackets`, tracking lines
// and indentation the way formatted source would, since the cost model reads
// both. Tokens are added to the current line until `EndLine` starts a new one.
class SourceBuilder {
 public:
  // Appends a token to the current line. Everything but a closing bracket,
  // `,`, `;`, and `.` is written with a space before it, as formatted code
  // would.
  auto Add(Kind kind) -> SourceBuilder& {
    bool spaced = !tokens_.empty() && !IsClosingBracket(kind) &&
                  kind != Kind::Comma && kind != Kind::Semi &&
                  kind != Kind::Period;
    tokens_.push_back({
        .token_index = TokenIndex(static_cast<int32_t>(tokens_.size())),
        .kind = kind,
        .line = line_,
        .line_indent = indent_,
        .has_leading_space = spaced && !at_line_start_,
    });
    at_line_start_ = false;
    return *this;
  }

  // Adds each of `kinds` to the current line.
  auto AddAll(std::initializer_list<Kind> kinds) -> SourceBuilder& {
    for (Kind kind : kinds) {
      Add(kind);
    }
    return *this;
  }

  // Ends the current line, and indents the next one by `indent` columns.
  auto EndLine(int32_t indent) -> SourceBuilder& {
    if (!tokens_.empty()) {
      tokens_.back().is_at_end_of_line = true;
    }
    ++line_;
    indent_ = indent;
    at_line_start_ = true;
    return *this;
  }

  // Ends the current line, keeping the same indentation.
  auto EndLine() -> SourceBuilder& { return EndLine(indent_); }

  // Terminates the sequence with the `FileEnd` token the algorithm expects.
  auto Finish() -> llvm::SmallVector<MismatchedBracketToken> {
    EndLine(0).Add(Kind::FileEnd);
    tokens_.back().is_at_end_of_line = true;
    return std::move(tokens_);
  }

  auto indent() const -> int32_t { return indent_; }

 private:
  llvm::SmallVector<MismatchedBracketToken> tokens_;
  int32_t line_ = 1;
  int32_t indent_ = 0;
  bool at_line_start_ = true;
};

// The bracket kinds a nesting pattern cycles through.
constexpr Kind OpenKinds[] = {Kind::OpenParen, Kind::OpenSquareBracket,
                              Kind::OpenCurlyBrace};

// Opens a top-level declaration: `fn Name(...) {` at column 0, which is where
// `FindRegionBoundaries` cuts one region from the next.
static auto AddDeclHeader(SourceBuilder& builder) -> void {
  builder.AddAll({Kind::StatementIntroducer, Kind::Leaf, Kind::OpenParen,
                  Kind::CloseParen, Kind::OpenCurlyBrace});
}

// `n` unmatched opening parens, one per line, each indenting further, as an
// unfinished call chain would.
static auto UnclosedOpeners(int n)
    -> llvm::SmallVector<MismatchedBracketToken> {
  SourceBuilder builder;
  AddDeclHeader(builder);
  for (int i = 0; i < n; ++i) {
    builder.EndLine(IndentWidth * (i + 1))
        .AddAll({Kind::Leaf, Kind::OpenParen});
  }
  return builder.Finish();
}

// `n` closing parens with nothing to close.
static auto UnmatchedClosers(int n)
    -> llvm::SmallVector<MismatchedBracketToken> {
  SourceBuilder builder;
  AddDeclHeader(builder);
  builder.EndLine(IndentWidth);
  for (int i = 0; i < n; ++i) {
    builder.AddAll({Kind::Leaf, Kind::CloseParen});
  }
  return builder.Finish();
}

// `n` balanced statements, then one whose opening paren is never closed, then
// `n` more balanced statements: the missing bracket is surrounded by correct
// code on both sides.
static auto BalancedRunWithGap(int n)
    -> llvm::SmallVector<MismatchedBracketToken> {
  SourceBuilder builder;
  AddDeclHeader(builder);
  auto add_call = [&](bool closed) {
    builder.EndLine(IndentWidth)
        .AddAll(
            {Kind::Leaf, Kind::OpenParen, Kind::Leaf, Kind::Comma, Kind::Leaf});
    if (closed) {
      builder.Add(Kind::CloseParen);
    }
    builder.Add(Kind::Semi);
  };
  for (int i = 0; i < n; ++i) {
    add_call(/*closed=*/true);
  }
  add_call(/*closed=*/false);
  for (int i = 0; i < n; ++i) {
    add_call(/*closed=*/true);
  }
  builder.EndLine(0).Add(Kind::CloseCurlyBrace);
  return builder.Finish();
}

// An `n`-deep nest of mixed bracket kinds with one line per level. `damage`
// picks which levels have their closer omitted: none, only the innermost, or
// every other level.
enum class NestDamage : uint8_t { None, Innermost, Alternating };

static auto DeepNest(int n, NestDamage damage)
    -> llvm::SmallVector<MismatchedBracketToken> {
  SourceBuilder builder;
  AddDeclHeader(builder);

  llvm::SmallVector<Kind> open_kinds;
  for (int i = 0; i < n; ++i) {
    Kind open = OpenKinds[i % std::size(OpenKinds)];
    open_kinds.push_back(open);
    builder.EndLine(IndentWidth * (i + 1)).AddAll({Kind::Leaf, open});
  }

  for (int i = n - 1; i >= 0; --i) {
    bool omit = damage == NestDamage::Alternating
                    ? i % 2 == 1
                    : damage == NestDamage::Innermost && i == n - 1;
    builder.EndLine(IndentWidth * (i + 1));
    if (omit) {
      builder.Add(Kind::Leaf);
    } else {
      builder.Add(MatchingClosingKind(open_kinds[i]));
    }
  }

  builder.EndLine(0).Add(Kind::CloseCurlyBrace);
  return builder.Finish();
}

// `n` unmatched openers of the same kind in a row, every one of which is an
// equally good place to insert the missing closer. This is what makes the
// search find many optimal repairs, which it then has to compare against each
// other to decide which corrections are tied.
static auto ManyTiedRepairs(int n)
    -> llvm::SmallVector<MismatchedBracketToken> {
  SourceBuilder builder;
  AddDeclHeader(builder);
  builder.EndLine(IndentWidth);
  for (int i = 0; i < n; ++i) {
    builder.AddAll({Kind::OpenParen, Kind::Leaf});
  }
  return builder.Finish();
}

// `n` top-level declarations, each with one bracket missing, so that the search
// runs once per region. Scaling here should be flat per declaration.
static auto ManyDamagedRegions(int n)
    -> llvm::SmallVector<MismatchedBracketToken> {
  SourceBuilder builder;
  for (int i = 0; i < n; ++i) {
    AddDeclHeader(builder);
    builder.EndLine(IndentWidth)
        .AddAll({Kind::Leaf, Kind::OpenParen, Kind::Leaf, Kind::Semi});
    builder.EndLine(0).Add(Kind::CloseCurlyBrace);
    builder.EndLine(0);
  }
  return builder.Finish();
}

// Runs `FixMismatchedBrackets` over `tokens`, reporting throughput against the
// input size so that a sweep shows how cost grows with it.
static auto RunBenchmark(benchmark::State& state,
                         llvm::ArrayRef<MismatchedBracketToken> tokens)
    -> void {
  for (auto _ : state) {
    auto corrections = FixMismatchedBrackets(tokens);
    benchmark::DoNotOptimize(corrections);
  }
  state.SetComplexityN(tokens.size());
  state.counters["tokens_per_second"] = benchmark::Counter(
      tokens.size(), benchmark::Counter::kIsIterationInvariantRate);
}

// A damaged region larger than `MaxRegionItemsForSearch` is handed to the naive
// greedy fallback instead of the search, which is orders of magnitude cheaper.
// The sweeps below stay under that threshold so that the reported complexity
// describes the search rather than averaging the two; `BM_RegionSizeCliff`
// covers the transition itself.
//
// Each pattern emits a couple of items per unit of `N`, so these bounds keep
// the region well under the threshold.
constexpr int MaxSweepN = 256;
constexpr int MaxNestSweepN = 128;
constexpr int MaxStatementSweepN = 64;

auto BM_UnclosedOpeners(benchmark::State& state) -> void {
  RunBenchmark(state, UnclosedOpeners(state.range(0)));
}
BENCHMARK(BM_UnclosedOpeners)
    ->RangeMultiplier(2)
    ->Range(2, MaxSweepN)
    ->Complexity();

auto BM_UnmatchedClosers(benchmark::State& state) -> void {
  RunBenchmark(state, UnmatchedClosers(state.range(0)));
}
BENCHMARK(BM_UnmatchedClosers)
    ->RangeMultiplier(2)
    ->Range(2, MaxSweepN)
    ->Complexity();

auto BM_BalancedRunWithGap(benchmark::State& state) -> void {
  RunBenchmark(state, BalancedRunWithGap(state.range(0)));
}
BENCHMARK(BM_BalancedRunWithGap)
    ->RangeMultiplier(2)
    ->Range(2, MaxStatementSweepN)
    ->Complexity();

// The control: nothing is damaged, so `RegionIsBalanced` skips the search and
// only the up-front whole-file analysis is measured.
auto BM_BalancedNest(benchmark::State& state) -> void {
  RunBenchmark(state, DeepNest(state.range(0), NestDamage::None));
}
BENCHMARK(BM_BalancedNest)->RangeMultiplier(2)->Range(2, 1024)->Complexity();

auto BM_NestInnermostMismatched(benchmark::State& state) -> void {
  RunBenchmark(state, DeepNest(state.range(0), NestDamage::Innermost));
}
BENCHMARK(BM_NestInnermostMismatched)
    ->RangeMultiplier(2)
    ->Range(2, MaxNestSweepN)
    ->Complexity();

auto BM_NestAlternatingMismatched(benchmark::State& state) -> void {
  RunBenchmark(state, DeepNest(state.range(0), NestDamage::Alternating));
}
BENCHMARK(BM_NestAlternatingMismatched)
    ->RangeMultiplier(2)
    ->Range(2, MaxNestSweepN)
    ->Complexity();

auto BM_ManyTiedRepairs(benchmark::State& state) -> void {
  RunBenchmark(state, ManyTiedRepairs(state.range(0)));
}
BENCHMARK(BM_ManyTiedRepairs)
    ->RangeMultiplier(2)
    ->Range(2, MaxSweepN)
    ->Complexity();

// Damage spread across many small regions rather than concentrated in one, to
// check that the per-region cost stays flat as a file grows.
auto BM_ManyDamagedRegions(benchmark::State& state) -> void {
  RunBenchmark(state, ManyDamagedRegions(state.range(0)));
}
BENCHMARK(BM_ManyDamagedRegions)
    ->RangeMultiplier(2)
    ->Range(2, 1024)
    ->Complexity();

// Sweeps one damaged region across `MaxRegionItemsForSearch`, where the search
// gives way to the naive fallback. The cost climbs to the threshold and then
// drops sharply, so this is the shape of the worst case: the most expensive
// input is the largest region the search still accepts.
auto BM_RegionSizeCliff(benchmark::State& state) -> void {
  RunBenchmark(state, UnclosedOpeners(state.range(0)));
}
BENCHMARK(BM_RegionSizeCliff)->RangeMultiplier(2)->Range(128, 2048);

// The benchmarks below run whole generated source files through the lexer, so
// they measure recovery in the place it actually runs, against source shaped
// like real Carbon code. They are modeled on the compile benchmarks, but where
// those sweep the phases of compilation, these sweep damage strategies applied
// to the generated source, including no damage at all: the difference between
// a damaged variant and the undamaged control is what recovery costs.
//
// Note that `SourceGen` seeds itself from entropy, so while the size and shape
// of the generated files are stable from run to run, their exact contents are
// not, and neither is where the damage below lands. Treat a single run of these
// as approximate; the benchmarks above are the reproducible ones.

// The byte offsets of every bracket in `text`. Brackets inside comments are
// included, which is fine: the generated comments contain none.
static auto BracketOffsets(llvm::StringRef text) -> llvm::SmallVector<size_t> {
  llvm::SmallVector<size_t> offsets;
  for (auto [offset, c] : llvm::enumerate(text)) {
    if (llvm::StringRef("()[]{}").contains(c)) {
      offsets.push_back(offset);
    }
  }
  return offsets;
}

// Deletes the brackets at `offsets`, which must be sorted, closing up the text
// so that nothing marks where they were.
static auto DeleteOffsets(llvm::StringRef text, llvm::ArrayRef<size_t> offsets)
    -> std::string {
  std::string result;
  result.reserve(text.size());
  size_t prev = 0;
  for (size_t offset : offsets) {
    result.append(text.substr(prev, offset - prev));
    prev = offset + 1;
  }
  result.append(text.substr(prev));
  return result;
}

// Deletes one in every `one_in` brackets, chosen at random. With `one_in` equal
// to the bracket count this deletes a single bracket.
static auto DeleteBrackets(llvm::StringRef text, int one_in) -> std::string {
  auto offsets = BracketOffsets(text);
  CARBON_CHECK(!offsets.empty(), "Generated source has no brackets.");
  absl::BitGen rng;

  llvm::SmallVector<size_t> deleted;
  int count = std::max<int>(1, offsets.size() / one_in);
  llvm::SmallVector<size_t> remaining = offsets;
  for (int i = 0; i < count && !remaining.empty(); ++i) {
    int pick = absl::Uniform<int>(rng, 0, remaining.size());
    deleted.push_back(remaining[pick]);
    remaining.erase(remaining.begin() + pick);
  }
  llvm::sort(deleted);
  return DeleteOffsets(text, deleted);
}

// Splits `text` into the blank-line separated hunks a generated file falls into
// naturally, returning each hunk's lines.
static auto SplitIntoHunks(llvm::StringRef text)
    -> llvm::SmallVector<llvm::SmallVector<llvm::StringRef>> {
  llvm::SmallVector<llvm::StringRef> lines;
  text.split(lines, '\n');

  llvm::SmallVector<llvm::SmallVector<llvm::StringRef>> hunks;
  hunks.emplace_back();
  for (llvm::StringRef line : lines) {
    if (line.trim().empty()) {
      hunks.emplace_back();
    } else {
      hunks.back().push_back(line);
    }
  }
  return hunks;
}

// Truncates one in every `one_in` hunks partway through, by dropping everything
// from a random line that closes a bracket to the end of that hunk. This is
// what a declaration still being typed looks like: the body is there, the
// closers that would end it are not.
static auto TruncateHunks(llvm::StringRef text, int one_in) -> std::string {
  auto hunks = SplitIntoHunks(text);
  absl::BitGen rng;

  // The hunks with a line that could be truncated at.
  llvm::SmallVector<size_t> candidates;
  for (auto [index, hunk] : llvm::enumerate(hunks)) {
    if (llvm::any_of(hunk, [](llvm::StringRef line) {
          return line.contains(')') || line.contains('}') || line.contains(']');
        })) {
      candidates.push_back(index);
    }
  }
  CARBON_CHECK(!candidates.empty(), "Generated source has no closing bracket.");

  int count = std::max<int>(1, candidates.size() / one_in);
  for (int i = 0; i < count && !candidates.empty(); ++i) {
    int pick = absl::Uniform<int>(rng, 0, candidates.size());
    auto& hunk = hunks[candidates[pick]];
    candidates.erase(candidates.begin() + pick);

    llvm::SmallVector<size_t> closing_lines;
    for (auto [index, line] : llvm::enumerate(hunk)) {
      if (line.contains(')') || line.contains('}') || line.contains(']')) {
        closing_lines.push_back(index);
      }
    }
    hunk.resize(
        closing_lines[absl::Uniform<int>(rng, 0, closing_lines.size())]);
  }

  llvm::SmallVector<llvm::StringRef> lines;
  for (const auto& hunk : hunks) {
    lines.append(hunk.begin(), hunk.end());
    lines.push_back("");
  }
  return llvm::join(lines, "\n");
}

// Lexes a fixed source text, which is what recovery runs inside of.
class LexBenchHelper {
 public:
  explicit LexBenchHelper(std::string text) : text_(std::move(text)) {
    CARBON_CHECK(fs_.addFile(filename_, /*ModificationTime=*/0,
                             llvm::MemoryBuffer::getMemBuffer(text_)));
    source_ = SourceBuffer::MakeFromFile(fs_, filename_,
                                         Diagnostics::ConsoleConsumer());
  }

  auto Lex() -> TokenizedBuffer {
    Lex::LexOptions options;
    options.consumer = &Diagnostics::NullConsumer();
    return Lex::Lex(value_stores_, *source_, options);
  }

 private:
  std::string text_;
  SharedValueStores value_stores_;
  llvm::vfs::InMemoryFileSystem fs_;
  std::string filename_ = "benchmark.carbon";
  std::optional<SourceBuffer> source_;
};

// The damage strategies the whole-file benchmarks sweep. `None` is the
// control: recovery never runs, so it measures the lexer alone.
enum class Damage : uint8_t {
  None,
  OneBracketDeleted,
  EighthOfBracketsDeleted,
  OneHunkTruncated,
  EighthOfHunksTruncated,
};

// Applies `damage` to `text`.
static auto ApplyDamage(std::string text, Damage damage) -> std::string {
  switch (damage) {
    case Damage::None:
      return text;
    case Damage::OneBracketDeleted:
      return DeleteBrackets(text, BracketOffsets(text).size());
    case Damage::EighthOfBracketsDeleted:
      return DeleteBrackets(text, 8);
    case Damage::OneHunkTruncated:
      return TruncateHunks(text, SplitIntoHunks(text).size());
    case Damage::EighthOfHunksTruncated:
      return TruncateHunks(text, 8);
  }
}

// Benchmark on multiple files of the same size but with different source code
// in order to avoid branch prediction perfectly learning a particular file's
// structure and shape, and to average over where the random damage lands. We
// enforce an upper bound to avoid excessive benchmark time and a lower bound to
// avoid anchoring on a single source file that may have unrepresentative
// content.
//
// For simplicity, we compute a number of files from the target line count as a
// heuristic, clamped to the range below.
static auto ComputeFileCount(int target_lines) -> int {
  constexpr int MinFiles = 8;
  [[maybe_unused]] constexpr int MaxFiles = 128;
  int file_count = (1024 * 1024) / target_lines;
#ifndef NDEBUG
  // Use a smaller number of files in debug builds where lexing is slower,
  // capping at the release-mode minimum.
  return std::max(1, std::min(MinFiles, file_count));
#else
  return std::max(MinFiles, std::min(MaxFiles, file_count));
#endif
}

// Lexes a batch of generated API files of `state.range(0)` target lines each,
// damaged according to `D`.
template <Damage D>
static auto BM_LexApiFileDenseDecls(benchmark::State& state) -> void {
  Testing::SourceGen gen;

  int target_lines = state.range(0);
  int num_files = ComputeFileCount(target_lines);

  llvm::SmallVector<std::unique_ptr<LexBenchHelper>> helpers;
  helpers.reserve(num_files);

  double total_bytes = 0.0;
  double total_lines = 0.0;
  for ([[maybe_unused]] auto _ : llvm::seq(num_files)) {
    std::string source =
        ApplyDamage(gen.GenApiFileDenseDecls(
                        target_lines, Testing::SourceGen::DenseDeclParams{}),
                    D);
    total_bytes += source.size();
    total_lines += llvm::count(source, '\n');
    helpers.push_back(std::make_unique<LexBenchHelper>(std::move(source)));
  }

  state.counters["Bytes"] = benchmark::Counter(
      total_bytes / num_files, benchmark::Counter::kIsIterationInvariantRate);
  state.counters["Lines"] = benchmark::Counter(
      total_lines / num_files, benchmark::Counter::kIsIterationInvariantRate);

  // We benchmark in batches of files to avoid benchmarking any peculiarities of
  // a single file.
  while (state.KeepRunningBatch(num_files)) {
    for (ssize_t i = 0; i < num_files;) {
      // We block optimizing `i` as that has proven both more effective at
      // blocking the loop from being optimized away and avoiding disruption of
      // the generated code that we're benchmarking.
      benchmark::DoNotOptimize(i);

      TokenizedBuffer buffer = helpers[i]->Lex();

      // We use the lex result to step through the files, establishing a
      // dependency between each lex and the next. This doesn't fully allow us
      // to measure latency rather than throughput, but minimizes any skew in
      // measurements from speculating the start of the next lex.
      i += static_cast<ssize_t>(buffer.size() != 0);
    }
  }
}

// Applies the shared range configuration used by every whole-file benchmark:
// 256-line test cases through 256k-line test cases, matching the compile
// benchmarks.
static auto ConfigureLexBenchmark(benchmark::Benchmark* b) -> void {
  b->RangeMultiplier(4)->Range(256, static_cast<int64_t>(256 * 1024));
}

BENCHMARK(BM_LexApiFileDenseDecls<Damage::None>)->Apply(ConfigureLexBenchmark);
BENCHMARK(BM_LexApiFileDenseDecls<Damage::OneBracketDeleted>)
    ->Apply(ConfigureLexBenchmark);
BENCHMARK(BM_LexApiFileDenseDecls<Damage::EighthOfBracketsDeleted>)
    ->Apply(ConfigureLexBenchmark);
BENCHMARK(BM_LexApiFileDenseDecls<Damage::OneHunkTruncated>)
    ->Apply(ConfigureLexBenchmark);
BENCHMARK(BM_LexApiFileDenseDecls<Damage::EighthOfHunksTruncated>)
    ->Apply(ConfigureLexBenchmark);

}  // namespace
}  // namespace Carbon::Lex
