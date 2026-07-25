// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "common/bazel_working_dir.h"
#include "common/check.h"
#include "common/command_line.h"
#include "common/init_llvm.h"
#include "common/ostream.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/diagnostics/consumer.h"
#include "toolchain/diagnostics/null_diagnostics.h"
#include "toolchain/lex/lex.h"
#include "toolchain/lex/mismatched_brackets.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon::Lex {
namespace {

enum class TestClassification {
  Correct,
  Partial,
  None,
  Incorrect,
};

struct DSpec {
  std::string label;
  bool is_percent = false;
  double percent_val = 0.0;
  int count_val = 0;
};

struct TrialStats {
  int total = 0;
  int correct = 0;
  int partial = 0;
  int none = 0;
  int incorrect = 0;

  auto Add(TestClassification classification) -> void {
    ++total;
    switch (classification) {
      case TestClassification::Correct:
        ++correct;
        break;
      case TestClassification::Partial:
        ++partial;
        break;
      case TestClassification::None:
        ++none;
        break;
      case TestClassification::Incorrect:
        ++incorrect;
        break;
    }
  }

  [[nodiscard]] auto CorrectPct() const -> double {
    return total == 0 ? 0.0 : (100.0 * correct) / total;
  }

  [[nodiscard]] auto PartialPct() const -> double {
    return total == 0 ? 0.0 : (100.0 * partial) / total;
  }

  [[nodiscard]] auto NonePct() const -> double {
    return total == 0 ? 0.0 : (100.0 * none) / total;
  }

  [[nodiscard]] auto IncorrectPct() const -> double {
    return total == 0 ? 0.0 : (100.0 * incorrect) / total;
  }

  [[nodiscard]] auto SafetyPct() const -> double {
    return total == 0 ? 0.0 : (100.0 * (correct + partial + none)) / total;
  }

  [[nodiscard]] auto AccuracyPct() const -> double {
    int decisive = correct + incorrect;
    return decisive == 0 ? 100.0 : (100.0 * correct) / decisive;
  }
};

struct BracketPair {
  TokenIndex open_token;
  TokenIndex close_token;
};

struct DeletedToken {
  TokenKind kind;
  int32_t byte_offset;
  int32_t length;
  int32_t line;
  int32_t column;
  int32_t next_token_byte_offset;
  int32_t next_token_line;
  int32_t next_token_column;
  llvm::SmallVector<int32_t, 4> valid_next_token_byte_offsets;
};

struct Suggestion {
  TokenKind kind;
  int32_t byte_offset;
  int32_t line;
  int32_t column;
  std::string origin;
};

auto MatchesDeletedToken(const DeletedToken& del, const Suggestion& sugg)
    -> bool {
  if (del.kind != sugg.kind) {
    return false;
  }
  return llvm::is_contained(del.valid_next_token_byte_offsets,
                            sugg.byte_offset);
}

auto ParseDSpecs(llvm::StringRef str) -> llvm::SmallVector<DSpec> {
  llvm::SmallVector<DSpec> specs;
  llvm::SmallVector<llvm::StringRef> parts;
  str.split(parts, ',');
  for (llvm::StringRef part : parts) {
    part = part.trim();
    if (part.empty()) {
      continue;
    }
    DSpec spec;
    spec.label = part.str();
    if (part.ends_with("%")) {
      spec.is_percent = true;
      llvm::StringRef num = part.drop_back(1);
      double val = 0.0;
      if (!num.getAsDouble(val)) {
        spec.percent_val = val / 100.0;
        specs.push_back(spec);
      }
    } else {
      spec.is_percent = false;
      int val = 0;
      if (!part.getAsInteger(10, val)) {
        spec.count_val = val;
        specs.push_back(spec);
      }
    }
  }
  return specs;
}

auto GetCleanBracketPairs(const TokenizedBuffer& buffer)
    -> llvm::SmallVector<BracketPair> {
  llvm::SmallVector<BracketPair> pairs;
  for (TokenIndex t : buffer.tokens()) {
    auto kind = buffer.GetKind(t);
    if (kind == TokenKind::OpenParen || kind == TokenKind::OpenSquareBracket ||
        kind == TokenKind::OpenCurlyBrace) {
      TokenIndex close = buffer.GetMatchedClosingToken(t);
      if (close != TokenIndex::None) {
        pairs.push_back({.open_token = t, .close_token = close});
      }
    }
  }
  return pairs;
}

auto ClassifyTrial(llvm::ArrayRef<DeletedToken> deleted_tokens,
                   llvm::ArrayRef<Suggestion> suggestions)
    -> TestClassification {
  if (deleted_tokens.empty()) {
    return suggestions.empty() ? TestClassification::Correct
                               : TestClassification::Incorrect;
  }

  std::vector<bool> deleted_matched(deleted_tokens.size(), false);
  int correct_suggestions = 0;

  for (const auto& sugg : suggestions) {
    bool found_match = false;
    for (size_t i = 0; i < deleted_tokens.size(); ++i) {
      if (!deleted_matched[i] && MatchesDeletedToken(deleted_tokens[i], sugg)) {
        deleted_matched[i] = true;
        found_match = true;
        ++correct_suggestions;
        break;
      }
    }
    if (!found_match) {
      return TestClassification::Incorrect;
    }
  }

  if (correct_suggestions == static_cast<int>(deleted_tokens.size())) {
    return TestClassification::Correct;
  }
  if (correct_suggestions > 0) {
    return TestClassification::Partial;
  }
  return TestClassification::None;
}

// How a clean file is corrupted for a trial. See `--mode` help for details.
enum class CorruptionMode {
  // Blank each deleted bracket with a space (byte offsets preserved).
  Blank,
  // Delete each bracket character, closing the gap (no leftover space). More
  // realistic, and doesn't leave the whitespace artifacts the algorithm can
  // key on.
  DeleteGap,
  // Truncate the file at a random token (models incomplete, in-development
  // code); recovery should close all still-open brackets at the new EOF.
  Truncate,
  // Delete from a random statement/element boundary inside one bracketed
  // region through that region's closing bracket (models typing new code
  // inside an existing class/scope, whose tail and close aren't there yet);
  // recovery must infer where the region should have ended.
  TruncateRegion,
};

// A corrupted source plus the ground-truth insertions recovery should make.
struct CorruptedCase {
  std::string text;
  std::vector<DeletedToken> expected;
};

// Remaps a byte offset from original to corrupted coordinates, given sorted,
// disjoint deleted ranges [begin, end). An offset inside a deleted range maps
// to where the gap closes.
auto RemapOffset(int32_t off,
                 llvm::ArrayRef<std::pair<int32_t, int32_t>> deleted)
    -> int32_t {
  int32_t shift = 0;
  for (auto [begin, end] : deleted) {
    if (end <= off) {
      shift += end - begin;
    } else if (begin <= off) {
      return begin - shift;
    } else {
      break;
    }
  }
  return off - shift;
}

// Returns `text` with the given sorted, disjoint byte ranges removed.
auto RemoveRanges(llvm::StringRef text,
                  llvm::ArrayRef<std::pair<int32_t, int32_t>> ranges)
    -> std::string {
  std::string out;
  int32_t pos = 0;
  for (auto [begin, end] : ranges) {
    out += text.substr(pos, begin - pos);
    pos = end;
  }
  out += text.substr(pos);
  return out;
}

// Builds a trial that deletes one endpoint of each of `d_count` random clean
// pairs, either blanking them or closing the gap.
auto MakeDeletionCase(const TokenizedBuffer& buffer,
                      llvm::StringRef source_text,
                      llvm::ArrayRef<BracketPair> pairs, int d_count,
                      bool close_gap, std::mt19937_64& rng)
    -> std::optional<CorruptedCase> {
  std::vector<int> pair_indices(pairs.size());
  std::iota(pair_indices.begin(), pair_indices.end(), 0);
  std::shuffle(pair_indices.begin(), pair_indices.end(), rng);
  pair_indices.resize(d_count);

  std::vector<bool> is_deleted(buffer.size(), false);
  std::vector<TokenIndex> sampled;
  sampled.reserve(d_count);
  for (int p_idx : pair_indices) {
    const auto& pair = pairs[p_idx];
    TokenIndex tok = (rng() % 2 == 0) ? pair.open_token : pair.close_token;
    is_deleted[tok.index] = true;
    sampled.push_back(tok);
  }

  std::vector<DeletedToken> deleted;
  deleted.reserve(d_count);
  for (TokenIndex tok : sampled) {
    TokenKind tok_kind = buffer.GetKind(tok);

    int32_t run_start = tok.index;
    while (run_start > 0 &&
           buffer.GetKind(TokenIndex(run_start - 1)) == tok_kind) {
      --run_start;
    }
    int32_t run_end = tok.index;
    while (run_end + 1 < static_cast<int32_t>(buffer.size()) &&
           buffer.GetKind(TokenIndex(run_end + 1)) == tok_kind) {
      ++run_end;
    }

    llvm::SmallVector<int32_t, 4> valid_offsets;
    for (int32_t idx = run_start; idx <= run_end; ++idx) {
      if (!is_deleted[idx]) {
        valid_offsets.push_back(buffer.GetByteOffset(TokenIndex(idx)));
      }
    }
    TokenIndex succ = TokenIndex(run_end + 1);
    while (succ.index < buffer.size() && is_deleted[succ.index]) {
      succ = TokenIndex(succ.index + 1);
    }
    int32_t succ_byte = (succ.index < buffer.size())
                            ? buffer.GetByteOffset(succ)
                            : static_cast<int32_t>(source_text.size());
    valid_offsets.push_back(succ_byte);

    deleted.push_back(DeletedToken{
        .kind = tok_kind,
        .byte_offset = buffer.GetByteOffset(tok),
        .length = static_cast<int32_t>(buffer.GetTokenText(tok).size()),
        .line = buffer.GetLineNumber(tok),
        .column = buffer.GetColumnNumber(tok),
        .next_token_byte_offset = succ_byte,
        .next_token_line =
            (succ.index < buffer.size()) ? buffer.GetLineNumber(succ) : -1,
        .next_token_column =
            (succ.index < buffer.size()) ? buffer.GetColumnNumber(succ) : -1,
        .valid_next_token_byte_offsets = std::move(valid_offsets),
    });
  }

  if (!close_gap) {
    std::string corrupted = source_text.str();
    for (const auto& del : deleted) {
      for (int i = 0; i < del.length; ++i) {
        corrupted[del.byte_offset + i] = ' ';
      }
    }
    return CorruptedCase{.text = std::move(corrupted),
                         .expected = std::move(deleted)};
  }

  llvm::SmallVector<std::pair<int32_t, int32_t>> ranges;
  for (TokenIndex tok : sampled) {
    int32_t off = buffer.GetByteOffset(tok);
    ranges.push_back(
        {off, off + static_cast<int32_t>(buffer.GetTokenText(tok).size())});
  }
  llvm::sort(ranges);
  std::string corrupted = RemoveRanges(source_text, ranges);
  for (auto& del : deleted) {
    del.byte_offset = RemapOffset(del.byte_offset, ranges);
    del.next_token_byte_offset =
        RemapOffset(del.next_token_byte_offset, ranges);
    for (auto& off : del.valid_next_token_byte_offsets) {
      off = RemapOffset(off, ranges);
    }
  }
  return CorruptedCase{.text = std::move(corrupted),
                       .expected = std::move(deleted)};
}

// Builds a trial that truncates the file at a random token boundary; recovery
// should close every bracket still open there, at the new EOF.
auto MakeTruncateCase(const TokenizedBuffer& buffer,
                      llvm::StringRef source_text, std::mt19937_64& rng)
    -> std::optional<CorruptedCase> {
  int32_t size = buffer.size();
  if (size <= 2) {
    return std::nullopt;
  }
  // Cut before a random token in (FileStart, FileEnd), keeping [0, cut).
  int32_t cut = 1 + static_cast<int32_t>(rng() % (size - 2));
  int32_t cut_byte = buffer.GetByteOffset(TokenIndex(cut));
  std::string corrupted = source_text.substr(0, cut_byte).str();

  llvm::SmallVector<TokenKind> stack;
  for (int32_t i = 0; i < cut; ++i) {
    auto kind = buffer.GetKind(TokenIndex(i));
    if (kind.is_opening_symbol()) {
      stack.push_back(kind);
    } else if (kind.is_closing_symbol() && !stack.empty() &&
               stack.back().closing_symbol() == kind) {
      stack.pop_back();
    }
  }

  int32_t eof = static_cast<int32_t>(corrupted.size());
  std::vector<DeletedToken> expected;
  for (TokenKind open_kind : llvm::reverse(stack)) {
    expected.push_back(DeletedToken{
        .kind = open_kind.closing_symbol(),
        .byte_offset = eof,
        .length = 1,
        .line = -1,
        .column = -1,
        .next_token_byte_offset = eof,
        .next_token_line = -1,
        .next_token_column = -1,
        // Every open bracket must close at EOF: the surviving token each
        // closer precedes is the FileEnd token. (The real FileEnd offset,
        // which excludes trailing whitespace, is added after lexing.)
        .valid_next_token_byte_offsets = {eof},
    });
  }
  return CorruptedCase{.text = std::move(corrupted),
                       .expected = std::move(expected)};
}

// Builds a trial that deletes from a region-top-level boundary inside a random
// pair through that pair's closing bracket. Recovery must reinsert the one
// closing bracket at the join, where the region should have ended.
auto MakeTruncateRegionCase(const TokenizedBuffer& buffer,
                            llvm::StringRef source_text,
                            llvm::ArrayRef<BracketPair> pairs,
                            std::mt19937_64& rng)
    -> std::optional<CorruptedCase> {
  if (pairs.empty()) {
    return std::nullopt;
  }
  const auto& pair = pairs[rng() % pairs.size()];
  int32_t open = pair.open_token.index;
  int32_t close = pair.close_token.index;

  // Collect cut points at the region's own nesting level (not inside a nested
  // pair), so that deleting through the close orphans only this one bracket.
  llvm::SmallVector<int32_t> candidates;
  int32_t depth = 0;
  for (int32_t i = open + 1; i <= close; ++i) {
    if (depth == 0) {
      candidates.push_back(i);
    }
    if (i < close) {
      auto kind = buffer.GetKind(TokenIndex(i));
      if (kind.is_opening_symbol()) {
        ++depth;
      } else if (kind.is_closing_symbol()) {
        --depth;
      }
    }
  }
  if (candidates.empty()) {
    return std::nullopt;
  }
  int32_t cut = candidates[rng() % candidates.size()];

  int32_t del_begin = buffer.GetByteOffset(TokenIndex(cut));
  int32_t del_end =
      buffer.GetByteOffset(pair.close_token) +
      static_cast<int32_t>(buffer.GetTokenText(pair.close_token).size());
  llvm::SmallVector<std::pair<int32_t, int32_t>> ranges = {
      {del_begin, del_end}};
  std::string corrupted = RemoveRanges(source_text, ranges);

  int32_t succ = close + 1;
  int32_t succ_byte = (succ < buffer.size())
                          ? buffer.GetByteOffset(TokenIndex(succ))
                          : static_cast<int32_t>(source_text.size());
  int32_t succ_corrupted = RemapOffset(succ_byte, ranges);

  TokenKind close_kind = buffer.GetKind(pair.open_token).closing_symbol();
  std::vector<DeletedToken> expected = {DeletedToken{
      .kind = close_kind,
      .byte_offset = del_begin,
      .length = 1,
      .line = buffer.GetLineNumber(pair.close_token),
      .column = buffer.GetColumnNumber(pair.close_token),
      .next_token_byte_offset = succ_corrupted,
      .next_token_line = -1,
      .next_token_column = -1,
      .valid_next_token_byte_offsets = {succ_corrupted},
  }};
  return CorruptedCase{.text = std::move(corrupted),
                       .expected = std::move(expected)};
}

auto CollectCarbonFiles(llvm::ArrayRef<llvm::StringRef> input_paths)
    -> llvm::SmallVector<std::string> {
  llvm::SmallVector<std::string> files;

  auto add_path_if_carbon = [&](llvm::StringRef path) {
    if (path.ends_with(".carbon")) {
      files.push_back(path.str());
    }
  };

  auto scan_directory = [&](llvm::StringRef dir_path) {
    std::error_code ec;
    for (llvm::sys::fs::recursive_directory_iterator it(dir_path, ec), end;
         it != end && !ec; it.increment(ec)) {
      if (it->path().ends_with(".carbon")) {
        files.push_back(it->path());
      }
    }
  };

  if (input_paths.empty()) {
    scan_directory("core");
    scan_directory("examples");
  } else {
    for (llvm::StringRef path : input_paths) {
      if (llvm::sys::fs::is_directory(path)) {
        scan_directory(path);
      } else {
        add_path_if_carbon(path);
      }
    }
  }

  std::sort(files.begin(), files.end());
  files.erase(std::unique(files.begin(), files.end()), files.end());
  return files;
}

constexpr CommandLine::CommandInfo CommandInfo = {
    .name = "mismatched_brackets_eval",
    .help = R"""(
A measurement and benchmarking tool for Carbon bracket recovery.

Evaluates how accurately and safely the bracket error recovery algorithm
recovers deleted subsets of brackets across Carbon source files.
)""",
};

auto Run(llvm::ArrayRef<llvm::StringRef> args) -> bool {
  llvm::SmallVector<llvm::StringRef> input_files;
  llvm::StringRef d_values_str = "1,2,5,10%,25%";
  llvm::StringRef mode_str = "blank";
  int total_trials = 1000;
  int base_seed = 42;
  bool verbose = false;
  bool json_output = false;
  int dump_incorrect = 0;
  int dump_none = 0;

  auto parse_result = CommandLine::Parse(
      args, llvm::outs(), CommandInfo, [&](CommandLine::CommandBuilder& b) {
        b.AddStringPositionalArg(
            {
                .name = "FILE",
                .help = "Input Carbon source file(s) or directories to test.",
            },
            [&](auto& arg_b) { arg_b.Append(&input_files); });

        b.AddStringOption(
            {
                .name = "d-values",
                .value_name = "LIST",
                .help =
                    "Comma-separated deletion levels (e.g. '1,2,5,10%,25%'). "
                    "Ignored by the truncate modes.",
            },
            [&](auto& arg_b) { arg_b.Set(&d_values_str); });

        b.AddStringOption(
            {
                .name = "mode",
                .value_name = "MODE",
                .help =
                    "How to corrupt each file: 'blank' (replace brackets with "
                    "spaces; the default), 'gap' (delete bracket characters, "
                    "closing the gap), 'truncate' (cut the file at a random "
                    "token; recovery should close all open brackets at EOF), "
                    "or 'truncate-region' (delete from inside a random pair "
                    "through its close, as when typing new code in an existing "
                    "class).",
            },
            [&](auto& arg_b) { arg_b.Set(&mode_str); });

        b.AddIntegerOption(
            {
                .name = "trials",
                .value_name = "N",
                .help = "Total number of trials per D configuration.",
            },
            [&](auto& arg_b) { arg_b.Set(&total_trials); });

        b.AddIntegerOption(
            {
                .name = "seed",
                .value_name = "N",
                .help = "Random seed for deterministic sampling.",
            },
            [&](auto& arg_b) { arg_b.Set(&base_seed); });

        b.AddFlag(
            {
                .name = "verbose",
                .help = "Print detailed per-file results.",
            },
            [&](auto& arg_b) { arg_b.Set(&verbose); });

        b.AddFlag(
            {
                .name = "json",
                .help = "Output results in JSON format.",
            },
            [&](auto& arg_b) { arg_b.Set(&json_output); });

        b.AddIntegerOption(
            {
                .name = "dump-incorrect",
                .value_name = "N",
                .help = "Print details for up to N incorrect trials.",
            },
            [&](auto& arg_b) { arg_b.Set(&dump_incorrect); });

        b.AddIntegerOption(
            {
                .name = "dump-none",
                .value_name = "N",
                .help = "Print details for up to N trials classified None.",
            },
            [&](auto& arg_b) { arg_b.Set(&dump_none); });

        b.Do([] {});
      });

  if (!parse_result.ok()) {
    llvm::errs() << "error: " << *parse_result << "\n";
    return false;
  } else if (*parse_result == CommandLine::ParseResult::MetaSuccess) {
    return true;
  }

  auto d_specs = ParseDSpecs(d_values_str);
  if (d_specs.empty()) {
    llvm::errs() << "error: No valid D deletion specifications provided.\n";
    return false;
  }

  CorruptionMode mode = CorruptionMode::Blank;
  if (mode_str == "blank") {
    mode = CorruptionMode::Blank;
  } else if (mode_str == "gap") {
    mode = CorruptionMode::DeleteGap;
  } else if (mode_str == "truncate") {
    mode = CorruptionMode::Truncate;
  } else if (mode_str == "truncate-region") {
    mode = CorruptionMode::TruncateRegion;
  } else {
    llvm::errs() << "error: Unknown --mode '" << mode_str << "'.\n";
    return false;
  }
  // The truncate modes don't delete a set number of brackets, so collapse the
  // D configurations to a single pass.
  if (mode == CorruptionMode::Truncate ||
      mode == CorruptionMode::TruncateRegion) {
    d_specs.resize(1);
    d_specs[0].label = mode_str.str();
  }

  auto files = CollectCarbonFiles(input_files);
  if (files.empty()) {
    llvm::errs() << "error: No Carbon source files found to test.\n";
    return false;
  }

  struct CandidateFile {
    std::string filename;
    int clean_pairs_count = 0;
  };

  std::vector<CandidateFile> valid_files;
  int total_clean_pairs = 0;

  for (const auto& filepath : files) {
    auto source = SourceBuffer::MakeFromFile(
        *llvm::vfs::getRealFileSystem(), filepath, Diagnostics::NullConsumer());
    if (!source) {
      continue;
    }

    SharedValueStores value_stores;
    LexOptions lex_options;
    lex_options.consumer = &Diagnostics::NullConsumer();
    auto clean_buffer = Lex::Lex(value_stores, *source, lex_options);

    if (clean_buffer.has_errors()) {
      continue;
    }

    auto clean_pairs = GetCleanBracketPairs(clean_buffer);
    if (clean_pairs.empty()) {
      continue;
    }

    total_clean_pairs += clean_pairs.size();
    valid_files.push_back({
        .filename = filepath,
        .clean_pairs_count = static_cast<int>(clean_pairs.size()),
    });
  }

  if (valid_files.empty() || total_clean_pairs == 0) {
    llvm::errs()
        << "error: No Carbon source files with bracket pairs found to test.\n";
    return false;
  }

  std::vector<std::vector<int>> scenario_file_trials(
      d_specs.size(), std::vector<int>(valid_files.size(), 0));

  if (total_trials > 0) {
    for (size_t s_idx = 0; s_idx < d_specs.size(); ++s_idx) {
      const auto& spec = d_specs[s_idx];
      double total_weight = spec.is_percent
                                ? static_cast<double>(valid_files.size())
                                : static_cast<double>(total_clean_pairs);

      int allocated = 0;
      std::vector<std::pair<double, size_t>> remainders;
      remainders.reserve(valid_files.size());

      for (size_t i = 0; i < valid_files.size(); ++i) {
        double weight =
            spec.is_percent
                ? 1.0
                : static_cast<double>(valid_files[i].clean_pairs_count);
        double exact_quota =
            static_cast<double>(total_trials) * weight / total_weight;
        int base_count = static_cast<int>(exact_quota);
        scenario_file_trials[s_idx][i] = base_count;
        allocated += base_count;
        remainders.push_back({exact_quota - base_count, i});
      }

      std::sort(remainders.begin(), remainders.end(),
                [&](const auto& a, const auto& b) {
                  if (a.first != b.first) {
                    return a.first > b.first;
                  }
                  return valid_files[a.second].filename <
                         valid_files[b.second].filename;
                });

      int remainder_trials = total_trials - allocated;
      for (int i = 0;
           i < remainder_trials && i < static_cast<int>(remainders.size());
           ++i) {
        ++scenario_file_trials[s_idx][remainders[i].second];
      }
    }
  }

  struct FileResult {
    std::string filename;
    int clean_pairs_count = 0;
    std::vector<TrialStats> scenario_stats;
  };

  struct OriginStat {
    int correct = 0;
    int incorrect = 0;
  };
  std::map<std::string, OriginStat> origin_stats;
  int merged_skips = 0;

  // For incorrect trials: how far (in tokens, signed; + = closed later /
  // swallowing following code) is the wrong close from the correct anchor,
  // bucketed by the deleted bracket's kind.
  struct DistStat {
    int later = 0;
    int earlier = 0;
    int no_close = 0;
    std::map<int, int> token_dist_hist;
    std::map<int, int> line_dist_hist;
  };
  std::map<std::string, DistStat> dist_by_kind;

  std::vector<FileResult> file_results;
  std::vector<TrialStats> overall_scenario_stats(d_specs.size());

  for (size_t f_idx = 0; f_idx < valid_files.size(); ++f_idx) {
    const auto& candidate = valid_files[f_idx];

    bool has_any_trials = false;
    for (size_t s_idx = 0; s_idx < d_specs.size(); ++s_idx) {
      if (scenario_file_trials[s_idx][f_idx] > 0) {
        has_any_trials = true;
        break;
      }
    }

    if (!verbose && !has_any_trials) {
      continue;
    }

    auto source = SourceBuffer::MakeFromFile(*llvm::vfs::getRealFileSystem(),
                                             candidate.filename,
                                             Diagnostics::NullConsumer());
    if (!source) {
      continue;
    }

    SharedValueStores value_stores;
    LexOptions lex_options;
    lex_options.consumer = &Diagnostics::NullConsumer();
    auto clean_buffer = Lex::Lex(value_stores, *source, lex_options);
    auto clean_pairs = GetCleanBracketPairs(clean_buffer);

    FileResult f_res;
    f_res.filename = candidate.filename;
    f_res.clean_pairs_count = candidate.clean_pairs_count;
    f_res.scenario_stats.resize(d_specs.size());

    llvm::StringRef source_text = source->text();

    for (size_t s_idx = 0; s_idx < d_specs.size(); ++s_idx) {
      const auto& spec = d_specs[s_idx];
      int num_trials = scenario_file_trials[s_idx][f_idx];

      int d_count = 0;
      if (spec.is_percent) {
        d_count = std::max(
            1, static_cast<int>(clean_pairs.size() * spec.percent_val));
      } else {
        d_count = spec.count_val;
      }
      d_count = std::min(d_count, static_cast<int>(clean_pairs.size()));

      for (int trial = 0; trial < num_trials; ++trial) {
        uint64_t file_hash = llvm::hash_value(candidate.filename);
        uint64_t spec_hash = llvm::hash_value(spec.label);
        uint64_t trial_seed =
            static_cast<uint64_t>(base_seed) ^ file_hash ^ (spec_hash << 16) ^
            (static_cast<uint64_t>(trial) * 0x9e3779b97f4a7c15ULL);

        std::mt19937_64 rng(trial_seed);

        std::optional<CorruptedCase> corrupted_case;
        switch (mode) {
          case CorruptionMode::Blank:
          case CorruptionMode::DeleteGap:
            corrupted_case = MakeDeletionCase(
                clean_buffer, source_text, clean_pairs, d_count,
                /*close_gap=*/mode == CorruptionMode::DeleteGap, rng);
            break;
          case CorruptionMode::Truncate:
            corrupted_case = MakeTruncateCase(clean_buffer, source_text, rng);
            break;
          case CorruptionMode::TruncateRegion:
            corrupted_case = MakeTruncateRegionCase(clean_buffer, source_text,
                                                    clean_pairs, rng);
            break;
        }
        if (!corrupted_case) {
          continue;
        }
        std::string corrupted_text = std::move(corrupted_case->text);
        std::vector<DeletedToken> deleted_tokens =
            std::move(corrupted_case->expected);

        auto corrupted_source = SourceBuffer::MakeFromStringCopy(
            candidate.filename, corrupted_text, Diagnostics::NullConsumer());
        if (!corrupted_source) {
          continue;
        }

        SharedValueStores c_value_stores;
        LexOptions c_lex_options;
        c_lex_options.consumer = &Diagnostics::NullConsumer();
        llvm::SmallVector<BracketCorrection> corrections;
        c_lex_options.bracket_corrections = &corrections;
        auto corrupted_buffer =
            Lex::Lex(c_value_stores, *corrupted_source, c_lex_options);

        // Ground-truth "at EOF" offsets are computed as the corrupted text
        // size, but recovery inserts before the FileEnd token, whose offset
        // excludes trailing whitespace. Accept the real FileEnd offset too.
        if (mode == CorruptionMode::Truncate ||
            mode == CorruptionMode::TruncateRegion) {
          int32_t eof = corrupted_buffer.GetByteOffset(
              TokenIndex(corrupted_buffer.size() - 1));
          int32_t text_size = static_cast<int32_t>(corrupted_text.size());
          for (auto& del : deleted_tokens) {
            if (llvm::is_contained(del.valid_next_token_byte_offsets,
                                   text_size)) {
              del.valid_next_token_byte_offsets.push_back(eof);
            }
          }
        }

        // Closing the gap can fuse two tokens into one (e.g. `f(x)` -> `fx)`),
        // which both is unrealistic and leaves no boundary to reinsert the
        // bracket at. Detect this by checking each ground-truth insertion still
        // has a real token boundary to land on, and skip the trial if not.
        if (mode == CorruptionMode::DeleteGap ||
            mode == CorruptionMode::TruncateRegion) {
          llvm::DenseSet<int32_t> token_offsets;
          for (TokenIndex t : corrupted_buffer.tokens()) {
            if (!corrupted_buffer.IsRecoveryToken(t)) {
              token_offsets.insert(corrupted_buffer.GetByteOffset(t));
            }
          }
          token_offsets.insert(static_cast<int32_t>(corrupted_text.size()));
          bool merged = false;
          for (const auto& del : deleted_tokens) {
            if (llvm::none_of(
                    del.valid_next_token_byte_offsets,
                    [&](int32_t off) { return token_offsets.contains(off); })) {
              merged = true;
              break;
            }
          }
          if (merged) {
            ++merged_skips;
            continue;
          }
        }

        // Corrections name tokens of this buffer, so an insertion must name a
        // recovery token of the kind it inserted. The origin lookup below
        // relies on that, and a wrong index would silently lose origins.
        for (const auto& c : corrections) {
          CARBON_CHECK(c.fix_token_index.index >= 0 &&
                           c.fix_token_index.index < corrupted_buffer.size(),
                       "Correction names a token outside the buffer.");
          if (c.fix_action != BracketFixAction::ReplaceWithError &&
              !c.is_tied) {
            CARBON_CHECK(corrupted_buffer.IsRecoveryToken(c.fix_token_index),
                         "Insertion doesn't name the token it inserted.");
            CARBON_CHECK(
                corrupted_buffer.GetKind(c.fix_token_index) == c.fix_token_kind,
                "Inserted token has the wrong kind.");
          }
        }

        llvm::SmallVector<Suggestion> suggestions;
        for (TokenIndex t : corrupted_buffer.tokens()) {
          if (corrupted_buffer.IsRecoveryToken(t)) {
            auto kind = corrupted_buffer.GetKind(t);
            if (kind.is_opening_symbol() || kind.is_closing_symbol()) {
              // Structure-equality: a fix is identified by the first
              // *surviving* token it precedes, not its raw offset. Skip other
              // inserted (recovery) tokens so a cascade of closers all point at
              // the same real anchor, and closing among trailing whitespace or
              // a deleted span still resolves to the token that structurally
              // follows.
              TokenIndex succ = TokenIndex(t.index + 1);
              while (succ.index < corrupted_buffer.size() &&
                     corrupted_buffer.IsRecoveryToken(succ)) {
                succ = TokenIndex(succ.index + 1);
              }
              int32_t byte_off =
                  (succ.index < corrupted_buffer.size())
                      ? corrupted_buffer.GetByteOffset(succ)
                      : static_cast<int32_t>(corrupted_text.size());
              int32_t line_num = (succ.index < corrupted_buffer.size())
                                     ? corrupted_buffer.GetLineNumber(succ)
                                     : -1;
              int32_t col_num = (succ.index < corrupted_buffer.size())
                                    ? corrupted_buffer.GetColumnNumber(succ)
                                    : -1;
              // Corrections name the tokens of this buffer, so the one that
              // inserted this token names it directly.
              std::string origin = "Unknown";
              for (const auto& c : corrections) {
                if (c.fix_action != BracketFixAction::ReplaceWithError &&
                    !c.is_tied && c.fix_token_index == t) {
                  origin = c.origin;
                  break;
                }
              }
              suggestions.push_back(Suggestion{
                  .kind = kind,
                  .byte_offset = byte_off,
                  .line = line_num,
                  .column = col_num,
                  .origin = origin,
              });
            }
          }
        }

        if (spec.label == "1" || mode == CorruptionMode::Truncate ||
            mode == CorruptionMode::TruncateRegion) {
          for (const auto& s : suggestions) {
            bool matched = false;
            for (const auto& del : deleted_tokens) {
              if (MatchesDeletedToken(del, s)) {
                matched = true;
                break;
              }
            }
            if (matched) {
              ++origin_stats[s.origin].correct;
            } else {
              ++origin_stats[s.origin].incorrect;
            }
          }
        }

        TestClassification classification =
            ClassifyTrial(deleted_tokens, suggestions);

        // Measure how far off each unmatched expected close is.
        if (classification == TestClassification::Incorrect) {
          llvm::DenseMap<int32_t, int32_t> off_to_tok;
          for (TokenIndex t : corrupted_buffer.tokens()) {
            off_to_tok[corrupted_buffer.GetByteOffset(t)] = t.index;
          }
          int32_t eof_idx = corrupted_buffer.size() - 1;
          auto to_tok = [&](int32_t off) -> int32_t {
            auto it = off_to_tok.find(off);
            return it != off_to_tok.end() ? it->second : eof_idx;
          };
          for (const auto& del : deleted_tokens) {
            bool matched = false;
            for (const auto& s : suggestions) {
              if (MatchesDeletedToken(del, s)) {
                matched = true;
                break;
              }
            }
            if (matched) {
              continue;
            }
            auto& stat = dist_by_kind[del.kind.name().str()];
            // Find the nearest same-kind close.
            int32_t exp_tok = to_tok(del.next_token_byte_offset);
            const Suggestion* best = nullptr;
            for (const auto& s : suggestions) {
              if (s.kind != del.kind) {
                continue;
              }
              if (best == nullptr ||
                  std::abs(to_tok(s.byte_offset) - exp_tok) <
                      std::abs(to_tok(best->byte_offset) - exp_tok)) {
                best = &s;
              }
            }
            if (best == nullptr) {
              ++stat.no_close;
              continue;
            }
            int32_t token_dist = to_tok(best->byte_offset) - exp_tok;
            (token_dist > 0 ? stat.later : stat.earlier)++;
            ++stat.token_dist_hist[token_dist];
            ++stat.line_dist_hist[best->line -
                                  corrupted_buffer.GetLineNumber(
                                      TokenIndex(std::min(exp_tok, eof_idx)))];
          }
        }

        bool do_dump = false;
        const char* dump_label = "";
        if (classification == TestClassification::Incorrect &&
            dump_incorrect > 0) {
          --dump_incorrect;
          do_dump = true;
          dump_label = "INCORRECT";
        } else if ((classification == TestClassification::None ||
                    classification == TestClassification::Partial) &&
                   dump_none > 0) {
          --dump_none;
          do_dump = true;
          dump_label =
              classification == TestClassification::None ? "NONE" : "PARTIAL";
        }
        if (do_dump) {
          llvm::errs() << "\n=== " << dump_label << " TRIAL in "
                       << candidate.filename << " (D=" << spec.label
                       << ") ===\n";
          for (const auto& del : deleted_tokens) {
            llvm::errs() << "  Deleted token: kind=" << del.kind.name()
                         << " at byte=" << del.byte_offset
                         << " (line=" << del.line << ", col=" << del.column
                         << ")\n";
          }
          llvm::errs() << "  Suggestions (" << suggestions.size() << "):\n";
          for (const auto& s : suggestions) {
            llvm::errs() << "    Suggestion (" << s.origin
                         << "): kind=" << s.kind.name()
                         << " at byte=" << s.byte_offset << " (line=" << s.line
                         << ", col=" << s.column << ")\n";
          }
          llvm::errs() << "  Raw corrections (" << corrections.size() << "):\n";
          for (const auto& c : corrections) {
            llvm::errs() << "    "
                         << (c.fix_action == BracketFixAction::InsertBefore
                                 ? "InsertBefore"
                             : c.fix_action == BracketFixAction::InsertAfter
                                 ? "InsertAfter"
                                 : "ReplaceWithError")
                         << " kind=" << c.fix_token_kind.name()
                         << " tok=" << c.fix_token_index.index
                         << (c.is_tied ? " TIED" : "") << " origin=" << c.origin
                         << "\n";
          }
          llvm::errs() << "--- Corrupted Text Sample ---\n";
          int32_t print_start =
              std::max(0, deleted_tokens[0].byte_offset - 100);
          int32_t print_end = std::min<int32_t>(
              corrupted_text.size(), deleted_tokens[0].byte_offset + 100);
          llvm::errs() << corrupted_text.substr(print_start,
                                                print_end - print_start)
                       << "\n";
          llvm::errs() << "===============================\n\n";
        }
        f_res.scenario_stats[s_idx].Add(classification);
        overall_scenario_stats[s_idx].Add(classification);
      }
    }

    file_results.push_back(std::move(f_res));
  }

  if (json_output) {
    llvm::outs() << "{\n";
    llvm::outs() << "  \"seed\": " << base_seed << ",\n";
    llvm::outs() << "  \"total_trials\": " << total_trials << ",\n";
    llvm::outs() << "  \"files_tested\": " << valid_files.size() << ",\n";
    llvm::outs() << "  \"total_bracket_pairs\": " << total_clean_pairs << ",\n";
    llvm::outs() << "  \"scenarios\": [\n";
    for (size_t i = 0; i < d_specs.size(); ++i) {
      const auto& spec = d_specs[i];
      const auto& stats = overall_scenario_stats[i];
      llvm::outs() << "    {\n";
      llvm::outs() << "      \"d_spec\": \"" << spec.label << "\",\n";
      llvm::outs() << "      \"total\": " << stats.total << ",\n";
      llvm::outs() << "      \"correct\": " << stats.correct << ",\n";
      llvm::outs() << "      \"partial\": " << stats.partial << ",\n";
      llvm::outs() << "      \"none\": " << stats.none << ",\n";
      llvm::outs() << "      \"incorrect\": " << stats.incorrect << ",\n";
      llvm::outs() << llvm::formatv("      \"correct_pct\": {0:F1},\n",
                                    stats.CorrectPct());
      llvm::outs() << llvm::formatv("      \"partial_pct\": {0:F1},\n",
                                    stats.PartialPct());
      llvm::outs() << llvm::formatv("      \"none_pct\": {0:F1},\n",
                                    stats.NonePct());
      llvm::outs() << llvm::formatv("      \"incorrect_pct\": {0:F1},\n",
                                    stats.IncorrectPct());
      llvm::outs() << llvm::formatv("      \"safety_pct\": {0:F1},\n",
                                    stats.SafetyPct());
      llvm::outs() << llvm::formatv("      \"accuracy_pct\": {0:F1}\n",
                                    stats.AccuracyPct());
      llvm::outs() << (i + 1 < d_specs.size() ? "    },\n" : "    }\n");
    }
    llvm::outs() << "  ]\n";
    llvm::outs() << "}\n";
    return true;
  }

  // Markdown Report Output
  llvm::outs() << "# Bracket Recovery Measurement Report\n\n";
  llvm::outs() << "- **Corruption mode**: " << mode_str << "\n";
  llvm::outs() << "- **Files tested**: " << valid_files.size() << " files ("
               << total_clean_pairs << " clean matched bracket pairs)\n";
  llvm::outs() << "- **Total trials per configuration**: " << total_trials
               << "\n";
  llvm::outs() << "- **Random seed**: " << base_seed << "\n";
  if (merged_skips > 0) {
    llvm::outs() << "- **Trials skipped (token fusion)**: " << merged_skips
                 << "\n";
  }
  llvm::outs() << "\n";

  llvm::outs() << "## Overall Performance by Deletion Level (D)\n\n";
  llvm::outs() << "| Deletion Level (D) | Total Trials | Correct | Partial | "
                  "None | Incorrect | Safety (%) | Accuracy (%) |\n";
  llvm::outs() << "|:---|---:|---:|---:|---:|---:|---:|---:|\n";

  for (size_t i = 0; i < d_specs.size(); ++i) {
    const auto& spec = d_specs[i];
    const auto& stats = overall_scenario_stats[i];
    llvm::outs() << llvm::formatv(
        "| D = {0,-6} | {1,12} | {2,5} ({3,4:F1}%) | {4,5} ({5,4:F1}%) | {6,5} "
        "({7,4:F1}%) | {8,5} ({9,4:F1}%) | {10,9:F1}% | {11,11:F1}% |\n",
        spec.label, stats.total, stats.correct, stats.CorrectPct(),
        stats.partial, stats.PartialPct(), stats.none, stats.NonePct(),
        stats.incorrect, stats.IncorrectPct(), stats.SafetyPct(),
        stats.AccuracyPct());
  }
  llvm::outs() << "\n";

  llvm::outs() << "## Suggestion Origin Breakdown (D = 1)\n\n";
  llvm::outs() << "| Origin Transition | Total | Correct | Incorrect | "
                  "Precision (%) |\n";
  llvm::outs() << "|:---|---:|---:|---:|---:|\n";
  for (const auto& [name, stat] : origin_stats) {
    int total = stat.correct + stat.incorrect;
    double prec = total == 0 ? 100.0 : (100.0 * stat.correct) / total;
    llvm::outs() << llvm::formatv(
        "| {0,-32} | {1,5} | {2,5} | {3,5} | {4,8:F1}% |\n", name, total,
        stat.correct, stat.incorrect, prec);
  }
  llvm::outs() << "\n";

  if (verbose) {
    llvm::outs() << "## Per-File Breakdown\n\n";
    for (const auto& fres : file_results) {
      llvm::outs() << "### `" << fres.filename << "` ("
                   << fres.clean_pairs_count << " pairs)\n\n";
      llvm::outs() << "| D | Total | Correct | Partial | None | Incorrect | "
                      "Safety | Accuracy |\n";
      llvm::outs() << "|:---|---:|---:|---:|---:|---:|---:|---:|\n";
      for (size_t i = 0; i < d_specs.size(); ++i) {
        const auto& spec = d_specs[i];
        const auto& stats = fres.scenario_stats[i];
        llvm::outs() << llvm::formatv(
            "| {0} | {1} | {2} ({3:F1}%) | {4} ({5:F1}%) | {6} ({7:F1}%) | {8} "
            "({9:F1}%) | {10:F1}% | {11:F1}% |\n",
            spec.label, stats.total, stats.correct, stats.CorrectPct(),
            stats.partial, stats.PartialPct(), stats.none, stats.NonePct(),
            stats.incorrect, stats.IncorrectPct(), stats.SafetyPct(),
            stats.AccuracyPct());
      }
      llvm::outs() << "\n";
    }
  }

  if (!dist_by_kind.empty()) {
    llvm::outs() << "## Wrong-Close Distance (incorrect trials)\n\n";
    llvm::outs()
        << "Signed token distance from the correct anchor to the "
           "nearest same-kind close (+ = closed later / swallowing).\n\n";
    llvm::outs() << "| Deleted kind | Wrong | Later | Earlier | No close | "
                    "Median | P90 | Max |\n";
    llvm::outs() << "|:---|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const auto& [kind, stat] : dist_by_kind) {
      int total = stat.later + stat.earlier + stat.no_close;
      // Median token distance.
      int median = 0;
      int seen = 0;
      int half = (stat.later + stat.earlier) / 2;
      for (const auto& [d, c] : stat.token_dist_hist) {
        seen += c;
        if (seen > half) {
          median = d;
          break;
        }
      }
      // 90th percentile and max token distance.
      int p90 = 0;
      int ninety = (stat.later + stat.earlier) * 9 / 10;
      seen = 0;
      int max_dist = 0;
      for (const auto& [d, c] : stat.token_dist_hist) {
        seen += c;
        if (seen <= ninety) {
          p90 = d;
        }
        max_dist = std::max(max_dist, std::abs(d));
      }
      llvm::outs() << llvm::formatv(
          "| {0,-18} | {1,5} | {2,5} | {3,7} | {4,8} | {5,6} | {6,4} | {7,4} "
          "|\n",
          kind, total, stat.later, stat.earlier, stat.no_close, median, p90,
          max_dist);
    }
    llvm::outs() << "\n";
  }

  llvm::outs() << "## Metric Definitions\n\n";
  llvm::outs()
      << "- **Correct**: Suggested correct locations for all removed tokens.\n";
  llvm::outs() << "- **Partial**: Suggested correct locations for some removed "
                  "tokens, and gave no suggestions for others.\n";
  llvm::outs()
      << "- **None**: Gave no suggestions for any removed tokens (e.g., "
         "recovered cleanly with errors and no hallucinated notes).\n";
  llvm::outs() << "- **Incorrect**: Suggested a location for any removed token "
                  "that was not where the token was removed from.\n";
  llvm::outs() << "- **Safety**: Percentage of trials with no incorrect "
                  "suggestions `(Correct + Partial + None) / Total`.\n";
  llvm::outs() << "- **Accuracy**: Precision of suggestions when suggestions "
                  "were made `Correct / (Correct + Incorrect)`.\n";

  return true;
}

}  // namespace
}  // namespace Carbon::Lex

auto main(int argc, char** argv) -> int {
  Carbon::InitLLVM init_llvm(argc, argv);
  Carbon::SetWorkingDirForBazelRun();
  llvm::SmallVector<llvm::StringRef> args(argv + 1, argv + argc);
  bool success = Carbon::Lex::Run(args);
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
