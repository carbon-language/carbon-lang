// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <map>
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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
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

// How well recovery did on one trial. See `PrintMetricDefinitions` for what
// each means.
namespace {
enum class TestClassification {
  Correct,
  Partial,
  None,
  Incorrect,
};
}  // namespace

// Lex options that discard diagnostics, since a trial only cares about the
// tokens and the corrections.
static auto QuietLexOptions() -> LexOptions {
  LexOptions options;
  options.consumer = &Diagnostics::NullConsumer();
  return options;
}

namespace {

// A deletion level from `--d-values`: how many brackets each trial deletes,
// either as an absolute count or as a percentage of the file's clean pairs.
struct DSpec {
  // The level as written on the command line, used to label it in the report.
  std::string label;
  bool is_percent = false;
  double percent_val = 0.0;
  int count_val = 0;

  // How many of a file's `num_clean_pairs` pairs a trial deletes, at least one
  // and never more than the file has.
  auto DeletionCount(int num_clean_pairs) const -> int {
    int count =
        is_percent
            ? std::max(1, static_cast<int>(num_clean_pairs * percent_val))
            : count_val;
    return std::min(count, num_clean_pairs);
  }
};

// Trial counts by classification, for one file and deletion level or for the
// whole run.
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

  auto CorrectPct() const -> double { return Pct(correct); }
  auto PartialPct() const -> double { return Pct(partial); }
  auto NonePct() const -> double { return Pct(none); }
  auto IncorrectPct() const -> double { return Pct(incorrect); }

  // Percentage of trials with no incorrect suggestion.
  auto SafetyPct() const -> double { return Pct(correct + partial + none); }

  // Precision of the suggestions that were made.
  auto AccuracyPct() const -> double {
    int decisive = correct + incorrect;
    return decisive == 0 ? 100.0 : (100.0 * correct) / decisive;
  }

 private:
  auto Pct(int count) const -> double {
    return total == 0 ? 0.0 : (100.0 * count) / total;
  }
};

struct BracketPair {
  TokenIndex open_token;
  TokenIndex close_token;
};

// One bracket the corruption removed, and where recovery would have to put it
// back for that to count as correct. Offsets are in the corrupted text.
struct DeletedToken {
  TokenKind kind;
  int32_t byte_offset;
  int32_t length;
  int32_t line;
  int32_t column;
  // The offset of the first token that survived after this one.
  int32_t next_token_byte_offset;
  // Every offset an insertion could name and still be the same repair: the
  // surviving members of the run of identical brackets this one belonged to,
  // plus `next_token_byte_offset`.
  llvm::SmallVector<int32_t, 4> valid_next_token_byte_offsets;
};

// One bracket recovery inserted, located by the first surviving token it
// precedes.
struct Suggestion {
  TokenKind kind;
  int32_t byte_offset;
  int32_t line;
  int32_t column;
  // The name of the rule that inserted this bracket. See
  // `BracketCorrection::rule_name`.
  llvm::StringRef rule_name;
};

}  // namespace

static auto MatchesDeletedToken(const DeletedToken& del, const Suggestion& sugg)
    -> bool {
  if (del.kind != sugg.kind) {
    return false;
  }
  return llvm::is_contained(del.valid_next_token_byte_offsets,
                            sugg.byte_offset);
}

// Whether any deleted bracket is the one `sugg` puts back, and the same
// question from the other side.
static auto AnyDeletionMatches(llvm::ArrayRef<DeletedToken> deleted,
                               const Suggestion& sugg) -> bool {
  return llvm::any_of(deleted, [&](const DeletedToken& del) {
    return MatchesDeletedToken(del, sugg);
  });
}
static auto AnySuggestionMatches(llvm::ArrayRef<Suggestion> suggestions,
                                 const DeletedToken& del) -> bool {
  return llvm::any_of(suggestions, [&](const Suggestion& sugg) {
    return MatchesDeletedToken(del, sugg);
  });
}

// Parses the `--d-values` list. Entries that don't parse are dropped; the
// caller errors out if nothing is left.
static auto ParseDSpecs(llvm::StringRef str) -> llvm::SmallVector<DSpec> {
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

// The bracket pairs a clean file matched up, which are the pairs a trial can
// delete an endpoint of.
static auto GetCleanBracketPairs(const TokenizedBuffer& buffer)
    -> llvm::SmallVector<BracketPair> {
  llvm::SmallVector<BracketPair> pairs;
  for (TokenIndex t : buffer.tokens()) {
    if (!buffer.GetKind(t).is_opening_symbol()) {
      continue;
    }
    TokenIndex close = buffer.GetMatchedClosingToken(t);
    if (close != TokenIndex::None) {
      pairs.push_back({.open_token = t, .close_token = close});
    }
  }
  return pairs;
}

// Scores one trial: every suggestion must restore a distinct deleted bracket,
// and a suggestion that restores none makes the whole trial incorrect.
static auto ClassifyTrial(llvm::ArrayRef<DeletedToken> deleted_tokens,
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

namespace {

// How a clean file is corrupted for a trial. See `--mode` help for details.
enum class CorruptionMode {
  // Blank each deleted bracket with a space (byte offsets preserved).
  Blank,
  // Delete each bracket character, closing the gap so that no space is left
  // behind. More realistic, and doesn't leave the whitespace artifacts the
  // algorithm can key on.
  Gapless,
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

}  // namespace

// Whether `mode` corrupts by cutting the file short rather than by deleting
// individual brackets. Such a mode ignores the deletion level, and its
// ground-truth insertions land at the new end of the file.
static auto IsTruncateMode(CorruptionMode mode) -> bool {
  return mode == CorruptionMode::Truncate ||
         mode == CorruptionMode::TruncateRegion;
}

// Remaps a byte offset from original to corrupted coordinates, given sorted,
// disjoint deleted ranges [begin, end). An offset inside a deleted range maps
// to where the gap closes.
static auto RemapOffset(int32_t off,
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
static auto RemoveRanges(llvm::StringRef text,
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
static auto MakeDeletionCase(const TokenizedBuffer& buffer,
                             llvm::StringRef source_text,
                             llvm::ArrayRef<BracketPair> pairs, int d_count,
                             bool close_gap, std::mt19937_64& rng)
    -> std::optional<CorruptedCase> {
  CARBON_CHECK(d_count <= static_cast<int>(pairs.size()),
               "Asked to delete more pairs than the file has.");
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

    // Any surviving bracket in a run of identical ones is an equally good place
    // to reinsert this one, so find the whole run.
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
    auto succ = TokenIndex(run_end + 1);
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
static auto MakeTruncateCase(const TokenizedBuffer& buffer,
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
  for (int32_t i : llvm::seq(0, cut)) {
    auto kind = buffer.GetKind(TokenIndex(i));
    if (kind.is_opening_symbol()) {
      stack.push_back(kind);
    } else if (kind.is_closing_symbol() && !stack.empty() &&
               stack.back().closing_symbol() == kind) {
      stack.pop_back();
    }
  }

  auto eof = static_cast<int32_t>(corrupted.size());
  std::vector<DeletedToken> expected;
  for (TokenKind open_kind : llvm::reverse(stack)) {
    expected.push_back(DeletedToken{
        .kind = open_kind.closing_symbol(),
        .byte_offset = eof,
        .length = 1,
        .line = -1,
        .column = -1,
        .next_token_byte_offset = eof,
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
static auto MakeTruncateRegionCase(const TokenizedBuffer& buffer,
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
  for (int32_t i : llvm::seq(open + 1, close + 1)) {
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
      .valid_next_token_byte_offsets = {succ_corrupted},
  }};
  return CorruptedCase{.text = std::move(corrupted),
                       .expected = std::move(expected)};
}

// Corrupts a clean file for one trial, per `mode`. Returns nullopt if the file
// has nothing this mode can corrupt.
static auto MakeCorruptedCase(CorruptionMode mode,
                              const TokenizedBuffer& buffer,
                              llvm::StringRef source_text,
                              llvm::ArrayRef<BracketPair> pairs, int d_count,
                              std::mt19937_64& rng)
    -> std::optional<CorruptedCase> {
  switch (mode) {
    case CorruptionMode::Blank:
    case CorruptionMode::Gapless:
      return MakeDeletionCase(buffer, source_text, pairs, d_count,
                              /*close_gap=*/mode == CorruptionMode::Gapless,
                              rng);
    case CorruptionMode::Truncate:
      return MakeTruncateCase(buffer, source_text, rng);
    case CorruptionMode::TruncateRegion:
      return MakeTruncateRegionCase(buffer, source_text, pairs, rng);
  }
}

// Accepts the real `FileEnd` offset for any ground-truth insertion at the end
// of the file. Recovery inserts before `FileEnd`, whose offset excludes
// trailing whitespace, whereas the ground truth was computed as the text size.
static auto AcceptFileEndOffset(const TokenizedBuffer& buffer,
                                llvm::StringRef text,
                                std::vector<DeletedToken>& deleted) -> void {
  int32_t eof = buffer.GetByteOffset(TokenIndex(buffer.size() - 1));
  auto text_size = static_cast<int32_t>(text.size());
  for (auto& del : deleted) {
    if (llvm::is_contained(del.valid_next_token_byte_offsets, text_size)) {
      del.valid_next_token_byte_offsets.push_back(eof);
    }
  }
}

// Whether closing the gap fused two tokens into one (e.g. `f(x)` -> `fx)`),
// leaving some ground-truth insertion with no token boundary to land on. That
// both is unrealistic and makes the trial unscoreable, so the caller skips it.
static auto TokensWereFused(const TokenizedBuffer& buffer, llvm::StringRef text,
                            llvm::ArrayRef<DeletedToken> deleted) -> bool {
  llvm::DenseSet<int32_t> token_offsets;
  for (TokenIndex t : buffer.tokens()) {
    if (!buffer.IsRecoveryToken(t)) {
      token_offsets.insert(buffer.GetByteOffset(t));
    }
  }
  token_offsets.insert(static_cast<int32_t>(text.size()));
  return llvm::any_of(deleted, [&](const DeletedToken& del) {
    return llvm::none_of(del.valid_next_token_byte_offsets, [&](int32_t off) {
      return token_offsets.contains(off);
    });
  });
}

// Checks the invariant the rule-name lookup relies on: corrections name tokens
// of the buffer recovery produced, so an insertion names the recovery token it
// inserted. A wrong index would silently lose rule names rather than fail.
static auto CheckCorrectionsNameRealTokens(
    const TokenizedBuffer& buffer,
    llvm::ArrayRef<BracketCorrection> corrections) -> void {
  for (const auto& c : corrections) {
    CARBON_CHECK(
        c.fix_token_index.index >= 0 && c.fix_token_index.index < buffer.size(),
        "Correction names a token outside the buffer.");
    if (c.fix_action != BracketFixAction::ReplaceWithError && !c.is_tied) {
      CARBON_CHECK(buffer.IsRecoveryToken(c.fix_token_index),
                   "Insertion doesn't name the token it inserted.");
      CARBON_CHECK(buffer.GetKind(c.fix_token_index) == c.fix_token_kind,
                   "Inserted token has the wrong kind.");
    }
  }
}

// Names the rule that inserted `token`. Corrections name the tokens of this
// buffer, so the one that inserted it names it directly. A tied correction was
// downgraded to an error token and has no rule to report.
static auto RuleNameOfInsertion(llvm::ArrayRef<BracketCorrection> corrections,
                                TokenIndex token) -> llvm::StringRef {
  for (const auto& c : corrections) {
    if (c.fix_action != BracketFixAction::ReplaceWithError && !c.is_tied &&
        c.fix_token_index == token) {
      return c.rule_name;
    }
  }
  return "Unknown";
}

// Turns the brackets recovery inserted into scoreable suggestions.
//
// Structure-equality: a fix is identified by the first *surviving* token it
// precedes, not its raw offset. Other inserted (recovery) tokens are skipped,
// so a cascade of closers all point at the same real anchor, and closing among
// trailing whitespace or a deleted span still resolves to the token that
// structurally follows.
static auto CollectSuggestions(const TokenizedBuffer& buffer,
                               llvm::StringRef text,
                               llvm::ArrayRef<BracketCorrection> corrections)
    -> llvm::SmallVector<Suggestion> {
  llvm::SmallVector<Suggestion> suggestions;
  for (TokenIndex t : buffer.tokens()) {
    auto kind = buffer.GetKind(t);
    if (!buffer.IsRecoveryToken(t) ||
        !(kind.is_opening_symbol() || kind.is_closing_symbol())) {
      continue;
    }
    auto succ = TokenIndex(t.index + 1);
    while (succ.index < buffer.size() && buffer.IsRecoveryToken(succ)) {
      succ = TokenIndex(succ.index + 1);
    }
    bool has_succ = succ.index < buffer.size();
    suggestions.push_back(Suggestion{
        .kind = kind,
        .byte_offset = has_succ ? buffer.GetByteOffset(succ)
                                : static_cast<int32_t>(text.size()),
        .line = has_succ ? buffer.GetLineNumber(succ) : -1,
        .column = has_succ ? buffer.GetColumnNumber(succ) : -1,
        .rule_name = RuleNameOfInsertion(corrections, t),
    });
  }
  return suggestions;
}

namespace {

// A file worth testing: it lexes cleanly and has at least one matched pair, so
// a trial can delete a bracket from it.
struct CandidateFile {
  std::string filename;
  int clean_pairs_count = 0;
};

// The corpus a run evaluates.
struct Corpus {
  llvm::SmallVector<CandidateFile> files;
  int total_clean_pairs = 0;
};

}  // namespace

static auto CollectCarbonFiles(llvm::ArrayRef<llvm::StringRef> input_paths)
    -> llvm::SmallVector<std::string> {
  llvm::SmallVector<std::string> files;

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
      } else if (path.ends_with(".carbon")) {
        files.push_back(path.str());
      }
    }
  }

  llvm::sort(files);
  files.erase(std::unique(files.begin(), files.end()), files.end());
  return files;
}

// Lexes each of `files` and keeps the ones a trial can be built from, which is
// what makes the corpus. A file that fails to lex cleanly can't serve as ground
// truth, and one with no matched pairs has no bracket to delete.
static auto FindCandidateFiles(llvm::ArrayRef<std::string> files) -> Corpus {
  Corpus corpus;
  for (const auto& filepath : files) {
    auto source = SourceBuffer::MakeFromFile(
        *llvm::vfs::getRealFileSystem(), filepath, Diagnostics::NullConsumer());
    if (!source) {
      continue;
    }

    SharedValueStores value_stores;
    auto clean_buffer = Lex::Lex(value_stores, *source, QuietLexOptions());
    if (clean_buffer.has_errors()) {
      continue;
    }

    auto clean_pairs = GetCleanBracketPairs(clean_buffer);
    if (clean_pairs.empty()) {
      continue;
    }

    auto num_pairs = static_cast<int>(clean_pairs.size());
    corpus.total_clean_pairs += num_pairs;
    corpus.files.push_back(
        {.filename = filepath, .clean_pairs_count = num_pairs});
  }
  return corpus;
}

// Apportions `total_trials` across the corpus for one deletion level, returning
// each file's share.
//
// A percentage level tests every file equally; an absolute level weights each
// file by its clean pair count, so that every pair in the corpus is equally
// likely to be picked. Fractional quotas are handed out
// largest-remainder-first, with ties broken by filename to keep the allocation
// deterministic.
static auto AllocateTrials(const Corpus& corpus, const DSpec& spec,
                           int total_trials) -> std::vector<int> {
  std::vector<int> allocation(corpus.files.size(), 0);
  if (total_trials <= 0) {
    return allocation;
  }

  double total_weight = spec.is_percent
                            ? static_cast<double>(corpus.files.size())
                            : static_cast<double>(corpus.total_clean_pairs);
  int allocated = 0;
  std::vector<std::pair<double, size_t>> remainders;
  remainders.reserve(corpus.files.size());
  for (auto [i, file] : llvm::enumerate(corpus.files)) {
    double weight =
        spec.is_percent ? 1.0 : static_cast<double>(file.clean_pairs_count);
    double exact_quota =
        static_cast<double>(total_trials) * weight / total_weight;
    auto base_count = static_cast<int>(exact_quota);
    allocation[i] = base_count;
    allocated += base_count;
    remainders.push_back({exact_quota - base_count, i});
  }

  std::sort(remainders.begin(), remainders.end(),
            [&](const auto& a, const auto& b) {
              if (a.first != b.first) {
                return a.first > b.first;
              }
              return corpus.files[a.second].filename <
                     corpus.files[b.second].filename;
            });
  int remainder_trials = total_trials - allocated;
  for (int i = 0;
       i < remainder_trials && i < static_cast<int>(remainders.size()); ++i) {
    ++allocation[remainders[i].second];
  }
  return allocation;
}

namespace {

struct FileResult {
  std::string filename;
  int clean_pairs_count = 0;
  // Indexed as the deletion levels.
  std::vector<TrialStats> stats_by_level;
};

// How often a named rule was right.
struct RuleStat {
  int correct = 0;
  int incorrect = 0;
};

// For incorrect trials: how far (in tokens, signed; + = closed later /
// swallowing following code) the wrong close is from the correct anchor.
struct DistStat {
  int later = 0;
  int earlier = 0;
  int no_close = 0;
  std::map<int, int> token_dist_hist;
};

// The shape of a distance histogram, for the report.
struct DistanceSummary {
  int median = 0;
  int p90 = 0;
  int max_abs = 0;
};

// Everything the trials measured.
struct Report {
  // Indexed as the deletion levels.
  std::vector<TrialStats> stats_by_level;
  std::vector<FileResult> results_by_file;
  std::map<llvm::StringRef, RuleStat> stats_by_rule;
  std::map<std::string, DistStat> wrong_close_by_kind;
  // Trials skipped because closing the gap fused two tokens.
  int merged_skips = 0;
};

// The evaluation configuration. `d_values` and `mode_name` hold what the
// command line said; `d_specs` and `mode` are the parsed forms that `Resolve`
// fills in.
struct EvalOptions {
  llvm::SmallVector<llvm::StringRef> input_files;
  llvm::StringRef d_values = "1,2,5,10%,25%";
  llvm::StringRef mode_name = "blank";
  int total_trials = 1000;
  int base_seed = 42;
  bool verbose = false;
  bool json_output = false;
  int dump_incorrect = 0;
  int dump_none = 0;

  llvm::SmallVector<DSpec> d_specs;
  CorruptionMode mode = CorruptionMode::Blank;

  // Parses `d_values` and `mode_name`, reporting to stderr and returning false
  // if either is invalid.
  auto Resolve() -> bool;

  // The one deletion level the rule table reports on, so that its precisions
  // aren't a blend of easy and hard configurations. The truncate modes have
  // only the single level; otherwise it's D=1, if that was asked for.
  auto RuleLevelLabel() const -> llvm::StringRef {
    return IsTruncateMode(mode) ? llvm::StringRef(d_specs.front().label) : "1";
  }
};

}  // namespace

auto EvalOptions::Resolve() -> bool {
  d_specs = ParseDSpecs(d_values);
  if (d_specs.empty()) {
    llvm::errs() << "error: No valid D deletion specifications provided.\n";
    return false;
  }

  if (mode_name == "blank") {
    mode = CorruptionMode::Blank;
  } else if (mode_name == "gapless") {
    mode = CorruptionMode::Gapless;
  } else if (mode_name == "truncate") {
    mode = CorruptionMode::Truncate;
  } else if (mode_name == "truncate-region") {
    mode = CorruptionMode::TruncateRegion;
  } else {
    llvm::errs() << "error: Unknown --mode '" << mode_name << "'.\n";
    return false;
  }

  // The truncate modes don't delete a set number of brackets, so collapse the
  // D configurations to a single pass.
  if (IsTruncateMode(mode)) {
    d_specs.resize(1);
    d_specs[0].label = mode_name.str();
  }
  return true;
}

// The seed for one trial, mixed from the file, level, and trial number so that
// any single trial can be reproduced on its own.
static auto TrialSeed(int base_seed, const std::string& filename,
                      const std::string& spec_label, int trial) -> uint64_t {
  uint64_t file_hash = llvm::hash_value(filename);
  uint64_t spec_hash = llvm::hash_value(spec_label);
  return static_cast<uint64_t>(base_seed) ^ file_hash ^ (spec_hash << 16) ^
         (static_cast<uint64_t>(trial) * 0x9e3779b97f4a7c15ULL);
}

namespace {

// Runs the trials over a corpus and accumulates the measurements the report is
// built from.
class Evaluator {
 public:
  Evaluator(const EvalOptions& options, const Corpus& corpus)
      : options_(options),
        corpus_(corpus),
        dump_incorrect_(options.dump_incorrect),
        dump_none_(options.dump_none) {}

  // Runs every trial and returns what they measured.
  auto Run() -> Report;

 private:
  // The clean file a trial corrupts, lexed once and shared by all its trials.
  struct FileContext {
    const CandidateFile& candidate;
    const TokenizedBuffer& clean_buffer;
    llvm::StringRef source_text;
    llvm::ArrayRef<BracketPair> clean_pairs;
  };

  // Runs every trial allocated to one file, appending its statistics to the
  // report.
  auto RunFile(size_t file_index, const CandidateFile& candidate) -> void;

  // Runs one trial, returning how it scored, or nullopt if it had to be
  // skipped.
  auto RunTrial(const FileContext& file, const DSpec& spec, int d_count,
                std::mt19937_64& rng) -> std::optional<TestClassification>;

  // Credits or blames the rule behind each suggestion, for the rule table.
  auto RecordRuleNames(llvm::ArrayRef<DeletedToken> deleted_tokens,
                       llvm::ArrayRef<Suggestion> suggestions) -> void;

  // Records, for each deleted bracket recovery failed to restore, how far the
  // nearest same-kind close it did suggest is from where the bracket belonged.
  auto RecordWrongCloseDistances(const TokenizedBuffer& buffer,
                                 llvm::ArrayRef<DeletedToken> deleted_tokens,
                                 llvm::ArrayRef<Suggestion> suggestions)
      -> void;

  // Prints a trial in full if the `--dump-*` budget for its classification
  // allows, spending one from that budget.
  auto MaybeDumpTrial(const FileContext& file, const DSpec& spec,
                      TestClassification classification,
                      llvm::StringRef corrupted_text,
                      llvm::ArrayRef<DeletedToken> deleted_tokens,
                      llvm::ArrayRef<Suggestion> suggestions,
                      llvm::ArrayRef<BracketCorrection> corrections) -> void;

  const EvalOptions& options_;
  const Corpus& corpus_;
  // Remaining `--dump-incorrect` and `--dump-none` budget.
  int dump_incorrect_;
  int dump_none_;
  // Trial counts indexed by [deletion level][file].
  std::vector<std::vector<int>> trials_;
  Report report_;
};

}  // namespace

auto Evaluator::Run() -> Report {
  report_.stats_by_level.resize(options_.d_specs.size());
  for (const DSpec& spec : options_.d_specs) {
    trials_.push_back(AllocateTrials(corpus_, spec, options_.total_trials));
  }
  for (auto [file_index, candidate] : llvm::enumerate(corpus_.files)) {
    RunFile(file_index, candidate);
  }
  return std::move(report_);
}

auto Evaluator::RunFile(size_t file_index, const CandidateFile& candidate)
    -> void {
  // Without `--verbose` a file with no trials has nothing to report, so don't
  // spend the time lexing it.
  bool has_any_trials =
      llvm::any_of(trials_, [&](const std::vector<int>& level_trials) {
        return level_trials[file_index] > 0;
      });
  if (!options_.verbose && !has_any_trials) {
    return;
  }

  auto source = SourceBuffer::MakeFromFile(*llvm::vfs::getRealFileSystem(),
                                           candidate.filename,
                                           Diagnostics::NullConsumer());
  if (!source) {
    return;
  }
  SharedValueStores value_stores;
  auto clean_buffer = Lex::Lex(value_stores, *source, QuietLexOptions());
  auto clean_pairs = GetCleanBracketPairs(clean_buffer);
  FileContext file = {.candidate = candidate,
                      .clean_buffer = clean_buffer,
                      .source_text = source->text(),
                      .clean_pairs = clean_pairs};

  FileResult result = {
      .filename = candidate.filename,
      .clean_pairs_count = candidate.clean_pairs_count,
      .stats_by_level = std::vector<TrialStats>(options_.d_specs.size())};

  for (auto [level, spec] : llvm::enumerate(options_.d_specs)) {
    int d_count = spec.DeletionCount(static_cast<int>(clean_pairs.size()));
    for (int trial : llvm::seq(0, trials_[level][file_index])) {
      std::mt19937_64 rng(
          TrialSeed(options_.base_seed, candidate.filename, spec.label, trial));
      auto classification = RunTrial(file, spec, d_count, rng);
      if (!classification) {
        continue;
      }
      result.stats_by_level[level].Add(*classification);
      report_.stats_by_level[level].Add(*classification);
    }
  }

  report_.results_by_file.push_back(std::move(result));
}

auto Evaluator::RunTrial(const FileContext& file, const DSpec& spec,
                         int d_count, std::mt19937_64& rng)
    -> std::optional<TestClassification> {
  auto corrupted_case =
      MakeCorruptedCase(options_.mode, file.clean_buffer, file.source_text,
                        file.clean_pairs, d_count, rng);
  if (!corrupted_case) {
    return std::nullopt;
  }
  std::string corrupted_text = std::move(corrupted_case->text);
  std::vector<DeletedToken> deleted_tokens =
      std::move(corrupted_case->expected);

  auto corrupted_source = SourceBuffer::MakeFromStringCopy(
      file.candidate.filename, corrupted_text, Diagnostics::NullConsumer());
  if (!corrupted_source) {
    return std::nullopt;
  }

  SharedValueStores value_stores;
  LexOptions lex_options = QuietLexOptions();
  llvm::SmallVector<BracketCorrection> corrections;
  lex_options.bracket_corrections = &corrections;
  auto buffer = Lex::Lex(value_stores, *corrupted_source, lex_options);

  if (IsTruncateMode(options_.mode)) {
    AcceptFileEndOffset(buffer, corrupted_text, deleted_tokens);
  }
  if ((options_.mode == CorruptionMode::Gapless ||
       options_.mode == CorruptionMode::TruncateRegion) &&
      TokensWereFused(buffer, corrupted_text, deleted_tokens)) {
    ++report_.merged_skips;
    return std::nullopt;
  }

  CheckCorrectionsNameRealTokens(buffer, corrections);
  auto suggestions = CollectSuggestions(buffer, corrupted_text, corrections);

  if (spec.label == options_.RuleLevelLabel()) {
    RecordRuleNames(deleted_tokens, suggestions);
  }

  TestClassification classification =
      ClassifyTrial(deleted_tokens, suggestions);
  if (classification == TestClassification::Incorrect) {
    RecordWrongCloseDistances(buffer, deleted_tokens, suggestions);
  }
  MaybeDumpTrial(file, spec, classification, corrupted_text, deleted_tokens,
                 suggestions, corrections);
  return classification;
}

auto Evaluator::RecordRuleNames(llvm::ArrayRef<DeletedToken> deleted_tokens,
                                llvm::ArrayRef<Suggestion> suggestions)
    -> void {
  for (const auto& sugg : suggestions) {
    auto& stat = report_.stats_by_rule[sugg.rule_name];
    ++(AnyDeletionMatches(deleted_tokens, sugg) ? stat.correct
                                                : stat.incorrect);
  }
}

auto Evaluator::RecordWrongCloseDistances(
    const TokenizedBuffer& buffer, llvm::ArrayRef<DeletedToken> deleted_tokens,
    llvm::ArrayRef<Suggestion> suggestions) -> void {
  // Ground truth and suggestions meet in byte offsets, but distances are
  // measured in tokens, so map back. An offset that isn't a token boundary is
  // treated as the end of the file.
  llvm::DenseMap<int32_t, int32_t> offset_to_token;
  for (TokenIndex t : buffer.tokens()) {
    offset_to_token[buffer.GetByteOffset(t)] = t.index;
  }
  int32_t eof_index = buffer.size() - 1;
  auto to_token = [&](int32_t offset) -> int32_t {
    auto it = offset_to_token.find(offset);
    return it != offset_to_token.end() ? it->second : eof_index;
  };

  for (const auto& del : deleted_tokens) {
    if (AnySuggestionMatches(suggestions, del)) {
      continue;
    }
    auto& stat = report_.wrong_close_by_kind[del.kind.name().str()];
    // Blame the nearest same-kind close, as the suggestion that most likely
    // ended the group in the wrong place.
    int32_t expected = to_token(del.next_token_byte_offset);
    const Suggestion* best = nullptr;
    for (const auto& sugg : suggestions) {
      if (sugg.kind != del.kind) {
        continue;
      }
      if (best == nullptr ||
          std::abs(to_token(sugg.byte_offset) - expected) <
              std::abs(to_token(best->byte_offset) - expected)) {
        best = &sugg;
      }
    }
    if (best == nullptr) {
      ++stat.no_close;
      continue;
    }
    int32_t token_dist = to_token(best->byte_offset) - expected;
    ++(token_dist > 0 ? stat.later : stat.earlier);
    ++stat.token_dist_hist[token_dist];
  }
}

auto Evaluator::MaybeDumpTrial(const FileContext& file, const DSpec& spec,
                               TestClassification classification,
                               llvm::StringRef corrupted_text,
                               llvm::ArrayRef<DeletedToken> deleted_tokens,
                               llvm::ArrayRef<Suggestion> suggestions,
                               llvm::ArrayRef<BracketCorrection> corrections)
    -> void {
  const char* dump_label = nullptr;
  if (classification == TestClassification::Incorrect && dump_incorrect_ > 0) {
    --dump_incorrect_;
    dump_label = "INCORRECT";
  } else if ((classification == TestClassification::None ||
              classification == TestClassification::Partial) &&
             dump_none_ > 0) {
    --dump_none_;
    dump_label =
        classification == TestClassification::None ? "NONE" : "PARTIAL";
  }
  if (dump_label == nullptr) {
    return;
  }

  llvm::errs() << llvm::formatv(
      R"(
=== {0} TRIAL in {1} (D={2}) ===
)",
      dump_label, file.candidate.filename, spec.label);
  for (const auto& del : deleted_tokens) {
    llvm::errs() << llvm::formatv(
        "  Deleted token: kind={0} at byte={1} (line={2}, col={3})\n",
        del.kind.name(), del.byte_offset, del.line, del.column);
  }
  llvm::errs() << llvm::formatv("  Suggestions ({0}):\n", suggestions.size());
  for (const auto& s : suggestions) {
    llvm::errs() << llvm::formatv(
        "    Suggestion ({0}): kind={1} at byte={2} (line={3}, col={4})\n",
        s.rule_name, s.kind.name(), s.byte_offset, s.line, s.column);
  }
  llvm::errs() << llvm::formatv("  Raw corrections ({0}):\n",
                                corrections.size());
  for (const auto& c : corrections) {
    llvm::StringRef action =
        c.fix_action == BracketFixAction::InsertBefore  ? "InsertBefore"
        : c.fix_action == BracketFixAction::InsertAfter ? "InsertAfter"
                                                        : "ReplaceWithError";
    llvm::errs() << llvm::formatv("    {0} kind={1} tok={2}{3} rule={4}\n",
                                  action, c.fix_token_kind.name(),
                                  c.fix_token_index.index,
                                  c.is_tied ? " TIED" : "", c.rule_name);
  }
  // Center the excerpt on the first deletion, or on the first suggestion when
  // there was nothing to delete (a truncation with nothing left open) yet
  // recovery suggested something anyway.
  int32_t center = 0;
  if (!deleted_tokens.empty()) {
    center = deleted_tokens.front().byte_offset;
  } else if (!suggestions.empty()) {
    center = suggestions.front().byte_offset;
  }
  int32_t print_start = std::max(0, center - 100);
  int32_t print_end =
      std::min(static_cast<int32_t>(corrupted_text.size()), center + 100);
  llvm::errs() << llvm::formatv(
      R"(--- Corrupted Text Sample ---
{0}
===============================

)",
      corrupted_text.substr(print_start, print_end - print_start));
}

static auto PrintJsonReport(const EvalOptions& options, const Corpus& corpus,
                            const Report& report) -> void {
  llvm::outs() << llvm::formatv(
      R"({{
  "seed": {0},
  "total_trials": {1},
  "files_tested": {2},
  "total_bracket_pairs": {3},
  "scenarios": [)",
      options.base_seed, options.total_trials, corpus.files.size(),
      corpus.total_clean_pairs);
  llvm::ListSeparator sep(",");
  for (auto [i, spec] : llvm::enumerate(options.d_specs)) {
    const auto& stats = report.stats_by_level[i];
    llvm::outs() << llvm::formatv(
        R"({0}
    {{
      "d_spec": "{1}",
      "total": {2},
      "correct": {3},
      "partial": {4},
      "none": {5},
      "incorrect": {6},
      "correct_pct": {7:F1},
      "partial_pct": {8:F1},
      "none_pct": {9:F1},
      "incorrect_pct": {10:F1},
      "safety_pct": {11:F1},
      "accuracy_pct": {12:F1}
    })",
        llvm::StringRef(sep), spec.label, stats.total, stats.correct,
        stats.partial, stats.none, stats.incorrect, stats.CorrectPct(),
        stats.PartialPct(), stats.NonePct(), stats.IncorrectPct(),
        stats.SafetyPct(), stats.AccuracyPct());
  }
  llvm::outs() << "\n  ]\n}\n";
}

static auto PrintLevelTable(const EvalOptions& options, const Report& report)
    -> void {
  llvm::outs() << R"(## Overall Performance by Deletion Level (D)

| Deletion Level (D) | Total Trials | Correct | Partial | None | Incorrect | Safety (%) | Accuracy (%) |
|:---|---:|---:|---:|---:|---:|---:|---:|
)";
  for (auto [i, spec] : llvm::enumerate(options.d_specs)) {
    const auto& stats = report.stats_by_level[i];
    llvm::outs() << llvm::formatv(
        "| D = {0,-6} | {1,12} | {2,5} ({3,4:F1}%) | {4,5} ({5,4:F1}%) | {6,5} "
        "({7,4:F1}%) | {8,5} ({9,4:F1}%) | {10,9:F1}% | {11,11:F1}% |\n",
        spec.label, stats.total, stats.correct, stats.CorrectPct(),
        stats.partial, stats.PartialPct(), stats.none, stats.NonePct(),
        stats.incorrect, stats.IncorrectPct(), stats.SafetyPct(),
        stats.AccuracyPct());
  }
  llvm::outs() << "\n";
}

static auto PrintRuleTable(const EvalOptions& options, const Report& report)
    -> void {
  llvm::outs() << llvm::formatv(
      R"(## Suggestion Rule Breakdown (D = {0})

| Rule | Total | Correct | Incorrect | Precision (%) |
|:---|---:|---:|---:|---:|
)",
      options.RuleLevelLabel());
  for (const auto& [name, stat] : report.stats_by_rule) {
    int total = stat.correct + stat.incorrect;
    double prec = total == 0 ? 100.0 : (100.0 * stat.correct) / total;
    llvm::outs() << llvm::formatv(
        "| {0,-32} | {1,5} | {2,5} | {3,5} | {4,8:F1}% |\n", name, total,
        stat.correct, stat.incorrect, prec);
  }
  llvm::outs() << "\n";
}

static auto PrintPerFileTables(const EvalOptions& options, const Report& report)
    -> void {
  llvm::outs() << "## Per-File Breakdown\n\n";
  for (const auto& file : report.results_by_file) {
    llvm::outs() << llvm::formatv(
        R"(### `{0}` ({1} pairs)

| D | Total | Correct | Partial | None | Incorrect | Safety | Accuracy |
|:---|---:|---:|---:|---:|---:|---:|---:|
)",
        file.filename, file.clean_pairs_count);
    for (auto [i, spec] : llvm::enumerate(options.d_specs)) {
      const auto& stats = file.stats_by_level[i];
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

// Summarizes a distance histogram. The median is the first distance past the
// halfway point and the 90th percentile the last one at or below the 90% mark,
// so both name an observed distance rather than an interpolated one.
static auto SummarizeDistances(const std::map<int, int>& hist)
    -> DistanceSummary {
  int count = 0;
  DistanceSummary summary;
  for (const auto& [dist, n] : hist) {
    count += n;
    summary.max_abs = std::max(summary.max_abs, std::abs(dist));
  }
  bool have_median = false;
  int seen = 0;
  for (const auto& [dist, n] : hist) {
    seen += n;
    if (!have_median && seen > count / 2) {
      summary.median = dist;
      have_median = true;
    }
    if (seen <= count * 9 / 10) {
      summary.p90 = dist;
    }
  }
  return summary;
}

static auto PrintWrongCloseTable(const Report& report) -> void {
  llvm::outs() << R"(## Wrong-Close Distance (incorrect trials)

Signed token distance from the correct anchor to the nearest same-kind close (+ = closed later / swallowing).

| Deleted kind | Wrong | Later | Earlier | No close | Median | P90 | Max |
|:---|---:|---:|---:|---:|---:|---:|---:|
)";
  for (const auto& [kind, stat] : report.wrong_close_by_kind) {
    int total = stat.later + stat.earlier + stat.no_close;
    DistanceSummary dist = SummarizeDistances(stat.token_dist_hist);
    llvm::outs() << llvm::formatv(
        "| {0,-18} | {1,5} | {2,5} | {3,7} | {4,8} | {5,6} | {6,4} | {7,4} "
        "|\n",
        kind, total, stat.later, stat.earlier, stat.no_close, dist.median,
        dist.p90, dist.max_abs);
  }
  llvm::outs() << "\n";
}

static auto PrintMetricDefinitions() -> void {
  llvm::outs() << R"(## Metric Definitions

- **Correct**: Suggested correct locations for all removed tokens.
- **Partial**: Suggested correct locations for some removed tokens, and gave no suggestions for others.
- **None**: Gave no suggestions for any removed tokens (e.g., recovered cleanly with errors and no hallucinated notes).
- **Incorrect**: Suggested a location for any removed token that was not where the token was removed from.
- **Safety**: Percentage of trials with no incorrect suggestions `(Correct + Partial + None) / Total`.
- **Accuracy**: Precision of suggestions when suggestions were made `Correct / (Correct + Incorrect)`.
)";
}

static auto PrintMarkdownReport(const EvalOptions& options,
                                const Corpus& corpus, const Report& report)
    -> void {
  llvm::outs() << llvm::formatv(
      R"(# Bracket Recovery Measurement Report

- **Corruption mode**: {0}
- **Files tested**: {1} files ({2} clean matched bracket pairs)
- **Total trials per configuration**: {3}
- **Random seed**: {4}
)",
      options.mode_name, corpus.files.size(), corpus.total_clean_pairs,
      options.total_trials, options.base_seed);
  if (report.merged_skips > 0) {
    llvm::outs() << llvm::formatv("- **Trials skipped (token fusion)**: {0}\n",
                                  report.merged_skips);
  }
  llvm::outs() << "\n";

  PrintLevelTable(options, report);
  PrintRuleTable(options, report);
  if (options.verbose) {
    PrintPerFileTables(options, report);
  }
  if (!report.wrong_close_by_kind.empty()) {
    PrintWrongCloseTable(report);
  }
  PrintMetricDefinitions();
}

constexpr CommandLine::CommandInfo CommandInfo = {
    .name = "mismatched_brackets_eval",
    .help = R"""(
A measurement and benchmarking tool for Carbon bracket recovery.

Evaluates how accurately and safely the bracket error recovery algorithm
recovers deleted subsets of brackets across Carbon source files.
)""",
};

static auto AddOptions(CommandLine::CommandBuilder& b, EvalOptions& options)
    -> void {
  b.AddStringPositionalArg(
      {
          .name = "FILE",
          .help = "Input Carbon source file(s) or directories to test.",
      },
      [&](auto& arg_b) { arg_b.Append(&options.input_files); });

  b.AddStringOption(
      {
          .name = "d-values",
          .value_name = "LIST",
          .help = "Comma-separated deletion levels (e.g. '1,2,5,10%,25%'). "
                  "Ignored by the truncate modes.",
      },
      [&](auto& arg_b) { arg_b.Set(&options.d_values); });

  b.AddStringOption(
      {
          .name = "mode",
          .value_name = "MODE",
          .help = "How to corrupt each file: 'blank' (replace brackets with "
                  "spaces; the default), 'gapless' (delete bracket characters, "
                  "leaving no space behind), 'truncate' (cut the file at a "
                  "random token; recovery should close all open brackets at "
                  "EOF), or 'truncate-region' (delete from inside a random "
                  "pair through its close, as when typing new code in an "
                  "existing class).",
      },
      [&](auto& arg_b) { arg_b.Set(&options.mode_name); });

  b.AddIntegerOption(
      {
          .name = "trials",
          .value_name = "N",
          .help = "Total number of trials per D configuration.",
      },
      [&](auto& arg_b) { arg_b.Set(&options.total_trials); });

  b.AddIntegerOption(
      {
          .name = "seed",
          .value_name = "N",
          .help = "Random seed for deterministic sampling.",
      },
      [&](auto& arg_b) { arg_b.Set(&options.base_seed); });

  b.AddFlag(
      {
          .name = "verbose",
          .help = "Print detailed per-file results.",
      },
      [&](auto& arg_b) { arg_b.Set(&options.verbose); });

  b.AddFlag(
      {
          .name = "json",
          .help = "Output results in JSON format.",
      },
      [&](auto& arg_b) { arg_b.Set(&options.json_output); });

  b.AddIntegerOption(
      {
          .name = "dump-incorrect",
          .value_name = "N",
          .help = "Print details for up to N incorrect trials.",
      },
      [&](auto& arg_b) { arg_b.Set(&options.dump_incorrect); });

  b.AddIntegerOption(
      {
          .name = "dump-none",
          .value_name = "N",
          .help = "Print details for up to N trials classified None.",
      },
      [&](auto& arg_b) { arg_b.Set(&options.dump_none); });

  b.Do([] {});
}

static auto Run(llvm::ArrayRef<llvm::StringRef> args) -> bool {
  EvalOptions options;
  auto parse_result = CommandLine::Parse(
      args, llvm::outs(), CommandInfo,
      [&](CommandLine::CommandBuilder& b) { AddOptions(b, options); });
  if (!parse_result.ok()) {
    llvm::errs() << "error: " << *parse_result << "\n";
    return false;
  }
  if (*parse_result == CommandLine::ParseResult::MetaSuccess) {
    return true;
  }
  if (!options.Resolve()) {
    return false;
  }

  auto files = CollectCarbonFiles(options.input_files);
  if (files.empty()) {
    llvm::errs() << "error: No Carbon source files found to test.\n";
    return false;
  }
  Corpus corpus = FindCandidateFiles(files);
  if (corpus.files.empty() || corpus.total_clean_pairs == 0) {
    llvm::errs()
        << "error: No Carbon source files with bracket pairs found to test.\n";
    return false;
  }

  Report report = Evaluator(options, corpus).Run();
  if (options.json_output) {
    PrintJsonReport(options, corpus, report);
  } else {
    PrintMarkdownReport(options, corpus, report);
  }
  return true;
}

}  // namespace Carbon::Lex

auto main(int argc, char** argv) -> int {
  Carbon::InitLLVM init_llvm(argc, argv);
  Carbon::SetWorkingDirForBazelRun();
  llvm::SmallVector<llvm::StringRef> args(argv + 1, argv + argc);
  bool success = Carbon::Lex::Run(args);
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
