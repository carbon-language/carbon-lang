// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <numeric>
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
  int total_trials = 1000;
  int base_seed = 42;
  bool verbose = false;
  bool json_output = false;
  int dump_incorrect = 0;

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
                    "Comma-separated deletion levels (e.g. '1,2,5,10%,25%').",
            },
            [&](auto& arg_b) { arg_b.Set(&d_values_str); });

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

        std::vector<int> pair_indices(clean_pairs.size());
        std::iota(pair_indices.begin(), pair_indices.end(), 0);
        std::shuffle(pair_indices.begin(), pair_indices.end(), rng);
        pair_indices.resize(d_count);

        std::vector<bool> is_deleted_token(clean_buffer.size(), false);
        std::vector<TokenIndex> sampled_tokens;
        sampled_tokens.reserve(d_count);

        for (int p_idx : pair_indices) {
          const auto& pair = clean_pairs[p_idx];
          TokenIndex tok =
              (rng() % 2 == 0) ? pair.open_token : pair.close_token;
          is_deleted_token[tok.index] = true;
          sampled_tokens.push_back(tok);
        }

        std::vector<DeletedToken> deleted_tokens;
        deleted_tokens.reserve(d_count);

        for (TokenIndex tok : sampled_tokens) {
          TokenKind tok_kind = clean_buffer.GetKind(tok);

          int32_t run_start = tok.index;
          while (run_start > 0 &&
                 clean_buffer.GetKind(TokenIndex(run_start - 1)) == tok_kind) {
            --run_start;
          }
          int32_t run_end = tok.index;
          while (run_end + 1 < static_cast<int32_t>(clean_buffer.size()) &&
                 clean_buffer.GetKind(TokenIndex(run_end + 1)) == tok_kind) {
            ++run_end;
          }

          llvm::SmallVector<int32_t, 4> valid_offsets;
          for (int32_t idx = run_start; idx <= run_end; ++idx) {
            if (!is_deleted_token[idx]) {
              valid_offsets.push_back(
                  clean_buffer.GetByteOffset(TokenIndex(idx)));
            }
          }

          TokenIndex succ = TokenIndex(run_end + 1);
          while (succ.index < clean_buffer.size() &&
                 is_deleted_token[succ.index]) {
            succ = TokenIndex(succ.index + 1);
          }

          int32_t succ_byte = (succ.index < clean_buffer.size())
                                  ? clean_buffer.GetByteOffset(succ)
                                  : static_cast<int32_t>(source_text.size());
          valid_offsets.push_back(succ_byte);

          int32_t succ_line = (succ.index < clean_buffer.size())
                                  ? clean_buffer.GetLineNumber(succ)
                                  : -1;
          int32_t succ_col = (succ.index < clean_buffer.size())
                                 ? clean_buffer.GetColumnNumber(succ)
                                 : -1;

          deleted_tokens.push_back(DeletedToken{
              .kind = tok_kind,
              .byte_offset = clean_buffer.GetByteOffset(tok),
              .length =
                  static_cast<int32_t>(clean_buffer.GetTokenText(tok).size()),
              .line = clean_buffer.GetLineNumber(tok),
              .column = clean_buffer.GetColumnNumber(tok),
              .next_token_byte_offset = succ_byte,
              .next_token_line = succ_line,
              .next_token_column = succ_col,
              .valid_next_token_byte_offsets = std::move(valid_offsets),
          });
        }

        std::string corrupted_text = source_text.str();
        for (const auto& del : deleted_tokens) {
          for (int i = 0; i < del.length; ++i) {
            corrupted_text[del.byte_offset + i] = ' ';
          }
        }

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

        llvm::SmallVector<Suggestion> suggestions;
        for (TokenIndex t : corrupted_buffer.tokens()) {
          if (corrupted_buffer.IsRecoveryToken(t)) {
            auto kind = corrupted_buffer.GetKind(t);
            if (kind.is_opening_symbol() || kind.is_closing_symbol()) {
              TokenIndex succ = TokenIndex(t.index + 1);
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
              std::string origin = "Unknown";
              for (const auto& c : corrections) {
                if ((c.fix_action == BracketFixAction::InsertBefore ||
                     c.fix_action == BracketFixAction::InsertAfter) &&
                    !c.is_tied && c.fix_token_kind == kind) {
                  if (c.fix_byte_offset == byte_off) {
                    origin = c.origin;
                    break;
                  }
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

        if (spec.label == "1") {
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
        if (classification == TestClassification::Incorrect &&
            dump_incorrect > 0) {
          --dump_incorrect;
          llvm::errs() << "\n=== INCORRECT TRIAL in " << candidate.filename
                       << " (D=" << spec.label << ") ===\n";
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
  llvm::outs() << "- **Files tested**: " << valid_files.size() << " files ("
               << total_clean_pairs << " clean matched bracket pairs)\n";
  llvm::outs() << "- **Total trials per configuration**: " << total_trials
               << "\n";
  llvm::outs() << "- **Random seed**: " << base_seed << "\n\n";

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
