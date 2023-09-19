// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/file_test/autoupdate.h"

#include <fstream>

#include "absl/strings/string_view.h"
#include "common/check.h"
#include "common/ostream.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"

namespace Carbon::Testing {

// Converts a matched line number to an int, trimming whitespace.
static auto ParseLineNumber(absl::string_view matched_line_number) -> int {
  llvm::StringRef trimmed = matched_line_number;
  trimmed = trimmed.trim();
  // NOLINTNEXTLINE(google-runtime-int): API requirement.
  long long val;
  CARBON_CHECK(!llvm::getAsSignedInteger(trimmed, 10, val))
      << matched_line_number;
  return val;
}

FileTestAutoupdater::FileAndLineNumber::FileAndLineNumber(
    const LineNumberReplacement* replacement, int file_number,
    absl::string_view line_number)
    : replacement(replacement),
      file_number(file_number),
      line_number(ParseLineNumber(line_number)) {}

auto FileTestAutoupdater::CheckLine::RemapLineNumbers(
    const llvm::DenseMap<std::pair<int, int>, int>& output_line_remap,
    const llvm::SmallVector<int>& new_last_line_numbers) -> void {
  // Only need to do remappings when there's a line number replacement.
  if (!replacement_) {
    return;
  }

  bool found_one = false;
  // Use a cursor for the line so that we can't keep matching the same
  // content, which may occur when we keep a literal line number.
  int line_offset = 0;
  while (true) {
    // Rebuild the cursor each time because we're editing the line, which
    // could cause a reallocation.
    absl::string_view line_cursor = line_;
    line_cursor.remove_prefix(line_offset);
    // Look for a line number to replace. There may be multiple, so we
    // repeatedly check.
    absl::string_view matched_line_number;
    if (replacement_->has_file) {
      RE2::PartialMatch(line_cursor, *replacement_->re, nullptr,
                        &matched_line_number);
    } else {
      RE2::PartialMatch(line_cursor, *replacement_->re, &matched_line_number);
    }
    if (matched_line_number.empty()) {
      CARBON_CHECK(found_one) << line_;
      return;
    }
    found_one = true;

    // Update the cursor offset from the match.
    line_offset = matched_line_number.begin() - line_.c_str();

    // Calculate the new line number (possibly with new CHECK lines added, or
    // some removed).
    int old_line_number = ParseLineNumber(matched_line_number);
    int new_line_number = -1;
    if (auto remapped =
            output_line_remap.find({file_number(), old_line_number});
        remapped != output_line_remap.end()) {
      // Map old non-check lines to their new line numbers.
      new_line_number = remapped->second;
    } else {
      // We assume unmapped references point to the end-of-file.
      new_line_number = new_last_line_numbers[file_number()];
    }

    std::string replacement;
    if (output_file_number_ == file_number()) {
      int offset = new_line_number - output_line_number_;
      // Update the line offset in the CHECK line.
      const char* offset_prefix = offset < 0 ? "" : "+";
      replacement = llvm::formatv(
          replacement_->line_formatv.c_str(),
          llvm::formatv("[[@LINE{0}{1}]]", offset_prefix, offset));
    } else {
      // If the CHECK was written to a different file from the file that it
      // refers to, leave behind an absolute line reference rather than a
      // cross-file offset.
      replacement =
          llvm::formatv(replacement_->line_formatv.c_str(), new_line_number);
    }
    line_.replace(matched_line_number.data() - line_.data(),
                  matched_line_number.size(), replacement);
  }
}

auto FileTestAutoupdater::GetFileAndLineNumber(
    llvm::DenseMap<llvm::StringRef, int> file_to_number_map,
    int default_file_number, const std::string& check_line)
    -> FileAndLineNumber {
  for (const auto& replacement : line_number_replacements_) {
    if (replacement.has_file) {
      absl::string_view filename;
      absl::string_view line_number;
      if (RE2::PartialMatch(check_line, *replacement.re, &filename,
                            &line_number)) {
        if (auto it = file_to_number_map.find(filename);
            it != file_to_number_map.end()) {
          return FileAndLineNumber(&replacement, it->second, line_number);
        } else {
          return FileAndLineNumber(default_file_number);
        }
      }
    } else {
      // There's no file association, so we only look at the line, and assume
      // it refers to the default file.
      absl::string_view line_number;
      if (RE2::PartialMatch(check_line, *replacement.re, &line_number)) {
        return FileAndLineNumber(&replacement, default_file_number,
                                 line_number);
      }
    }
  }
  return FileAndLineNumber(default_file_number);
}

auto FileTestAutoupdater::BuildCheckLines(llvm::StringRef output,
                                          const char* label) -> CheckLines {
  if (output.empty()) {
    return CheckLines({});
  }

  // Prepare to look for filenames in lines.
  llvm::DenseMap<llvm::StringRef, int> file_to_number_map;
  for (auto [number, name] : llvm::enumerate(filenames_)) {
    file_to_number_map.insert({name, number});
  }

  // %t substitution means we may see TEST_TMPDIR in output.
  char* tmpdir_env = getenv("TEST_TMPDIR");
  CARBON_CHECK(tmpdir_env != nullptr);
  llvm::StringRef tmpdir = tmpdir_env;

  llvm::SmallVector<llvm::StringRef> lines(llvm::split(output, '\n'));
  // It's typical that output ends with a newline, but we don't want to add a
  // blank CHECK for it.
  if (lines.back().empty()) {
    lines.pop_back();
  }

  // `{{` and `[[` are escaped as a regex matcher.
  RE2 double_brace_re(R"(\{\{)");
  RE2 double_square_bracket_re(R"(\[\[)");
  // End-of-line whitespace is replaced with a regex matcher to make it visible.
  RE2 end_of_line_whitespace_re(R"((\s+)$)");

  // The default file number for when no specific file is found.
  int default_file_number = 0;

  llvm::SmallVector<CheckLine> check_lines;
  for (const auto& line : lines) {
    std::string check_line = llvm::formatv("// CHECK:{0}:{1}{2}", label,
                                           line.empty() ? "" : " ", line);
    RE2::Replace(&check_line, double_brace_re, R"({{\\{\\{}})");
    RE2::Replace(&check_line, double_square_bracket_re, R"({{\\[\\[}})");
    RE2::Replace(&check_line, end_of_line_whitespace_re, R"({{\1}})");

    // Ignore TEST_TMPDIR in output.
    if (auto pos = check_line.find(tmpdir); pos != std::string::npos) {
      check_line.replace(pos, tmpdir.size(), "{{.+}}");
    }

    do_extra_check_replacements_(check_line);

    if (default_file_re_) {
      absl::string_view filename;
      if (RE2::PartialMatch(line, *default_file_re_, &filename)) {
        auto it = file_to_number_map.find(filename);
        CARBON_CHECK(it != file_to_number_map.end())
            << "default_file_re had unexpected match in '" << line << "' (`"
            << default_file_re_->pattern() << "`)";
        default_file_number = it->second;
      }
    }
    auto file_and_line = GetFileAndLineNumber(file_to_number_map,
                                              default_file_number, check_line);
    check_lines.push_back(CheckLine(file_and_line, check_line));
  }
  return CheckLines(check_lines);
}

auto FileTestAutoupdater::AddRemappedNonCheckLine() -> void {
  new_lines_.push_back(non_check_line_);
  CARBON_CHECK(output_line_remap_
                   .insert({{non_check_line_->file_number(),
                             non_check_line_->line_number()},
                            ++output_line_number_})
                   .second);
}

auto FileTestAutoupdater::ShouldAddCheckLine(const CheckLines& check_lines,
                                             bool to_file_end) const -> bool {
  return check_lines.cursor != check_lines.lines.end() &&
         (check_lines.cursor->file_number() < output_file_number_ ||
          (check_lines.cursor->file_number() == output_file_number_ &&
           (to_file_end || check_lines.cursor->line_number() <=
                               non_check_line_->line_number())));
}

auto FileTestAutoupdater::AddCheckLines(CheckLines& check_lines,
                                        bool to_file_end) -> void {
  for (; ShouldAddCheckLine(check_lines, to_file_end); ++check_lines.cursor) {
    new_lines_.push_back(check_lines.cursor);
    check_lines.cursor->SetOutputLine(
        to_file_end ? "" : non_check_line_->indent(), output_file_number_,
        ++output_line_number_);
  }
}

auto FileTestAutoupdater::FinishFile(bool is_last_file) -> void {
  bool include_stdout = any_attached_stdout_lines_ || is_last_file;

  // At the end of each file, print any remaining lines which are associated
  // with the file.
  if (ShouldAddCheckLine(stderr_, /*to_file_end=*/true) ||
      (include_stdout && ShouldAddCheckLine(stdout_, /*to_file_end=*/true))) {
    // Ensure there's a blank line before any trailing CHECKs.
    if (!new_lines_.empty() && !new_lines_.back()->is_blank()) {
      new_lines_.push_back(&blank_line_);
      ++output_line_number_;
    }

    AddCheckLines(stderr_, /*to_file_end=*/true);
    if (include_stdout) {
      AddCheckLines(stdout_, /*to_file_end=*/true);
    }
  }

  new_last_line_numbers_.push_back(output_line_number_);
}

auto FileTestAutoupdater::StartSplitFile() -> void {
  // Advance the file.
  ++output_file_number_;
  output_line_number_ = 0;
  CARBON_CHECK(output_file_number_ == non_check_line_->file_number())
      << "Non-sequential file: " << non_check_line_->file_number();

  // Each following file has precisely one split line.
  CARBON_CHECK(non_check_line_->line_number() < 1)
      << "Expected a split line, got " << *non_check_line_;
  // The split line is ignored when calculating line counts.
  new_lines_.push_back(non_check_line_);

  // Add any file-specific but line-unattached STDOUT messages here. STDERR is
  // handled through the main loop because it's before the next line.
  if (any_attached_stdout_lines_) {
    AddCheckLines(stdout_, /*to_file_end=*/false);
  }

  ++non_check_line_;
}

auto FileTestAutoupdater::Run(bool dry_run) -> bool {
  // Print everything until the autoupdate line.
  while (non_check_line_->line_number() != autoupdate_line_number_) {
    CARBON_CHECK(non_check_line_ != non_check_lines_.end() &&
                 non_check_line_->file_number() == 0)
        << "Missed autoupdate?";
    AddRemappedNonCheckLine();
    ++non_check_line_;
  }

  // Add the AUTOUPDATE line along with any early STDERR lines, so that the
  // initial batch of CHECK lines have STDERR before STDOUT. This also ensures
  // we don't insert a blank line before the STDERR checks if there are no more
  // lines after AUTOUPDATE.
  AddRemappedNonCheckLine();
  AddCheckLines(stderr_, /*to_file_end=*/false);
  if (any_attached_stdout_lines_) {
    AddCheckLines(stdout_, /*to_file_end=*/false);
  }
  ++non_check_line_;

  // Loop through remaining content.
  while (non_check_line_ != non_check_lines_.end()) {
    if (output_file_number_ < non_check_line_->file_number()) {
      FinishFile(/*is_last_file=*/false);
      StartSplitFile();
      continue;
    }

    // STDERR check lines are placed before the line they refer to, or as
    // early as possible if they don't refer to a line. Include all STDERR
    // lines until we find one that wants to go later in the file.
    AddCheckLines(stderr_, /*to_file_end=*/false);
    AddRemappedNonCheckLine();

    // STDOUT check lines are placed after the line they refer to, or at the
    // end of the file if none of them refers to a line.
    if (any_attached_stdout_lines_) {
      AddCheckLines(stdout_, /*to_file_end=*/false);
    }

    ++non_check_line_;
  }

  FinishFile(/*is_last_file=*/true);

  for (auto& check_line : stdout_.lines) {
    check_line.RemapLineNumbers(output_line_remap_, new_last_line_numbers_);
  }
  for (auto& check_line : stderr_.lines) {
    check_line.RemapLineNumbers(output_line_remap_, new_last_line_numbers_);
  }

  // Generate the autoupdated file.
  std::string new_content;
  llvm::raw_string_ostream new_content_stream(new_content);
  for (const auto& line : new_lines_) {
    new_content_stream << *line << '\n';
  }

  // Update the file on disk if needed.
  if (new_content == input_content_) {
    return false;
  }
  if (!dry_run) {
    std::ofstream out(file_test_path_);
    out << new_content;
  }
  return true;
}

}  // namespace Carbon::Testing
