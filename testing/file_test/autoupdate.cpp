// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/file_test/autoupdate.h"

#include <fstream>

#include "absl/strings/string_view.h"
#include "common/check.h"
#include "common/ostream.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"

namespace Carbon::Testing {

// Put helper classes in an anonymous namespace.
namespace {

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

// The file and line number that a CHECK line refers to, and the
// replacement from which they were determined, if any.
struct FileAndLineNumber {
  explicit FileAndLineNumber(int file_number) : file_number(file_number) {}

  explicit FileAndLineNumber(const FileTestLineNumberReplacement* replacement,
                             int file_number, absl::string_view line_number)
      : replacement(replacement),
        file_number(file_number),
        line_number(ParseLineNumber(line_number)) {}

  const FileTestLineNumberReplacement* replacement = nullptr;
  int file_number;
  int line_number = -1;
};

class CheckLine : public FileTestLineBase {
 public:
  // RE2 is passed by a pointer because it doesn't support std::optional.
  explicit CheckLine(FileAndLineNumber file_and_line_number, std::string line)
      : FileTestLineBase(file_and_line_number.line_number),
        file_number_(file_and_line_number.file_number),
        replacement_(file_and_line_number.replacement),
        line_(std::move(line)) {}

  auto Print(llvm::raw_ostream& out) const -> void override {
    out << indent_ << line_;
  }

  // When the location of the CHECK in output is known, we can set the indent
  // and its line.
  auto SetOutputLine(llvm::StringRef indent, int output_file_number,
                     int output_line_number) -> void {
    indent_ = indent;
    output_file_number_ = output_file_number;
    output_line_number_ = output_line_number;
  }

  // When the location of all lines in a file are known, we can set the line
  // offset based on the target line.
  auto RemapLineNumbers(
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
              output_line_remap.find({file_number_, old_line_number});
          remapped != output_line_remap.end()) {
        // Map old non-check lines to their new line numbers.
        new_line_number = remapped->second;
      } else {
        // We assume unmapped references point to the end-of-file.
        new_line_number = new_last_line_numbers[file_number_];
      }

      std::string replacement;
      if (output_file_number_ == file_number_) {
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

  auto file_number() const -> int { return file_number_; }

  auto is_blank() const -> bool override { return false; }

 private:
  int file_number_;
  const FileTestLineNumberReplacement* replacement_;
  std::string line_;
  llvm::StringRef indent_;
  int output_file_number_ = -1;
  int output_line_number_ = -1;
};

}  // namespace

// Looks for the patterns in the line. Returns the first match, or defaulted
// information if not found.
static auto GetFileAndLineNumber(
    const llvm::SmallVector<FileTestLineNumberReplacement>& replacements,
    llvm::DenseMap<llvm::StringRef, int> file_to_number_map,
    int default_file_number, const std::string& check_line)
    -> FileAndLineNumber {
  for (const auto& replacement : replacements) {
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

// Builds CheckLine lists for autoupdate.
static auto BuildCheckLines(
    llvm::StringRef output, const char* label,
    const llvm::SmallVector<llvm::StringRef>& filenames,
    const llvm::SmallVector<std::shared_ptr<RE2>>& default_file_res,
    const llvm::SmallVector<FileTestLineNumberReplacement>& replacements,
    std::function<void(std::string&)> do_extra_check_replacements)
    -> llvm::SmallVector<CheckLine> {
  llvm::SmallVector<CheckLine> check_lines;
  if (output.empty()) {
    return check_lines;
  }

  // Prepare to look for filenames in lines.
  llvm::DenseMap<llvm::StringRef, int> file_to_number_map;
  for (auto [number, name] : llvm::enumerate(filenames)) {
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

    do_extra_check_replacements(check_line);

    for (const auto& re : default_file_res) {
      absl::string_view filename;
      if (RE2::PartialMatch(line, *re, &filename)) {
        auto it = file_to_number_map.find(filename);
        CARBON_CHECK(it != file_to_number_map.end())
            << "default_file_re had unexpected match in '" << line << "' (`"
            << re->pattern() << "`)";
        default_file_number = it->second;
        break;
      }
    }
    auto file_and_line = GetFileAndLineNumber(replacements, file_to_number_map,
                                              default_file_number, check_line);
    check_lines.push_back(CheckLine(file_and_line, check_line));
  }

  return check_lines;
}

auto AutoupdateFileTest(
    const std::filesystem::path& file_test_path, llvm::StringRef input_content,
    const llvm::SmallVector<llvm::StringRef>& filenames,
    int autoupdate_line_number,
    const llvm::SmallVector<llvm::SmallVector<FileTestLine>>& non_check_lines,
    llvm::StringRef stdout, llvm::StringRef stderr,
    const llvm::SmallVector<std::shared_ptr<RE2>>& default_file_res,
    const llvm::SmallVector<FileTestLineNumberReplacement>&
        line_number_replacements,
    std::function<void(std::string&)> do_extra_check_replacements) -> bool {
  for (const auto& replacement : line_number_replacements) {
    CARBON_CHECK(replacement.has_file || !default_file_res.empty())
        << "For replacement with pattern `" << replacement.re->pattern()
        << "` to have has_file=false, override GetDefaultFileRE.";
    CARBON_CHECK(replacement.re->ok())
        << "Invalid line replacement RE2: " << replacement.re->error();
  }

  // Prepare CHECK lines.
  llvm::SmallVector<CheckLine> stdout_check_lines =
      BuildCheckLines(stdout, "STDOUT", filenames, default_file_res,
                      line_number_replacements, do_extra_check_replacements);
  llvm::SmallVector<CheckLine> stderr_check_lines =
      BuildCheckLines(stderr, "STDERR", filenames, default_file_res,
                      line_number_replacements, do_extra_check_replacements);
  auto* stdout_check_line = stdout_check_lines.begin();
  auto* stderr_check_line = stderr_check_lines.begin();

  bool any_attached_stdout_lines = std::any_of(
      stdout_check_lines.begin(), stdout_check_lines.end(),
      [&](const CheckLine& line) { return line.line_number() != -1; });

  // All CHECK lines are suppressed until we reach AUTOUPDATE.
  bool reached_autoupdate = false;

  const FileTestLine blank_line(-1, "");

  // Maps {file_number, original line number} to a new line number.
  llvm::DenseMap<std::pair<int, int>, int> output_line_remap;

  // Tracks the new last line numbers for each file.
  llvm::SmallVector<int> new_last_line_numbers;
  new_last_line_numbers.reserve(filenames.size());

  // Stitch together content.
  llvm::SmallVector<const FileTestLineBase*> new_lines;
  for (auto [file_number_as_size_t, filename, non_check_file] :
       llvm::enumerate(filenames, non_check_lines)) {
    auto file_number = static_cast<int>(file_number_as_size_t);
    int output_line_number = 0;

    // Track the offset in new_lines to later determine the line count.
    int file_offset_in_new_lines = new_lines.size();

    // Add all check lines from the given vector until we reach a check line
    // attached to a line later than `to_line_number`.
    auto add_check_lines = [&](const llvm::SmallVector<CheckLine>& lines,
                               CheckLine*& line, int to_line_number,
                               llvm::StringRef indent) {
      for (; line != lines.end() && (line->file_number() < file_number ||
                                     (line->file_number() == file_number &&
                                      line->line_number() <= to_line_number));
           ++line) {
        new_lines.push_back(line);
        line->SetOutputLine(indent, file_number, ++output_line_number);
      }
    };

    // Looping through the original file, print check lines preceding each
    // original line.
    for (const auto& non_check_line : non_check_file) {
      // If there are any non-check lines with an invalid line_number, it's
      // something like a split directive which shouldn't increment
      // output_line_number.
      if (non_check_line.line_number() < 1) {
        // These are ignored when calculating line counts.
        CARBON_CHECK(file_offset_in_new_lines ==
                     static_cast<int>(new_lines.size()));
        ++file_offset_in_new_lines;
        new_lines.push_back(&non_check_line);
        continue;
      }

      // STDERR check lines are placed before the line they refer to, or as
      // early as possible if they don't refer to a line. Include all STDERR
      // lines until we find one that wants to go later in the file.
      if (reached_autoupdate) {
        add_check_lines(stderr_check_lines, stderr_check_line,
                        non_check_line.line_number(), non_check_line.indent());
      } else if (autoupdate_line_number == non_check_line.line_number()) {
        // This is the AUTOUPDATE line, so we'll print it, then start printing
        // CHECK lines.
        reached_autoupdate = true;
      }

      new_lines.push_back(&non_check_line);
      CARBON_CHECK(output_line_remap
                       .insert({{file_number, non_check_line.line_number()},
                                ++output_line_number})
                       .second);

      // If we just added the AUTOUPDATE line, include any early STDERR lines
      // now, so that the initial batch of CHECK lines have STDERR before
      // STDOUT. This also ensures we don't insert a blank line before the
      // STDERR checks if there are no more lines after AUTOUPDATE.
      if (autoupdate_line_number == non_check_line.line_number()) {
        add_check_lines(stderr_check_lines, stderr_check_line,
                        non_check_line.line_number(), non_check_line.indent());
      }

      // STDOUT check lines are placed after the line they refer to, or at the
      // end of the file if none of them refers to a line.
      if (reached_autoupdate && any_attached_stdout_lines) {
        add_check_lines(stdout_check_lines, stdout_check_line,
                        non_check_line.line_number(), non_check_line.indent());
      }
    }

    // This should always be true after the first file is processed.
    CARBON_CHECK(reached_autoupdate);

    // At the end of each file, print any remaining lines which are associated
    // with the file.
    if ((stderr_check_line != stderr_check_lines.end() &&
         stderr_check_line->file_number() == file_number) ||
        (stdout_check_line != stdout_check_lines.end() &&
         stdout_check_line->file_number() == file_number)) {
      // Ensure there's a blank line before any trailing CHECKs.
      if (!new_lines.empty() && !new_lines.back()->is_blank()) {
        new_lines.push_back(&blank_line);
        ++output_line_number;
      }

      add_check_lines(stderr_check_lines, stderr_check_line, INT_MAX, "");
      add_check_lines(stdout_check_lines, stdout_check_line, INT_MAX, "");
    }

    new_last_line_numbers.push_back(new_lines.size() -
                                    file_offset_in_new_lines);
  }

  for (auto& check_line : stdout_check_lines) {
    check_line.RemapLineNumbers(output_line_remap, new_last_line_numbers);
  }
  for (auto& check_line : stderr_check_lines) {
    check_line.RemapLineNumbers(output_line_remap, new_last_line_numbers);
  }

  // Generate the autoupdated file.
  std::string new_content;
  llvm::raw_string_ostream new_content_stream(new_content);
  for (const auto& line : new_lines) {
    line->Print(new_content_stream);
    new_content_stream << '\n';
  }

  // Update the file on disk if needed.
  if (new_content == input_content) {
    return false;
  }
  std::ofstream out(file_test_path);
  out << new_content;
  return true;
}

}  // namespace Carbon::Testing
