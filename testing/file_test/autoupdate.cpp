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
#include "re2/re2.h"

namespace Carbon::Testing {

// Put helper classes in an anonymous namespace.
namespace {

// Converts a matched line number to an int, trimming whitespace.
static auto ParseLineNumber(absl::string_view matched_line_number) -> int {
  llvm::StringRef trimmed = matched_line_number;
  trimmed = trimmed.trim();
  // NOLINTNEXTLINE(google-runtime-int): API requirement.
  long long val;
  CARBON_CHECK(!llvm::getAsSignedInteger(trimmed, 10, val));
  return val;
}

class CheckLine : public FileTestLineBase {
 public:
  // RE2 is passed by a pointer because it doesn't support std::optional.
  explicit CheckLine(int line_number, bool line_number_re_has_file,
                     const RE2* line_number_re, std::string line)
      : FileTestLineBase(line_number),
        line_number_re_has_file_(line_number_re_has_file),
        line_number_re_(line_number_re),
        line_(std::move(line)) {}

  auto Print(llvm::raw_ostream& out) const -> void override {
    out << indent_ << line_;
  }

  // When the location of the CHECK in output is known, we can set the indent
  // and its line.
  auto SetOutputLine(llvm::StringRef indent, int output_line_number) -> void {
    indent_ = indent;
    output_line_number_ = output_line_number;
  }

  // When the location of all lines in a file are known, we can set the line
  // offset based on the target line.
  auto RemapLineNumbers(const std::string& line_formatv,
                        llvm::function_ref<int(int)> line_remap) -> void {
    // Only need to do remappings when there's a regex.
    if (!line_number_re_) {
      return;
    }

    bool found_one = false;
    while (true) {
      // Look for a line number to replace. There may be multiple, so we
      // repeatedly check.
      absl::string_view matched_line_number;
      if (line_number_re_has_file_) {
        RE2::PartialMatch(line_, *line_number_re_, nullptr,
                          &matched_line_number);
      } else {
        RE2::PartialMatch(line_, *line_number_re_, &matched_line_number);
      }
      if (matched_line_number.empty()) {
        CARBON_CHECK(found_one) << line_;
        return;
      }
      found_one = true;

      // Calculate the offset from the CHECK line to the new line number
      // (possibly with new CHECK lines added, or some removed).
      int new_line_number = line_remap(ParseLineNumber(matched_line_number));
      int offset = new_line_number - output_line_number_;

      // Update the line offset in the CHECK line.
      const char* offset_prefix = offset < 0 ? "" : "+";
      std::string replacement = llvm::formatv(
          line_formatv.c_str(),
          llvm::formatv("[[@LINE{0}{1}]]", offset_prefix, offset));
      line_.replace(matched_line_number.data() - line_.data(),
                    matched_line_number.size(), replacement);
    }
  }

  auto is_blank() const -> bool override { return false; }

 private:
  bool line_number_re_has_file_;
  const RE2* line_number_re_;
  std::string line_;
  llvm::StringRef indent_;
  int output_line_number_ = -1;
};

}  // namespace

// Builds CheckLine lists for autoupdate.
static auto BuildCheckLines(
    llvm::StringRef output, const char* label,
    const llvm::SmallVector<llvm::StringRef>& filenames,
    bool line_number_re_has_file, const RE2& line_number_re,
    std::function<void(std::string&)> do_extra_check_replacements)
    -> llvm::SmallVector<llvm::SmallVector<CheckLine>> {
  llvm::SmallVector<llvm::SmallVector<CheckLine>> check_lines;
  check_lines.resize(filenames.size());
  if (output.empty()) {
    return check_lines;
  }

  // Prepare to look for filenames in lines.
  llvm::StringRef current_filename = filenames[0];
  const auto* remaining_filenames = filenames.begin() + 1;

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

  int append_to = 0;
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

    // Look for line information in the output. use_line_number is only set if
    // the match is correct.
    std::optional<llvm::StringRef> use_line_number;
    absl::string_view match_line_number;
    if (line_number_re_has_file) {
      absl::string_view match_filename;
      if (RE2::PartialMatch(check_line, line_number_re, &match_filename,
                            &match_line_number)) {
        llvm::StringRef match_filename_ref = match_filename;
        if (match_filename_ref != current_filename) {
          // If the filename doesn't match, it may be still usable if it refers
          // to a later file.
          const auto* pos = std::find(remaining_filenames, filenames.end(),
                                      match_filename_ref);
          if (pos != filenames.end()) {
            remaining_filenames = pos + 1;
            append_to = pos - filenames.begin();
            use_line_number = match_line_number;
          }
        } else {
          // The line applies to the current file.
          use_line_number = match_line_number;
        }
      }
    } else {
      // There's no file association, so we only look at the line.
      if (RE2::PartialMatch(check_line, line_number_re, &match_line_number)) {
        use_line_number = match_line_number;
      }
    }
    int line_number = use_line_number ? ParseLineNumber(*use_line_number) : -1;
    check_lines[append_to].push_back(
        CheckLine(line_number, line_number_re_has_file,
                  use_line_number ? &line_number_re : nullptr, check_line));
  }

  return check_lines;
}

auto AutoupdateFileTest(
    const std::filesystem::path& file_test_path, llvm::StringRef input_content,
    const llvm::SmallVector<llvm::StringRef>& filenames,
    int autoupdate_line_number,
    llvm::SmallVector<llvm::SmallVector<FileTestLine>>& non_check_lines,
    llvm::StringRef stdout, llvm::StringRef stderr,
    FileTestLineNumberReplacement line_number_replacement,
    std::function<void(std::string&)> do_extra_check_replacements) -> bool {
  RE2 line_number_re(line_number_replacement.pattern);
  CARBON_CHECK(line_number_re.ok()) << "Invalid line replacement RE2: `"
                                    << line_number_replacement.pattern << "`";

  // Prepare CHECK lines.
  llvm::SmallVector<llvm::SmallVector<CheckLine>> stdout_check_lines =
      BuildCheckLines(stdout, "STDOUT", filenames,
                      line_number_replacement.has_file, line_number_re,
                      do_extra_check_replacements);
  llvm::SmallVector<llvm::SmallVector<CheckLine>> stderr_check_lines =
      BuildCheckLines(stderr, "STDERR", filenames,
                      line_number_replacement.has_file, line_number_re,
                      do_extra_check_replacements);

  // All CHECK lines are suppressed until we reach AUTOUPDATE.
  bool reached_autoupdate = false;

  const FileTestLine blank_line(-1, "");

  // Stitch together content.
  llvm::SmallVector<const FileTestLineBase*> new_lines;
  for (auto [filename, non_check_file, stdout_check_file, stderr_check_file] :
       llvm::zip(filenames, non_check_lines, stdout_check_lines,
                 stderr_check_lines)) {
    llvm::DenseMap<int, int> output_line_remap;
    int output_line_number = 0;
    auto* stdout_check_line = stdout_check_file.begin();
    auto* stderr_check_line = stderr_check_file.begin();

    // Add all check lines from the given vector until we reach a check line
    // attached to a line later than `to_line_number`.
    auto add_check_lines = [&](const llvm::SmallVector<CheckLine>& lines,
                               CheckLine*& line, int to_line_number,
                               llvm::StringRef indent) {
      for (; line != lines.end() && line->line_number() <= to_line_number;
           ++line) {
        new_lines.push_back(line);
        line->SetOutputLine(indent, ++output_line_number);
      }
    };

    bool any_attached_stdout_lines = std::any_of(
        stdout_check_file.begin(), stdout_check_file.end(),
        [&](const CheckLine& line) { return line.line_number() != -1; });

    // Looping through the original file, print check lines preceding each
    // original line.
    for (const auto& non_check_line : non_check_file) {
      // If there are any non-check lines with an invalid line_number, it's
      // something like a split directive which shouldn't increment
      // output_line_number.
      if (non_check_line.line_number() < 1) {
        new_lines.push_back(&non_check_line);
        continue;
      }

      // STDERR check lines are placed before the line they refer to, or as
      // early as possible if they don't refer to a line. Include all STDERR
      // lines until we find one that wants to go later in the file.
      if (reached_autoupdate) {
        add_check_lines(stderr_check_file, stderr_check_line,
                        non_check_line.line_number(), non_check_line.indent());
      } else if (autoupdate_line_number == non_check_line.line_number()) {
        // This is the AUTOUPDATE line, so we'll print it, then start printing
        // CHECK lines.
        reached_autoupdate = true;
      }

      new_lines.push_back(&non_check_line);
      CARBON_CHECK(
          output_line_remap
              .insert({non_check_line.line_number(), ++output_line_number})
              .second);

      // STDOUT check lines are placed after the line they refer to, or at the
      // end of the file if none of them refers to a line.
      if (reached_autoupdate && any_attached_stdout_lines) {
        add_check_lines(stdout_check_file, stdout_check_line,
                        non_check_line.line_number(), non_check_line.indent());
      }
    }

    // This should always be true after the first file is processed.
    CARBON_CHECK(reached_autoupdate);

    // Print remaining check lines which -- for whatever reason -- come after
    // all original lines.
    if (stderr_check_line != stderr_check_file.end() ||
        stdout_check_line != stdout_check_file.end()) {
      // Ensure there's a blank line before any trailing CHECKs.
      if (!new_lines.empty() && !new_lines.back()->is_blank()) {
        new_lines.push_back(&blank_line);
        ++output_line_number;
      }

      add_check_lines(stderr_check_file, stderr_check_line, INT_MAX, "");
      add_check_lines(stdout_check_file, stdout_check_line, INT_MAX, "");
    }

    // Update all remapped lines in CHECK output.
    for (auto* check_file : {&stdout_check_file, &stderr_check_file}) {
      for (auto& offset_check_line : *check_file) {
        int last_non_check_line = non_check_file.back().line_number();
        offset_check_line.RemapLineNumbers(
            line_number_replacement.line_formatv, [&](int old_line_number) {
              // Map old non-check lines to their new line numbers.
              auto remapped = output_line_remap.find(old_line_number);
              if (remapped != output_line_remap.end()) {
                return remapped->second;
              }

              // Map any reference to a line past the final non-check line to
              // the new end-of-file. We assume that any such reference is
              // referring to the end of file, not to some specific CHECK
              // comment.
              if (old_line_number > last_non_check_line) {
                return output_line_number;
              }

              // Line didn't get remapped; maybe it refers to a CHECK line.
              // We can't express that as an offset, just leave it as-is.
              return old_line_number;
            });
      }
    }
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
