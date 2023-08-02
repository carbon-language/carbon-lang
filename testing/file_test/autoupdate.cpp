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
#include "re2/re2.h"

namespace Carbon::Testing {

// Put helper classes in an anonymous namespace.
namespace {

class CheckLine : public FileTestLineBase {
 public:
  // RE2 is passed by a pointer because it doesn't support std::optional.
  explicit CheckLine(int line_number, const RE2* line_number_re,
                     std::string line)
      : FileTestLineBase(line_number),
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
  auto SetRemappedLine(const std::string& sub_for_formatv,
                       int target_line_number) -> void {
    // Should only be called when we have a regex.
    CARBON_CHECK(line_number_re_);

    int offset = target_line_number - output_line_number_;
    const char* offset_prefix = offset < 0 ? "" : "+";
    std::string replacement =
        llvm::formatv(sub_for_formatv.data(),
                      llvm::formatv("[[@LINE{0}{1}]]", offset_prefix, offset));
    RE2::Replace(&line_, *line_number_re_, replacement);
  }

 private:
  const RE2* line_number_re_;
  std::string line_;
  llvm::StringRef indent_;
  int output_line_number_ = -1;
};

}  // namespace

// Adds output lines for autoupdate.
static auto AddCheckLines(
    llvm::StringRef output, const char* label,
    const llvm::SmallVector<llvm::StringRef>& filenames,
    bool line_number_re_has_file, const RE2& line_number_re,
    std::function<void(std::string&)> do_extra_check_replacements,
    llvm::SmallVector<llvm::SmallVector<CheckLine>>& check_lines) -> void {
  if (output.empty()) {
    return;
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

  int append_to = 0;
  for (const auto& line : lines) {
    std::string check_line = llvm::formatv("// CHECK:{0}:{1}{2}", label,
                                           line.empty() ? "" : " ", line);

    // Ignore TEST_TMPDIR in output.
    if (auto pos = check_line.find(tmpdir); pos != std::string::npos) {
      check_line.replace(pos, tmpdir.size(), "{{.+}}");
    }

    do_extra_check_replacements(check_line);

    // Look for line information in the output. line_number is only set if the
    // match is correct.
    int line_number = -1;
    int match_line_number;
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
            line_number = match_line_number;
          }
        } else {
          // The line applies to the current file.
          line_number = match_line_number;
        }
      }
    } else {
      // There's no file association, so we only look at the line.
      if (RE2::PartialMatch(check_line, line_number_re, &match_line_number)) {
        line_number = match_line_number;
      }
    }
    check_lines[append_to].push_back(
        CheckLine(line_number, line_number == -1 ? nullptr : &line_number_re,
                  check_line));
  }
}

auto AutoupdateFileTest(
    const std::filesystem::path& file_test_path, llvm::StringRef input_content,
    const llvm::SmallVector<llvm::StringRef>& filenames,
    int autoupdate_line_number,
    llvm::SmallVector<llvm::SmallVector<FileTestLine>>& non_check_lines,
    llvm::StringRef stdout, llvm::StringRef stderr,
    FileTestLineNumberReplacement line_number_replacement,
    std::function<void(std::string&)> do_extra_check_replacements) -> bool {
  // Prepare CHECK lines.
  llvm::SmallVector<llvm::SmallVector<CheckLine>> check_lines;
  check_lines.resize(filenames.size());
  RE2 line_number_re(line_number_replacement.pattern);
  CARBON_CHECK(line_number_re.ok()) << "Invalid line replacement RE2: `"
                                    << line_number_replacement.pattern << "`";

  AddCheckLines(stdout, "STDOUT", filenames, line_number_replacement.has_file,
                line_number_re, do_extra_check_replacements, check_lines);
  AddCheckLines(stderr, "STDERR", filenames, line_number_replacement.has_file,
                line_number_re, do_extra_check_replacements, check_lines);

  // All CHECK lines are suppressed until we reach AUTOUPDATE.
  bool reached_autoupdate = false;

  // Stitch together content.
  llvm::SmallVector<const FileTestLineBase*> new_lines;
  for (auto [filename, non_check_file, check_file] :
       llvm::zip(filenames, non_check_lines, check_lines)) {
    llvm::DenseMap<int, int> output_line_remap;
    int output_line_number = 0;
    auto* check_line = check_file.begin();

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

      if (reached_autoupdate) {
        for (; check_line != check_file.end() &&
               check_line->line_number() <= non_check_line.line_number();
             ++check_line) {
          new_lines.push_back(check_line);
          check_line->SetOutputLine(non_check_line.indent(),
                                    ++output_line_number);
        }
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
    }

    // This should always be true after the first file is processed.
    CARBON_CHECK(reached_autoupdate);

    // Print remaining check lines which -- for whatever reason -- come after
    // all original lines.
    for (; check_line != check_file.end(); ++check_line) {
      new_lines.push_back(check_line);
      check_line->SetOutputLine("", ++output_line_number);
    }

    // Update all remapped lines in CHECK output.
    for (auto& offset_check_line : check_file) {
      if (offset_check_line.line_number() >= 1) {
        auto new_line = output_line_remap.find(offset_check_line.line_number());
        CARBON_CHECK(new_line != output_line_remap.end());
        offset_check_line.SetRemappedLine(
            line_number_replacement.sub_for_formatv, new_line->second);
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
