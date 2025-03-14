// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/file_test/test_file.h"

#include <fstream>

#include "common/check.h"
#include "common/error.h"
#include "common/raw_string_ostream.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "testing/base/file_helpers.h"

namespace Carbon::Testing {

using ::testing::Matcher;
using ::testing::MatchesRegex;
using ::testing::StrEq;

// Processes conflict markers, including tracking of whether code is within a
// conflict marker. Returns true if the line is consumed.
static auto TryConsumeConflictMarker(bool running_autoupdate,
                                     llvm::StringRef line,
                                     llvm::StringRef line_trimmed,
                                     bool* inside_conflict_marker)
    -> ErrorOr<bool> {
  bool is_start = line.starts_with("<<<<<<<");
  bool is_middle = line.starts_with("=======") || line.starts_with("|||||||");
  bool is_end = line.starts_with(">>>>>>>");

  // When running the test, any conflict marker is an error.
  if (!running_autoupdate && (is_start || is_middle || is_end)) {
    return ErrorBuilder() << "Conflict marker found:\n" << line;
  }

  // Autoupdate tracks conflict markers for context, and will discard
  // conflicting lines when it can autoupdate them.
  if (*inside_conflict_marker) {
    if (is_start) {
      return ErrorBuilder() << "Unexpected conflict marker inside conflict:\n"
                            << line;
    }
    if (is_middle) {
      return true;
    }
    if (is_end) {
      *inside_conflict_marker = false;
      return true;
    }

    // Look for CHECK and TIP lines, which can be discarded.
    if (line_trimmed.starts_with("// CHECK:STDOUT:") ||
        line_trimmed.starts_with("// CHECK:STDERR:") ||
        line_trimmed.starts_with("// TIP:")) {
      return true;
    }

    return ErrorBuilder()
           << "Autoupdate can't discard non-CHECK lines inside conflicts:\n"
           << line;
  } else {
    if (is_start) {
      *inside_conflict_marker = true;
      return true;
    }
    if (is_middle || is_end) {
      return ErrorBuilder() << "Unexpected conflict marker outside conflict:\n"
                            << line;
    }
    return false;
  }
}

// State for file splitting logic: TryConsumeSplit and FinishSplit.
struct SplitState {
  auto has_splits() const -> bool { return file_index > 0; }

  auto add_content(llvm::StringRef line) -> void {
    content.append(line.str());
    content.append("\n");
  }

  // Whether content has been found. Only updated before a file split is found
  // (which may be never).
  bool found_code_pre_split = false;

  // The current file name, considering splits. Empty for the default file.
  llvm::StringRef filename = "";

  // The accumulated content for the file being built. This may elide some of
  // the original content, such as conflict markers.
  std::string content;

  // The current file index.
  int file_index = 0;
};

// Given a `file:/<filename>` URI, returns the filename.
static auto ExtractFilePathFromUri(llvm::StringRef uri)
    -> ErrorOr<llvm::StringRef> {
  static constexpr llvm::StringRef FilePrefix = "file:/";
  if (!uri.starts_with(FilePrefix)) {
    return ErrorBuilder("uri `") << uri << "` is not a file uri";
  }
  return uri.drop_front(FilePrefix.size());
}

// When `FROM_FILE_SPLIT` is used in path `textDocument.text`, populate the
// value from the split matching the `uri`. Only used for
// `textDocument/didOpen`.
static auto AutoFillDidOpenParams(llvm::json::Object* params,
                                  llvm::ArrayRef<TestFile::Split> splits)
    -> ErrorOr<Success> {
  auto* text_document = params->getObject("textDocument");
  if (text_document == nullptr) {
    return Success();
  }

  auto attr_it = text_document->find("text");
  if (attr_it == text_document->end() || attr_it->second != "FROM_FILE_SPLIT") {
    return Success();
  }

  auto uri = text_document->getString("uri");
  if (!uri) {
    return Error("missing uri in params.textDocument");
  }

  CARBON_ASSIGN_OR_RETURN(auto file_path, ExtractFilePathFromUri(*uri));
  const auto* split_it =
      llvm::find_if(splits, [&](const TestFile::Split& split) {
        return split.filename == file_path;
      });
  if (split_it == splits.end()) {
    return ErrorBuilder() << "No split found for uri: " << *uri;
  }
  attr_it->second = split_it->content;
  return Success();
}

// Reformats `[[@LSP:` and similar keyword as an LSP call with headers.
static auto ReplaceLspKeywordAt(std::string* content, size_t keyword_pos,
                                int& lsp_call_id,
                                llvm::ArrayRef<TestFile::Split> splits)
    -> ErrorOr<size_t> {
  llvm::StringRef content_at_keyword =
      llvm::StringRef(*content).substr(keyword_pos);

  auto [keyword, body_start] = content_at_keyword.split(":");
  if (body_start.empty()) {
    return ErrorBuilder() << "Missing `:` for `"
                          << content_at_keyword.take_front(10) << "`";
  }

  // Whether the first param is a method or id.
  llvm::StringRef method_or_id_label = "method";
  // Whether to attach the `lsp_call_id`.
  bool use_call_id = false;
  // The JSON label for extra content.
  llvm::StringRef extra_content_label;
  if (keyword == "[[@LSP-CALL") {
    use_call_id = true;
    extra_content_label = "params";
  } else if (keyword == "[[@LSP-NOTIFY") {
    extra_content_label = "params";
  } else if (keyword == "[[@LSP-REPLY") {
    method_or_id_label = "id";
    extra_content_label = "result";
  } else if (keyword != "[[@LSP") {
    return ErrorBuilder() << "Unrecognized @LSP keyword at `"
                          << keyword.take_front(10) << "`";
  }

  static constexpr llvm::StringLiteral LspEnd = "]]";
  auto body_end = body_start.find(LspEnd);
  if (body_end == std::string::npos) {
    return ErrorBuilder() << "Missing `" << LspEnd << "` after `" << keyword
                          << "`";
  }
  llvm::StringRef body = body_start.take_front(body_end);
  auto [method_or_id, extra_content] = body.split(":");

  llvm::json::Value parsed_extra_content = nullptr;
  if (!extra_content.empty()) {
    std::string extra_content_as_object =
        llvm::formatv("{{{0}}", extra_content);
    auto parse_result = llvm::json::parse(extra_content_as_object);
    if (auto err = parse_result.takeError()) {
      return ErrorBuilder() << "Error parsing extra content: " << err;
    }
    parsed_extra_content = std::move(*parse_result);
    CARBON_CHECK(parsed_extra_content.kind() == llvm::json::Value::Object);
    if (extra_content_label == "params" &&
        method_or_id == "textDocument/didOpen") {
      CARBON_RETURN_IF_ERROR(
          AutoFillDidOpenParams(parsed_extra_content.getAsObject(), splits));
    }
  }

  // Form the JSON.
  RawStringOstream buffer;
  llvm::json::OStream json(buffer);

  json.object([&] {
    json.attribute("jsonrpc", "2.0");
    json.attribute(method_or_id_label, method_or_id);

    if (use_call_id) {
      json.attribute("id", ++lsp_call_id);
    }
    if (parsed_extra_content != nullptr) {
      if (!extra_content_label.empty()) {
        json.attribute(extra_content_label, parsed_extra_content);
      } else {
        for (const auto& [key, value] : *parsed_extra_content.getAsObject()) {
          json.attribute(key, value);
        }
      }
    }
  });

  // Add the Content-Length header. The `2` accounts for extra newlines.
  int content_length = buffer.size() + 2;
  auto json_with_header = llvm::formatv("Content-Length: {0}\n\n{1}\n",
                                        content_length, buffer.TakeStr())
                              .str();
  int keyword_len =
      (body_start.data() + body_end + LspEnd.size()) - keyword.data();
  content->replace(keyword_pos, keyword_len, json_with_header);
  return keyword_pos + json_with_header.size();
}

// Replaces the keyword at the given position. Returns the position to start a
// find for the next keyword.
static auto ReplaceContentKeywordAt(std::string* content, size_t keyword_pos,
                                    llvm::StringRef test_name, int& lsp_call_id,
                                    llvm::ArrayRef<TestFile::Split> splits)
    -> ErrorOr<size_t> {
  auto keyword = llvm::StringRef(*content).substr(keyword_pos);

  // Line replacements aren't handled here.
  static constexpr llvm::StringLiteral Line = "[[@LINE";
  if (keyword.starts_with(Line)) {
    // Just move past the prefix to find the next one.
    return keyword_pos + Line.size();
  }

  // Replaced with the actual test name.
  static constexpr llvm::StringLiteral TestName = "[[@TEST_NAME]]";
  if (keyword.starts_with(TestName)) {
    content->replace(keyword_pos, TestName.size(), test_name);
    return keyword_pos + test_name.size();
  }

  if (keyword.starts_with("[[@LSP")) {
    return ReplaceLspKeywordAt(content, keyword_pos, lsp_call_id, splits);
  }

  return ErrorBuilder() << "Unexpected use of `[[@` at `"
                        << keyword.substr(0, 5) << "`";
}

// Replaces the content keywords.
//
// TEST_NAME is the only content keyword at present, but we do validate that
// other names are reserved.
static auto ReplaceContentKeywords(llvm::StringRef filename,
                                   std::string* content,
                                   llvm::ArrayRef<TestFile::Split> splits)
    -> ErrorOr<Success> {
  static constexpr llvm::StringLiteral Prefix = "[[@";

  auto keyword_pos = content->find(Prefix);
  // Return early if not finding anything.
  if (keyword_pos == std::string::npos) {
    return Success();
  }

  // Construct the test name by getting the base name without the extension,
  // then removing any "fail_" or "todo_" prefixes.
  llvm::StringRef test_name = filename;
  if (auto last_slash = test_name.rfind("/");
      last_slash != llvm::StringRef::npos) {
    test_name = test_name.substr(last_slash + 1);
  }
  if (auto ext_dot = test_name.find("."); ext_dot != llvm::StringRef::npos) {
    test_name = test_name.substr(0, ext_dot);
  }
  // Note this also handles `fail_todo_` and `todo_fail_`.
  test_name.consume_front("todo_");
  test_name.consume_front("fail_");
  test_name.consume_front("todo_");

  // A counter for LSP calls.
  int lsp_call_id = 0;
  while (keyword_pos != std::string::npos) {
    CARBON_ASSIGN_OR_RETURN(
        auto keyword_end,
        ReplaceContentKeywordAt(content, keyword_pos, test_name, lsp_call_id,
                                splits));
    keyword_pos = content->find(Prefix, keyword_end);
  }
  return Success();
}

// Adds a file. Used for both split and unsplit test files.
static auto AddSplit(llvm::StringRef filename, std::string* content,
                     llvm::SmallVector<TestFile::Split>* file_splits)
    -> ErrorOr<Success> {
  CARBON_RETURN_IF_ERROR(
      ReplaceContentKeywords(filename, content, *file_splits));
  file_splits->push_back(
      {.filename = filename.str(), .content = std::move(*content)});
  content->clear();
  return Success();
}

// Process file split ("---") lines when found. Returns true if the line is
// consumed.
static auto TryConsumeSplit(llvm::StringRef line, llvm::StringRef line_trimmed,
                            bool found_autoupdate, int* line_index,
                            SplitState* split,
                            llvm::SmallVector<TestFile::Split>* file_splits,
                            llvm::SmallVector<FileTestLine>* non_check_lines)
    -> ErrorOr<bool> {
  if (!line_trimmed.consume_front("// ---")) {
    if (!split->has_splits() && !line_trimmed.starts_with("//") &&
        !line_trimmed.empty()) {
      split->found_code_pre_split = true;
    }

    // Add the line to the current file's content (which may not be a split
    // file).
    split->add_content(line);
    return false;
  }

  if (!found_autoupdate) {
    // If there's a split, all output is appended at the end of each file
    // before AUTOUPDATE. We may want to change that, but it's not
    // necessary to handle right now.
    return Error(
        "AUTOUPDATE/NOAUTOUPDATE setting must be in "
        "the first file.");
  }

  // On a file split, add the previous file, then start a new one.
  if (split->has_splits()) {
    CARBON_RETURN_IF_ERROR(
        AddSplit(split->filename, &split->content, file_splits));
  } else {
    split->content.clear();
    if (split->found_code_pre_split) {
      // For the first split, we make sure there was no content prior.
      return Error(
          "When using split files, there must be no content before the first "
          "split file.");
    }
  }

  ++split->file_index;
  split->filename = line_trimmed.trim();
  if (split->filename.empty()) {
    return Error("Missing filename for split.");
  }
  // The split line is added to non_check_lines for retention in autoupdate, but
  // is not added to the test file content.
  *line_index = 0;
  non_check_lines->push_back(
      FileTestLine(split->file_index, *line_index, line));
  return true;
}

// Converts a `FileCheck`-style expectation string into a single complete regex
// string by escaping all regex characters outside of the designated `{{...}}`
// regex sequences, and switching those to a normal regex sub-pattern syntax.
static auto ConvertExpectationStringToRegex(std::string& str) -> void {
  for (int pos = 0; pos < static_cast<int>(str.size());) {
    switch (str[pos]) {
      case '(':
      case ')':
      case '[':
      case ']':
      case '}':
      case '.':
      case '^':
      case '$':
      case '*':
      case '+':
      case '?':
      case '|':
      case '\\': {
        // Escape regex characters.
        str.insert(pos, "\\");
        pos += 2;
        break;
      }
      case '{': {
        if (pos + 1 == static_cast<int>(str.size()) || str[pos + 1] != '{') {
          // Single `{`, escape it.
          str.insert(pos, "\\");
          pos += 2;
          break;
        }

        // Replace the `{{...}}` regex syntax with standard `(...)` syntax.
        str.replace(pos, 2, "(");
        for (++pos; pos < static_cast<int>(str.size() - 1); ++pos) {
          if (str[pos] == '}' && str[pos + 1] == '}') {
            str.replace(pos, 2, ")");
            ++pos;
            break;
          }
        }
        break;
      }
      default: {
        ++pos;
      }
    }
  }
}

// Transforms an expectation on a given line from `FileCheck` syntax into a
// standard regex matcher.
static auto TransformExpectation(int line_index, llvm::StringRef in)
    -> ErrorOr<Matcher<std::string>> {
  if (in.empty()) {
    return Matcher<std::string>{StrEq("")};
  }
  if (!in.consume_front(" ")) {
    return ErrorBuilder() << "Malformated CHECK line: " << in;
  }

  // Check early if we have a regex component as we can avoid building an
  // expensive matcher when not using those.
  bool has_regex = in.find("{{") != llvm::StringRef::npos;

  // Now scan the string and expand any keywords. Note that this needs to be
  // `size_t` to correctly store `npos`.
  size_t keyword_pos = in.find("[[");

  // If there are neither keywords nor regex sequences, we can match the
  // incoming string directly.
  if (!has_regex && keyword_pos == llvm::StringRef::npos) {
    return Matcher<std::string>{StrEq(in)};
  }

  std::string str = in.str();

  // First expand the keywords.
  while (keyword_pos != std::string::npos) {
    llvm::StringRef line_keyword_cursor =
        llvm::StringRef(str).substr(keyword_pos);
    CARBON_CHECK(line_keyword_cursor.consume_front("[["));

    static constexpr llvm::StringLiteral LineKeyword = "@LINE";
    if (!line_keyword_cursor.consume_front(LineKeyword)) {
      return ErrorBuilder()
             << "Unexpected [[, should be {{\\[\\[}} at `"
             << line_keyword_cursor.substr(0, 5) << "` in: " << in;
    }

    // Allow + or - here; consumeInteger handles -.
    line_keyword_cursor.consume_front("+");
    int offset;
    // consumeInteger returns true for errors, not false.
    if (line_keyword_cursor.consumeInteger(10, offset) ||
        !line_keyword_cursor.consume_front("]]")) {
      return ErrorBuilder()
             << "Unexpected @LINE offset at `"
             << line_keyword_cursor.substr(0, 5) << "` in: " << in;
    }
    std::string int_str = llvm::Twine(line_index + offset).str();
    int remove_len = (line_keyword_cursor.data() - str.data()) - keyword_pos;
    str.replace(keyword_pos, remove_len, int_str);
    keyword_pos += int_str.size();
    // Find the next keyword start or the end of the string.
    keyword_pos = str.find("[[", keyword_pos);
  }

  // If there was no regex, we can directly match the adjusted string.
  if (!has_regex) {
    return Matcher<std::string>{StrEq(str)};
  }

  // Otherwise, we need to turn the entire string into a regex by escaping
  // things outside the regex region and transforming the regex region into a
  // normal syntax.
  ConvertExpectationStringToRegex(str);
  return Matcher<std::string>{MatchesRegex(str)};
}

// Once all content is processed, do any remaining split processing.
static auto FinishSplit(llvm::StringRef test_name, SplitState* split,
                        llvm::SmallVector<TestFile::Split>* file_splits)
    -> ErrorOr<Success> {
  if (split->has_splits()) {
    return AddSplit(split->filename, &split->content, file_splits);
  } else {
    // If no file splitting happened, use the main file as the test file.
    // There will always be a `/` unless tests are in the repo root.
    return AddSplit(test_name.drop_front(test_name.rfind("/") + 1),
                    &split->content, file_splits);
  }
}

// Process CHECK lines when found. Returns true if the line is consumed.
static auto TryConsumeCheck(
    bool running_autoupdate, int line_index, llvm::StringRef line,
    llvm::StringRef line_trimmed,
    llvm::SmallVector<testing::Matcher<std::string>>* expected_stdout,
    llvm::SmallVector<testing::Matcher<std::string>>* expected_stderr)
    -> ErrorOr<bool> {
  if (!line_trimmed.consume_front("// CHECK")) {
    return false;
  }

  // Don't build expectations when doing an autoupdate. We don't want to
  // break the autoupdate on an invalid CHECK line.
  if (!running_autoupdate) {
    llvm::SmallVector<Matcher<std::string>>* expected;
    if (line_trimmed.consume_front(":STDOUT:")) {
      expected = expected_stdout;
    } else if (line_trimmed.consume_front(":STDERR:")) {
      expected = expected_stderr;
    } else {
      return ErrorBuilder() << "Unexpected CHECK in input: " << line.str();
    }
    CARBON_ASSIGN_OR_RETURN(Matcher<std::string> check_matcher,
                            TransformExpectation(line_index, line_trimmed));
    expected->push_back(check_matcher);
  }
  return true;
}

// Processes ARGS and EXTRA-ARGS lines when found. Returns true if the line is
// consumed.
static auto TryConsumeArgs(llvm::StringRef line, llvm::StringRef line_trimmed,
                           llvm::SmallVector<std::string>* args,
                           llvm::SmallVector<std::string>* extra_args)
    -> ErrorOr<bool> {
  llvm::SmallVector<std::string>* arg_list = nullptr;
  if (line_trimmed.consume_front("// ARGS: ")) {
    arg_list = args;
  } else if (line_trimmed.consume_front("// EXTRA-ARGS: ")) {
    arg_list = extra_args;
  } else {
    return false;
  }

  if (!args->empty() || !extra_args->empty()) {
    return ErrorBuilder() << "ARGS / EXTRA-ARGS specified multiple times: "
                          << line.str();
  }

  // Split the line into arguments.
  std::pair<llvm::StringRef, llvm::StringRef> cursor =
      llvm::getToken(line_trimmed);
  while (!cursor.first.empty()) {
    arg_list->push_back(std::string(cursor.first));
    cursor = llvm::getToken(cursor.second);
  }

  return true;
}

static auto TryConsumeIncludeFile(llvm::StringRef line_trimmed,
                                  llvm::SmallVector<std::string>* include_files)
    -> ErrorOr<bool> {
  if (!line_trimmed.consume_front("// INCLUDE-FILE: ")) {
    return false;
  }

  include_files->push_back(std::string(line_trimmed));
  return true;
}

// Processes AUTOUPDATE lines when found. Returns true if the line is consumed.
static auto TryConsumeAutoupdate(int line_index, llvm::StringRef line_trimmed,
                                 bool* found_autoupdate,
                                 std::optional<int>* autoupdate_line_number)
    -> ErrorOr<bool> {
  static constexpr llvm::StringLiteral Autoupdate = "// AUTOUPDATE";
  static constexpr llvm::StringLiteral NoAutoupdate = "// NOAUTOUPDATE";
  if (line_trimmed != Autoupdate && line_trimmed != NoAutoupdate) {
    return false;
  }
  if (*found_autoupdate) {
    return Error("Multiple AUTOUPDATE/NOAUTOUPDATE settings found");
  }
  *found_autoupdate = true;
  if (line_trimmed == Autoupdate) {
    *autoupdate_line_number = line_index;
  }
  return true;
}

// Processes SET-* lines when found. Returns true if the line is consumed.
static auto TryConsumeSetFlag(llvm::StringRef line_trimmed,
                              llvm::StringLiteral flag_name, bool* flag)
    -> ErrorOr<bool> {
  if (!line_trimmed.consume_front("// ") || line_trimmed != flag_name) {
    return false;
  }
  if (*flag) {
    return ErrorBuilder() << flag_name << " was specified multiple times";
  }
  *flag = true;
  return true;
}

auto ProcessTestFile(llvm::StringRef test_name, bool running_autoupdate)
    -> ErrorOr<TestFile> {
  TestFile test_file;

  // Store the file so that file_splits can use references to content.
  CARBON_ASSIGN_OR_RETURN(test_file.input_content, ReadFile(test_name.str()));

  // Original file content, and a cursor for walking through it.
  llvm::StringRef file_content = test_file.input_content;
  llvm::StringRef cursor = file_content;

  // Whether either AUTOUDPATE or NOAUTOUPDATE was found.
  bool found_autoupdate = false;

  // The index in the current test file. Will be reset on splits.
  int line_index = 0;

  SplitState split;

  // When autoupdating, we track whether we're inside conflict markers.
  // Otherwise conflict markers are errors.
  bool inside_conflict_marker = false;

  while (!cursor.empty()) {
    auto [line, next_cursor] = cursor.split("\n");
    cursor = next_cursor;
    auto line_trimmed = line.ltrim();

    bool is_consumed = false;
    CARBON_ASSIGN_OR_RETURN(
        is_consumed,
        TryConsumeConflictMarker(running_autoupdate, line, line_trimmed,
                                 &inside_conflict_marker));
    if (is_consumed) {
      continue;
    }

    // At this point, remaining lines are part of the test input.
    CARBON_ASSIGN_OR_RETURN(
        is_consumed,
        TryConsumeSplit(line, line_trimmed, found_autoupdate, &line_index,
                        &split, &test_file.file_splits,
                        &test_file.non_check_lines));
    if (is_consumed) {
      continue;
    }

    ++line_index;

    // TIP lines have no impact on validation.
    if (line_trimmed.starts_with("// TIP:")) {
      continue;
    }

    CARBON_ASSIGN_OR_RETURN(
        is_consumed, TryConsumeCheck(running_autoupdate, line_index, line,
                                     line_trimmed, &test_file.expected_stdout,
                                     &test_file.expected_stderr));
    if (is_consumed) {
      continue;
    }

    // At this point, lines are retained as non-CHECK lines.
    test_file.non_check_lines.push_back(
        FileTestLine(split.file_index, line_index, line));

    CARBON_ASSIGN_OR_RETURN(
        is_consumed, TryConsumeArgs(line, line_trimmed, &test_file.test_args,
                                    &test_file.extra_args));
    if (is_consumed) {
      continue;
    }
    CARBON_ASSIGN_OR_RETURN(
        is_consumed,
        TryConsumeIncludeFile(line_trimmed, &test_file.include_files));
    if (is_consumed) {
      continue;
    }
    CARBON_ASSIGN_OR_RETURN(
        is_consumed,
        TryConsumeAutoupdate(line_index, line_trimmed, &found_autoupdate,
                             &test_file.autoupdate_line_number));
    if (is_consumed) {
      continue;
    }
    CARBON_ASSIGN_OR_RETURN(
        is_consumed,
        TryConsumeSetFlag(line_trimmed, "SET-CAPTURE-CONSOLE-OUTPUT",
                          &test_file.capture_console_output));
    if (is_consumed) {
      continue;
    }
    CARBON_ASSIGN_OR_RETURN(is_consumed,
                            TryConsumeSetFlag(line_trimmed, "SET-CHECK-SUBSET",
                                              &test_file.check_subset));
    if (is_consumed) {
      continue;
    }
  }

  if (!found_autoupdate) {
    return ErrorBuilder() << "Missing AUTOUPDATE/NOAUTOUPDATE setting: "
                          << test_name;
  }

  test_file.has_splits = split.has_splits();
  CARBON_RETURN_IF_ERROR(
      FinishSplit(test_name, &split, &test_file.file_splits));

  // Validate AUTOUPDATE-SPLIT use, and remove it from test files if present.
  if (test_file.has_splits) {
    constexpr llvm::StringLiteral AutoupdateSplit = "AUTOUPDATE-SPLIT";
    for (const auto& test_file :
         llvm::ArrayRef(test_file.file_splits).drop_back()) {
      if (test_file.filename == AutoupdateSplit) {
        return Error("AUTOUPDATE-SPLIT must be the last split");
      }
    }
    if (test_file.file_splits.back().filename == AutoupdateSplit) {
      if (!test_file.autoupdate_line_number) {
        return Error("AUTOUPDATE-SPLIT requires AUTOUPDATE");
      }
      test_file.autoupdate_split = true;
      test_file.file_splits.pop_back();
    }
  }

  // Assume there is always a suffix `\n` in output.
  if (!test_file.expected_stdout.empty()) {
    test_file.expected_stdout.push_back(StrEq(""));
  }
  if (!test_file.expected_stderr.empty()) {
    test_file.expected_stderr.push_back(StrEq(""));
  }

  return std::move(test_file);
}

}  // namespace Carbon::Testing
