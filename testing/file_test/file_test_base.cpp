// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/file_test/file_test_base.h"

#include <gmock/gmock.h>

#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "common/check.h"
#include "common/error.h"
#include "common/exe_path.h"
#include "common/init_llvm.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/ThreadPool.h"
#include "testing/file_test/autoupdate.h"

ABSL_FLAG(std::vector<std::string>, file_tests, {},
          "A comma-separated list of repo-relative names of test files. "
          "Overrides test_targets_file.");
ABSL_FLAG(std::string, test_targets_file, "",
          "A path to a file containing repo-relative names of test files.");
ABSL_FLAG(bool, autoupdate, false,
          "Instead of verifying files match test output, autoupdate files "
          "based on test output.");
ABSL_FLAG(unsigned int, threads, 0,
          "Number of threads to use when autoupdating tests, or 0 to "
          "automatically determine a thread count.");
ABSL_FLAG(bool, dump_output, false,
          "Instead of verifying files match test output, directly dump output "
          "to stderr.");

namespace Carbon::Testing {

// While these are marked as "internal" APIs, they seem to work and be pretty
// widely used for their exact documented behavior.
using ::testing::internal::CaptureStderr;
using ::testing::internal::CaptureStdout;
using ::testing::internal::GetCapturedStderr;
using ::testing::internal::GetCapturedStdout;

using ::testing::Matcher;
using ::testing::MatchesRegex;
using ::testing::StrEq;

// Reads a file to string.
static auto ReadFile(std::string_view path) -> ErrorOr<std::string> {
  std::ifstream proto_file{std::string(path)};
  if (proto_file.fail()) {
    return Error(llvm::formatv("Error opening file: {0}", path));
  }
  std::stringstream buffer;
  buffer << proto_file.rdbuf();
  if (proto_file.fail()) {
    return Error(llvm::formatv("Error reading file: {0}", path));
  }
  proto_file.close();
  return buffer.str();
}

// Splits outputs to string_view because gtest handles string_view by default.
static auto SplitOutput(llvm::StringRef output)
    -> llvm::SmallVector<std::string_view> {
  if (output.empty()) {
    return {};
  }
  llvm::SmallVector<llvm::StringRef> lines;
  llvm::StringRef(output).split(lines, "\n");
  return llvm::SmallVector<std::string_view>(lines.begin(), lines.end());
}

// Verify that the success and `fail_` prefix use correspond. Separately handle
// both cases for clearer test failures.
static auto CompareFailPrefix(llvm::StringRef filename, bool success) -> void {
  if (success) {
    EXPECT_FALSE(filename.starts_with("fail_"))
        << "`" << filename
        << "` succeeded; if success is expected, remove the `fail_` "
           "prefix.";
  } else {
    EXPECT_TRUE(filename.starts_with("fail_"))
        << "`" << filename
        << "` failed; if failure is expected, add the `fail_` prefix.";
  }
}

// Modes for GetBazelCommand.
enum class BazelMode {
  Autoupdate,
  Dump,
  Test,
};

// Returns the requested bazel command string for the given execution mode.
static auto GetBazelCommand(BazelMode mode, llvm::StringRef test_name)
    -> std::string {
  std::string args_str;
  llvm::raw_string_ostream args(args_str);

  const char* target = getenv("TEST_TARGET");
  args << "bazel " << ((mode == BazelMode::Test) ? "test" : "run") << " "
       << (target ? target : "<target>") << " ";

  switch (mode) {
    case BazelMode::Autoupdate:
      args << "-- --autoupdate ";
      break;

    case BazelMode::Dump:
      args << "-- --dump_output ";
      break;

    case BazelMode::Test:
      args << "--test_arg=";
      break;
  }

  args << "--file_tests=";
  args << test_name;
  return args_str;
}

// Runs a test and compares output. This keeps output split by line so that
// issues are a little easier to identify by the different line.
auto FileTestBase::TestBody() -> void {
  // Add a crash trace entry with the single-file test command.
  std::string test_command = GetBazelCommand(BazelMode::Test, test_name_);
  llvm::PrettyStackTraceString stack_trace_entry(test_command.c_str());
  llvm::errs() << "\nTo test this file alone, run:\n  " << test_command
               << "\n\n";

  TestContext context;
  auto run_result = ProcessTestFileAndRun(context);
  ASSERT_TRUE(run_result.ok()) << run_result.error();
  ValidateRun();
  auto test_filename = std::filesystem::path(test_name_.str()).filename();

  // Check success/failure against `fail_` prefixes.
  if (context.run_result.per_file_success.empty()) {
    CompareFailPrefix(test_filename.string(), context.run_result.success);
  } else {
    bool require_overall_failure = false;
    for (const auto& [filename, success] :
         context.run_result.per_file_success) {
      CompareFailPrefix(filename, success);
      if (!success) {
        require_overall_failure = true;
      }
    }

    if (require_overall_failure) {
      EXPECT_FALSE(context.run_result.success)
          << "There is a per-file failure expectation, so the overall result "
             "should have been a failure.";
    } else {
      // Individual files all succeeded, so the prefix is enforced on the main
      // test file.
      CompareFailPrefix(test_filename.string(), context.run_result.success);
    }
  }

  // Check results. Include a reminder of the autoupdate command for any
  // stdout/stderr differences.
  std::string update_message;
  if (context.autoupdate_line_number) {
    update_message = llvm::formatv(
        "If these differences are expected, try the autoupdater:\n  {0}",
        GetBazelCommand(BazelMode::Autoupdate, test_name_));
  } else {
    update_message =
        "If these differences are expected, content must be updated manually.";
  }
  SCOPED_TRACE(update_message);
  if (context.check_subset) {
    EXPECT_THAT(SplitOutput(context.stdout),
                IsSupersetOf(context.expected_stdout));
    EXPECT_THAT(SplitOutput(context.stderr),
                IsSupersetOf(context.expected_stderr));

  } else {
    EXPECT_THAT(SplitOutput(context.stdout),
                ElementsAreArray(context.expected_stdout));
    EXPECT_THAT(SplitOutput(context.stderr),
                ElementsAreArray(context.expected_stderr));
  }

  // If there are no other test failures, check if autoupdate would make
  // changes. We don't do this when there _are_ failures because the
  // SCOPED_TRACE already contains the autoupdate reminder.
  if (!HasFailure() && RunAutoupdater(context, /*dry_run=*/true)) {
    ADD_FAILURE() << "Autoupdate would make changes to the file content.";
  }
}

auto FileTestBase::RunAutoupdater(const TestContext& context, bool dry_run)
    -> bool {
  if (!context.autoupdate_line_number) {
    return false;
  }

  llvm::SmallVector<llvm::StringRef> filenames;
  filenames.reserve(context.non_check_lines.size());
  if (context.has_splits) {
    // There are splits, so we provide an empty name for the first file.
    filenames.push_back({});
  }
  for (const auto& file : context.test_files) {
    filenames.push_back(file.filename);
  }

  llvm::ArrayRef expected_filenames = filenames;
  if (filenames.size() > 1) {
    expected_filenames = expected_filenames.drop_front();
  }

  return FileTestAutoupdater(
             std::filesystem::absolute(test_name_.str()),
             GetBazelCommand(BazelMode::Test, test_name_),
             GetBazelCommand(BazelMode::Dump, test_name_),
             context.input_content, filenames, *context.autoupdate_line_number,
             context.autoupdate_split, context.non_check_lines, context.stdout,
             context.stderr, GetDefaultFileRE(expected_filenames),
             GetLineNumberReplacements(expected_filenames),
             [&](std::string& line) { DoExtraCheckReplacements(line); })
      .Run(dry_run);
}

auto FileTestBase::Autoupdate() -> ErrorOr<bool> {
  // Add a crash trace entry mentioning which file we're updating.
  std::string stack_trace_string =
      llvm::formatv("performing autoupdate for {0}", test_name_);
  llvm::PrettyStackTraceString stack_trace_entry(stack_trace_string.c_str());

  TestContext context;
  auto run_result = ProcessTestFileAndRun(context);
  if (!run_result.ok()) {
    return ErrorBuilder() << "Error updating " << test_name_ << ": "
                          << run_result.error();
  }
  return RunAutoupdater(context, /*dry_run=*/false);
}

auto FileTestBase::DumpOutput() -> ErrorOr<Success> {
  TestContext context;
  context.dump_output = true;
  std::string banner(79, '=');
  banner.append("\n");
  llvm::errs() << banner << "= " << test_name_ << "\n";

  auto run_result = ProcessTestFileAndRun(context);
  if (!run_result.ok()) {
    return ErrorBuilder() << "Error updating " << test_name_ << ": "
                          << run_result.error();
  }
  llvm::errs() << banner << context.stdout << banner << "= Exit with success: "
               << (context.run_result.success ? "true" : "false") << "\n"
               << banner;
  return Success();
}

auto FileTestBase::GetLineNumberReplacements(
    llvm::ArrayRef<llvm::StringRef> filenames)
    -> llvm::SmallVector<LineNumberReplacement> {
  return {{.has_file = true,
           .re = std::make_shared<RE2>(
               llvm::formatv(R"(({0}):(\d+))", llvm::join(filenames, "|"))),
           .line_formatv = R"({0})"}};
}

auto FileTestBase::ProcessTestFileAndRun(TestContext& context)
    -> ErrorOr<Success> {
  // Store the file so that test_files can use references to content.
  CARBON_ASSIGN_OR_RETURN(context.input_content, ReadFile(test_name_));

  // Load expected output.
  CARBON_RETURN_IF_ERROR(ProcessTestFile(context));

  // Process arguments.
  if (context.test_args.empty()) {
    context.test_args = GetDefaultArgs();
  }
  CARBON_RETURN_IF_ERROR(
      DoArgReplacements(context.test_args, context.test_files));

  // Create the files in-memory.
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> fs =
      new llvm::vfs::InMemoryFileSystem;
  for (const auto& test_file : context.test_files) {
    if (!fs->addFile(test_file.filename, /*ModificationTime=*/0,
                     llvm::MemoryBuffer::getMemBuffer(
                         test_file.content, test_file.filename,
                         /*RequiresNullTerminator=*/false))) {
      return ErrorBuilder() << "File is repeated: " << test_file.filename;
    }
  }

  // Convert the arguments to StringRef and const char* to match the
  // expectations of PrettyStackTraceProgram and Run.
  llvm::SmallVector<llvm::StringRef> test_args_ref;
  llvm::SmallVector<const char*> test_argv_for_stack_trace;
  test_args_ref.reserve(context.test_args.size());
  test_argv_for_stack_trace.reserve(context.test_args.size() + 1);
  for (const auto& arg : context.test_args) {
    test_args_ref.push_back(arg);
    test_argv_for_stack_trace.push_back(arg.c_str());
  }
  // Add a trailing null so that this is a proper argv.
  test_argv_for_stack_trace.push_back(nullptr);

  // Add a stack trace entry for the test invocation.
  llvm::PrettyStackTraceProgram stack_trace_entry(
      test_argv_for_stack_trace.size() - 1, test_argv_for_stack_trace.data());

  // Conditionally capture console output. We use a scope exit to ensure the
  // captures terminate even on run failures.
  std::unique_lock<std::mutex> output_lock;
  if (context.capture_console_output) {
    if (output_mutex_) {
      output_lock = std::unique_lock<std::mutex>(*output_mutex_);
    }
    CaptureStderr();
    CaptureStdout();
  }
  auto add_output_on_exit = llvm::make_scope_exit([&]() {
    if (context.capture_console_output) {
      // No need to flush stderr.
      llvm::outs().flush();
      context.stdout += GetCapturedStdout();
      context.stderr += GetCapturedStderr();
    }
  });

  // Prepare string streams to capture output. In order to address casting
  // constraints, we split calls to Run as a ternary based on whether we want to
  // capture output.
  llvm::raw_svector_ostream stdout(context.stdout);
  llvm::raw_svector_ostream stderr(context.stderr);
  CARBON_ASSIGN_OR_RETURN(
      context.run_result,
      context.dump_output ? Run(test_args_ref, fs, llvm::outs(), llvm::errs())
                          : Run(test_args_ref, fs, stdout, stderr));

  return Success();
}

auto FileTestBase::DoArgReplacements(
    llvm::SmallVector<std::string>& test_args,
    const llvm::SmallVector<TestFile>& test_files) -> ErrorOr<Success> {
  auto replacements = GetArgReplacements();
  for (auto* it = test_args.begin(); it != test_args.end(); ++it) {
    auto percent = it->find("%");
    if (percent == std::string::npos) {
      continue;
    }

    if (percent + 1 >= it->size()) {
      return ErrorBuilder() << "% is not allowed on its own: " << *it;
    }
    char c = (*it)[percent + 1];
    switch (c) {
      case 's': {
        if (*it != "%s") {
          return ErrorBuilder() << "%s must be the full argument: " << *it;
        }
        it = test_args.erase(it);
        for (const auto& file : test_files) {
          it = test_args.insert(it, file.filename);
          ++it;
        }
        // Back up once because the for loop will advance.
        --it;
        break;
      }
      case 't': {
        char* tmpdir = getenv("TEST_TMPDIR");
        CARBON_CHECK(tmpdir != nullptr);
        it->replace(percent, 2, llvm::formatv("{0}/temp_file", tmpdir));
        break;
      }
      case '{': {
        auto end_brace = it->find('}', percent);
        if (end_brace == std::string::npos) {
          return ErrorBuilder() << "%{ without closing }: " << *it;
        }
        llvm::StringRef substr(&*(it->begin() + percent + 2),
                               end_brace - percent - 2);
        auto replacement = replacements.find(substr);
        if (replacement == replacements.end()) {
          return ErrorBuilder()
                 << "unknown substitution: %{" << substr << "}: " << *it;
        }
        it->replace(percent, end_brace - percent + 1, replacement->second);
        break;
      }
      default:
        return ErrorBuilder() << "%" << c << " is not supported: " << *it;
    }
  }
  return Success();
}

// Processes conflict markers, including tracking of whether code is within a
// conflict marker. Returns true if the line is consumed.
static auto TryConsumeConflictMarker(llvm::StringRef line,
                                     llvm::StringRef line_trimmed,
                                     bool* inside_conflict_marker)
    -> ErrorOr<bool> {
  bool is_start = line.starts_with("<<<<<<<");
  bool is_middle = line.starts_with("=======") || line.starts_with("|||||||");
  bool is_end = line.starts_with(">>>>>>>");

  // When running the test, any conflict marker is an error.
  if (!absl::GetFlag(FLAGS_autoupdate) && (is_start || is_middle || is_end)) {
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

// Replaces the content keywords.
//
// TEST_NAME is the only content keyword at present, but we do validate that
// other names are reserved.
static auto ReplaceContentKeywords(llvm::StringRef filename,
                                   std::string* content) -> ErrorOr<Success> {
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

  while (keyword_pos != std::string::npos) {
    static constexpr llvm::StringLiteral TestName = "[[@TEST_NAME]]";
    auto keyword = llvm::StringRef(*content).substr(keyword_pos);
    if (keyword.starts_with(TestName)) {
      content->replace(keyword_pos, TestName.size(), test_name);
      keyword_pos += test_name.size();
    } else if (keyword.starts_with("[[@LINE")) {
      // Just move past the prefix to find the next one.
      keyword_pos += Prefix.size();
    } else {
      return ErrorBuilder()
             << "Unexpected use of `[[@` at `" << keyword.substr(0, 5) << "`";
    }
    keyword_pos = content->find(Prefix, keyword_pos);
  }
  return Success();
}

// Adds a file. Used for both split and unsplit test files.
static auto AddTestFile(llvm::StringRef filename, std::string* content,
                        llvm::SmallVector<FileTestBase::TestFile>* test_files)
    -> ErrorOr<Success> {
  CARBON_RETURN_IF_ERROR(ReplaceContentKeywords(filename, content));

  test_files->push_back(
      {.filename = filename.str(), .content = std::move(*content)});
  content->clear();
  return Success();
}

// Process file split ("---") lines when found. Returns true if the line is
// consumed.
static auto TryConsumeSplit(
    llvm::StringRef line, llvm::StringRef line_trimmed, bool found_autoupdate,
    int* line_index, SplitState* split,
    llvm::SmallVector<FileTestBase::TestFile>* test_files,
    llvm::SmallVector<FileTestLine>* non_check_lines) -> ErrorOr<bool> {
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
    return ErrorBuilder() << "AUTOUPDATE/NOAUTOUPDATE setting must be in "
                             "the first file.";
  }

  // On a file split, add the previous file, then start a new one.
  if (split->has_splits()) {
    CARBON_RETURN_IF_ERROR(
        AddTestFile(split->filename, &split->content, test_files));
  } else {
    split->content.clear();
    if (split->found_code_pre_split) {
      // For the first split, we make sure there was no content prior.
      return ErrorBuilder() << "When using split files, there must be no "
                               "content before the first split file.";
    }
  }

  ++split->file_index;
  split->filename = line_trimmed.trim();
  if (split->filename.empty()) {
    return ErrorBuilder() << "Missing filename for split.";
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
static void ConvertExpectationStringToRegex(std::string& str) {
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
                        llvm::SmallVector<FileTestBase::TestFile>* test_files)
    -> ErrorOr<Success> {
  if (split->has_splits()) {
    return AddTestFile(split->filename, &split->content, test_files);
  } else {
    // If no file splitting happened, use the main file as the test file.
    // There will always be a `/` unless tests are in the repo root.
    return AddTestFile(test_name.drop_front(test_name.rfind("/") + 1),
                       &split->content, test_files);
  }
}

// Process CHECK lines when found. Returns true if the line is consumed.
static auto TryConsumeCheck(
    int line_index, llvm::StringRef line, llvm::StringRef line_trimmed,
    llvm::SmallVector<testing::Matcher<std::string>>* expected_stdout,
    llvm::SmallVector<testing::Matcher<std::string>>* expected_stderr)
    -> ErrorOr<bool> {
  if (!line_trimmed.consume_front("// CHECK")) {
    return false;
  }

  // Don't build expectations when doing an autoupdate. We don't want to
  // break the autoupdate on an invalid CHECK line.
  if (!absl::GetFlag(FLAGS_autoupdate)) {
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

// Processes ARGS lines when found. Returns true if the line is consumed.
static auto TryConsumeArgs(llvm::StringRef line, llvm::StringRef line_trimmed,
                           llvm::SmallVector<std::string>* args)
    -> ErrorOr<bool> {
  if (!line_trimmed.consume_front("// ARGS: ")) {
    return false;
  }

  if (!args->empty()) {
    return ErrorBuilder() << "ARGS was specified multiple times: "
                          << line.str();
  }

  // Split the line into arguments.
  std::pair<llvm::StringRef, llvm::StringRef> cursor =
      llvm::getToken(line_trimmed);
  while (!cursor.first.empty()) {
    args->push_back(std::string(cursor.first));
    cursor = llvm::getToken(cursor.second);
  }

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
    return ErrorBuilder() << "Multiple AUTOUPDATE/NOAUTOUPDATE settings found";
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

auto FileTestBase::ProcessTestFile(TestContext& context) -> ErrorOr<Success> {
  // Original file content, and a cursor for walking through it.
  llvm::StringRef file_content = context.input_content;
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
        TryConsumeConflictMarker(line, line_trimmed, &inside_conflict_marker));
    if (is_consumed) {
      continue;
    }

    // At this point, remaining lines are part of the test input.
    CARBON_ASSIGN_OR_RETURN(
        is_consumed,
        TryConsumeSplit(line, line_trimmed, found_autoupdate, &line_index,
                        &split, &context.test_files, &context.non_check_lines));
    if (is_consumed) {
      continue;
    }

    ++line_index;

    // TIP lines have no impact on validation.
    if (line_trimmed.starts_with("// TIP:")) {
      continue;
    }

    CARBON_ASSIGN_OR_RETURN(
        is_consumed,
        TryConsumeCheck(line_index, line, line_trimmed,
                        &context.expected_stdout, &context.expected_stderr));
    if (is_consumed) {
      continue;
    }

    // At this point, lines are retained as non-CHECK lines.
    context.non_check_lines.push_back(
        FileTestLine(split.file_index, line_index, line));

    CARBON_ASSIGN_OR_RETURN(
        is_consumed, TryConsumeArgs(line, line_trimmed, &context.test_args));
    if (is_consumed) {
      continue;
    }
    CARBON_ASSIGN_OR_RETURN(
        is_consumed,
        TryConsumeAutoupdate(line_index, line_trimmed, &found_autoupdate,
                             &context.autoupdate_line_number));
    if (is_consumed) {
      continue;
    }
    CARBON_ASSIGN_OR_RETURN(
        is_consumed,
        TryConsumeSetFlag(line_trimmed, "SET-CAPTURE-CONSOLE-OUTPUT",
                          &context.capture_console_output));
    if (is_consumed) {
      continue;
    }
    CARBON_ASSIGN_OR_RETURN(is_consumed,
                            TryConsumeSetFlag(line_trimmed, "SET-CHECK-SUBSET",
                                              &context.check_subset));
    if (is_consumed) {
      continue;
    }
  }

  if (!found_autoupdate) {
    return Error("Missing AUTOUPDATE/NOAUTOUPDATE setting");
  }

  context.has_splits = split.has_splits();
  CARBON_RETURN_IF_ERROR(FinishSplit(test_name_, &split, &context.test_files));

  // Validate AUTOUPDATE-SPLIT use, and remove it from test files if present.
  if (context.has_splits) {
    constexpr llvm::StringLiteral AutoupdateSplit = "AUTOUPDATE-SPLIT";
    for (const auto& test_file :
         llvm::ArrayRef(context.test_files).drop_back()) {
      if (test_file.filename == AutoupdateSplit) {
        return Error("AUTOUPDATE-SPLIT must be the last split");
      }
    }
    if (context.test_files.back().filename == AutoupdateSplit) {
      if (!context.autoupdate_line_number) {
        return Error("AUTOUPDATE-SPLIT requires AUTOUPDATE");
      }
      context.autoupdate_split = true;
      context.test_files.pop_back();
    }
  }

  // Assume there is always a suffix `\n` in output.
  if (!context.expected_stdout.empty()) {
    context.expected_stdout.push_back(StrEq(""));
  }
  if (!context.expected_stderr.empty()) {
    context.expected_stderr.push_back(StrEq(""));
  }

  return Success();
}

// Returns the tests to run.
static auto GetTests() -> llvm::SmallVector<std::string> {
  // Prefer a user-specified list if present.
  auto specific_tests = absl::GetFlag(FLAGS_file_tests);
  if (!specific_tests.empty()) {
    return llvm::SmallVector<std::string>(specific_tests.begin(),
                                          specific_tests.end());
  }

  // Extracts tests from the target file.
  CARBON_CHECK(!absl::GetFlag(FLAGS_test_targets_file).empty(),
               "Missing --test_targets_file.");
  auto content = ReadFile(absl::GetFlag(FLAGS_test_targets_file));
  CARBON_CHECK(content.ok(), "{0}", content.error());
  llvm::SmallVector<std::string> all_tests;
  for (llvm::StringRef file_ref : llvm::split(*content, "\n")) {
    if (file_ref.empty()) {
      continue;
    }
    all_tests.push_back(file_ref.str());
  }
  return all_tests;
}

// Runs autoupdate for the given tests. This is multi-threaded to try to get a
// little extra speed.
static auto RunAutoupdate(llvm::StringRef exe_path,
                          llvm::ArrayRef<std::string> tests,
                          FileTestFactory& test_factory) -> int {
  llvm::CrashRecoveryContext::Enable();
  llvm::DefaultThreadPool pool(
      {.ThreadsRequested = absl::GetFlag(FLAGS_threads)});

  // Guard access to both `llvm::errs` and `crashed`.
  std::mutex mutex;
  bool crashed = false;

  for (const auto& test_name : tests) {
    pool.async([&test_factory, &mutex, &exe_path, &crashed, test_name] {
      // If any thread crashed, don't try running more.
      {
        std::unique_lock<std::mutex> lock(mutex);
        if (crashed) {
          return;
        }
      }

      // Use a crash recovery context to try to get a stack trace when
      // multiple threads may crash in parallel, which otherwise leads to the
      // program aborting without printing a stack trace.
      llvm::CrashRecoveryContext crc;
      crc.DumpStackAndCleanupOnFailure = true;
      bool thread_crashed = !crc.RunSafely([&] {
        std::unique_ptr<FileTestBase> test(
            test_factory.factory_fn(exe_path, &mutex, test_name));
        auto result = test->Autoupdate();

        std::unique_lock<std::mutex> lock(mutex);
        if (result.ok()) {
          llvm::errs() << (*result ? "!" : ".");
        } else {
          llvm::errs() << "\n" << result.error().message() << "\n";
        }
      });
      if (thread_crashed) {
        std::unique_lock<std::mutex> lock(mutex);
        crashed = true;
      }
    });
  }

  pool.wait();
  if (crashed) {
    // Abort rather than returning so that we don't get a LeakSanitizer report.
    // We expect to have leaked memory if one or more of our tests crashed.
    std::abort();
  }
  llvm::errs() << "\nDone!\n";
  return EXIT_SUCCESS;
}

// Implements main() within the Carbon::Testing namespace for convenience.
static auto Main(int argc, char** argv) -> int {
  Carbon::InitLLVM init_llvm(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  auto args = absl::ParseCommandLine(argc, argv);

  if (args.size() > 1) {
    llvm::errs() << "Unexpected arguments:";
    for (char* arg : llvm::ArrayRef(args).drop_front()) {
      llvm::errs() << " ";
      llvm::errs().write_escaped(arg);
    }
    llvm::errs() << "\n";
    return EXIT_FAILURE;
  }

  std::string exe_path = FindExecutablePath(argv[0]);

  // Tests might try to read from stdin. Ensure those reads fail by closing
  // stdin and reopening it as /dev/null. Note that STDIN_FILENO doesn't exist
  // on Windows, but POSIX requires it to be 0.
  if (std::error_code error =
          llvm::sys::Process::SafelyCloseFileDescriptor(0)) {
    llvm::errs() << "Unable to close standard input: " << error.message()
                 << "\n";
    return EXIT_FAILURE;
  }
  if (std::error_code error =
          llvm::sys::Process::FixupStandardFileDescriptors()) {
    llvm::errs() << "Unable to correct standard file descriptors: "
                 << error.message() << "\n";
    return EXIT_FAILURE;
  }
  if (absl::GetFlag(FLAGS_autoupdate) && absl::GetFlag(FLAGS_dump_output)) {
    llvm::errs() << "--autoupdate and --dump_output are mutually exclusive.\n";
    return EXIT_FAILURE;
  }

  llvm::SmallVector<std::string> tests = GetTests();
  auto test_factory = GetFileTestFactory();
  if (absl::GetFlag(FLAGS_autoupdate)) {
    return RunAutoupdate(exe_path, tests, test_factory);
  } else if (absl::GetFlag(FLAGS_dump_output)) {
    for (const auto& test_name : tests) {
      std::unique_ptr<FileTestBase> test(
          test_factory.factory_fn(exe_path, nullptr, test_name));
      auto result = test->DumpOutput();
      if (!result.ok()) {
        llvm::errs() << "\n" << result.error().message() << "\n";
      }
    }
    llvm::errs() << "\nDone!\n";
    return EXIT_SUCCESS;
  } else {
    for (llvm::StringRef test_name : tests) {
      testing::RegisterTest(
          test_factory.name, test_name.data(), nullptr, test_name.data(),
          __FILE__, __LINE__,
          [&test_factory, &exe_path, test_name = test_name]() {
            return test_factory.factory_fn(exe_path, nullptr, test_name);
          });
    }
    return RUN_ALL_TESTS();
  }
}

}  // namespace Carbon::Testing

auto main(int argc, char** argv) -> int {
  return Carbon::Testing::Main(argc, argv);
}
