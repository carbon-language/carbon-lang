// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/file_test/file_test_base.h"

#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "common/check.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "testing/file_test/autoupdate.h"

ABSL_FLAG(std::vector<std::string>, file_tests, {},
          "A comma-separated list of repo-relative names of test files. "
          "Overrides test_targets_file.");
ABSL_FLAG(std::string, test_targets_file, "",
          "A path to a file containing repo-relative names of test files.");
ABSL_FLAG(bool, autoupdate, false,
          "Instead of verifying files match test output, autoupdate files "
          "based on test output.");

namespace Carbon::Testing {

using ::testing::Eq;
using ::testing::Matcher;
using ::testing::MatchesRegex;
using ::testing::StrEq;

// Reads a file to string.
static auto ReadFile(std::string_view path) -> std::string {
  std::ifstream proto_file(path);
  std::stringstream buffer;
  buffer << proto_file.rdbuf();
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

// Runs a test and compares output. This keeps output split by line so that
// issues are a little easier to identify by the different line.
auto FileTestBase::TestBody() -> void {
  std::optional<llvm::PrettyStackTraceFormat> stack_trace_entry;

  // If we're being run from bazel, provide some assistance for understanding
  // and reproducing failures.
  const char* target = getenv("TEST_TARGET");
  if (target) {
    // This advice overrides the --file_tests flag provided by the file_test
    // rule.
    llvm::errs() << "\nTo test this file alone, run:\n  bazel test " << target
                 << " --test_arg=--file_tests=" << test_name_ << "\n\n";

    // Add a crash trace entry with a command that runs this test in isolation.
    stack_trace_entry.emplace("bazel test %s --test_arg=--file_tests=%s",
                              target, test_name_);
  }

  TestContext context;
  auto run_result = ProcessTestFileAndRun(context);
  ASSERT_TRUE(run_result.ok()) << run_result.error();
  ValidateRun();
  auto test_filename = std::filesystem::path(test_name_.str()).filename();
  EXPECT_THAT(!llvm::StringRef(test_filename).starts_with("fail_"),
              Eq(context.exit_with_success))
      << "Tests should be prefixed with `fail_` if and only if running them "
         "is expected to fail.";

  // Check results. Include a reminder of the autoupdate command for any
  // stdout/stderr differences.
  std::string update_message;
  if (target && context.autoupdate_line_number) {
    update_message = llvm::formatv(
        "If these differences are expected, try the autoupdater:\n"
        "\tbazel run {0} -- --autoupdate --file_tests={1}",
        target, test_name_);
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
             std::filesystem::absolute(test_name_.str()), context.input_content,
             filenames, *context.autoupdate_line_number,
             context.non_check_lines, context.stdout, context.stderr,
             GetDefaultFileRE(expected_filenames),
             GetLineNumberReplacements(expected_filenames),
             [&](std::string& line) { DoExtraCheckReplacements(line); })
      .Run(dry_run);
}

auto FileTestBase::Autoupdate() -> ErrorOr<bool> {
  // Add a crash trace entry mentioning which file we're updating.
  llvm::PrettyStackTraceFormat stack_trace_entry("performing autoupdate for %s",
                                                 test_name_);

  TestContext context;
  auto run_result = ProcessTestFileAndRun(context);
  if (!run_result.ok()) {
    return ErrorBuilder() << "Error updating " << test_name_ << ": "
                          << run_result.error();
  }
  return RunAutoupdater(context, /*dry_run=*/false);
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
  context.input_content = ReadFile(test_name_);

  // Load expected output.
  CARBON_RETURN_IF_ERROR(ProcessTestFile(context));

  // Process arguments.
  if (context.test_args.empty()) {
    context.test_args = GetDefaultArgs();
  }
  CARBON_RETURN_IF_ERROR(
      DoArgReplacements(context.test_args, context.test_files));

  // Create the files in-memory.
  llvm::vfs::InMemoryFileSystem fs;
  for (const auto& test_file : context.test_files) {
    if (!fs.addFile(test_file.filename, /*ModificationTime=*/0,
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

  // Capture trace streaming, but only when in debug mode.
  llvm::raw_svector_ostream stdout(context.stdout);
  llvm::raw_svector_ostream stderr(context.stderr);
  CARBON_ASSIGN_OR_RETURN(context.exit_with_success,
                          Run(test_args_ref, fs, stdout, stderr));
  return Success();
}

auto FileTestBase::DoArgReplacements(
    llvm::SmallVector<std::string>& test_args,
    const llvm::SmallVector<TestFile>& test_files) -> ErrorOr<Success> {
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
      default:
        return ErrorBuilder() << "%" << c << " is not supported: " << *it;
    }
  }
  return Success();
}

auto FileTestBase::ProcessTestFile(TestContext& context) -> ErrorOr<Success> {
  // Original file content, and a cursor for walking through it.
  llvm::StringRef file_content = context.input_content;
  llvm::StringRef cursor = file_content;

  // Whether content has been found, only updated before a file split is found
  // (which may be never).
  bool found_content_pre_split = false;

  // Whether either AUTOUDPATE or NOAUTOUPDATE was found.
  bool found_autoupdate = false;

  // The index in the current test file. Will be reset on splits.
  int line_index = 0;

  // The current file name, considering splits. Not set for the default file.
  llvm::StringRef current_file_name;

  // The current file's start.
  const char* current_file_start = nullptr;

  int file_number = 0;
  while (!cursor.empty()) {
    auto [line, next_cursor] = cursor.split("\n");
    cursor = next_cursor;
    auto line_trimmed = line.ltrim();

    static constexpr llvm::StringLiteral SplitPrefix = "// ---";
    if (line_trimmed.consume_front(SplitPrefix)) {
      if (!found_autoupdate) {
        // If there's a split, all output is appended at the end of each file
        // before AUTOUPDATE. We may want to change that, but it's not necessary
        // to handle right now.
        return ErrorBuilder()
               << "AUTOUPDATE/NOAUTOUPDATE setting must be in the first file.";
      }

      context.has_splits = true;
      ++file_number;
      context.non_check_lines.push_back(FileTestLine(file_number, 0, line));
      // On a file split, add the previous file, then start a new one.
      if (current_file_start) {
        context.test_files.push_back(TestFile(
            current_file_name.str(),
            llvm::StringRef(current_file_start, line_trimmed.begin() -
                                                    current_file_start -
                                                    SplitPrefix.size())));
      } else if (found_content_pre_split) {
        // For the first split, we make sure there was no content prior.
        return ErrorBuilder()
               << "When using split files, there must be no content before the "
                  "first split file.";
      }
      current_file_name = line_trimmed.trim();
      current_file_start = cursor.begin();
      line_index = 0;
      continue;
    } else if (!current_file_start && !line_trimmed.starts_with("//") &&
               !line_trimmed.empty()) {
      found_content_pre_split = true;
    }
    ++line_index;

    // Process expectations when found.
    if (line_trimmed.consume_front("// CHECK")) {
      // Don't build expectations when doing an autoupdate. We don't want to
      // break the autoupdate on an invalid CHECK line.
      if (!absl::GetFlag(FLAGS_autoupdate)) {
        llvm::SmallVector<Matcher<std::string>>* expected = nullptr;
        if (line_trimmed.consume_front(":STDOUT:")) {
          expected = &context.expected_stdout;
        } else if (line_trimmed.consume_front(":STDERR:")) {
          expected = &context.expected_stderr;
        } else {
          return ErrorBuilder() << "Unexpected CHECK in input: " << line.str();
        }
        CARBON_ASSIGN_OR_RETURN(Matcher<std::string> check_matcher,
                                TransformExpectation(line_index, line_trimmed));
        expected->push_back(check_matcher);
      }
    } else {
      context.non_check_lines.push_back(
          FileTestLine(file_number, line_index, line));
      if (line_trimmed.consume_front("// ARGS: ")) {
        if (context.test_args.empty()) {
          // Split the line into arguments.
          std::pair<llvm::StringRef, llvm::StringRef> cursor =
              llvm::getToken(line_trimmed);
          while (!cursor.first.empty()) {
            context.test_args.push_back(std::string(cursor.first));
            cursor = llvm::getToken(cursor.second);
          }
        } else {
          return ErrorBuilder()
                 << "ARGS was specified multiple times: " << line.str();
        }
      } else if (line_trimmed == "// AUTOUPDATE" ||
                 line_trimmed == "// NOAUTOUPDATE") {
        if (found_autoupdate) {
          return ErrorBuilder()
                 << "Multiple AUTOUPDATE/NOAUTOUPDATE settings found";
        }
        found_autoupdate = true;
        if (line_trimmed == "// AUTOUPDATE") {
          context.autoupdate_line_number = line_index;
        }
      } else if (line_trimmed == "// SET-CHECK-SUBSET") {
        if (!context.check_subset) {
          context.check_subset = true;
        } else {
          return ErrorBuilder()
                 << "SET-CHECK-SUBSET was specified multiple times";
        }
      }
    }
  }

  if (!found_autoupdate) {
    return ErrorBuilder() << "Missing AUTOUPDATE/NOAUTOUPDATE setting";
  }

  if (current_file_start) {
    context.test_files.push_back(
        TestFile(current_file_name.str(),
                 llvm::StringRef(current_file_start,
                                 file_content.end() - current_file_start)));
  } else {
    // If no file splitting happened, use the main file as the test file.
    // There will always be a `/` unless tests are in the repo root.
    context.test_files.push_back(TestFile(
        test_name_.drop_front(test_name_.rfind("/") + 1).str(), file_content));
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

auto FileTestBase::TransformExpectation(int line_index, llvm::StringRef in)
    -> ErrorOr<Matcher<std::string>> {
  if (in.empty()) {
    return Matcher<std::string>{StrEq("")};
  }
  if (in[0] != ' ') {
    return ErrorBuilder() << "Malformated CHECK line: " << in;
  }
  std::string str = in.substr(1).str();
  for (int pos = 0; pos < static_cast<int>(str.size());) {
    switch (str[pos]) {
      case '(':
      case ')':
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
      case '[': {
        llvm::StringRef line_keyword_cursor = llvm::StringRef(str).substr(pos);
        if (line_keyword_cursor.consume_front("[[")) {
          static constexpr llvm::StringLiteral LineKeyword = "@LINE";
          if (line_keyword_cursor.consume_front(LineKeyword)) {
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
            int remove_len = (line_keyword_cursor.data() - str.data()) - pos;
            str.replace(pos, remove_len, int_str);
            pos += int_str.size();
          } else {
            return ErrorBuilder()
                   << "Unexpected [[, should be {{\\[\\[}} at `"
                   << line_keyword_cursor.substr(0, 5) << "` in: " << in;
          }
        } else {
          // Escape the `[`.
          str.insert(pos, "\\");
          pos += 2;
        }
        break;
      }
      case '{': {
        if (pos + 1 == static_cast<int>(str.size()) || str[pos + 1] != '{') {
          // Single `{`, escape it.
          str.insert(pos, "\\");
          pos += 2;
        } else {
          // Replace the `{{...}}` regex syntax with standard `(...)` syntax.
          str.replace(pos, 2, "(");
          for (++pos; pos < static_cast<int>(str.size() - 1); ++pos) {
            if (str[pos] == '}' && str[pos + 1] == '}') {
              str.replace(pos, 2, ")");
              ++pos;
              break;
            }
          }
        }
        break;
      }
      default: {
        ++pos;
      }
    }
  }

  return Matcher<std::string>{MatchesRegex(str)};
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
  CARBON_CHECK(!absl::GetFlag(FLAGS_test_targets_file).empty())
      << "Missing --test_targets_file.";
  auto content = ReadFile(absl::GetFlag(FLAGS_test_targets_file));
  llvm::SmallVector<std::string> all_tests;
  for (llvm::StringRef file_ref : llvm::split(content, "\n")) {
    if (file_ref.empty()) {
      continue;
    }
    all_tests.push_back(file_ref.str());
  }
  return all_tests;
}

// Implements main() within the Carbon::Testing namespace for convenience.
static auto Main(int argc, char** argv) -> int {
  absl::ParseCommandLine(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  llvm::setBugReportMsg(
      "Please report issues to "
      "https://github.com/carbon-language/carbon-lang/issues and include the "
      "crash backtrace.\n");
  llvm::InitLLVM init_llvm(argc, argv);

  if (argc > 1) {
    llvm::errs() << "Unexpected arguments starting at: " << argv[1] << "\n";
    return EXIT_FAILURE;
  }

  llvm::SmallVector<std::string> tests = GetTests();
  auto test_factory = GetFileTestFactory();
  if (absl::GetFlag(FLAGS_autoupdate)) {
    for (const auto& test_name : tests) {
      std::unique_ptr<FileTestBase> test(test_factory.factory_fn(test_name));
      auto result = test->Autoupdate();
      llvm::errs() << (result.ok() ? (*result ? "!" : ".")
                                   : result.error().message());
    }
    llvm::errs() << "\nDone!\n";
    return EXIT_SUCCESS;
  } else {
    for (llvm::StringRef test_name : tests) {
      testing::RegisterTest(test_factory.name, test_name.data(), nullptr,
                            test_name.data(), __FILE__, __LINE__,
                            [&test_factory, test_name = test_name]() {
                              return test_factory.factory_fn(test_name);
                            });
    }
    return RUN_ALL_TESTS();
  }
}

}  // namespace Carbon::Testing

auto main(int argc, char** argv) -> int {
  return Carbon::Testing::Main(argc, argv);
}
