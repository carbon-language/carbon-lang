// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/file_test/file_test_base.h"

#include <filesystem>
#include <fstream>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "common/check.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"

ABSL_FLAG(std::vector<std::string>, file_tests, {},
          "A comma-separated list of tests for file_test infrastructure.");
ABSL_FLAG(bool, autoupdate, false,
          "Instead of verifying files match test output, autoupdate files "
          "based on test output.");

namespace Carbon::Testing {

using ::testing::Eq;
using ::testing::Matcher;
using ::testing::MatchesRegex;
using ::testing::StrEq;

// Reads a file to string.
static auto ReadFile(std::filesystem::path path) -> std::string {
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
  const char* src_dir = getenv("TEST_SRCDIR");
  CARBON_CHECK(src_dir);
  std::string test_file = path().lexically_relative(
      std::filesystem::path(src_dir).append("carbon"));
  const char* target = getenv("TEST_TARGET");
  CARBON_CHECK(target);
  // This advice overrides the --file_tests flag provided by the file_test rule.
  llvm::errs() << "\nTo test this file alone, run:\n  bazel test " << target
               << " --test_arg=--file_tests=" << test_file << "\n\n";

  TestContext context;
  auto run_result = ProcessTestFileAndRun(context);
  ASSERT_TRUE(run_result.ok()) << run_result.error();
  ValidateRun(context.test_files);
  EXPECT_THAT(!llvm::StringRef(path().filename()).starts_with("fail_"),
              Eq(context.exit_with_success))
      << "Tests should be prefixed with `fail_` if and only if running them "
         "is expected to fail.";

  // Check results.
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
}

auto FileTestBase::Autoupdate() -> bool {
  TestContext context;
  auto run_result = ProcessTestFileAndRun(context);
  CARBON_CHECK(run_result.ok()) << run_result.error();
  if (!context.autoupdate_line_number) {
    return false;
  }

  llvm::SmallVector<llvm::StringRef> filenames;
  filenames.reserve(context.non_check_lines.size());
  if (context.non_check_lines.size() > 1) {
    // There are splits, so we provide an empty name for the first file.
    filenames.push_back({});
  }
  for (const auto& file : context.test_files) {
    filenames.push_back(file.filename);
  }

  llvm::ArrayRef filenames_for_line_number = filenames;
  if (filenames.size() > 1) {
    filenames_for_line_number = filenames_for_line_number.drop_front();
  }

  return AutoupdateFileTest(
      path(), context.input_content, filenames, *context.autoupdate_line_number,
      context.non_check_lines, context.stdout, context.stderr,
      GetLineNumberReplacement(filenames_for_line_number),
      [&](std::string& line) { DoExtraCheckReplacements(line); });
}

auto FileTestBase::GetLineNumberReplacement(
    llvm::ArrayRef<llvm::StringRef> filenames) -> LineNumberReplacement {
  return {
      .has_file = true,
      .pattern = llvm::formatv(R"(({0}):(\d+):)", llvm::join(filenames, "|")),
      .sub_for_formatv = R"(\1:{0}:)"};
}

auto FileTestBase::ProcessTestFileAndRun(TestContext& context)
    -> ErrorOr<Success> {
  // Store the file so that test_files can use references to content.
  context.input_content = ReadFile(path());

  // Load expected output.
  CARBON_RETURN_IF_ERROR(ProcessTestFile(context));

  // Process arguments.
  if (context.test_args.empty()) {
    context.test_args = GetDefaultArgs();
  }
  CARBON_RETURN_IF_ERROR(
      DoArgReplacements(context.test_args, context.test_files));

  // Pass arguments as StringRef.
  llvm::SmallVector<llvm::StringRef> test_args_ref;
  test_args_ref.reserve(context.test_args.size());
  for (const auto& arg : context.test_args) {
    test_args_ref.push_back(arg);
  }

  // Capture trace streaming, but only when in debug mode.
  llvm::raw_svector_ostream stdout(context.stdout);
  llvm::raw_svector_ostream stderr(context.stderr);
  CARBON_ASSIGN_OR_RETURN(
      context.exit_with_success,
      Run(test_args_ref, context.test_files, stdout, stderr));
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

  context.non_check_lines.resize(1);
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

      context.non_check_lines.push_back({FileTestLine(0, line)});
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
      llvm::SmallVector<Matcher<std::string>>* expected = nullptr;
      if (line_trimmed.consume_front(":STDOUT:")) {
        expected = &context.expected_stdout;
      } else if (line_trimmed.consume_front(":STDERR:")) {
        expected = &context.expected_stderr;
      } else {
        return ErrorBuilder() << "Unexpected CHECK in input: " << line.str();
      }
      expected->push_back(TransformExpectation(line_index, line_trimmed));
    } else {
      context.non_check_lines.back().push_back(FileTestLine(line_index, line));
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
    context.test_files.push_back(
        TestFile(path().filename().string(), file_content));
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
    -> Matcher<std::string> {
  if (in.empty()) {
    return StrEq("");
  }
  CARBON_CHECK(in[0] == ' ') << "Malformated input: " << in;
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
            CARBON_CHECK(!line_keyword_cursor.consumeInteger(10, offset) &&
                         line_keyword_cursor.consume_front("]]"))
                << "Unexpected @LINE offset at `"
                << line_keyword_cursor.substr(0, 5) << "` in: " << in;
            std::string int_str = llvm::Twine(line_index + offset).str();
            int remove_len = (line_keyword_cursor.data() - str.data()) - pos;
            str.replace(pos, remove_len, int_str);
            pos += int_str.size();
          } else {
            CARBON_FATAL() << "Unexpected [[, should be {{\\[\\[}} at `"
                           << line_keyword_cursor.substr(0, 5)
                           << "` in: " << in;
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

  return MatchesRegex(str);
}

}  // namespace Carbon::Testing

auto main(int argc, char** argv) -> int {
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

  // Configure the base directory for test names.
  const char* target = getenv("TEST_TARGET");
  CARBON_CHECK(target != nullptr);
  llvm::StringRef target_dir = target;
  std::error_code ec;
  std::filesystem::path working_dir = std::filesystem::current_path(ec);
  CARBON_CHECK(!ec) << ec.message();
  // Leaves one slash.
  CARBON_CHECK(target_dir.consume_front("/"));
  target_dir = target_dir.substr(0, target_dir.rfind(":"));
  std::string base_dir = working_dir.string() + target_dir.str() + "/";

  auto test_factory = Carbon::Testing::GetFileTestFactory();

  for (const auto& file_test : absl::GetFlag(FLAGS_file_tests)) {
    // Pass the absolute path to the factory function.
    auto path = std::filesystem::absolute(file_test, ec);
    CARBON_CHECK(!ec) << file_test << ": " << ec.message();
    CARBON_CHECK(llvm::StringRef(path.string()).starts_with(base_dir))
        << "\n  " << path << "\n  should start with\n  " << base_dir;
    if (absl::GetFlag(FLAGS_autoupdate)) {
      std::unique_ptr<Carbon::Testing::FileTestBase> test(
          test_factory.factory_fn(path));
      llvm::errs() << (test->Autoupdate() ? "!" : ".");
    } else {
      std::string test_name = path.string().substr(base_dir.size());
      testing::RegisterTest(test_factory.name, test_name.c_str(), nullptr,
                            test_name.c_str(), __FILE__, __LINE__,
                            [=]() { return test_factory.factory_fn(path); });
    }
  }
  if (absl::GetFlag(FLAGS_autoupdate)) {
    llvm::errs() << "\nDone!\n";
  }

  if (absl::GetFlag(FLAGS_autoupdate)) {
    return EXIT_SUCCESS;
  } else {
    return RUN_ALL_TESTS();
  }
}
