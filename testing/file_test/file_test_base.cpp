// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/file_test/file_test_base.h"

#include <filesystem>
#include <fstream>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "common/check.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "testing/util/test_raw_ostream.h"

ABSL_FLAG(std::vector<std::string>, file_tests, {},
          "A comma-separated list of tests for file_test infrastructure.");

namespace Carbon::Testing {

// The length of the base directory.
static int base_dir_len = 0;

using ::testing::Eq;
using ::testing::Matcher;
using ::testing::MatchesRegex;
using ::testing::StrEq;

void FileTestBase::RegisterTests(
    const char* fixture_label,
    const llvm::SmallVector<std::filesystem::path>& paths,
    std::function<FileTestBase*(const std::filesystem::path&)> factory) {
  // Use RegisterTest instead of INSTANTIATE_TEST_CASE_P because of ordering
  // issues between container initialization and test instantiation by
  // InitGoogleTest.
  for (const auto& path : paths) {
    std::string test_name = path.string().substr(base_dir_len);
    testing::RegisterTest(fixture_label, test_name.c_str(), nullptr,
                          test_name.c_str(), __FILE__, __LINE__,
                          [=]() { return factory(path); });
  }
}

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

  // Store the file so that test_files can use references to content.
  std::string test_content = ReadFile(path());

  // Load expected output.
  llvm::SmallVector<std::string> test_args;
  llvm::SmallVector<TestFile> test_files;
  llvm::SmallVector<Matcher<std::string>> expected_stdout;
  llvm::SmallVector<Matcher<std::string>> expected_stderr;
  bool check_subset = false;
  ProcessTestFile(test_content, test_args, test_files, expected_stdout,
                  expected_stderr, check_subset);
  if (HasFailure()) {
    return;
  }

  // Process arguments.
  if (test_args.empty()) {
    test_args = GetDefaultArgs();
  }
  DoArgReplacements(test_args, test_files);
  if (HasFailure()) {
    return;
  }

  // Pass arguments as StringRef.
  llvm::SmallVector<llvm::StringRef> test_args_ref;
  test_args_ref.reserve(test_args.size());
  for (const auto& arg : test_args) {
    test_args_ref.push_back(arg);
  }

  // Capture trace streaming, but only when in debug mode.
  TestRawOstream stdout;
  TestRawOstream stderr;
  bool run_succeeded = RunWithFiles(test_args_ref, test_files, stdout, stderr);
  if (HasFailure()) {
    return;
  }
  EXPECT_THAT(!llvm::StringRef(path().filename()).starts_with("fail_"),
              Eq(run_succeeded))
      << "Tests should be prefixed with `fail_` if and only if running them "
         "is expected to fail.";

  // Check results.
  if (check_subset) {
    EXPECT_THAT(SplitOutput(stdout.TakeStr()), IsSupersetOf(expected_stdout));
    EXPECT_THAT(SplitOutput(stderr.TakeStr()), IsSupersetOf(expected_stderr));

  } else {
    EXPECT_THAT(SplitOutput(stdout.TakeStr()),
                ElementsAreArray(expected_stdout));
    EXPECT_THAT(SplitOutput(stderr.TakeStr()),
                ElementsAreArray(expected_stderr));
  }
}

auto FileTestBase::DoArgReplacements(
    llvm::SmallVector<std::string>& test_args,
    const llvm::SmallVector<TestFile>& test_files) -> void {
  for (auto* it = test_args.begin(); it != test_args.end(); ++it) {
    auto percent = it->find("%");
    if (percent == std::string::npos) {
      continue;
    }

    if (percent + 1 >= it->size()) {
      FAIL() << "% is not allowed on its own: " << *it;
    }
    char c = (*it)[percent + 1];
    switch (c) {
      case 's': {
        if (*it != "%s") {
          FAIL() << "%s must be the full argument: " << *it;
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
        char* temp = getenv("TEST_TMPDIR");
        CARBON_CHECK(temp != nullptr);
        it->replace(percent, 2, llvm::formatv("{0}/temp_file", temp));
        break;
      }
      default:
        FAIL() << "%" << c << " is not supported: " << *it;
    }
  }
}

auto FileTestBase::ProcessTestFile(
    llvm::StringRef file_content, llvm::SmallVector<std::string>& test_args,
    llvm::SmallVector<TestFile>& test_files,
    llvm::SmallVector<Matcher<std::string>>& expected_stdout,
    llvm::SmallVector<Matcher<std::string>>& expected_stderr,
    bool& check_subset) -> void {
  llvm::StringRef cursor = file_content;
  bool found_content_pre_split = false;
  int line_index = 0;
  llvm::StringRef current_file_name;
  const char* current_file_start = nullptr;
  while (!cursor.empty()) {
    auto [line, next_cursor] = cursor.split("\n");
    cursor = next_cursor;

    static constexpr llvm::StringLiteral SplitPrefix = "// ---";
    if (line.consume_front(SplitPrefix)) {
      // On a file split, add the previous file, then start a new one.
      if (current_file_start) {
        test_files.push_back(TestFile(
            current_file_name.str(),
            llvm::StringRef(
                current_file_start,
                line.begin() - current_file_start - SplitPrefix.size())));
      } else {
        // For the first split, we make sure there was no content prior.
        ASSERT_FALSE(found_content_pre_split)
            << "When using split files, there must be no content before the "
               "first split file.";
      }
      current_file_name = line.trim();
      current_file_start = cursor.begin();
      line_index = 0;
      continue;
    } else if (!current_file_start && !line.starts_with("//") &&
               !line.trim().empty()) {
      found_content_pre_split = true;
    }
    ++line_index;

    // Process expectations when found.
    auto line_trimmed = line.ltrim();
    if (line_trimmed.consume_front("// ARGS: ")) {
      if (test_args.empty()) {
        // Split the line into arguments.
        std::pair<llvm::StringRef, llvm::StringRef> cursor =
            llvm::getToken(line_trimmed);
        while (!cursor.first.empty()) {
          test_args.push_back(std::string(cursor.first));
          cursor = llvm::getToken(cursor.second);
        }
      } else {
        FAIL() << "ARGS was specified multiple times: " << line.str();
      }
    } else if (line_trimmed == "// SET-CHECK-SUBSET") {
      if (!check_subset) {
        check_subset = true;
      } else {
        FAIL() << "SET-CHECK-SUBSET was specified multiple times";
      }
    } else if (line_trimmed.consume_front("// CHECK")) {
      if (line_trimmed.consume_front(":STDOUT:")) {
        expected_stdout.push_back(
            TransformExpectation(line_index, line_trimmed));
      } else if (line_trimmed.consume_front(":STDERR:")) {
        expected_stderr.push_back(
            TransformExpectation(line_index, line_trimmed));
      } else {
        FAIL() << "Unexpected CHECK in input: " << line.str();
      }
    }
  }

  if (current_file_start) {
    test_files.push_back(
        TestFile(current_file_name.str(),
                 llvm::StringRef(current_file_start,
                                 file_content.end() - current_file_start)));
  } else {
    // If no file splitting happened, use the main file as the test file.
    test_files.push_back(TestFile(path().filename().string(), file_content));
  }

  // Assume there is always a suffix `\n` in output.
  if (!expected_stdout.empty()) {
    expected_stdout.push_back(StrEq(""));
  }
  if (!expected_stderr.empty()) {
    expected_stderr.push_back(StrEq(""));
  }
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
  Carbon::Testing::base_dir_len = base_dir.size();

  // Register tests based on their absolute path.
  llvm::SmallVector<std::filesystem::path> paths;
  for (const auto& file_test : absl::GetFlag(FLAGS_file_tests)) {
    auto path = std::filesystem::absolute(file_test, ec);
    CARBON_CHECK(!ec) << file_test << ": " << ec.message();
    CARBON_CHECK(llvm::StringRef(path.string()).starts_with(base_dir))
        << "\n  " << path << "\n  should start with\n  " << base_dir;
    paths.push_back(path);
  }
  Carbon::Testing::RegisterFileTests(paths);

  return RUN_ALL_TESTS();
}
