// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/file_test/file_test_base.h"

#include <fstream>

#include "common/check.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/InitLLVM.h"

static std::string* subset_target = nullptr;

namespace Carbon::Testing {

using ::testing::Eq;

void FileTestBase::RegisterTests(
    const char* fixture_label, const std::vector<llvm::StringRef>& paths,
    std::function<FileTestBase*(llvm::StringRef)> factory) {
  // Use RegisterTest instead of INSTANTIATE_TEST_CASE_P because of ordering
  // issues between container initialization and test instantiation by
  // InitGoogleTest.
  for (auto path : paths) {
    testing::RegisterTest(fixture_label, path.data(), nullptr, path.data(),
                          __FILE__, __LINE__, [=]() { return factory(path); });
  }
}

// Splits outputs to string_view because gtest handles string_view by default.
static auto SplitOutput(llvm::StringRef output)
    -> std::vector<std::string_view> {
  if (output.empty()) {
    return {};
  }
  llvm::SmallVector<llvm::StringRef> lines;
  llvm::StringRef(output).split(lines, "\n");
  return std::vector<std::string_view>(lines.begin(), lines.end());
}

// Runs a test and compares output. This keeps output split by line so that
// issues are a little easier to identify by the different line.
auto FileTestBase::TestBody() -> void {
  llvm::errs() << "\nTo test this file alone, run:\n  bazel test "
               << *subset_target << " --test_arg=" << path() << "\n\n";

  // Load expected output.
  std::vector<testing::Matcher<std::string>> expected_stdout;
  std::vector<testing::Matcher<std::string>> expected_stderr;
  std::ifstream file_content(path_.str());
  int line_index = 0;
  std::string line_str;
  while (std::getline(file_content, line_str)) {
    ++line_index;
    llvm::StringRef line = line_str;
    line = line.ltrim();
    if (!line.consume_front("// CHECK")) {
      continue;
    }
    if (line.consume_front(":STDOUT:")) {
      expected_stdout.push_back(TransformExpectation(line_index, line));
    } else if (line.consume_front(":STDERR:")) {
      expected_stderr.push_back(TransformExpectation(line_index, line));
    } else {
      FAIL() << "Unexpected CHECK in input: " << line_str;
    }
  }

  // Assume there is always a suffix `\n` in output.
  if (!expected_stdout.empty()) {
    expected_stdout.push_back(testing::StrEq(""));
  }
  if (!expected_stderr.empty()) {
    expected_stderr.push_back(testing::StrEq(""));
  }

  // Capture trace streaming, but only when in debug mode.
  std::string stdout;
  std::string stderr;
  llvm::raw_string_ostream stdout_ostream(stdout);
  llvm::raw_string_ostream stderr_ostream(stderr);
  bool run_succeeded = RunOverFile(stdout_ostream, stderr_ostream);
  if (HasFailure()) {
    return;
  }
  EXPECT_THAT(!filename().starts_with("fail_"), Eq(run_succeeded))
      << "Tests should be prefixed with `fail_` if and only if running them "
         "is expected to fail.";

  // Check results.
  EXPECT_THAT(SplitOutput(stdout), ElementsAreArray(expected_stdout));
  EXPECT_THAT(SplitOutput(stderr), ElementsAreArray(expected_stderr));
}

auto FileTestBase::TransformExpectation(int line_index, llvm::StringRef in)
    -> testing::Matcher<std::string> {
  if (in.empty()) {
    return testing::StrEq("");
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

  return testing::MatchesRegex(str);
}

auto FileTestBase::filename() -> llvm::StringRef {
  auto last_slash = path_.rfind("/");
  if (last_slash == llvm::StringRef::npos) {
    return path_;
  } else {
    return path_.substr(last_slash + 1);
  }
}

}  // namespace Carbon::Testing

// Returns the name of the subset target.
static auto GetSubsetTarget() -> std::string {
  char* name = getenv("TEST_TARGET");
  if (name == nullptr) {
    return "<missing TEST_TARGET>";
  }

  if (llvm::StringRef(name).ends_with(".subset")) {
    return name;
  } else {
    return std::string(name) + ".subset";
  }
}

auto main(int argc, char** argv) -> int {
  testing::InitGoogleTest(&argc, argv);
  llvm::setBugReportMsg(
      "Please report issues to "
      "https://github.com/carbon-language/carbon-lang/issues and include the "
      "crash backtrace.\n");
  llvm::InitLLVM init_llvm(argc, argv);

  if (argc < 2) {
    llvm::errs() << "At least one test file must be provided.\n";
    return EXIT_FAILURE;
  }

  std::string subset_target_storage = GetSubsetTarget();
  ::subset_target = &subset_target_storage;

  std::vector<llvm::StringRef> paths(argv + 1, argv + argc);
  Carbon::Testing::RegisterFileTests(paths);

  return RUN_ALL_TESTS();
}
