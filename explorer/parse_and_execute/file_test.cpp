// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <fstream>
#include <variant>

#include "explorer/parse_and_execute/parse_and_execute.h"
#include "gtest/gtest.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon::Testing {
namespace {

using ::testing::ElementsAreArray;
using ::testing::Matcher;
using ::testing::MatchesRegex;
using ::testing::RegisterTest;

static constexpr char PreludePath[] = "explorer/data/prelude.carbon";

class ParseAndExecuteTestFile : public ::testing::Test {
 public:
  explicit ParseAndExecuteTestFile(llvm::StringRef path, bool parser_debug)
      : path_(path), parser_debug_(parser_debug) {}

  auto TestBody() -> void override {
    // Load expected output.
    std::vector<Matcher<std::string>> expected_stdout;
    std::vector<Matcher<std::string>> expected_stderr;
    std::ifstream file_content(path_.str());
    for (std::string line_str; std::getline(file_content, line_str);) {
      llvm::StringRef line = line_str;
      line = line.drop_while([](char c) { return c == ' '; });
      if (!line.consume_front("// CHECK")) {
        continue;
      }
      if (line.consume_front(":STDOUT: ")) {
        expected_stdout.push_back(TransformExpectation(line));
      } else if (line.consume_front(":STDERR: ")) {
        expected_stderr.push_back(TransformExpectation(line));
      } else {
        FAIL() << "Unexpected CHECK in input: " << line_str;
      }
    }

    // Capture trace streaming, but only when in debug mode.
    TraceStream trace_stream;
    std::string trace_stream_str;
    llvm::raw_string_ostream trace_stream_ostream(trace_stream_str);
    if (parser_debug_) {
      trace_stream.set_stream(&trace_stream_ostream);
    }

    // Run the parse.
    auto result = ParseAndExecuteFile(PreludePath, path_.str(), parser_debug_,
                                      &trace_stream);

    // Check results.
    if (filename().starts_with("fail_")) {
      if (result.ok()) {
        FAIL() << "Expected error, but got success";
      } else {
        llvm::SmallVector<llvm::StringRef> error_lines;
        llvm::StringRef(result.error().message()).split(error_lines, "\n");
        EXPECT_THAT(error_lines, ElementsAreArray(expected_stderr));
      }
    } else {
      EXPECT_TRUE(result.ok()) << result.error();
    }
    if (parser_debug_) {
      EXPECT_FALSE(trace_stream_str.empty())
          << "Tracing should always do something";
    }
  }

  static auto TransformExpectation(llvm::StringRef in) -> Matcher<std::string> {
    std::string str = in.str();
    static constexpr llvm::StringLiteral PathBefore = "{{.*}}/explorer/";
    static constexpr llvm::StringLiteral PathAfter = "explorer/";

    for (int pos = 0; pos < static_cast<int>(str.size()); ++pos) {
      switch (str[pos]) {
        case '(':
          str.replace(pos, 1, "\\(");
          ++pos;
          break;
        case ')':
          str.replace(pos, 1, "\\)");
          ++pos;
          break;
        case '{':
          if (pos + 1 == static_cast<int>(str.size()) || str[pos + 1] != '{') {
            // Single `{`, escape it.
            str.replace(pos, 1, "\\{");
            ++pos;
          } else if (llvm::StringRef(str).substr(pos).starts_with(PathBefore)) {
            str.replace(pos, PathBefore.size(), PathAfter);
          } else {
            // Eat the regex.
            str.replace(pos, 2, "(");
            for (++pos; pos < static_cast<int>(str.size() - 1); ++pos) {
              if (str[pos] == '}' && str[pos + 1] == '}') {
                str.replace(pos, 2, ")");
                break;
              }
            }
          }
          break;
        case '}':
          // The regex case is already handled, so just escape this.
          str.replace(pos, 1, "\\}");
          ++pos;
          break;
      }
    }

    return MatchesRegex(str);
  }

  auto filename() -> llvm::StringRef {
    auto last_slash = path_.rfind("/");
    if (last_slash == llvm::StringRef::npos) {
      return path_;
    } else {
      return path_.substr(last_slash + 1);
    }
  }

 private:
  llvm::StringRef path_;
  bool parser_debug_;
};

static void RegisterTests(int argc, char** argv) {
  // Use RegisterTest instead of INSTANTIATE_TEST_CASE_P because of ordering
  // issues between container initialization and test instantiation by
  // InitGoogleTest.
  for (int i = 1; i < argc; ++i) {
    llvm::StringRef path = argv[i];
    RegisterTest("ParseAndExecuteTestForFile", path.data(), nullptr,
                 path.data(), __FILE__, __LINE__, [=]() {
                   return new ParseAndExecuteTestFile(path,
                                                      /*parser_debug=*/false);
                 });
    // "limits" tests check for various limit conditions (such as an infinite
    // loop). The tests collectively don't test tracing because it creates
    // substantial additional overhead.
    if (!path.find("/limits/")) {
      RegisterTest("ParseAndExecuteTestForFile", path.data(), nullptr,
                   path.data(), __FILE__, __LINE__, [=]() {
                     return new ParseAndExecuteTestFile(path,
                                                        /*parser_debug=*/true);
                   });
    }
  }
}

}  // namespace
}  // namespace Carbon::Testing

auto main(int argc, char** argv) -> int {
  ::testing::InitGoogleTest(&argc, argv);
  Carbon::Testing::RegisterTests(argc, argv);
  return RUN_ALL_TESTS();
}
