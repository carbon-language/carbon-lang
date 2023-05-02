// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/parse_and_execute/parse_and_execute.h"
#include "testing/file_test/file_test_base.h"

namespace Carbon::Testing {
namespace {

static constexpr char PreludePath[] = "explorer/data/prelude.carbon";

class ParseAndExecuteTestFile : public FileTestBase {
 public:
  explicit ParseAndExecuteTestFile(llvm::StringRef path, bool trace)
      : FileTestBase(path), trace_(trace) {}

  auto SetUp() -> void override {
    if (trace_) {
      if (path().find("/limits/") != llvm::StringRef::npos) {
        GTEST_SKIP()
            << "`limits` tests check for various limit conditions (such as an "
               "infinite loop). The tests collectively don't test tracing "
               "because it creates substantial additional overhead.";
      } else if (path().endswith(
                     "testdata/assoc_const/rewrite_large_type.carbon") ||
                 path().endswith(
                     "testdata/linked_list/typed_linked_list.carbon")) {
        GTEST_SKIP() << "Expensive test to trace";
      }
    }
  }

  auto RunOverFile(llvm::raw_ostream& stdout, llvm::raw_ostream& stderr)
      -> void override {
    // Capture trace streaming, but only when in debug mode.
    TraceStream trace_stream;
    std::string trace_stream_str;
    llvm::raw_string_ostream trace_stream_ostream(trace_stream_str);
    if (trace_) {
      trace_stream.set_stream(&trace_stream_ostream);
    }

    // Run the parse. Parser debug output is always off because it's difficult
    // to redirect.
    auto result = ParseAndExecuteFile(PreludePath, path().str(),
                                      /*parser_debug=*/false, &trace_stream);
    if (result.ok()) {
      stdout << "Result: " << *result << "\n";
    } else {
      stderr << result.error() << "\n";
    }

    if (trace_) {
      EXPECT_FALSE(trace_stream_str.empty())
          << "Tracing should always do something";
    }
  }

 private:
  bool trace_;
};

}  // namespace
}  // namespace Carbon::Testing

auto main(int argc, char** argv) -> int {
  ::testing::InitGoogleTest(&argc, argv);
  using Carbon::Testing::ParseAndExecuteTestFile;
  ParseAndExecuteTestFile::RegisterTests(
      "ParseAndExecuteTestFile", argc, argv, [=](llvm::StringRef path) {
        return new ParseAndExecuteTestFile(path, /*trace=*/true);
      });
  ParseAndExecuteTestFile::RegisterTests(
      "ParseAndExecuteTestFile.debug", argc, argv, [=](llvm::StringRef path) {
        return new ParseAndExecuteTestFile(path, /*trace=*/false);
      });
  return RUN_ALL_TESTS();
}
