// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/parse_and_execute/parse_and_execute.h"
#include "testing/file_test/file_test_base.h"
#include "testing/util/test_raw_ostream.h"

namespace Carbon::Testing {
namespace {

class ParseAndExecuteTestFile : public FileTestBase {
 public:
  explicit ParseAndExecuteTestFile(const std::filesystem::path& path,
                                   bool trace)
      : FileTestBase(path), trace_(trace) {}

  auto SetUp() -> void override {
    std::string path_str = path().string();
    llvm::StringRef path_ref = path_str;

    if (path_ref.find("trace_testdata") != llvm::StringRef::npos) {
      is_trace_test = true;
    }

    if (trace_) {
      if (path_ref.find("/limits/") != llvm::StringRef::npos) {
        GTEST_SKIP()
            << "`limits` tests check for various limit conditions (such as an "
               "infinite loop). The tests collectively don't test tracing "
               "because it creates substantial additional overhead.";
      } else if (path_ref.endswith(
                     "testdata/assoc_const/rewrite_large_type.carbon") ||
                 path_ref.endswith(
                     "testdata/linked_list/typed_linked_list.carbon")) {
        GTEST_SKIP() << "Expensive test to trace";
      }
    } else {
      if (is_trace_test) {
        GTEST_SKIP() << "`trace` tests only check for trace output.";
        is_trace_test = true;
      }
    }
  }

  auto RunWithFiles(const llvm::SmallVector<TestFile>& test_files,
                    llvm::raw_pwrite_stream& stdout,
                    llvm::raw_pwrite_stream& stderr) -> bool override {
    if (test_files.size() != 1) {
      ADD_FAILURE() << "Only 1 file is supported: " << test_files.size()
                    << " provided";
      return false;
    }

    // Capture trace streaming, but only when in debug mode.
    TraceStream trace_stream;
    TestRawOstream trace_stream_ostream;
    if (trace_) {
      trace_stream.set_stream(is_trace_test ? &stdout : &trace_stream_ostream);
      trace_stream.set_allowed_phases({ProgramPhase::All});
    }

    // Set the location of the prelude.
    char* test_srcdir = getenv("TEST_SRCDIR");
    CARBON_CHECK(test_srcdir != nullptr);
    std::string prelude_path(test_srcdir);
    prelude_path += "/carbon/explorer/data/prelude.carbon";

    // Run the parse. Parser debug output is always off because it's difficult
    // to redirect.
    auto result = ParseAndExecute(
        prelude_path, test_files[0].filename, test_files[0].content,
        /*parser_debug=*/false, &trace_stream, &stdout);
    // This mirrors printing currently done by main.cpp.
    if (result.ok()) {
      stdout << "result: " << *result << "\n";
    } else {
      stderr << result.error() << "\n";
    }

    // Skip trace test check as they use stdout stream instead of
    // trace_stream_ostream
    if (trace_ && !is_trace_test) {
      EXPECT_FALSE(trace_stream_ostream.TakeStr().empty())
          << "Tracing should always do something";
    }

    return result.ok();
  }

 private:
  bool trace_;
  bool is_trace_test = false;
};

}  // namespace

extern auto RegisterFileTests(
    const llvm::SmallVector<std::filesystem::path>& paths) -> void {
  ParseAndExecuteTestFile::RegisterTests(
      "ParseAndExecuteTestFile", paths, [](const std::filesystem::path& path) {
        return new ParseAndExecuteTestFile(path, /*trace=*/false);
      });
  ParseAndExecuteTestFile::RegisterTests("ParseAndExecuteTestFile.trace", paths,
                                         [](const std::filesystem::path& path) {
                                           return new ParseAndExecuteTestFile(
                                               path, /*trace=*/true);
                                         });
}

}  // namespace Carbon::Testing
