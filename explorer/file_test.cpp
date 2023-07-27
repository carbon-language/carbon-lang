// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "absl/flags/flag.h"
#include "explorer/parse_and_execute/parse_and_execute.h"
#include "testing/file_test/file_test_base.h"
#include "testing/util/test_raw_ostream.h"

ABSL_FLAG(bool, trace, false,
          "Set to true to run tests with tracing enabled, even if they don't "
          "otherwise specify it. This does not result in checking trace output "
          "contents; it essentially only verifies there's not a crash bug.");

namespace Carbon::Testing {
namespace {

class ParseAndExecuteTestFile : public FileTestBase {
 public:
  explicit ParseAndExecuteTestFile(const std::filesystem::path& path)
      : FileTestBase(path) {}

  auto RunWithFiles(const llvm::SmallVector<llvm::StringRef>& test_args,
                    const llvm::SmallVector<TestFile>& test_files,
                    llvm::raw_pwrite_stream& stdout,
                    llvm::raw_pwrite_stream& stderr) -> bool override {
    CARBON_CHECK(test_args.empty())
        << "ARGS are not currently used in explorer's file_test.";

    if (test_files.size() != 1) {
      ADD_FAILURE() << "Only 1 file is supported: " << test_files.size()
                    << " provided";
      return false;
    }

    // Trace output is only checked for a few tests.
    bool check_trace_output =
        path().string().find("/trace_testdata/") != std::string::npos;

    // Capture trace streaming, but only when in debug mode.
    TraceStream trace_stream;
    TestRawOstream trace_stream_ostream;
    if (check_trace_output || absl::GetFlag(FLAGS_trace)) {
      trace_stream.set_stream(check_trace_output ? &stdout
                                                 : &trace_stream_ostream);
      trace_stream.set_allowed_phases({ProgramPhase::All});
      trace_stream.set_allowed_file_kinds({FileKind::Main});
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
    if (absl::GetFlag(FLAGS_trace)) {
      CARBON_CHECK(!check_trace_output)
          << "trace tests should only be run in the default mode.";
      EXPECT_FALSE(trace_stream_ostream.TakeStr().empty())
          << "Tracing should always do something";
    }

    return result.ok();
  }

  auto GetDefaultArgs() -> llvm::SmallVector<std::string> override {
    return {};
  }
};

}  // namespace

extern auto RegisterFileTests(
    const llvm::SmallVector<std::filesystem::path>& paths) -> void {
  ParseAndExecuteTestFile::RegisterTests(
      "ParseAndExecuteTestFile", paths, [](const std::filesystem::path& path) {
        return new ParseAndExecuteTestFile(path);
      });
}

}  // namespace Carbon::Testing
