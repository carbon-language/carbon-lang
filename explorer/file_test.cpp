// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "absl/flags/flag.h"
#include "explorer/main.h"
#include "re2/re2.h"
#include "testing/file_test/file_test_base.h"
#include "testing/util/test_raw_ostream.h"

ABSL_FLAG(bool, trace, false,
          "Set to true to run tests with tracing enabled, even if they don't "
          "otherwise specify it. This does not result in checking trace output "
          "contents; it essentially only verifies there's not a crash bug.");

namespace Carbon::Testing {
namespace {

class ExplorerFileTest : public FileTestBase {
 public:
  explicit ExplorerFileTest(std::filesystem::path path)
      : FileTestBase(std::move(path)),
        prelude_line_re_(R"(prelude.carbon:(\d+))"),
        timing_re_(R"((Time elapsed in \w+: )\d+(ms))") {
    CARBON_CHECK(prelude_line_re_.ok()) << prelude_line_re_.error();
  }

  auto Run(const llvm::SmallVector<llvm::StringRef>& test_args,
           const llvm::SmallVector<TestFile>& test_files,
           llvm::raw_pwrite_stream& stdout, llvm::raw_pwrite_stream& stderr)
      -> ErrorOr<bool> override {
    // Create the files in-memory.
    llvm::vfs::InMemoryFileSystem fs(new llvm::vfs::InMemoryFileSystem());
    for (const auto& test_file : test_files) {
      if (!fs.addFile(test_file.filename, /*ModificationTime=*/0,
                      llvm::MemoryBuffer::getMemBuffer(test_file.content))) {
        return ErrorBuilder() << "File is repeated: " << test_file.filename;
      }
    }

    // Add the prelude.
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> prelude =
        llvm::MemoryBuffer::getFile("explorer/data/prelude.carbon");
    if (prelude.getError()) {
      return ErrorBuilder() << prelude.getError().message();
    }
    // TODO: This path is long with a prefix / because of the path expectations
    // in tests. Change those to allow a shorter path (e.g., `prelude.carbon`)
    // here.
    static constexpr llvm::StringLiteral PreludePath =
        "/explorer/data/prelude.carbon";
    if (!fs.addFile(PreludePath, /*ModificationTime=*/0, std::move(*prelude))) {
      return ErrorBuilder() << "Duplicate prelude.carbon";
    }

    llvm::SmallVector<const char*> args = {"explorer"};
    for (auto arg : test_args) {
      args.push_back(arg.data());
    }

    int exit_code = ExplorerMain(
        args.size(), args.data(), /*install_path=*/"", PreludePath, stdout,
        stderr, check_trace_output() ? stdout : trace_stream_, fs);

    return exit_code == EXIT_SUCCESS;
  }

  auto ValidateRun(const llvm::SmallVector<TestFile>& /*test_files*/)
      -> void override {
    // Skip trace test check as they use stdout stream instead of
    // trace_stream_ostream
    if (absl::GetFlag(FLAGS_trace)) {
      EXPECT_FALSE(trace_stream_.TakeStr().empty())
          << "Tracing should always do something";
    }
  }

  auto GetDefaultArgs() -> llvm::SmallVector<std::string> override {
    llvm::SmallVector<std::string> args;
    if (absl::GetFlag(FLAGS_trace)) {
      args.push_back("--trace_file=-");
      args.push_back("--trace_phase=all");
    }
    args.push_back("%s");
    return args;
  }

  auto GetLineNumberReplacement(llvm::ArrayRef<llvm::StringRef> filenames)
      -> LineNumberReplacement override {
    if (check_trace_output()) {
      return {.has_file = false,
              .pattern = R"((DO NOT MATCH))",
              // The `{{{{` becomes `{{`.
              .line_formatv = "{{{{ *}}{0}"};
    }
    return FileTestBase::GetLineNumberReplacement(filenames);
  }

  auto DoExtraCheckReplacements(std::string& check_line) -> void override {
    // Ignore the resulting column of EndOfFile because it's often the end of
    // the CHECK comment.
    RE2::GlobalReplace(&check_line, prelude_line_re_,
                       R"(prelude.carbon:{{\\d+}})");
    if (check_trace_output()) {
      // Replace timings in trace output.
      RE2::GlobalReplace(&check_line, timing_re_, R"(\1{{\\d+}}\2)");
    }
  }

 private:
  // Trace output is directly checked for a few tests.
  auto check_trace_output() -> bool {
    return path().string().find("/trace/") != std::string::npos;
  }

  TestRawOstream trace_stream_;
  RE2 prelude_line_re_;
  RE2 timing_re_;
};

}  // namespace

CARBON_FILE_TEST_FACTORY(ExplorerFileTest);

}  // namespace Carbon::Testing
