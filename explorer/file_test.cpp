// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "absl/flags/flag.h"
#include "explorer/main.h"
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
  using FileTestBase::FileTestBase;

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

    // Trace output is only checked for a few tests.
    bool check_trace_output =
        path().string().find("trace_testdata/") != std::string::npos;

    int exit_code = ExplorerMain(
        args.size(), args.data(), /*install_path=*/"", PreludePath, stdout,
        stderr, check_trace_output ? stdout : trace_stream_, fs);

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

  TestRawOstream trace_stream_;
};

}  // namespace

CARBON_FILE_TEST_FACTORY(ExplorerFileTest);

}  // namespace Carbon::Testing
