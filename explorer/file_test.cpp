// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "absl/flags/flag.h"
#include "absl/strings/str_split.h"
#include "common/raw_string_ostream.h"
#include "explorer/main.h"
#include "re2/re2.h"
#include "testing/base/file_helpers.h"
#include "testing/file_test/file_test_base.h"
#include "testing/file_test/manifest.h"

ABSL_FLAG(bool, trace, false,
          "Set to true to run tests with tracing enabled, even if they don't "
          "otherwise specify it. This does not result in checking trace output "
          "contents; it essentially only verifies there's not a crash bug.");
ABSL_FLAG(std::string, explorer_test_targets_file, "",
          "A path to a file containing repo-relative names of test files.");

namespace Carbon::Testing {
namespace {

class ExplorerFileTest : public FileTestBase {
 public:
  explicit ExplorerFileTest(llvm::StringRef /*exe_path*/,
                            llvm::StringRef test_name)
      : FileTestBase(test_name),
        prelude_line_re_(R"(prelude.carbon:(\d+))"),
        timing_re_(R"((Time elapsed in \w+: )\d+(ms))") {
    CARBON_CHECK(prelude_line_re_.ok(), "{0}", prelude_line_re_.error());
    CARBON_CHECK(timing_re_.ok(), "{0}", timing_re_.error());
  }

  auto Run(const llvm::SmallVector<llvm::StringRef>& test_args,
           llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem>& fs,
           FILE* /*input_stream*/, llvm::raw_pwrite_stream& output_stream,
           llvm::raw_pwrite_stream& error_stream) const
      -> ErrorOr<RunResult> override {
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
    if (!fs->addFile(PreludePath, /*ModificationTime=*/0,
                     std::move(*prelude))) {
      return ErrorBuilder() << "Duplicate prelude.carbon";
    }

    llvm::SmallVector<const char*> args = {"explorer"};
    for (auto arg : test_args) {
      args.push_back(arg.data());
    }

    RawStringOstream trace_stream;
    int exit_code =
        ExplorerMain(args.size(), args.data(), /*install_path=*/"", PreludePath,
                     output_stream, error_stream,
                     check_trace_output() ? output_stream : trace_stream, *fs);

    // Skip trace test check as they use stdout stream instead of
    // trace_stream_ostream
    if (absl::GetFlag(FLAGS_trace) && trace_stream.TakeStr().empty()) {
      return Error("Tracing should always do something");
    }

    return {{.success = exit_code == EXIT_SUCCESS}};
  }

  auto GetDefaultArgs() const -> llvm::SmallVector<std::string> override {
    llvm::SmallVector<std::string> args;
    if (absl::GetFlag(FLAGS_trace)) {
      args.push_back("--trace_file=-");
      args.push_back("--trace_phase=all");
    }
    args.push_back("%s");
    return args;
  }

  auto GetLineNumberReplacements(llvm::ArrayRef<llvm::StringRef> filenames)
      const -> llvm::SmallVector<LineNumberReplacement> override {
    if (check_trace_output()) {
      return {};
    }
    return FileTestBase::GetLineNumberReplacements(filenames);
  }

  auto DoExtraCheckReplacements(std::string& check_line) const
      -> void override {
    // Ignore the resulting column of EndOfFile because it's often the end of
    // the CHECK comment.
    RE2::GlobalReplace(&check_line, prelude_line_re_,
                       R"(prelude.carbon:{{\\d+}})");
    if (check_trace_output()) {
      // Replace timings in trace output.
      RE2::GlobalReplace(&check_line, timing_re_, R"(\1{{\\d+}}\2)");
    }
  }

  // Cannot execute in parallel.
  auto AllowParallelRun() const -> bool override { return false; }

 private:
  // Trace output is directly checked for a few tests.
  auto check_trace_output() const -> bool {
    return test_name().find("/trace/") != std::string::npos;
  }

  RE2 prelude_line_re_;
  RE2 timing_re_;
};

}  // namespace

// Explorer uses a non-standard approach to getting the manifest path.
auto GetFileTestManifest() -> llvm::SmallVector<std::string> {
  llvm::SmallVector<std::string> manifest;
  auto content = ReadFile(absl::GetFlag(FLAGS_explorer_test_targets_file));
  for (const auto& line : absl::StrSplit(*content, '\n', absl::SkipEmpty())) {
    manifest.push_back(std::string(line));
  }
  return manifest;
}

CARBON_FILE_TEST_FACTORY(ExplorerFileTest)

}  // namespace Carbon::Testing
