// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/file_test/file_test_base.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common/ostream.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FormatVariadic.h"

namespace Carbon::Testing {
namespace {

class FileTestBaseTest : public FileTestBase {
 public:
  FileTestBaseTest(llvm::StringRef /*exe_path*/, std::mutex* output_mutex,
                   llvm::StringRef test_name)
      : FileTestBase(output_mutex, test_name) {}

  auto Run(const llvm::SmallVector<llvm::StringRef>& test_args,
           llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem>& fs,
           llvm::raw_pwrite_stream& stdout, llvm::raw_pwrite_stream& stderr)
      -> ErrorOr<RunResult> override;

  auto GetArgReplacements() -> llvm::StringMap<std::string> override {
    return {{"replacement", "replaced"}};
  }

  auto GetDefaultArgs() -> llvm::SmallVector<std::string> override {
    return {"default_args", "%s"};
  }

  auto GetDefaultFileRE(llvm::ArrayRef<llvm::StringRef> filenames)
      -> std::optional<RE2> override {
    return std::make_optional<RE2>(
        llvm::formatv(R"(file: ({0}))", llvm::join(filenames, "|")));
  }

  auto GetLineNumberReplacements(llvm::ArrayRef<llvm::StringRef> filenames)
      -> llvm::SmallVector<LineNumberReplacement> override {
    auto replacements = FileTestBase::GetLineNumberReplacements(filenames);
    auto filename = std::filesystem::path(test_name().str()).filename();
    if (llvm::StringRef(filename).starts_with("file_only_re_")) {
      replacements.push_back({.has_file = false,
                              .re = std::make_shared<RE2>(R"(line: (\d+))"),
                              .line_formatv = "{0}"});
    }
    return replacements;
  }
};

// Prints arguments so that they can be validated in tests.
static auto PrintArgs(llvm::ArrayRef<llvm::StringRef> args,
                      llvm::raw_pwrite_stream& stdout) -> void {
  llvm::ListSeparator sep;
  stdout << args.size() << " args: ";
  for (auto arg : args) {
    stdout << sep << "`" << arg << "`";
  }
  stdout << "\n";
}

// Verifies arguments are well-structured, and returns the files in them.
static auto GetFilesFromArgs(llvm::ArrayRef<llvm::StringRef> args,
                             llvm::vfs::InMemoryFileSystem& fs)
    -> ErrorOr<llvm::ArrayRef<llvm::StringRef>> {
  if (args.empty() || args.front() != "default_args") {
    return ErrorBuilder() << "missing `default_args` argument";
  }
  args = args.drop_front();

  for (auto arg : args) {
    if (!fs.exists(arg)) {
      return ErrorBuilder() << "Missing file: " << arg;
    }
  }
  return args;
}

// Parameters used to by individual test handlers, for easy value forwarding.
struct TestParams {
  // These are the arguments to `Run()`.
  llvm::vfs::InMemoryFileSystem& fs;
  llvm::raw_pwrite_stream& stdout;
  llvm::raw_pwrite_stream& stderr;

  // This is assigned after construction.
  llvm::ArrayRef<llvm::StringRef> files;
};

// Does printing and returns expected results for alternating_files.carbon.
static auto TestAlternatingFiles(TestParams& params)
    -> ErrorOr<FileTestBaseTest::RunResult> {
  params.stdout << "unattached message 1\n"
                << "a.carbon:2: message 2\n"
                << "b.carbon:5: message 3\n"
                << "a.carbon:2: message 4\n"
                << "b.carbon:5: message 5\n"
                << "unattached message 6\n";
  params.stderr << "unattached message 1\n"
                << "a.carbon:2: message 2\n"
                << "b.carbon:5: message 3\n"
                << "a.carbon:2: message 4\n"
                << "b.carbon:5: message 5\n"
                << "unattached message 6\n";
  return {{.success = true}};
}

// Does printing and returns expected results for capture_console_output.carbon.
static auto TestCaptureConsoleOutput(TestParams& params)
    -> ErrorOr<FileTestBaseTest::RunResult> {
  llvm::errs() << "llvm::errs\n";
  params.stderr << "params.stderr\n";
  llvm::outs() << "llvm::outs\n";
  params.stdout << "params.stdout\n";
  return {{.success = true}};
}

// Does printing and returns expected results for example.carbon.
static auto TestExample(TestParams& params)
    -> ErrorOr<FileTestBaseTest::RunResult> {
  int delta_line = 10;
  params.stdout << "something\n"
                << "\n"
                << "example.carbon:" << delta_line + 1 << ": Line delta\n"
                << "example.carbon:" << delta_line << ": Negative line delta\n"
                << "+*[]{}\n"
                << "Foo baz\n";
  return {{.success = true}};
}

// Does printing and returns expected results for fail_example.carbon.
static auto TestFailExample(TestParams& params)
    -> ErrorOr<FileTestBaseTest::RunResult> {
  params.stderr << "Oops\n";
  return {{.success = false}};
}

// Does printing and returns expected results for
// file_only_re_multi_file.carbon.
static auto TestFileOnlyREMultiFile(TestParams& params)
    -> ErrorOr<FileTestBaseTest::RunResult> {
  int msg_count = 0;
  params.stdout << "unattached message " << ++msg_count << "\n"
                << "file: a.carbon\n"
                << "unattached message " << ++msg_count << "\n"
                << "line: 3: attached message " << ++msg_count << "\n"
                << "unattached message " << ++msg_count << "\n"
                << "line: 8: late message " << ++msg_count << "\n"
                << "unattached message " << ++msg_count << "\n"
                << "file: b.carbon\n"
                << "line: 2: attached message " << ++msg_count << "\n"
                << "unattached message " << ++msg_count << "\n"
                << "line: 7: late message " << ++msg_count << "\n"
                << "unattached message " << ++msg_count << "\n";
  return {{.success = true}};
}

// Does printing and returns expected results for file_only_re_one_file.carbon.
static auto TestFileOnlyREOneFile(TestParams& params)
    -> ErrorOr<FileTestBaseTest::RunResult> {
  params.stdout << "unattached message 1\n"
                << "file: file_only_re_one_file.carbon\n"
                << "line: 1\n"
                << "unattached message 2\n";
  return {{.success = true}};
}

// Does printing and returns expected results for unattached_multi_file.carbon.
static auto TestUnattachedMultiFile(TestParams& params)
    -> ErrorOr<FileTestBaseTest::RunResult> {
  params.stdout << "unattached message 1\n"
                << "unattached message 2\n";
  params.stderr << "unattached message 3\n"
                << "unattached message 4\n";
  return {{.success = true}};
}

// Does printing and returns expected results for:
// - fail_multi_success_overall_fail.carbon
// - multi_success.carbon
// - multi_success_and_fail.carbon
//
// Parameters indicate overall and per-file success.
static auto HandleMultiSuccessTests(bool overall, bool a, bool b)
    -> ErrorOr<FileTestBaseTest::RunResult> {
  FileTestBaseTest::RunResult result = {.success = overall};
  result.per_file_success.push_back({a ? "a.carbon" : "fail_a.carbon", a});
  result.per_file_success.push_back({b ? "b.carbon" : "fail_b.carbon", b});
  return result;
}

// Echoes back non-comment file content. Used for default file handling.
static auto EchoFileContent(TestParams& params)
    -> ErrorOr<FileTestBaseTest::RunResult> {
  // By default, echo non-comment content of files back.
  for (auto test_file : params.files) {
    // Describe file contents to stdout to validate splitting.
    auto file = params.fs.getBufferForFile(test_file, /*FileSize=*/-1,
                                           /*RequiresNullTerminator=*/false);
    if (file.getError()) {
      return Error(file.getError().message());
    }
    llvm::StringRef buffer = file.get()->getBuffer();
    for (int line_number = 1; !buffer.empty(); ++line_number) {
      auto [line, remainder] = buffer.split('\n');
      if (!line.empty() && !line.starts_with("//")) {
        params.stdout << test_file << ":" << line_number << ": " << line
                      << "\n";
      }
      buffer = remainder;
    }
  }
  return {{.success = true}};
}

auto FileTestBaseTest::Run(
    const llvm::SmallVector<llvm::StringRef>& test_args,
    llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem>& fs,
    llvm::raw_pwrite_stream& stdout, llvm::raw_pwrite_stream& stderr)
    -> ErrorOr<RunResult> {
  PrintArgs(test_args, stdout);

  auto filename = std::filesystem::path(test_name().str()).filename();
  if (filename == "args.carbon") {
    // 'args.carbon' has custom arguments, so don't do regular argument
    // validation for it.
    return {{.success = true}};
  }

  // Choose the test function based on filename.
  auto test_fn =
      llvm::StringSwitch<std::function<ErrorOr<RunResult>(TestParams&)>>(
          filename.string())
          .Case("alternating_files.carbon", &TestAlternatingFiles)
          .Case("capture_console_output.carbon", &TestCaptureConsoleOutput)
          .Case("example.carbon", &TestExample)
          .Case("fail_example.carbon", &TestFailExample)
          .Case("file_only_re_one_file.carbon", &TestFileOnlyREOneFile)
          .Case("file_only_re_multi_file.carbon", &TestFileOnlyREMultiFile)
          .Case("unattached_multi_file.carbon", &TestUnattachedMultiFile)
          .Case("fail_multi_success_overall_fail.carbon",
                [&](TestParams&) {
                  return HandleMultiSuccessTests(/*overall=*/false, /*a=*/true,
                                                 /*b=*/true);
                })
          .Case("multi_success.carbon",
                [&](TestParams&) {
                  return HandleMultiSuccessTests(/*overall=*/true, /*a=*/true,
                                                 /*b=*/true);
                })
          .Case("multi_success_and_fail.carbon",
                [&](TestParams&) {
                  return HandleMultiSuccessTests(/*overall=*/false, /*a=*/true,
                                                 /*b=*/false);
                })
          .Default(&EchoFileContent);

  // Call the appropriate test function for the file.
  TestParams params = {.fs = *fs, .stdout = stdout, .stderr = stderr};
  CARBON_ASSIGN_OR_RETURN(params.files, GetFilesFromArgs(test_args, *fs));
  return test_fn(params);
}

}  // namespace

CARBON_FILE_TEST_FACTORY(FileTestBaseTest)

}  // namespace Carbon::Testing
