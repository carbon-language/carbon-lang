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
  FileTestBaseTest(llvm::StringRef /*exe_path*/, llvm::StringRef test_name)
      : FileTestBase(test_name) {}

  auto Run(const llvm::SmallVector<llvm::StringRef>& test_args,
           llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem>& fs,
           FILE* input_stream, llvm::raw_pwrite_stream& output_stream,
           llvm::raw_pwrite_stream& error_stream) const
      -> ErrorOr<RunResult> override;

  auto GetArgReplacements() const -> llvm::StringMap<std::string> override {
    return {{"replacement", "replaced"}};
  }

  auto GetDefaultArgs() const -> llvm::SmallVector<std::string> override {
    return {"default_args", "%s"};
  }

  auto GetDefaultFileRE(llvm::ArrayRef<llvm::StringRef> filenames) const
      -> std::optional<RE2> override {
    return std::make_optional<RE2>(
        llvm::formatv(R"(file: ({0}))", llvm::join(filenames, "|")));
  }

  auto GetLineNumberReplacements(llvm::ArrayRef<llvm::StringRef> filenames)
      const -> llvm::SmallVector<LineNumberReplacement> override {
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
                      llvm::raw_pwrite_stream& output_stream) -> void {
  llvm::ListSeparator sep;
  output_stream << args.size() << " args: ";
  for (auto arg : args) {
    output_stream << sep << "`" << arg << "`";
  }
  output_stream << "\n";
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
  FILE* input_stream;
  llvm::raw_pwrite_stream& output_stream;
  llvm::raw_pwrite_stream& error_stream;

  // This is assigned after construction.
  llvm::ArrayRef<llvm::StringRef> files;
};

// Prints and returns expected results for alternating_files.carbon.
static auto TestAlternatingFiles(TestParams& params)
    -> ErrorOr<FileTestBaseTest::RunResult> {
  params.output_stream << "unattached message 1\n"
                       << "a.carbon:2: message 2\n"
                       << "b.carbon:5: message 3\n"
                       << "a.carbon:2: message 4\n"
                       << "b.carbon:5: message 5\n"
                       << "unattached message 6\n";
  params.error_stream << "unattached message 1\n"
                      << "a.carbon:2: message 2\n"
                      << "b.carbon:5: message 3\n"
                      << "a.carbon:2: message 4\n"
                      << "b.carbon:5: message 5\n"
                      << "unattached message 6\n";
  return {{.success = true}};
}

// Prints and returns expected results for capture_console_output.carbon.
static auto TestCaptureConsoleOutput(TestParams& params)
    -> ErrorOr<FileTestBaseTest::RunResult> {
  llvm::errs() << "llvm::errs\n";
  params.error_stream << "params.error_stream\n";
  llvm::outs() << "llvm::outs\n";
  params.output_stream << "params.output_stream\n";
  return {{.success = true}};
}

// Prints and returns expected results for example.carbon.
static auto TestExample(TestParams& params)
    -> ErrorOr<FileTestBaseTest::RunResult> {
  int delta_line = 10;
  params.output_stream << "something\n"
                       << "\n"
                       << "example.carbon:" << delta_line + 1
                       << ": Line delta\n"
                       << "example.carbon:" << delta_line
                       << ": Negative line delta\n"
                       << "+*[]{}\n"
                       << "Foo baz\n";
  return {{.success = true}};
}

// Prints and returns expected results for fail_example.carbon.
static auto TestFailExample(TestParams& params)
    -> ErrorOr<FileTestBaseTest::RunResult> {
  params.error_stream << "Oops\n";
  return {{.success = false}};
}

// Prints and returns expected results for
// file_only_re_multi_file.carbon.
static auto TestFileOnlyREMultiFile(TestParams& params)
    -> ErrorOr<FileTestBaseTest::RunResult> {
  int msg_count = 0;
  params.output_stream << "unattached message " << ++msg_count << "\n"
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

// Prints and returns expected results for file_only_re_one_file.carbon.
static auto TestFileOnlyREOneFile(TestParams& params)
    -> ErrorOr<FileTestBaseTest::RunResult> {
  params.output_stream << "unattached message 1\n"
                       << "file: file_only_re_one_file.carbon\n"
                       << "line: 1\n"
                       << "unattached message 2\n";
  return {{.success = true}};
}

// Prints and returns expected results for no_line_number.carbon.
static auto TestNoLineNumber(TestParams& params)
    -> ErrorOr<FileTestBaseTest::RunResult> {
  params.output_stream << "a.carbon: msg1\n"
                          "msg2\n"
                          "b.carbon: msg3\n"
                          "msg4\n"
                          "a.carbon: msg5\n";
  return {{.success = true}};
}

// Prints and returns expected results for escaping.carbon.
static auto TestEscaping(TestParams& params)
    -> ErrorOr<FileTestBaseTest::RunResult> {
  params.error_stream << "carriage return\r\n"
                         "{one brace}\n"
                         "{{two braces}}\n"
                         "[one bracket]\n"
                         "[[two brackets]]\n"
                         "end of line whitespace   \n"
                         "\ttabs\t\n";
  return {{.success = true}};
}

// Prints and returns expected results for unattached_multi_file.carbon.
static auto TestUnattachedMultiFile(TestParams& params)
    -> ErrorOr<FileTestBaseTest::RunResult> {
  params.output_stream << "unattached message 1\n"
                       << "unattached message 2\n";
  params.error_stream << "unattached message 3\n"
                      << "unattached message 4\n";
  return {{.success = true}};
}

// Prints and returns expected results for:
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
        params.output_stream << test_file << ":" << line_number << ": " << line
                             << "\n";
      }
      buffer = remainder;
    }
  }
  if (params.input_stream) {
    params.error_stream << "--- STDIN:\n";
    constexpr int ReadSize = 1024;
    char buf[ReadSize];
    while (feof(params.input_stream) == 0) {
      auto read = fread(&buf, sizeof(char), ReadSize, params.input_stream);
      if (read > 0) {
        params.error_stream.write(buf, read);
      }
    }
  }
  return {{.success = true}};
}

auto FileTestBaseTest::Run(
    const llvm::SmallVector<llvm::StringRef>& test_args,
    llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem>& fs,
    FILE* input_stream, llvm::raw_pwrite_stream& output_stream,
    llvm::raw_pwrite_stream& error_stream) const -> ErrorOr<RunResult> {
  PrintArgs(test_args, output_stream);

  auto filename = std::filesystem::path(test_name().str()).filename();
  if (filename == "args.carbon" || filename == "include_file.carbon") {
    // 'args.carbon' and 'include_file.carbon' have custom arguments, so don't
    // do regular argument validation for them.
    return {{.success = true}};
  }

  // Choose the test function based on filename.
  auto test_fn =
      llvm::StringSwitch<std::function<auto(TestParams&)->ErrorOr<RunResult>>>(
          filename.string())
          .Case("alternating_files.carbon", &TestAlternatingFiles)
          .Case("capture_console_output.carbon", &TestCaptureConsoleOutput)
          .Case("escaping.carbon", &TestEscaping)
          .Case("example.carbon", &TestExample)
          .Case("fail_example.carbon", &TestFailExample)
          .Case("file_only_re_one_file.carbon", &TestFileOnlyREOneFile)
          .Case("file_only_re_multi_file.carbon", &TestFileOnlyREMultiFile)
          .Case("no_line_number.carbon", &TestNoLineNumber)
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
  TestParams params = {.fs = *fs,
                       .input_stream = input_stream,
                       .output_stream = output_stream,
                       .error_stream = error_stream};
  CARBON_ASSIGN_OR_RETURN(params.files, GetFilesFromArgs(test_args, *fs));
  return test_fn(params);
}

}  // namespace

CARBON_FILE_TEST_FACTORY(FileTestBaseTest)

}  // namespace Carbon::Testing
