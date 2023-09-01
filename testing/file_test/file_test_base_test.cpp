// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/file_test/file_test_base.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common/ostream.h"
#include "llvm/ADT/StringExtras.h"

namespace Carbon::Testing {
namespace {

using ::testing::Eq;

// Helper to validate file content.
static auto CheckFileContent(llvm::vfs::InMemoryFileSystem& fs,
                             llvm::StringRef filename,
                             llvm::StringRef expected_content)
    -> ErrorOr<Success> {
  auto file = fs.getBufferForFile(filename, /*FileSize=*/-1,
                                  /*RequiresNullTerminator=*/false);
  if (file.getError()) {
    return ErrorBuilder() << "Missing " << filename;
  }
  if (file->get()->getBuffer() != expected_content) {
    return ErrorBuilder() << "Unexpected file content for " << filename
                          << ".\n--- Actual:\n"
                          << file->get()->getBuffer() << "\n--- Expected:\n"
                          << expected_content << "\n---";
  }
  return Success();
}

class FileTestBaseTest : public FileTestBase {
 public:
  using FileTestBase::FileTestBase;

  auto Run(const llvm::SmallVector<llvm::StringRef>& test_args,
           llvm::vfs::InMemoryFileSystem& fs, llvm::raw_pwrite_stream& stdout,
           llvm::raw_pwrite_stream& stderr) -> ErrorOr<bool> override {
    if (!test_args.empty()) {
      llvm::ListSeparator sep;
      stdout << test_args.size() << " args: ";
      for (const auto& arg : test_args) {
        stdout << sep << "`" << arg << "`";
      }
      stdout << "\n";
    }

    auto filename = path().filename().string();
    if (filename == "two_files.carbon") {
      // Verify the split.
      CARBON_RETURN_IF_ERROR(CheckFileContent(
          fs, "a.carbon", "aaa\n// CHECK:STDOUT: a.carbon:[[@LINE-1]]: 1\n\n"));
      CARBON_RETURN_IF_ERROR(CheckFileContent(
          fs, "b.carbon", "bbb\n// CHECK:STDOUT: b.carbon:[[@LINE-1]]: 2\n"));
    } else {
      // Other files should be copied directly, so aren't as interesting.
      if (!fs.exists(filename)) {
        return ErrorBuilder() << "Missing file: " << filename;
      }
    }

    if (filename == "args.carbon") {
      return true;
    } else if (filename == "example.carbon") {
      int delta_line = 10;
      stdout << "something\n"
             << "\n"
             << "example.carbon:" << delta_line + 1 << ": Line delta\n"
             << "example.carbon:" << delta_line << ": Negative line delta\n"
             << "+*[]{}\n"
             << "Foo baz\n";
      return true;
    } else if (filename == "fail_example.carbon") {
      stderr << "Oops\n";
      return false;
    } else if (filename == "two_files.carbon") {
      // Prints line numbers to validate per-file.
      stdout << "a.carbon:1: 1\nb.carbon:1: 2\n";
      return true;
    } else {
      return ErrorBuilder() << "Unexpected file: " << filename;
    }
  }

  auto GetDefaultArgs() -> llvm::SmallVector<std::string> override {
    return {"default_args", "%s"};
  }
};

}  // namespace

CARBON_FILE_TEST_FACTORY(FileTestBaseTest);

}  // namespace Carbon::Testing
