// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/file_test/file_test_base.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "llvm/ADT/StringExtras.h"

namespace Carbon::Testing {
namespace {

using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Field;
using ::testing::Matcher;

class FileTestBaseTest : public FileTestBase {
 public:
  using FileTestBase::FileTestBase;

  static auto HasFilename(std::string filename) -> Matcher<TestFile> {
    return Field("filename", &TestFile::filename, Eq(filename));
  }

  static auto HasContent(std::string content) -> Matcher<TestFile> {
    return Field("content", &TestFile::content, Eq(content));
  }

  auto Run(const llvm::SmallVector<llvm::StringRef>& test_args,
           const llvm::SmallVector<TestFile>& test_files,
           llvm::raw_pwrite_stream& stdout, llvm::raw_pwrite_stream& stderr)
      -> ErrorOr<bool> override {
    if (!test_args.empty()) {
      llvm::ListSeparator sep;
      stdout << test_args.size() << " args: ";
      for (const auto& arg : test_args) {
        stdout << sep << "`" << arg << "`";
      }
      stdout << "\n";
    }

    auto filename = path().filename();
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
      int i = 0;
      for (const auto& file : test_files) {
        // Prints line numbers to validate per-file.
        stdout << file.filename << ":1: " << ++i << "\n";
      }
      return true;
    } else if (filename == "alternating_files.carbon") {
      stdout << "unattached message 1\n"
             << "a.carbon:2: message 2\n"
             << "b.carbon:5: message 3\n"
             << "a.carbon:2: message 4\n"
             << "b.carbon:5: message 5\n"
             << "unattached message 6\n";
      stderr << "unattached message 1\n"
             << "a.carbon:2: message 2\n"
             << "b.carbon:5: message 3\n"
             << "a.carbon:2: message 4\n"
             << "b.carbon:5: message 5\n"
             << "unattached message 6\n";
      return true;
    } else if (filename == "unattached_multi_file.carbon") {
      stdout << "unattached message 1\n"
             << "unattached message 2\n";
      stderr << "unattached message 3\n"
             << "unattached message 4\n";
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
