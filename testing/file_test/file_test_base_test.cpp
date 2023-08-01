// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/file_test/file_test_base.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <fstream>
#include <vector>

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

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

  auto RunWithFiles(const llvm::SmallVector<llvm::StringRef>& test_args,
                    const llvm::SmallVector<TestFile>& test_files,
                    llvm::raw_pwrite_stream& stdout,
                    llvm::raw_pwrite_stream& stderr) -> bool override {
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
      EXPECT_THAT(test_files, ElementsAre(HasFilename("args.carbon")));
      return true;
    } else if (filename == "example.carbon") {
      EXPECT_THAT(test_files, ElementsAre(HasFilename("example.carbon")));
      stdout << "something\n"
                "\n"
                "9: Line delta\n"
                "8: Negative line delta\n"
                "+*[]{}\n"
                "Foo baz\n";
      return true;
    } else if (filename == "fail_example.carbon") {
      EXPECT_THAT(test_files, ElementsAre(HasFilename("fail_example.carbon")));
      stderr << "Oops\n";
      return false;
    } else if (filename == "two_files.carbon") {
      int i = 0;
      for (const auto& file : test_files) {
        // Prints line numbers to validate per-file.
        stdout << file.filename << ": " << ++i << "\n";
      }
      EXPECT_THAT(
          test_files,
          ElementsAre(
              AllOf(HasFilename("a.carbon"),
                    HasContent("// CHECK:STDOUT: a.carbon: [[@LINE+0]]\n\n")),
              AllOf(HasFilename("b.carbon"),
                    HasContent("// CHECK:STDOUT: b.carbon: [[@LINE+1]]\n"))));
      return true;
    } else {
      ADD_FAILURE() << "Unexpected file: " << filename;
      return false;
    }
  }

  auto GetDefaultArgs() -> llvm::SmallVector<std::string> override {
    return {"default_args", "%s"};
  }
};

}  // namespace

CARBON_FILE_TEST_FACTORY(FileTestBaseTest);

}  // namespace Carbon::Testing
