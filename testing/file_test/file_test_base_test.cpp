// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/file_test/file_test_base.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <fstream>
#include <vector>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon::Testing {
namespace {

using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Field;

class FileTestBaseTest : public FileTestBase {
 public:
  explicit FileTestBaseTest(const std::filesystem::path& path)
      : FileTestBase(path) {}

  static auto HasFilename(std::string filename) -> testing::Matcher<TestFile> {
    return Field("filename", &TestFile::filename, Eq(filename));
  }

  static auto HasContent(std::string content) -> testing::Matcher<TestFile> {
    return Field("content", &TestFile::content, Eq(content));
  }

  auto RunWithFiles(const llvm::SmallVector<TestFile>& test_files,
                    llvm::raw_pwrite_stream& stdout,
                    llvm::raw_pwrite_stream& stderr) -> bool override {
    auto filename = path().filename();
    if (filename == "example.carbon") {
      EXPECT_THAT(test_files, ElementsAre(HasFilename("example.carbon")));
      stdout << "something\n"
                "\n"
                "8: Line delta\n"
                "7: Negative line delta\n"
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
};

}  // namespace

auto RegisterFileTests(const llvm::SmallVector<std::filesystem::path>& paths)
    -> void {
  FileTestBaseTest::RegisterTests<FileTestBaseTest>("FileTestBaseTest", paths);
}

}  // namespace Carbon::Testing
