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

using ::testing::ElementsAre;

class FileTestBaseTest : public FileTestBase {
 public:
  explicit FileTestBaseTest(const std::filesystem::path& path)
      : FileTestBase(path) {}

  auto RunWithFiles(const llvm::SmallVector<std::string>& test_files,
                    llvm::raw_ostream& stdout, llvm::raw_ostream& stderr)
      -> bool override {
    auto filename = path().filename();
    if (filename == "example.carbon") {
      EXPECT_THAT(test_files, ElementsAre("example.carbon"));
      stdout << "something\n"
                "\n"
                "8: Line delta\n"
                "7: Negative line delta\n"
                "+*[]{}\n"
                "Foo baz\n";
      return true;
    } else if (filename == "fail_example.carbon") {
      EXPECT_THAT(test_files, ElementsAre("fail_example.carbon"));
      stderr << "Oops\n";
      return false;
    } else if (filename == "two_files.carbon") {
      int i = 0;
      for (const auto& file : test_files) {
        // Prints line numbers to validate per-file.
        stdout << file << ": " << ++i << "\n";

        // Make sure the split files have appropriate content.
        std::ifstream file_in(file);
        std::stringstream content;
        content << file_in.rdbuf();
        if (file == "a.carbon") {
          EXPECT_THAT(content.str(),
                      testing::Eq("// CHECK:STDOUT: a.carbon: [[@LINE+0]]\n\n"))
              << "Checking " << file;
        } else {
          EXPECT_THAT(content.str(),
                      testing::Eq("// CHECK:STDOUT: b.carbon: [[@LINE+1]]\n"))
              << "Checking " << file;
        }
      }
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
  FileTestBaseTest::RegisterTests("FileTestBaseTest", paths,
                                  [](const std::filesystem::path& path) {
                                    return new FileTestBaseTest(path);
                                  });
}

}  // namespace Carbon::Testing
