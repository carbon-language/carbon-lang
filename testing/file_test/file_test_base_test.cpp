// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/file_test/file_test_base.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <vector>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon::Testing {
namespace {

class FileTestBaseTest : public FileTestBase {
 public:
  explicit FileTestBaseTest(const std::filesystem::path& path)
      : FileTestBase(path) {}

  auto RunOverFile(llvm::raw_ostream& stdout, llvm::raw_ostream& stderr)
      -> bool override {
    auto filename = path().filename();
    if (filename == "example.carbon") {
      stdout << "something\n"
                "\n"
                "8: Line delta\n"
                "7: Negative line delta\n"
                "+*[]{}\n"
                "Foo baz\n";
      return true;
    } else if (filename == "fail_example.carbon") {
      stderr << "Oops\n";
      return false;
    } else {
      ADD_FAILURE() << "Unexpected file: " << filename;
      return false;
    }
  }
};

}  // namespace

auto RegisterFileTests(const std::vector<std::filesystem::path>& paths)
    -> void {
  FileTestBaseTest::RegisterTests("FileTestBaseTest", paths,
                                  [](const std::filesystem::path& path) {
                                    return new FileTestBaseTest(path);
                                  });
}

}  // namespace Carbon::Testing
