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
  explicit FileTestBaseTest(llvm::StringRef path) : FileTestBase(path) {}

  auto RunOverFile(llvm::raw_ostream& stdout, llvm::raw_ostream& stderr)
      -> bool override {
    if (filename() == "example.carbon") {
      stdout << "something\n"
                "\n"
                "8: Line delta\n"
                "7: Negative line delta\n"
                "+*[]{}\n"
                "Foo baz\n";
      return true;
    } else if (filename() == "fail_example.carbon") {
      stderr << "Oops\n";
      return false;
    } else {
      ADD_FAILURE() << "Unexpected file: " << path().str();
      return false;
    }
  }
};

}  // namespace

auto RegisterFileTests(const std::vector<llvm::StringRef>& paths) -> void {
  FileTestBaseTest::RegisterTests(
      "FileTestBaseTest", paths,
      [](llvm::StringRef path) { return new FileTestBaseTest(path); });
}

}  // namespace Carbon::Testing
