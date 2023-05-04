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

  void RunOverFile(llvm::raw_ostream& stdout_stream,
                   llvm::raw_ostream& /*stderr*/) override {
    ASSERT_THAT(filename(), testing::StrEq("example.carbon"));

    stdout_stream << "something\n"
                     "\n"
                     "8: Line delta\n"
                     "7: Negative line delta\n"
                     "+*[]{}\n"
                     "Foo baz\n";
  }
};

}  // namespace

auto RegisterFileTests(const std::vector<llvm::StringRef>& paths) -> void {
  Carbon::Testing::FileTestBaseTest::RegisterTests(
      "FileTestBaseTest", paths, [](llvm::StringRef path) {
        return new Carbon::Testing::FileTestBaseTest(path);
      });
}

}  // namespace Carbon::Testing
