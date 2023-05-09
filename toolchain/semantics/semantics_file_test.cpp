// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <vector>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "testing/file_test/file_test_base.h"
#include "toolchain/driver/driver.h"

namespace Carbon::Testing {
namespace {

class SemanticsFileTest : public FileTestBase {
 public:
  explicit SemanticsFileTest(llvm::StringRef path) : FileTestBase(path) {}

  void RunOverFile(llvm::raw_ostream& stdout,
                   llvm::raw_ostream& stderr) override {
    Driver driver(stdout, stderr);
    driver.RunFullCommand({"dump", "semantics-ir", path()});
  }
};

}  // namespace

auto RegisterFileTests(const std::vector<llvm::StringRef>& paths) -> void {
  SemanticsFileTest::RegisterTests(
      "SemanticsFileTest", paths,
      [](llvm::StringRef path) { return new SemanticsFileTest(path); });
}

}  // namespace Carbon::Testing
