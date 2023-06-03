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

class LoweringFileTest : public FileTestBase {
 public:
  explicit LoweringFileTest(const std::filesystem::path& path)
      : FileTestBase(path) {}

  auto RunOverFile(llvm::raw_ostream& stdout, llvm::raw_ostream& stderr)
      -> bool override {
    Driver driver(stdout, stderr);
    return driver.RunFullCommand(
        {"dump", "llvm-ir", path().filename().string()});
  }
};

}  // namespace

auto RegisterFileTests(const std::vector<std::filesystem::path>& paths)
    -> void {
  LoweringFileTest::RegisterTests("LoweringFileTest", paths,
                                  [=](const std::filesystem::path& path) {
                                    return new LoweringFileTest(path);
                                  });
}

}  // namespace Carbon::Testing
