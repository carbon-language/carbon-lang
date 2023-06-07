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

class ParserFileTest : public FileTestBase {
 public:
  explicit ParserFileTest(const std::filesystem::path& path)
      : FileTestBase(path) {}

  auto RunWithFiles(const llvm::SmallVector<std::string>& test_files,
                    llvm::raw_ostream& stdout, llvm::raw_ostream& stderr)
      -> bool override {
    llvm::SmallVector<llvm::StringRef> args({"dump", "parse-tree"});
    for (const auto& file : test_files) {
      args.push_back(file);
    }
    Driver driver(stdout, stderr);
    return driver.RunFullCommand(args);
  }
};

}  // namespace

auto RegisterFileTests(const llvm::SmallVector<std::filesystem::path>& paths)
    -> void {
  ParserFileTest::RegisterTests("ParserFileTest", paths,
                                [](const std::filesystem::path& path) {
                                  return new ParserFileTest(path);
                                });
}

}  // namespace Carbon::Testing
