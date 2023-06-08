// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <filesystem>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/driver/driver_file_test_base.h"

namespace Carbon::Testing {
namespace {

class ParseTreeFileTest : public DriverFileTestBase {
 public:
  using DriverFileTestBase::DriverFileTestBase;

  auto MakeArgs(const llvm::SmallVector<std::string>& test_files)
      -> llvm::SmallVector<llvm::StringRef> override {
    llvm::SmallVector<llvm::StringRef> args({"dump", "parse-tree"});
    for (const auto& file : test_files) {
      args.push_back(file);
    }
    return args;
  }
};

}  // namespace

auto RegisterFileTests(const llvm::SmallVector<std::filesystem::path>& paths)
    -> void {
  ParseTreeFileTest::RegisterTests<ParseTreeFileTest>("ParseTreeFileTest",
                                                      paths);
}

}  // namespace Carbon::Testing
