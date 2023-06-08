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

class LoweringFileTest : public DriverFileTestBase {
 public:
  using DriverFileTestBase::DriverFileTestBase;

  auto MakeArgs(const llvm::SmallVector<std::string>& test_files)
      -> llvm::SmallVector<llvm::StringRef> override {
    llvm::SmallVector<llvm::StringRef> args({"dump", "llvm-ir"});
    for (const auto& file : test_files) {
      args.push_back(file);
    }
    return args;
  }
};

}  // namespace

auto RegisterFileTests(const llvm::SmallVector<std::filesystem::path>& paths)
    -> void {
  LoweringFileTest::RegisterTests<LoweringFileTest>("LoweringFileTest", paths);
}

}  // namespace Carbon::Testing
