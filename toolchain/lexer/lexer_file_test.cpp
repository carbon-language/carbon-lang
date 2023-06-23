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

class LexerFileTest : public DriverFileTestBase {
 public:
  using DriverFileTestBase::DriverFileTestBase;

  auto MakeArgs(const llvm::SmallVector<llvm::StringRef>& test_files)
      -> llvm::SmallVector<llvm::StringRef> override {
    llvm::SmallVector<llvm::StringRef> args({"dump", "tokens"});
    args.insert(args.end(), test_files.begin(), test_files.end());
    return args;
  }
};

}  // namespace

auto RegisterFileTests(const llvm::SmallVector<std::filesystem::path>& paths)
    -> void {
  LexerFileTest::RegisterTests<LexerFileTest>("LexerFileTest", paths);
}

}  // namespace Carbon::Testing
