// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "llvm/ADT/SmallVector.h"
#include "toolchain/driver/driver_file_test_base.h"

// Test commit with a no-op comment.

namespace Carbon::Testing {
namespace {

class ParseFileTest : public DriverFileTestBase {
 public:
  using DriverFileTestBase::DriverFileTestBase;

  auto GetDefaultArgs() -> llvm::SmallVector<std::string> override {
    return {"compile", "--phase=parse", "--dump-parse-tree", "%s"};
  }
};

}  // namespace

CARBON_FILE_TEST_FACTORY(ParseFileTest);

}  // namespace Carbon::Testing
