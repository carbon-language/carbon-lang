// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "llvm/ADT/SmallVector.h"
#include "toolchain/driver/driver_file_test_base.h"

namespace Carbon::Testing {
namespace {

class DriverFileTest : public DriverFileTestBase {
 public:
  using DriverFileTestBase::DriverFileTestBase;

  auto GetDefaultArgs() -> llvm::SmallVector<std::string> override {
    CARBON_FATAL() << "ARGS is always set in these tests";
  }

  auto DoExtraCheckReplacements(std::string& check_line) -> void override {
    // TODO: Disable token output, it's not interesting for these tests.
    if (llvm::StringRef(check_line).starts_with("// CHECK:STDOUT: {")) {
      check_line = "// CHECK:STDOUT: {{.*}}";
    }
  }
};

}  // namespace

CARBON_FILE_TEST_FACTORY(DriverFileTest);

}  // namespace Carbon::Testing
