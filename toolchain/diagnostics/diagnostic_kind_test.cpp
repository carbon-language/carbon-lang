// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/diagnostic_kind.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "llvm/Support/raw_ostream.h"

namespace Carbon::Testing {
namespace {

TEST(DiagnosticKindTest, Name) {
  std::string buffer;
  llvm::raw_string_ostream(buffer) << DiagnosticKind::TestDiagnostic;
  EXPECT_EQ(buffer, "TestDiagnostic");
}

}  // namespace
}  // namespace Carbon::Testing
