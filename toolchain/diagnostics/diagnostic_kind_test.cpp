// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/diagnostic_kind.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace Carbon::Testing {
namespace {

TEST(DiagnosticKindTest, Name) {
  EXPECT_EQ(DiagnosticKind::TestDiagnostic().name(), "TestDiagnostic");
}

}  // namespace
}  // namespace Carbon::Testing
