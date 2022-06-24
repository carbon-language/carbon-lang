// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_TEST_HELPERS_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_TEST_HELPERS_H_

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common/check.h"
#include "common/ostream.h"
#include "llvm/ADT/StringExtras.h"
#include "toolchain/semantics/nodes/binary_operator_test_matchers.h"
#include "toolchain/semantics/nodes/function_test_matchers.h"
#include "toolchain/semantics/nodes/integer_literal_test_matchers.h"
#include "toolchain/semantics/nodes/return_test_matchers.h"
#include "toolchain/semantics/nodes/set_name_test_matchers.h"
#include "toolchain/semantics/semantics_ir_for_test.h"

namespace Carbon::Testing {

// Avoids gtest confusion of how to print llvm::None.
MATCHER(IsNone, "is llvm::None") { return arg == llvm::None; }

}  // namespace Carbon::Testing

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_TEST_HELPERS_H_
