// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TESTING_BASE_GTEST_MAIN_H_
#define CARBON_TESTING_BASE_GTEST_MAIN_H_

#include "llvm/ADT/StringRef.h"

// When using the Carbon `main` function for GoogleTest, we export some extra
// information about the test binary that can be accessed with this header.

namespace Carbon::Testing {

// The executable path of the test binary.
auto TestExePath() -> llvm::StringRef;

}  // namespace Carbon::Testing

#endif  // CARBON_TESTING_BASE_GTEST_MAIN_H_
