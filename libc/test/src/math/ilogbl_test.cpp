//===-- Unittests for ilogbl ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ILogbTest.h"

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/ManipulationFunctions.h"
#include "src/__support/FPUtil/TestHelpers.h"
#include "src/math/ilogbl.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

using RunContext = __llvm_libc::testing::RunContext;

TEST_F(LlvmLibcILogbTest, SpecialNumbers_ilogbl) {
  testSpecialNumbers<long double>(&__llvm_libc::ilogbl);
}

TEST_F(LlvmLibcILogbTest, PowersOfTwo_ilogbl) {
  testPowersOfTwo<long double>(&__llvm_libc::ilogbl);
}

TEST_F(LlvmLibcILogbTest, SomeIntegers_ilogbl) {
  testSomeIntegers<long double>(&__llvm_libc::ilogbl);
}

TEST_F(LlvmLibcILogbTest, SubnormalRange_ilogbl) {
  testSubnormalRange<long double>(&__llvm_libc::ilogbl);
}

TEST_F(LlvmLibcILogbTest, NormalRange_ilogbl) {
  testNormalRange<long double>(&__llvm_libc::ilogbl);
}
