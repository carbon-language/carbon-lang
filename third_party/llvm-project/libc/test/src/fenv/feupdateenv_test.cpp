//===-- Unittests for feupdateenv -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fenv/feupdateenv.h"

#include "src/__support/FPUtil/FEnvUtils.h"
#include "utils/UnitTest/Test.h"

#include <fenv.h>
#include <signal.h>

TEST(LlvmLibcFEnvTest, UpdateEnvTest) {
  __llvm_libc::fputil::disableExcept(FE_ALL_EXCEPT);
  __llvm_libc::fputil::clearExcept(FE_ALL_EXCEPT);

  fenv_t env;
  ASSERT_EQ(__llvm_libc::fputil::getEnv(&env), 0);
  __llvm_libc::fputil::setExcept(FE_INVALID | FE_INEXACT);
  ASSERT_EQ(__llvm_libc::feupdateenv(&env), 0);
  ASSERT_EQ(__llvm_libc::fputil::testExcept(FE_INVALID | FE_INEXACT),
            FE_INVALID | FE_INEXACT);
}
