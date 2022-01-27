//===-- Unittests for feraiseexcept with exceptions enabled ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fenv/feclearexcept.h"
#include "src/fenv/feraiseexcept.h"
#include "src/fenv/fetestexcept.h"

#include "src/__support/FPUtil/FEnvUtils.h"
#include "src/__support/architectures.h"
#include "utils/UnitTest/FPExceptMatcher.h"
#include "utils/UnitTest/Test.h"

#include <fenv.h>
#include <signal.h>

// This test enables an exception and verifies that raising that exception
// triggers SIGFPE.
TEST(LlvmLibcExceptionStatusTest, RaiseAndCrash) {
#if defined(LLVM_LIBC_ARCH_AARCH64)
  // Few aarch64 HW implementations do not trap exceptions. We skip this test
  // completely on such HW.
  //
  // Whether HW supports trapping exceptions or not is deduced by enabling an
  // exception and reading back to see if the exception got enabled. If the
  // exception did not get enabled, then it means that the HW does not support
  // trapping exceptions.
  __llvm_libc::fputil::disable_except(FE_ALL_EXCEPT);
  __llvm_libc::fputil::enable_except(FE_DIVBYZERO);
  if (__llvm_libc::fputil::get_except() == 0)
    return;
#endif // defined(LLVM_LIBC_ARCH_AARCH64)

  // TODO: Install a floating point exception handler and verify that the
  // the expected exception was raised. One will have to longjmp back from
  // that exception handler, so such a testing can be done after we have
  // longjmp implemented.

  int excepts[] = {FE_DIVBYZERO, FE_INVALID, FE_INEXACT, FE_OVERFLOW,
                   FE_UNDERFLOW};

  // We '|' the individual exception flags instead of using FE_ALL_EXCEPT
  // as it can include non-standard extensions. Note that we should be able
  // to compile this file with headers from other libcs as well.
  constexpr int ALL_EXCEPTS =
      FE_DIVBYZERO | FE_INVALID | FE_INEXACT | FE_OVERFLOW | FE_UNDERFLOW;

  for (int e : excepts) {
    __llvm_libc::fputil::disable_except(FE_ALL_EXCEPT);
    __llvm_libc::fputil::enable_except(e);
    ASSERT_EQ(__llvm_libc::feclearexcept(FE_ALL_EXCEPT), 0);
    // Raising all exceptions except |e| should not call the
    // SIGFPE handler. They should set the exception flag though,
    // so we verify that. Since other exceptions like FE_DIVBYZERO
    // can raise FE_INEXACT as well, we don't verify the other
    // exception flags when FE_INEXACT is enabled.
    if (e != FE_INEXACT) {
      int others = ALL_EXCEPTS & ~e;
      ASSERT_EQ(__llvm_libc::feraiseexcept(others), 0);
      ASSERT_EQ(__llvm_libc::fetestexcept(others), others);
    }

    ASSERT_RAISES_FP_EXCEPT([=] {
      // In test frameworks like Fuchsia's zxtest, this translates to
      // a death test which runs this closure in a different thread. So,
      // we enable the exception again inside this closure so that the
      // exception gets enabled for the thread running this closure.
      __llvm_libc::fputil::enable_except(e);
      __llvm_libc::feraiseexcept(e);
    });

    // Cleanup.
    __llvm_libc::fputil::disable_except(FE_ALL_EXCEPT);
    ASSERT_EQ(__llvm_libc::feclearexcept(FE_ALL_EXCEPT), 0);
  }
}
