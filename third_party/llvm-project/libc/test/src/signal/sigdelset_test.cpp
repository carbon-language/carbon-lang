//===-- Unittests for sigdelset -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/errno.h"
#include "include/signal.h"
#include "src/signal/raise.h"
#include "src/signal/sigdelset.h"
#include "src/signal/sigfillset.h"
#include "src/signal/sigprocmask.h"

#include "test/ErrnoSetterMatcher.h"
#include "utils/UnitTest/Test.h"

TEST(LlvmLibcSigdelset, Invalid) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Fails;
  // Invalid set.
  EXPECT_THAT(__llvm_libc::sigdelset(nullptr, SIGUSR1), Fails(EINVAL));

  sigset_t set;
  // Valid set, invalid signum.
  EXPECT_THAT(__llvm_libc::sigdelset(&set, -1), Fails(EINVAL));
}

TEST(LlvmLibcSigdelset, UnblockOne) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Succeeds;
  sigset_t set;
  EXPECT_THAT(__llvm_libc::sigfillset(&set), Succeeds());
  EXPECT_THAT(__llvm_libc::sigdelset(&set, SIGUSR1), Succeeds());
  EXPECT_THAT(__llvm_libc::sigprocmask(SIG_SETMASK, &set, nullptr), Succeeds());
  EXPECT_DEATH([] { __llvm_libc::raise(SIGUSR1); }, WITH_SIGNAL(SIGUSR1));
}
