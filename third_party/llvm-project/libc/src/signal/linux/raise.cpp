//===-- Linux implementation of signal ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/signal/raise.h"
#include "src/signal/linux/signal.h"

#include "src/__support/common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, raise, (int sig)) {
  __llvm_libc::Sigset sigset;
  __llvm_libc::block_all_signals(sigset);
  long pid = __llvm_libc::syscall(SYS_getpid);
  long tid = __llvm_libc::syscall(SYS_gettid);
  int ret = __llvm_libc::syscall(SYS_tgkill, pid, tid, sig);
  __llvm_libc::restore_signals(sigset);
  return ret;
}

} // namespace __llvm_libc
