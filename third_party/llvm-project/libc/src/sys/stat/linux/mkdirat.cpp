//===-- Linux implementation of mkdirat -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/stat/mkdirat.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include <errno.h>
#include <sys/stat.h>
#include <sys/syscall.h> // For syscall numbers.

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, mkdirat, (int dfd, const char *path, mode_t mode)) {
#ifdef SYS_mkdirat
  long ret = __llvm_libc::syscall(SYS_mkdirat, dfd, path, mode);
#else
#error "mkdirat syscalls not available."
#endif

  if (ret < 0) {
    errno = -ret;
    return -1;
  }
  return 0;
}

} // namespace __llvm_libc
