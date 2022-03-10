//===-- Linux implementation of fsync -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/fsync.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include <errno.h>
#include <sys/syscall.h> // For syscall numbers.

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, fsync, (int fd)) {
  long ret = __llvm_libc::syscall(SYS_fsync, fd);
  if (ret < 0) {
    errno = -ret;
    return -1;
  }
  return ret;
}

} // namespace __llvm_libc
