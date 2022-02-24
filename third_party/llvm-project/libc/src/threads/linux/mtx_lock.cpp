//===-- Linux implementation of the mtx_lock function ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/threads/mtx_lock.h"
#include "include/threads.h"      // For mtx_t definition.
#include "src/__support/common.h"
#include "src/threads/linux/Mutex.h"

namespace __llvm_libc {

// The implementation currently handles only plain mutexes.
LLVM_LIBC_FUNCTION(int, mtx_lock, (mtx_t * mutex)) {
  auto *m = reinterpret_cast<Mutex *>(mutex);
  return m->lock();
}

} // namespace __llvm_libc
