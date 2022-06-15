//===-- Implementation of the pthread_mutexattr_settype -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_mutexattr_settype.h"
#include "pthread_mutexattr.h"

#include "src/__support/common.h"

#include <errno.h>
#include <pthread.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, pthread_mutexattr_settype,
                   (pthread_mutexattr_t *__restrict attr, int type)) {
  if (type != PTHREAD_MUTEX_NORMAL && type != PTHREAD_MUTEX_ERRORCHECK &&
      type != PTHREAD_MUTEX_RECURSIVE) {
    return EINVAL;
  }
  pthread_mutexattr_t old = *attr;
  old &= ~unsigned(PThreadMutexAttrPos::TYPE_MASK);
  *attr = old | (type << unsigned(PThreadMutexAttrPos::TYPE_SHIFT));
  return 0;
}

} // namespace __llvm_libc
