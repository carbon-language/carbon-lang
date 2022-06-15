//===-- Implementation of the pthread_attr_setdetachstate -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_attr_setdetachstate.h"

#include "src/__support/common.h"

#include <errno.h>
#include <pthread.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, pthread_attr_setdetachstate,
                   (pthread_attr_t * attr, int detachstate)) {
  if (detachstate != PTHREAD_CREATE_DETACHED &&
      detachstate != PTHREAD_CREATE_JOINABLE)
    return EINVAL;
  attr->__detachstate = detachstate;
  return 0;
}

} // namespace __llvm_libc
