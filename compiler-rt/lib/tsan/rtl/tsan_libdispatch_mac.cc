//===-- tsan_libdispatch_mac.cc -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
// Mac-specific libdispatch (GCD) support.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_MAC

#include "sanitizer_common/sanitizer_common.h"
#include "interception/interception.h"
#include "tsan_interceptors.h"
#include "tsan_platform.h"
#include "tsan_rtl.h"

#include <dispatch/dispatch.h>
#include <pthread.h>

namespace __tsan {

// GCD's dispatch_once implementation has a fast path that contains a racy read
// and it's inlined into user's code. Furthermore, this fast path doesn't
// establish a proper happens-before relations between the initialization and
// code following the call to dispatch_once. We could deal with this in
// instrumented code, but there's not much we can do about it in system
// libraries. Let's disable the fast path (by never storing the value ~0 to
// predicate), so the interceptor is always called, and let's add proper release
// and acquire semantics. Since TSan does not see its own atomic stores, the
// race on predicate won't be reported - the only accesses to it that TSan sees
// are the loads on the fast path. Loads don't race. Secondly, dispatch_once is
// both a macro and a real function, we want to intercept the function, so we
// need to undefine the macro.
#undef dispatch_once
TSAN_INTERCEPTOR(void, dispatch_once, dispatch_once_t *predicate,
                 dispatch_block_t block) {
  SCOPED_TSAN_INTERCEPTOR(dispatch_once, predicate, block);
  atomic_uint32_t *a = reinterpret_cast<atomic_uint32_t *>(predicate);
  u32 v = atomic_load(a, memory_order_acquire);
  if (v == 0 &&
      atomic_compare_exchange_strong(a, &v, 1, memory_order_relaxed)) {
    block();
    Release(thr, pc, (uptr)a);
    atomic_store(a, 2, memory_order_release);
  } else {
    while (v != 2) {
      internal_sched_yield();
      v = atomic_load(a, memory_order_acquire);
    }
    Acquire(thr, pc, (uptr)a);
  }
}

#undef dispatch_once_f
TSAN_INTERCEPTOR(void, dispatch_once_f, dispatch_once_t *predicate,
                 void *context, dispatch_function_t function) {
  SCOPED_TSAN_INTERCEPTOR(dispatch_once_f, predicate, context, function);
  WRAP(dispatch_once)(predicate, ^(void) {
    function(context);
  });
}

}  // namespace __tsan

#endif  // SANITIZER_MAC
