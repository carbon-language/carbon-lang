//===-- tsan_interceptors_mac.cc ------------------------------------------===//
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
// Mac-specific interceptors.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_MAC

#include "interception/interception.h"
#include "tsan_interceptors.h"

#include <libkern/OSAtomic.h>
#include <xpc/xpc.h>

namespace __tsan {

TSAN_INTERCEPTOR(void, OSSpinLockLock, volatile OSSpinLock *lock) {
  CHECK(!cur_thread()->is_dead);
  if (!cur_thread()->is_inited) {
    return REAL(OSSpinLockLock)(lock);
  }
  SCOPED_TSAN_INTERCEPTOR(OSSpinLockLock, lock);
  REAL(OSSpinLockLock)(lock);
  Acquire(thr, pc, (uptr)lock);
}

TSAN_INTERCEPTOR(bool, OSSpinLockTry, volatile OSSpinLock *lock) {
  CHECK(!cur_thread()->is_dead);
  if (!cur_thread()->is_inited) {
    return REAL(OSSpinLockTry)(lock);
  }
  SCOPED_TSAN_INTERCEPTOR(OSSpinLockTry, lock);
  bool result = REAL(OSSpinLockTry)(lock);
  if (result)
    Acquire(thr, pc, (uptr)lock);
  return result;
}

TSAN_INTERCEPTOR(void, OSSpinLockUnlock, volatile OSSpinLock *lock) {
  CHECK(!cur_thread()->is_dead);
  if (!cur_thread()->is_inited) {
    return REAL(OSSpinLockUnlock)(lock);
  }
  SCOPED_TSAN_INTERCEPTOR(OSSpinLockUnlock, lock);
  Release(thr, pc, (uptr)lock);
  REAL(OSSpinLockUnlock)(lock);
}

TSAN_INTERCEPTOR(void, os_lock_lock, void *lock) {
  CHECK(!cur_thread()->is_dead);
  if (!cur_thread()->is_inited) {
    return REAL(os_lock_lock)(lock);
  }
  SCOPED_TSAN_INTERCEPTOR(os_lock_lock, lock);
  REAL(os_lock_lock)(lock);
  Acquire(thr, pc, (uptr)lock);
}

TSAN_INTERCEPTOR(bool, os_lock_trylock, void *lock) {
  CHECK(!cur_thread()->is_dead);
  if (!cur_thread()->is_inited) {
    return REAL(os_lock_trylock)(lock);
  }
  SCOPED_TSAN_INTERCEPTOR(os_lock_trylock, lock);
  bool result = REAL(os_lock_trylock)(lock);
  if (result)
    Acquire(thr, pc, (uptr)lock);
  return result;
}

TSAN_INTERCEPTOR(void, os_lock_unlock, void *lock) {
  CHECK(!cur_thread()->is_dead);
  if (!cur_thread()->is_inited) {
    return REAL(os_lock_unlock)(lock);
  }
  SCOPED_TSAN_INTERCEPTOR(os_lock_unlock, lock);
  Release(thr, pc, (uptr)lock);
  REAL(os_lock_unlock)(lock);
}

TSAN_INTERCEPTOR(void, xpc_connection_set_event_handler,
                 xpc_connection_t connection, xpc_handler_t handler) {
  SCOPED_TSAN_INTERCEPTOR(xpc_connection_set_event_handler, connection,
                          handler);
  Release(thr, pc, (uptr)connection);
  xpc_handler_t new_handler = ^(xpc_object_t object) {
    {
      SCOPED_INTERCEPTOR_RAW(xpc_connection_set_event_handler);
      Acquire(thr, pc, (uptr)connection);
    }
    handler(object);
  };
  REAL(xpc_connection_set_event_handler)(connection, new_handler);
}

TSAN_INTERCEPTOR(void, xpc_connection_send_barrier, xpc_connection_t connection,
                 dispatch_block_t barrier) {
  SCOPED_TSAN_INTERCEPTOR(xpc_connection_send_barrier, connection, barrier);
  Release(thr, pc, (uptr)connection);
  dispatch_block_t new_barrier = ^() {
    {
      SCOPED_INTERCEPTOR_RAW(xpc_connection_send_barrier);
      Acquire(thr, pc, (uptr)connection);
    }
    barrier();
  };
  REAL(xpc_connection_send_barrier)(connection, new_barrier);
}

TSAN_INTERCEPTOR(void, xpc_connection_send_message_with_reply,
                 xpc_connection_t connection, xpc_object_t message,
                 dispatch_queue_t replyq, xpc_handler_t handler) {
  SCOPED_TSAN_INTERCEPTOR(xpc_connection_send_message_with_reply, connection,
                          message, replyq, handler);
  Release(thr, pc, (uptr)connection);
  xpc_handler_t new_handler = ^(xpc_object_t object) {
    {
      SCOPED_INTERCEPTOR_RAW(xpc_connection_send_message_with_reply);
      Acquire(thr, pc, (uptr)connection);
    }
    handler(object);
  };
  REAL(xpc_connection_send_message_with_reply)
  (connection, message, replyq, new_handler);
}

}  // namespace __tsan

#endif  // SANITIZER_MAC
