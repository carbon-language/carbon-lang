//===-- tsan_interface.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
// Public interface header for TSan.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_TSAN_INTERFACE_H
#define SANITIZER_TSAN_INTERFACE_H

#include <sanitizer/common_interface_defs.h>

#ifdef __cplusplus
extern "C" {
#endif

// __tsan_release establishes a happens-before relation with a preceding
// __tsan_acquire on the same address.
void __tsan_acquire(void *addr);
void __tsan_release(void *addr);

// Annotations for custom mutexes.
// The annotations allow to get better reports (with sets of locked mutexes),
// detect more types of bugs (e.g. mutex misuses, races between lock/unlock and
// destruction and potential deadlocks) and improve precision and performance
// (by ignoring individual atomic operations in mutex code). However, the
// downside is that annotated mutex code itself is not checked for correctness.

// Mutex creation flags are passed to __tsan_mutex_create annotation.
// If mutex has no constructor and __tsan_mutex_create is not called,
// the flags may be passed to __tsan_mutex_pre_lock/__tsan_mutex_post_lock
// annotations.

// Mutex has static storage duration and no-op constructor and destructor.
// This effectively makes tsan ignore destroy annotation.
static const unsigned __tsan_mutex_linker_init      = 1 << 0;
// Mutex is write reentrant.
static const unsigned __tsan_mutex_write_reentrant  = 1 << 1;
// Mutex is read reentrant.
static const unsigned __tsan_mutex_read_reentrant   = 1 << 2;
// Mutex does not have static storage duration, and must not be used after
// its destructor runs.  The opposite of __tsan_mutex_linker_init.
// If this flag is passed to __tsan_mutex_destroy, then the destruction
// is ignored unless this flag was previously set on the mutex.
static const unsigned __tsan_mutex_not_static       = 1 << 8;

// Mutex operation flags:

// Denotes read lock operation.
static const unsigned __tsan_mutex_read_lock        = 1 << 3;
// Denotes try lock operation.
static const unsigned __tsan_mutex_try_lock         = 1 << 4;
// Denotes that a try lock operation has failed to acquire the mutex.
static const unsigned __tsan_mutex_try_lock_failed  = 1 << 5;
// Denotes that the lock operation acquires multiple recursion levels.
// Number of levels is passed in recursion parameter.
// This is useful for annotation of e.g. Java builtin monitors,
// for which wait operation releases all recursive acquisitions of the mutex.
static const unsigned __tsan_mutex_recursive_lock   = 1 << 6;
// Denotes that the unlock operation releases all recursion levels.
// Number of released levels is returned and later must be passed to
// the corresponding __tsan_mutex_post_lock annotation.
static const unsigned __tsan_mutex_recursive_unlock = 1 << 7;

// Convenient composed constants.
static const unsigned __tsan_mutex_try_read_lock =
    __tsan_mutex_read_lock | __tsan_mutex_try_lock;
static const unsigned __tsan_mutex_try_read_lock_failed =
    __tsan_mutex_try_read_lock | __tsan_mutex_try_lock_failed;

// Annotate creation of a mutex.
// Supported flags: mutex creation flags.
void __tsan_mutex_create(void *addr, unsigned flags);

// Annotate destruction of a mutex.
// Supported flags:
//   - __tsan_mutex_linker_init
//   - __tsan_mutex_not_static
void __tsan_mutex_destroy(void *addr, unsigned flags);

// Annotate start of lock operation.
// Supported flags:
//   - __tsan_mutex_read_lock
//   - __tsan_mutex_try_lock
//   - all mutex creation flags
void __tsan_mutex_pre_lock(void *addr, unsigned flags);

// Annotate end of lock operation.
// Supported flags:
//   - __tsan_mutex_read_lock (must match __tsan_mutex_pre_lock)
//   - __tsan_mutex_try_lock (must match __tsan_mutex_pre_lock)
//   - __tsan_mutex_try_lock_failed
//   - __tsan_mutex_recursive_lock
//   - all mutex creation flags
void __tsan_mutex_post_lock(void *addr, unsigned flags, int recursion);

// Annotate start of unlock operation.
// Supported flags:
//   - __tsan_mutex_read_lock
//   - __tsan_mutex_recursive_unlock
int __tsan_mutex_pre_unlock(void *addr, unsigned flags);

// Annotate end of unlock operation.
// Supported flags:
//   - __tsan_mutex_read_lock (must match __tsan_mutex_pre_unlock)
void __tsan_mutex_post_unlock(void *addr, unsigned flags);

// Annotate start/end of notify/signal/broadcast operation.
// Supported flags: none.
void __tsan_mutex_pre_signal(void *addr, unsigned flags);
void __tsan_mutex_post_signal(void *addr, unsigned flags);

// Annotate start/end of a region of code where lock/unlock/signal operation
// diverts to do something else unrelated to the mutex. This can be used to
// annotate, for example, calls into cooperative scheduler or contention
// profiling code.
// These annotations must be called only from within
// __tsan_mutex_pre/post_lock, __tsan_mutex_pre/post_unlock,
// __tsan_mutex_pre/post_signal regions.
// Supported flags: none.
void __tsan_mutex_pre_divert(void *addr, unsigned flags);
void __tsan_mutex_post_divert(void *addr, unsigned flags);

// External race detection API.
// Can be used by non-instrumented libraries to detect when their objects are
// being used in an unsafe manner.
//   - __tsan_external_read/__tsan_external_write annotates the logical reads
//       and writes of the object at the specified address. 'caller_pc' should
//       be the PC of the library user, which the library can obtain with e.g.
//       `__builtin_return_address(0)`.
//   - __tsan_external_register_tag registers a 'tag' with the specified name,
//       which is later used in read/write annotations to denote the object type
//   - __tsan_external_assign_tag can optionally mark a heap object with a tag
void *__tsan_external_register_tag(const char *object_type);
void __tsan_external_register_header(void *tag, const char *header);
void __tsan_external_assign_tag(void *addr, void *tag);
void __tsan_external_read(void *addr, void *caller_pc, void *tag);
void __tsan_external_write(void *addr, void *caller_pc, void *tag);

// Fiber switching API.
//   - TSAN context for fiber can be created by __tsan_create_fiber
//     and freed by __tsan_destroy_fiber.
//   - TSAN context of current fiber or thread can be obtained
//     by calling __tsan_get_current_fiber.
//   - __tsan_switch_to_fiber should be called immediately before switch
//     to fiber, such as call of swapcontext.
//   - Fiber name can be set by __tsan_set_fiber_name.
void *__tsan_get_current_fiber(void);
void *__tsan_create_fiber(unsigned flags);
void __tsan_destroy_fiber(void *fiber);
void __tsan_switch_to_fiber(void *fiber, unsigned flags);
void __tsan_set_fiber_name(void *fiber, const char *name);

// Flags for __tsan_switch_to_fiber:
// Do not establish a happens-before relation between fibers
static const unsigned __tsan_switch_to_fiber_no_sync = 1 << 0;

// User-provided callback invoked on TSan initialization.
void __tsan_on_initialize();

// User-provided callback invoked on TSan shutdown.
// `failed` - Nonzero if TSan did detect issues, zero otherwise.
// Return `0` if TSan should exit as if no issues were detected.  Return nonzero
// if TSan should exit as if issues were detected.
int __tsan_on_finalize(int failed);

// Release TSan internal memory in a best-effort manner.
void __tsan_flush_memory();

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SANITIZER_TSAN_INTERFACE_H
