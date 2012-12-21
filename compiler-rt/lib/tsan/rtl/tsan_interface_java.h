//===-- tsan_interface_java.h -----------------------------------*- C++ -*-===//
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
// Interface for verification of Java or mixed Java/C++ programs.
// The interface is intended to be used from within a JVM and notify TSan
// about such events like Java locks and GC memory compaction.
//
// For plain memory accesses and function entry/exit a JVM is intended to use
// C++ interfaces: __tsan_readN/writeN and __tsan_func_enter/exit.
//
// For volatile memory accesses and atomic operations JVM is intended to use
// standard atomics API: __tsan_atomicN_load/store/etc.
//
// For usage examples see lit_tests/java_*.cc
//===----------------------------------------------------------------------===//
#ifndef TSAN_INTERFACE_JAVA_H
#define TSAN_INTERFACE_JAVA_H

#ifndef INTERFACE_ATTRIBUTE
# define INTERFACE_ATTRIBUTE __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned long jptr;  // NOLINT

// Must be called before any other callback from Java.
void __tsan_java_init(jptr heap_begin, jptr heap_size) INTERFACE_ATTRIBUTE;
// Must be called when the application exits.
// Not necessary the last callback (concurrently running threads are OK).
// Returns exit status or 0 if tsan does not want to override it.
int  __tsan_java_fini() INTERFACE_ATTRIBUTE;

// Callback for memory allocations.
// May be omitted for allocations that are not subject to data races
// nor contain synchronization objects (e.g. String).
void __tsan_java_alloc(jptr ptr, jptr size) INTERFACE_ATTRIBUTE;
// Callback for memory free.
// Can be aggregated for several objects (preferably).
void __tsan_java_free(jptr ptr, jptr size) INTERFACE_ATTRIBUTE;
// Callback for memory move by GC.
// Can be aggregated for several objects (preferably).
// The ranges must not overlap.
void __tsan_java_move(jptr src, jptr dst, jptr size) INTERFACE_ATTRIBUTE;

// Mutex lock.
// Addr is any unique address associated with the mutex.
// Must not be called on recursive reentry.
// Object.wait() is handled as a pair of unlock/lock.
void __tsan_java_mutex_lock(jptr addr) INTERFACE_ATTRIBUTE;
// Mutex unlock.
void __tsan_java_mutex_unlock(jptr addr) INTERFACE_ATTRIBUTE;
// Mutex read lock.
void __tsan_java_mutex_read_lock(jptr addr) INTERFACE_ATTRIBUTE;
// Mutex read unlock.
void __tsan_java_mutex_read_unlock(jptr addr) INTERFACE_ATTRIBUTE;

#ifdef __cplusplus
}  // extern "C"
#endif

#undef INTERFACE_ATTRIBUTE

#endif  // #ifndef TSAN_INTERFACE_JAVA_H
