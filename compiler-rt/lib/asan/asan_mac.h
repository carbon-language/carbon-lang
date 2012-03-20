//===-- asan_mac.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// ASan-private header for asan_mac.cc
//===----------------------------------------------------------------------===//
#ifdef __APPLE__

#ifndef ASAN_MAC_H
#define ASAN_MAC_H

#include "asan_interceptors.h"

#include <CoreFoundation/CFString.h>

typedef void* pthread_workqueue_t;
typedef void* pthread_workitem_handle_t;

typedef void* dispatch_group_t;
typedef void* dispatch_queue_t;
typedef uint64_t dispatch_time_t;
typedef void (*dispatch_function_t)(void *block);
typedef void* (*worker_t)(void *block);

DECLARE_REAL_AND_INTERCEPTOR(void, dispatch_async_f, dispatch_queue_t dq,
                                                     void *ctxt,
                                                     dispatch_function_t func);
DECLARE_REAL_AND_INTERCEPTOR(void, dispatch_sync_f, dispatch_queue_t dq,
                                                    void *ctxt,
                                                    dispatch_function_t func);
DECLARE_REAL_AND_INTERCEPTOR(void, dispatch_after_f, dispatch_time_t when,
                                                     dispatch_queue_t dq,
                                                     void *ctxt,
                                                     dispatch_function_t func);
DECLARE_REAL_AND_INTERCEPTOR(void, dispatch_barrier_async_f,
                                   dispatch_queue_t dq,
                                   void *ctxt,
                                   dispatch_function_t func);
DECLARE_REAL_AND_INTERCEPTOR(void, dispatch_group_async_f,
                                   dispatch_group_t group,
                                   dispatch_queue_t dq,
                                   void *ctxt,
                                   dispatch_function_t func);
DECLARE_REAL_AND_INTERCEPTOR(int, pthread_workqueue_additem_np,
                                  pthread_workqueue_t workq,
                                  void *(*workitem_func)(void *),
                                  void * workitem_arg,
                                  pthread_workitem_handle_t * itemhandlep,
                                  unsigned int *gencountp);
DECLARE_REAL_AND_INTERCEPTOR(CFStringRef, CFStringCreateCopy,
                                          CFAllocatorRef alloc,
                                          CFStringRef str);

// A wrapper for the ObjC blocks used to support libdispatch.
typedef struct {
  void *block;
  dispatch_function_t func;
  int parent_tid;
} asan_block_context_t;


extern "C" {
void dispatch_async_f(dispatch_queue_t dq, void *ctxt,
                      dispatch_function_t func);
void dispatch_sync_f(dispatch_queue_t dq, void *ctxt,
                     dispatch_function_t func);
void dispatch_after_f(dispatch_time_t when, dispatch_queue_t dq, void *ctxt,
                      dispatch_function_t func);
void dispatch_barrier_async_f(dispatch_queue_t dq, void *ctxt,
                              dispatch_function_t func);
void dispatch_group_async_f(dispatch_group_t group, dispatch_queue_t dq,
                            void *ctxt, dispatch_function_t func);
int pthread_workqueue_additem_np(pthread_workqueue_t workq,
    void *(*workitem_func)(void *), void * workitem_arg,
    pthread_workitem_handle_t * itemhandlep, unsigned int *gencountp);
}

#endif  // ASAN_MAC_H

#endif  // __APPLE__
