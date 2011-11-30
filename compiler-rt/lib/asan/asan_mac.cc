//===-- asan_mac.cc -------------------------------------------------------===//
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
// Mac-specific details.
//===----------------------------------------------------------------------===//

#ifndef __APPLE__
#error "This file should be used on Mac OS X only."
#endif

#include "asan_mac.h"

#include "asan_internal.h"
#include "asan_stack.h"
#include "asan_thread.h"
#include "asan_thread_registry.h"

#include <algorithm>

#include <sys/mman.h>
#include <unistd.h>

namespace __asan {

extern dispatch_async_f_f real_dispatch_async_f;
extern dispatch_sync_f_f real_dispatch_sync_f;
extern dispatch_after_f_f real_dispatch_after_f;
extern dispatch_barrier_async_f_f real_dispatch_barrier_async_f;
extern dispatch_group_async_f_f real_dispatch_group_async_f;
extern pthread_workqueue_additem_np_f real_pthread_workqueue_additem_np;

// No-op. Mac does not support static linkage anyway.
void *AsanDoesNotSupportStaticLinkage() {
  return NULL;
}

void *asan_mmap(void *addr, size_t length, int prot, int flags,
                int fd, uint64_t offset) {
  return mmap(addr, length, prot, flags, fd, offset);
}

ssize_t asan_write(int fd, const void *buf, size_t count) {
  return write(fd, buf, count);
}

// Support for the following functions from libdispatch on Mac OS:
//   dispatch_async_f()
//   dispatch_async()
//   dispatch_sync_f()
//   dispatch_sync()
//   dispatch_after_f()
//   dispatch_after()
//   dispatch_group_async_f()
//   dispatch_group_async()
// TODO(glider): libdispatch API contains other functions that we don't support
// yet.
//
// dispatch_sync() and dispatch_sync_f() are synchronous, although chances are
// they can cause jobs to run on a thread different from the current one.
// TODO(glider): if so, we need a test for this (otherwise we should remove
// them).
//
// The following functions use dispatch_barrier_async_f() (which isn't a library
// function but is exported) and are thus supported:
//   dispatch_source_set_cancel_handler_f()
//   dispatch_source_set_cancel_handler()
//   dispatch_source_set_event_handler_f()
//   dispatch_source_set_event_handler()
//
// The reference manual for Grand Central Dispatch is available at
//   http://developer.apple.com/library/mac/#documentation/Performance/Reference/GCD_libdispatch_Ref/Reference/reference.html
// The implementation details are at
//   http://libdispatch.macosforge.org/trac/browser/trunk/src/queue.c

extern "C"
void asan_dispatch_call_block_and_release(void *block) {
  GET_STACK_TRACE_HERE(kStackTraceMax, /*fast_unwind*/false);
  asan_block_context_t *context = (asan_block_context_t*)block;
  if (FLAG_v >= 2) {
    Report("asan_dispatch_call_block_and_release(): "
           "context: %p, pthread_self: %p\n",
           block, pthread_self());
  }
  AsanThread *t = asanThreadRegistry().GetCurrent();
  if (t) {
    // We've already executed a job on this worker thread. Let's reuse the
    // AsanThread object.
    if (t != asanThreadRegistry().GetMain()) {
      // Flush the statistics and update the current thread's tid.
      asanThreadRegistry().UnregisterThread(t);
      asanThreadRegistry().RegisterThread(t, context->parent_tid, &stack);
    }
    // Otherwise the worker is being executed on the main thread -- we are
    // draining the dispatch queue.
    // TODO(glider): any checks for that?
  } else {
    // It's incorrect to assert that the current thread is not dying: at least
    // the callbacks from dispatch_sync() are sometimes called after the TSD is
    // destroyed.
    t = (AsanThread*)asan_malloc(sizeof(AsanThread), &stack);
    new(t) AsanThread(context->parent_tid,
                      /*start_routine*/NULL, /*arg*/NULL, &stack);
    asanThreadRegistry().SetCurrent(t);
  }
  // Call the original dispatcher for the block.
  context->func(context->block);
  asan_free(context, &stack);
}

}  // namespace __asan

using namespace __asan;  // NOLINT

// Wrap |ctxt| and |func| into an asan_block_context_t.
// The caller retains control of the allocated context.
extern "C"
asan_block_context_t *alloc_asan_context(void *ctxt, dispatch_function_t func,
                                         AsanStackTrace *stack) {
  asan_block_context_t *asan_ctxt =
      (asan_block_context_t*) asan_malloc(sizeof(asan_block_context_t), stack);
  asan_ctxt->block = ctxt;
  asan_ctxt->func = func;
  AsanThread *curr_thread = asanThreadRegistry().GetCurrent();
  if (FLAG_debug) {
    // Sometimes at Chromium teardown this assertion is violated:
    //  -- a task is created via dispatch_async() on the "CFMachPort"
    //     thread while doing _dispatch_queue_drain();
    //  -- a task is created via dispatch_async_f() on the
    //     "com.apple.root.default-overcommit-priority" thread while doing
    //     _dispatch_dispose().
    // TODO(glider): find out what's going on.
    CHECK(curr_thread || asanThreadRegistry().IsCurrentThreadDying());
  }
  asan_ctxt->parent_tid = asanThreadRegistry().GetCurrentTidOrMinusOne();
  return asan_ctxt;
}

// TODO(glider): can we reduce code duplication by introducing a macro?
extern "C"
int WRAP(dispatch_async_f)(dispatch_queue_t dq,
                           void *ctxt,
                           dispatch_function_t func) {
  GET_STACK_TRACE_HERE(kStackTraceMax, /*fast_unwind*/false);
  asan_block_context_t *asan_ctxt = alloc_asan_context(ctxt, func, &stack);
  if (FLAG_v >= 2) {
    Report("dispatch_async_f(): context: %p, pthread_self: %p\n",
        asan_ctxt, pthread_self());
    PRINT_CURRENT_STACK();
  }
  return real_dispatch_async_f(dq, (void*)asan_ctxt,
                               asan_dispatch_call_block_and_release);
}

extern "C"
int WRAP(dispatch_sync_f)(dispatch_queue_t dq,
                          void *ctxt,
                          dispatch_function_t func) {
  GET_STACK_TRACE_HERE(kStackTraceMax, /*fast_unwind*/false);
  asan_block_context_t *asan_ctxt = alloc_asan_context(ctxt, func, &stack);
  if (FLAG_v >= 2) {
    Report("dispatch_sync_f(): context: %p, pthread_self: %p\n",
        asan_ctxt, pthread_self());
    PRINT_CURRENT_STACK();
  }
  return real_dispatch_sync_f(dq, (void*)asan_ctxt,
                              asan_dispatch_call_block_and_release);
}

extern "C"
int WRAP(dispatch_after_f)(dispatch_time_t when,
                           dispatch_queue_t dq,
                           void *ctxt,
                           dispatch_function_t func) {
  GET_STACK_TRACE_HERE(kStackTraceMax, /*fast_unwind*/false);
  asan_block_context_t *asan_ctxt = alloc_asan_context(ctxt, func, &stack);
  if (FLAG_v >= 2) {
    Report("dispatch_after_f: %p\n", asan_ctxt);
    PRINT_CURRENT_STACK();
  }
  return real_dispatch_after_f(when, dq, (void*)asan_ctxt,
                               asan_dispatch_call_block_and_release);
}

extern "C"
void WRAP(dispatch_barrier_async_f)(dispatch_queue_t dq,
                                    void *ctxt, dispatch_function_t func) {
  GET_STACK_TRACE_HERE(kStackTraceMax, /*fast_unwind*/false);
  asan_block_context_t *asan_ctxt = alloc_asan_context(ctxt, func, &stack);
  if (FLAG_v >= 2) {
    Report("dispatch_barrier_async_f(): context: %p, pthread_self: %p\n",
           asan_ctxt, pthread_self());
    PRINT_CURRENT_STACK();
  }
  real_dispatch_barrier_async_f(dq, (void*)asan_ctxt,
                                asan_dispatch_call_block_and_release);
}

extern "C"
void WRAP(dispatch_group_async_f)(dispatch_group_t group,
                                  dispatch_queue_t dq,
                                  void *ctxt, dispatch_function_t func) {
  GET_STACK_TRACE_HERE(kStackTraceMax, /*fast_unwind*/false);
  asan_block_context_t *asan_ctxt = alloc_asan_context(ctxt, func, &stack);
  if (FLAG_v >= 2) {
    Report("dispatch_group_async_f(): context: %p, pthread_self: %p\n",
           asan_ctxt, pthread_self());
    PRINT_CURRENT_STACK();
  }
  real_dispatch_group_async_f(group, dq, (void*)asan_ctxt,
                              asan_dispatch_call_block_and_release);
}

// The following stuff has been extremely helpful while looking for the
// unhandled functions that spawned jobs on Chromium shutdown. If the verbosity
// level is 2 or greater, we wrap pthread_workqueue_additem_np() in order to
// find the points of worker thread creation (each of such threads may be used
// to run several tasks, that's why this is not enough to support the whole
// libdispatch API.
extern "C"
void *wrap_workitem_func(void *arg) {
  if (FLAG_v >= 2) {
    Report("wrap_workitem_func: %p, pthread_self: %p\n", arg, pthread_self());
  }
  asan_block_context_t *ctxt = (asan_block_context_t*)arg;
  worker_t fn = (worker_t)(ctxt->func);
  void *result =  fn(ctxt->block);
  GET_STACK_TRACE_HERE(kStackTraceMax, /*fast_unwind*/false);
  asan_free(arg, &stack);
  return result;
}

extern "C"
int WRAP(pthread_workqueue_additem_np)(pthread_workqueue_t workq,
    void *(*workitem_func)(void *), void * workitem_arg,
    pthread_workitem_handle_t * itemhandlep, unsigned int *gencountp) {
  GET_STACK_TRACE_HERE(kStackTraceMax, /*fast_unwind*/false);
  asan_block_context_t *asan_ctxt =
      (asan_block_context_t*) asan_malloc(sizeof(asan_block_context_t), &stack);
  asan_ctxt->block = workitem_arg;
  asan_ctxt->func = (dispatch_function_t)workitem_func;
  asan_ctxt->parent_tid = asanThreadRegistry().GetCurrentTidOrMinusOne();
  if (FLAG_v >= 2) {
    Report("pthread_workqueue_additem_np: %p\n", asan_ctxt);
    PRINT_CURRENT_STACK();
  }
  return real_pthread_workqueue_additem_np(workq, wrap_workitem_func, asan_ctxt,
                                           itemhandlep, gencountp);
}
