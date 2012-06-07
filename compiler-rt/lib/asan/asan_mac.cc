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

#ifdef __APPLE__

#include "asan_interceptors.h"
#include "asan_internal.h"
#include "asan_mapping.h"
#include "asan_procmaps.h"
#include "asan_stack.h"
#include "asan_thread.h"
#include "asan_thread_registry.h"
#include "sanitizer_common/sanitizer_libc.h"

#include <crt_externs.h>  // for _NSGetEnviron
#include <mach-o/dyld.h>
#include <mach-o/loader.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <sys/ucontext.h>
#include <pthread.h>
#include <fcntl.h>
#include <unistd.h>
#include <libkern/OSAtomic.h>
#include <CoreFoundation/CFString.h>

namespace __asan {

void GetPcSpBp(void *context, uptr *pc, uptr *sp, uptr *bp) {
  ucontext_t *ucontext = (ucontext_t*)context;
# if __WORDSIZE == 64
  *pc = ucontext->uc_mcontext->__ss.__rip;
  *bp = ucontext->uc_mcontext->__ss.__rbp;
  *sp = ucontext->uc_mcontext->__ss.__rsp;
# else
  *pc = ucontext->uc_mcontext->__ss.__eip;
  *bp = ucontext->uc_mcontext->__ss.__ebp;
  *sp = ucontext->uc_mcontext->__ss.__esp;
# endif  // __WORDSIZE
}

enum {
  MACOS_VERSION_UNKNOWN = 0,
  MACOS_VERSION_LEOPARD,
  MACOS_VERSION_SNOW_LEOPARD,
  MACOS_VERSION_LION,
};

static int GetMacosVersion() {
  int mib[2] = { CTL_KERN, KERN_OSRELEASE };
  char version[100];
  uptr len = 0, maxlen = sizeof(version) / sizeof(version[0]);
  for (int i = 0; i < maxlen; i++) version[i] = '\0';
  // Get the version length.
  CHECK(sysctl(mib, 2, 0, &len, 0, 0) != -1);
  CHECK(len < maxlen);
  CHECK(sysctl(mib, 2, version, &len, 0, 0) != -1);
  switch (version[0]) {
    case '9': return MACOS_VERSION_LEOPARD;
    case '1': {
      switch (version[1]) {
        case '0': return MACOS_VERSION_SNOW_LEOPARD;
        case '1': return MACOS_VERSION_LION;
        default: return MACOS_VERSION_UNKNOWN;
      }
    }
    default: return MACOS_VERSION_UNKNOWN;
  }
}

bool PlatformHasDifferentMemcpyAndMemmove() {
  // On OS X 10.7 memcpy() and memmove() are both resolved
  // into memmove$VARIANT$sse42.
  // See also http://code.google.com/p/address-sanitizer/issues/detail?id=34.
  // TODO(glider): need to check dynamically that memcpy() and memmove() are
  // actually the same function.
  return GetMacosVersion() == MACOS_VERSION_SNOW_LEOPARD;
}

// No-op. Mac does not support static linkage anyway.
void *AsanDoesNotSupportStaticLinkage() {
  return 0;
}

bool AsanInterceptsSignal(int signum) {
  return (signum == SIGSEGV || signum == SIGBUS) && FLAG_handle_segv;
}

void *AsanMmapFixedNoReserve(uptr fixed_addr, uptr size) {
  return internal_mmap((void*)fixed_addr, size,
                      PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANON | MAP_FIXED | MAP_NORESERVE,
                      0, 0);
}

void *AsanMprotect(uptr fixed_addr, uptr size) {
  return internal_mmap((void*)fixed_addr, size,
                       PROT_NONE,
                       MAP_PRIVATE | MAP_ANON | MAP_FIXED | MAP_NORESERVE,
                       0, 0);
}

const char *AsanGetEnv(const char *name) {
  char ***env_ptr = _NSGetEnviron();
  CHECK(env_ptr);
  char **environ = *env_ptr;
  CHECK(environ);
  uptr name_len = internal_strlen(name);
  while (*environ != 0) {
    uptr len = internal_strlen(*environ);
    if (len > name_len) {
      const char *p = *environ;
      if (!internal_memcmp(p, name, name_len) &&
          p[name_len] == '=') {  // Match.
        return *environ + name_len + 1;  // String starting after =.
      }
    }
    environ++;
  }
  return 0;
}

AsanLock::AsanLock(LinkerInitialized) {
  // We assume that OS_SPINLOCK_INIT is zero
}

void AsanLock::Lock() {
  CHECK(sizeof(OSSpinLock) <= sizeof(opaque_storage_));
  CHECK(OS_SPINLOCK_INIT == 0);
  CHECK(owner_ != (uptr)pthread_self());
  OSSpinLockLock((OSSpinLock*)&opaque_storage_);
  CHECK(!owner_);
  owner_ = (uptr)pthread_self();
}

void AsanLock::Unlock() {
  CHECK(owner_ == (uptr)pthread_self());
  owner_ = 0;
  OSSpinLockUnlock((OSSpinLock*)&opaque_storage_);
}

void AsanStackTrace::GetStackTrace(uptr max_s, uptr pc, uptr bp) {
  size = 0;
  trace[0] = pc;
  if ((max_s) > 1) {
    max_size = max_s;
    FastUnwindStack(pc, bp);
  }
}

// The range of pages to be used for escape islands.
// TODO(glider): instead of mapping a fixed range we must find a range of
// unmapped pages in vmmap and take them.
// These constants were chosen empirically and may not work if the shadow
// memory layout changes. Unfortunately they do necessarily depend on
// kHighMemBeg or kHighMemEnd.
static void *island_allocator_pos = 0;

#if __WORDSIZE == 32
# define kIslandEnd (0xffdf0000 - kPageSize)
# define kIslandBeg (kIslandEnd - 256 * kPageSize)
#else
# define kIslandEnd (0x7fffffdf0000 - kPageSize)
# define kIslandBeg (kIslandEnd - 256 * kPageSize)
#endif

extern "C"
mach_error_t __interception_allocate_island(void **ptr,
                                            uptr unused_size,
                                            void *unused_hint) {
  if (!island_allocator_pos) {
    island_allocator_pos =
        internal_mmap((void*)kIslandBeg, kIslandEnd - kIslandBeg,
                      PROT_READ | PROT_WRITE | PROT_EXEC,
                      MAP_PRIVATE | MAP_ANON | MAP_FIXED,
                      -1, 0);
    if (island_allocator_pos != (void*)kIslandBeg) {
      return KERN_NO_SPACE;
    }
    if (FLAG_v) {
      Report("Mapped pages %p--%p for branch islands.\n",
             (void*)kIslandBeg, (void*)kIslandEnd);
    }
    // Should not be very performance-critical.
    internal_memset(island_allocator_pos, 0xCC, kIslandEnd - kIslandBeg);
  };
  *ptr = island_allocator_pos;
  island_allocator_pos = (char*)island_allocator_pos + kPageSize;
  if (FLAG_v) {
    Report("Branch island allocated at %p\n", *ptr);
  }
  return err_none;
}

extern "C"
mach_error_t __interception_deallocate_island(void *ptr) {
  // Do nothing.
  // TODO(glider): allow to free and reuse the island memory.
  return err_none;
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

typedef void* pthread_workqueue_t;
typedef void* pthread_workitem_handle_t;

typedef void* dispatch_group_t;
typedef void* dispatch_queue_t;
typedef u64 dispatch_time_t;
typedef void (*dispatch_function_t)(void *block);
typedef void* (*worker_t)(void *block);

// A wrapper for the ObjC blocks used to support libdispatch.
typedef struct {
  void *block;
  dispatch_function_t func;
  u32 parent_tid;
} asan_block_context_t;

// We use extern declarations of libdispatch functions here instead
// of including <dispatch/dispatch.h>. This header is not present on
// Mac OS X Leopard and eariler, and although we don't expect ASan to
// work on legacy systems, it's bad to break the build of
// LLVM compiler-rt there.
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
}  // extern "C"

extern "C"
void asan_dispatch_call_block_and_release(void *block) {
  GET_STACK_TRACE_HERE(kStackTraceMax);
  asan_block_context_t *context = (asan_block_context_t*)block;
  if (FLAG_v >= 2) {
    Report("asan_dispatch_call_block_and_release(): "
           "context: %p, pthread_self: %p\n",
           block, pthread_self());
  }
  AsanThread *t = asanThreadRegistry().GetCurrent();
  if (!t) {
    t = AsanThread::Create(context->parent_tid, 0, 0, &stack);
    asanThreadRegistry().RegisterThread(t);
    t->Init();
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
  asan_ctxt->parent_tid = asanThreadRegistry().GetCurrentTidOrInvalid();
  return asan_ctxt;
}

// TODO(glider): can we reduce code duplication by introducing a macro?
INTERCEPTOR(void, dispatch_async_f, dispatch_queue_t dq, void *ctxt,
                                    dispatch_function_t func) {
  GET_STACK_TRACE_HERE(kStackTraceMax);
  asan_block_context_t *asan_ctxt = alloc_asan_context(ctxt, func, &stack);
  if (FLAG_v >= 2) {
    Report("dispatch_async_f(): context: %p, pthread_self: %p\n",
        asan_ctxt, pthread_self());
    PRINT_CURRENT_STACK();
  }
  return REAL(dispatch_async_f)(dq, (void*)asan_ctxt,
                                asan_dispatch_call_block_and_release);
}

INTERCEPTOR(void, dispatch_sync_f, dispatch_queue_t dq, void *ctxt,
                                   dispatch_function_t func) {
  GET_STACK_TRACE_HERE(kStackTraceMax);
  asan_block_context_t *asan_ctxt = alloc_asan_context(ctxt, func, &stack);
  if (FLAG_v >= 2) {
    Report("dispatch_sync_f(): context: %p, pthread_self: %p\n",
        asan_ctxt, pthread_self());
    PRINT_CURRENT_STACK();
  }
  return REAL(dispatch_sync_f)(dq, (void*)asan_ctxt,
                               asan_dispatch_call_block_and_release);
}

INTERCEPTOR(void, dispatch_after_f, dispatch_time_t when,
                                    dispatch_queue_t dq, void *ctxt,
                                    dispatch_function_t func) {
  GET_STACK_TRACE_HERE(kStackTraceMax);
  asan_block_context_t *asan_ctxt = alloc_asan_context(ctxt, func, &stack);
  if (FLAG_v >= 2) {
    Report("dispatch_after_f: %p\n", asan_ctxt);
    PRINT_CURRENT_STACK();
  }
  return REAL(dispatch_after_f)(when, dq, (void*)asan_ctxt,
                                asan_dispatch_call_block_and_release);
}

INTERCEPTOR(void, dispatch_barrier_async_f, dispatch_queue_t dq, void *ctxt,
                                            dispatch_function_t func) {
  GET_STACK_TRACE_HERE(kStackTraceMax);
  asan_block_context_t *asan_ctxt = alloc_asan_context(ctxt, func, &stack);
  if (FLAG_v >= 2) {
    Report("dispatch_barrier_async_f(): context: %p, pthread_self: %p\n",
           asan_ctxt, pthread_self());
    PRINT_CURRENT_STACK();
  }
  REAL(dispatch_barrier_async_f)(dq, (void*)asan_ctxt,
                                 asan_dispatch_call_block_and_release);
}

INTERCEPTOR(void, dispatch_group_async_f, dispatch_group_t group,
                                          dispatch_queue_t dq, void *ctxt,
                                          dispatch_function_t func) {
  GET_STACK_TRACE_HERE(kStackTraceMax);
  asan_block_context_t *asan_ctxt = alloc_asan_context(ctxt, func, &stack);
  if (FLAG_v >= 2) {
    Report("dispatch_group_async_f(): context: %p, pthread_self: %p\n",
           asan_ctxt, pthread_self());
    PRINT_CURRENT_STACK();
  }
  REAL(dispatch_group_async_f)(group, dq, (void*)asan_ctxt,
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
  GET_STACK_TRACE_HERE(kStackTraceMax);
  asan_free(arg, &stack);
  return result;
}

INTERCEPTOR(int, pthread_workqueue_additem_np, pthread_workqueue_t workq,
    void *(*workitem_func)(void *), void * workitem_arg,
    pthread_workitem_handle_t * itemhandlep, unsigned int *gencountp) {
  GET_STACK_TRACE_HERE(kStackTraceMax);
  asan_block_context_t *asan_ctxt =
      (asan_block_context_t*) asan_malloc(sizeof(asan_block_context_t), &stack);
  asan_ctxt->block = workitem_arg;
  asan_ctxt->func = (dispatch_function_t)workitem_func;
  asan_ctxt->parent_tid = asanThreadRegistry().GetCurrentTidOrInvalid();
  if (FLAG_v >= 2) {
    Report("pthread_workqueue_additem_np: %p\n", asan_ctxt);
    PRINT_CURRENT_STACK();
  }
  return REAL(pthread_workqueue_additem_np)(workq, wrap_workitem_func,
                                            asan_ctxt, itemhandlep,
                                            gencountp);
}

// CF_RC_BITS, the layout of CFRuntimeBase and __CFStrIsConstant are internal
// and subject to change in further CoreFoundation versions. Apple does not
// guarantee any binary compatibility from release to release.

// See http://opensource.apple.com/source/CF/CF-635.15/CFInternal.h
#if defined(__BIG_ENDIAN__)
#define CF_RC_BITS 0
#endif

#if defined(__LITTLE_ENDIAN__)
#define CF_RC_BITS 3
#endif

// See http://opensource.apple.com/source/CF/CF-635.15/CFRuntime.h
typedef struct __CFRuntimeBase {
  uptr _cfisa;
  u8 _cfinfo[4];
#if __LP64__
  u32 _rc;
#endif
} CFRuntimeBase;

// See http://opensource.apple.com/source/CF/CF-635.15/CFString.c
int __CFStrIsConstant(CFStringRef str) {
  CFRuntimeBase *base = (CFRuntimeBase*)str;
#if __LP64__
  return base->_rc == 0;
#else
  return (base->_cfinfo[CF_RC_BITS]) == 0;
#endif
}

INTERCEPTOR(CFStringRef, CFStringCreateCopy, CFAllocatorRef alloc,
                                             CFStringRef str) {
  if (__CFStrIsConstant(str)) {
    return str;
  } else {
    return REAL(CFStringCreateCopy)(alloc, str);
  }
}

namespace __asan {

void InitializeMacInterceptors() {
  CHECK(INTERCEPT_FUNCTION(dispatch_async_f));
  CHECK(INTERCEPT_FUNCTION(dispatch_sync_f));
  CHECK(INTERCEPT_FUNCTION(dispatch_after_f));
  CHECK(INTERCEPT_FUNCTION(dispatch_barrier_async_f));
  CHECK(INTERCEPT_FUNCTION(dispatch_group_async_f));
  // We don't need to intercept pthread_workqueue_additem_np() to support the
  // libdispatch API, but it helps us to debug the unsupported functions. Let's
  // intercept it only during verbose runs.
  if (FLAG_v >= 2) {
    CHECK(INTERCEPT_FUNCTION(pthread_workqueue_additem_np));
  }
  // Normally CFStringCreateCopy should not copy constant CF strings.
  // Replacing the default CFAllocator causes constant strings to be copied
  // rather than just returned, which leads to bugs in big applications like
  // Chromium and WebKit, see
  // http://code.google.com/p/address-sanitizer/issues/detail?id=10
  // Until this problem is fixed we need to check that the string is
  // non-constant before calling CFStringCreateCopy.
  CHECK(INTERCEPT_FUNCTION(CFStringCreateCopy));
}

}  // namespace __asan

#endif  // __APPLE__
