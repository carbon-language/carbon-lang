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
#include "asan_mac.h"
#include "asan_mapping.h"
#include "asan_stack.h"
#include "asan_thread.h"
#include "asan_thread_registry.h"
#include "sanitizer_common/sanitizer_libc.h"

#include <crt_externs.h>  // for _NSGetArgv
#include <dlfcn.h>  // for dladdr()
#include <mach-o/dyld.h>
#include <mach-o/loader.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <sys/ucontext.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdlib.h>  // for free()
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

int GetMacosVersion() {
  int mib[2] = { CTL_KERN, KERN_OSRELEASE };
  char version[100];
  uptr len = 0, maxlen = sizeof(version) / sizeof(version[0]);
  for (uptr i = 0; i < maxlen; i++) version[i] = '\0';
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

extern "C"
void __asan_init();

static const char kDyldInsertLibraries[] = "DYLD_INSERT_LIBRARIES";

void MaybeReexec() {
  if (!flags()->allow_reexec) return;
#if MAC_INTERPOSE_FUNCTIONS
  // If the program is linked with the dynamic ASan runtime library, make sure
  // the library is preloaded so that the wrappers work. If it is not, set
  // DYLD_INSERT_LIBRARIES and re-exec ourselves.
  Dl_info info;
  int result = dladdr((void*)__asan_init, &info);
  const char *dyld_insert_libraries = GetEnv(kDyldInsertLibraries);
  if (!dyld_insert_libraries ||
      !REAL(strstr)(dyld_insert_libraries, info.dli_fname)) {
    // DYLD_INSERT_LIBRARIES is not set or does not contain the runtime
    // library.
    char program_name[1024];
    uint32_t buf_size = sizeof(program_name);
    _NSGetExecutablePath(program_name, &buf_size);
    // Ok to use setenv() since the wrappers don't depend on the value of
    // asan_inited.
    setenv(kDyldInsertLibraries, info.dli_fname, /*overwrite*/0);
    if (flags()->verbosity >= 1) {
      Report("exec()-ing the program with\n");
      Report("%s=%s\n", kDyldInsertLibraries, info.dli_fname);
      Report("to enable ASan wrappers.\n");
      Report("Set ASAN_OPTIONS=allow_reexec=0 to disable this.\n");
    }
    execv(program_name, *_NSGetArgv());
  }
#endif  // MAC_INTERPOSE_FUNCTIONS
  // If we're not using the dynamic runtime, do nothing.
}

// No-op. Mac does not support static linkage anyway.
void *AsanDoesNotSupportStaticLinkage() {
  return 0;
}

bool AsanInterceptsSignal(int signum) {
  return (signum == SIGSEGV || signum == SIGBUS) && flags()->handle_segv;
}

void AsanPlatformThreadInit() {
  ReplaceCFAllocator();
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

void GetStackTrace(StackTrace *stack, uptr max_s, uptr pc, uptr bp) {
  stack->size = 0;
  stack->trace[0] = pc;
  if ((max_s) > 1) {
    stack->max_size = max_s;
    if (!asan_inited) return;
    if (AsanThread *t = asanThreadRegistry().GetCurrent())
      stack->FastUnwindStack(pc, bp, t->stack_top(), t->stack_bottom());
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
    if (flags()->verbosity) {
      Report("Mapped pages %p--%p for branch islands.\n",
             (void*)kIslandBeg, (void*)kIslandEnd);
    }
    // Should not be very performance-critical.
    internal_memset(island_allocator_pos, 0xCC, kIslandEnd - kIslandBeg);
  };
  *ptr = island_allocator_pos;
  island_allocator_pos = (char*)island_allocator_pos + kPageSize;
  if (flags()->verbosity) {
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
typedef void* dispatch_source_t;
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

static ALWAYS_INLINE
void asan_register_worker_thread(int parent_tid, StackTrace *stack) {
  AsanThread *t = asanThreadRegistry().GetCurrent();
  if (!t) {
    t = AsanThread::Create(parent_tid, 0, 0, stack);
    asanThreadRegistry().RegisterThread(t);
    t->Init();
    asanThreadRegistry().SetCurrent(t);
  }
}

// For use by only those functions that allocated the context via
// alloc_asan_context().
extern "C"
void asan_dispatch_call_block_and_release(void *block) {
  GET_STACK_TRACE_HERE(kStackTraceMax);
  asan_block_context_t *context = (asan_block_context_t*)block;
  if (flags()->verbosity >= 2) {
    Report("asan_dispatch_call_block_and_release(): "
           "context: %p, pthread_self: %p\n",
           block, pthread_self());
  }
  asan_register_worker_thread(context->parent_tid, &stack);
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
                                         StackTrace *stack) {
  asan_block_context_t *asan_ctxt =
      (asan_block_context_t*) asan_malloc(sizeof(asan_block_context_t), stack);
  asan_ctxt->block = ctxt;
  asan_ctxt->func = func;
  asan_ctxt->parent_tid = asanThreadRegistry().GetCurrentTidOrInvalid();
  return asan_ctxt;
}

// Define interceptor for dispatch_*_f function with the three most common
// parameters: dispatch_queue_t, context, dispatch_function_t.
#define INTERCEPT_DISPATCH_X_F_3(dispatch_x_f)                                \
  INTERCEPTOR(void, dispatch_x_f, dispatch_queue_t dq, void *ctxt,            \
                                  dispatch_function_t func) {                 \
    GET_STACK_TRACE_HERE(kStackTraceMax);                                     \
    asan_block_context_t *asan_ctxt = alloc_asan_context(ctxt, func, &stack); \
    if (flags()->verbosity >= 2) {                                            \
      Report(#dispatch_x_f "(): context: %p, pthread_self: %p\n",             \
             asan_ctxt, pthread_self());                                      \
       PRINT_CURRENT_STACK();                                                 \
     }                                                                        \
     return REAL(dispatch_x_f)(dq, (void*)asan_ctxt,                          \
                               asan_dispatch_call_block_and_release);         \
  }

INTERCEPT_DISPATCH_X_F_3(dispatch_async_f)
INTERCEPT_DISPATCH_X_F_3(dispatch_sync_f)
INTERCEPT_DISPATCH_X_F_3(dispatch_barrier_async_f)

INTERCEPTOR(void, dispatch_after_f, dispatch_time_t when,
                                    dispatch_queue_t dq, void *ctxt,
                                    dispatch_function_t func) {
  GET_STACK_TRACE_HERE(kStackTraceMax);
  asan_block_context_t *asan_ctxt = alloc_asan_context(ctxt, func, &stack);
  if (flags()->verbosity >= 2) {
    Report("dispatch_after_f: %p\n", asan_ctxt);
    PRINT_CURRENT_STACK();
  }
  return REAL(dispatch_after_f)(when, dq, (void*)asan_ctxt,
                                asan_dispatch_call_block_and_release);
}

INTERCEPTOR(void, dispatch_group_async_f, dispatch_group_t group,
                                          dispatch_queue_t dq, void *ctxt,
                                          dispatch_function_t func) {
  GET_STACK_TRACE_HERE(kStackTraceMax);
  asan_block_context_t *asan_ctxt = alloc_asan_context(ctxt, func, &stack);
  if (flags()->verbosity >= 2) {
    Report("dispatch_group_async_f(): context: %p, pthread_self: %p\n",
           asan_ctxt, pthread_self());
    PRINT_CURRENT_STACK();
  }
  REAL(dispatch_group_async_f)(group, dq, (void*)asan_ctxt,
                               asan_dispatch_call_block_and_release);
}

#if MAC_INTERPOSE_FUNCTIONS
// dispatch_async, dispatch_group_async and others tailcall the corresponding
// dispatch_*_f functions. When wrapping functions with mach_override, those
// dispatch_*_f are intercepted automatically. But with dylib interposition
// this does not work, because the calls within the same library are not
// interposed.
// Therefore we need to re-implement dispatch_async and friends.

extern "C" {
// FIXME: consolidate these declarations with asan_intercepted_functions.h.
void dispatch_async(dispatch_queue_t dq, void(^work)(void));
void dispatch_group_async(dispatch_group_t dg, dispatch_queue_t dq,
                          void(^work)(void));
void dispatch_after(dispatch_time_t when, dispatch_queue_t queue,
                    void(^work)(void));
void dispatch_source_set_cancel_handler(dispatch_source_t ds,
                                        void(^work)(void));
void dispatch_source_set_event_handler(dispatch_source_t ds, void(^work)(void));
}

#define GET_ASAN_BLOCK(work) \
  void (^asan_block)(void);  \
  int parent_tid = asanThreadRegistry().GetCurrentTidOrInvalid(); \
  asan_block = ^(void) { \
    GET_STACK_TRACE_HERE(kStackTraceMax); \
    asan_register_worker_thread(parent_tid, &stack); \
    work(); \
  }

INTERCEPTOR(void, dispatch_async,
            dispatch_queue_t dq, void(^work)(void)) {
  GET_ASAN_BLOCK(work);
  REAL(dispatch_async)(dq, asan_block);
}

INTERCEPTOR(void, dispatch_group_async,
            dispatch_group_t dg, dispatch_queue_t dq, void(^work)(void)) {
  GET_ASAN_BLOCK(work);
  REAL(dispatch_group_async)(dg, dq, asan_block);
}

INTERCEPTOR(void, dispatch_after,
            dispatch_time_t when, dispatch_queue_t queue, void(^work)(void)) {
  GET_ASAN_BLOCK(work);
  REAL(dispatch_after)(when, queue, asan_block);
}

INTERCEPTOR(void, dispatch_source_set_cancel_handler,
            dispatch_source_t ds, void(^work)(void)) {
  GET_ASAN_BLOCK(work);
  REAL(dispatch_source_set_cancel_handler)(ds, asan_block);
}

INTERCEPTOR(void, dispatch_source_set_event_handler,
            dispatch_source_t ds, void(^work)(void)) {
  GET_ASAN_BLOCK(work);
  REAL(dispatch_source_set_event_handler)(ds, asan_block);
}
#endif

// The following stuff has been extremely helpful while looking for the
// unhandled functions that spawned jobs on Chromium shutdown. If the verbosity
// level is 2 or greater, we wrap pthread_workqueue_additem_np() in order to
// find the points of worker thread creation (each of such threads may be used
// to run several tasks, that's why this is not enough to support the whole
// libdispatch API.
extern "C"
void *wrap_workitem_func(void *arg) {
  if (flags()->verbosity >= 2) {
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
  if (flags()->verbosity >= 2) {
    Report("pthread_workqueue_additem_np: %p\n", asan_ctxt);
    PRINT_CURRENT_STACK();
  }
  return REAL(pthread_workqueue_additem_np)(workq, wrap_workitem_func,
                                            asan_ctxt, itemhandlep,
                                            gencountp);
}

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

DECLARE_REAL_AND_INTERCEPTOR(void, free, void *ptr)

DECLARE_REAL_AND_INTERCEPTOR(void, __CFInitialize)

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
  if (flags()->verbosity >= 2) {
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
  // Some of the library functions call free() directly, so we have to
  // intercept it.
  CHECK(INTERCEPT_FUNCTION(free));
  if (flags()->replace_cfallocator) {
    CHECK(INTERCEPT_FUNCTION(__CFInitialize));
  }
}

}  // namespace __asan

#endif  // __APPLE__
