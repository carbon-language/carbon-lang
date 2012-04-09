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

void GetPcSpBp(void *context, uintptr_t *pc, uintptr_t *sp, uintptr_t *bp) {
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
  size_t len = 0, maxlen = sizeof(version) / sizeof(version[0]);
  for (int i = 0; i < maxlen; i++) version[i] = '\0';
  // Get the version length.
  CHECK(sysctl(mib, 2, NULL, &len, NULL, 0) != -1);
  CHECK(len < maxlen);
  CHECK(sysctl(mib, 2, version, &len, NULL, 0) != -1);
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
  return NULL;
}

static inline bool IntervalsAreSeparate(uintptr_t start1, uintptr_t end1,
                                        uintptr_t start2, uintptr_t end2) {
  CHECK(start1 <= end1);
  CHECK(start2 <= end2);
  return (end1 < start2) || (end2 < start1);
}

// FIXME: this is thread-unsafe, but should not cause problems most of the time.
// When the shadow is mapped only a single thread usually exists (plus maybe
// several worker threads on Mac, which aren't expected to map big chunks of
// memory).
bool AsanShadowRangeIsAvailable() {
  AsanProcMaps procmaps;
  uintptr_t start, end;
  bool available = true;
  while (procmaps.Next(&start, &end,
                       /*offset*/NULL, /*filename*/NULL, /*filename_size*/0)) {
    if (!IntervalsAreSeparate(start, end,
                              kLowShadowBeg - kMmapGranularity,
                              kHighShadowEnd)) {
      available = false;
      break;
    }
  }
  return available;
}

bool AsanInterceptsSignal(int signum) {
  return (signum == SIGSEGV || signum == SIGBUS) && FLAG_handle_segv;
}

static void *asan_mmap(void *addr, size_t length, int prot, int flags,
                int fd, uint64_t offset) {
  return mmap(addr, length, prot, flags, fd, offset);
}

size_t AsanWrite(int fd, const void *buf, size_t count) {
  return write(fd, buf, count);
}

void *AsanMmapSomewhereOrDie(size_t size, const char *mem_type) {
  size = RoundUpTo(size, kPageSize);
  void *res = asan_mmap(0, size,
                        PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANON, -1, 0);
  if (res == (void*)-1) {
    OutOfMemoryMessageAndDie(mem_type, size);
  }
  return res;
}

void *AsanMmapFixedNoReserve(uintptr_t fixed_addr, size_t size) {
  return asan_mmap((void*)fixed_addr, size,
                   PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANON | MAP_FIXED | MAP_NORESERVE,
                   0, 0);
}

void *AsanMprotect(uintptr_t fixed_addr, size_t size) {
  return asan_mmap((void*)fixed_addr, size,
                   PROT_NONE,
                   MAP_PRIVATE | MAP_ANON | MAP_FIXED | MAP_NORESERVE,
                   0, 0);
}

void AsanUnmapOrDie(void *addr, size_t size) {
  if (!addr || !size) return;
  int res = munmap(addr, size);
  if (res != 0) {
    Report("Failed to unmap\n");
    AsanDie();
  }
}

int AsanOpenReadonly(const char* filename) {
  return open(filename, O_RDONLY);
}

const char *AsanGetEnv(const char *name) {
  char ***env_ptr = _NSGetEnviron();
  CHECK(env_ptr);
  char **environ = *env_ptr;
  CHECK(environ);
  size_t name_len = internal_strlen(name);
  while (*environ != NULL) {
    size_t len = internal_strlen(*environ);
    if (len > name_len) {
      const char *p = *environ;
      if (!internal_memcmp(p, name, name_len) &&
          p[name_len] == '=') {  // Match.
        return *environ + name_len + 1;  // String starting after =.
      }
    }
    environ++;
  }
  return NULL;
}

size_t AsanRead(int fd, void *buf, size_t count) {
  return read(fd, buf, count);
}

int AsanClose(int fd) {
  return close(fd);
}

AsanProcMaps::AsanProcMaps() {
  Reset();
}

AsanProcMaps::~AsanProcMaps() {
}

// More information about Mach-O headers can be found in mach-o/loader.h
// Each Mach-O image has a header (mach_header or mach_header_64) starting with
// a magic number, and a list of linker load commands directly following the
// header.
// A load command is at least two 32-bit words: the command type and the
// command size in bytes. We're interested only in segment load commands
// (LC_SEGMENT and LC_SEGMENT_64), which tell that a part of the file is mapped
// into the task's address space.
// The |vmaddr|, |vmsize| and |fileoff| fields of segment_command or
// segment_command_64 correspond to the memory address, memory size and the
// file offset of the current memory segment.
// Because these fields are taken from the images as is, one needs to add
// _dyld_get_image_vmaddr_slide() to get the actual addresses at runtime.

void AsanProcMaps::Reset() {
  // Count down from the top.
  // TODO(glider): as per man 3 dyld, iterating over the headers with
  // _dyld_image_count is thread-unsafe. We need to register callbacks for
  // adding and removing images which will invalidate the AsanProcMaps state.
  current_image_ = _dyld_image_count();
  current_load_cmd_count_ = -1;
  current_load_cmd_addr_ = NULL;
  current_magic_ = 0;
}

// Next and NextSegmentLoad were inspired by base/sysinfo.cc in
// Google Perftools, http://code.google.com/p/google-perftools.

// NextSegmentLoad scans the current image for the next segment load command
// and returns the start and end addresses and file offset of the corresponding
// segment.
// Note that the segment addresses are not necessarily sorted.
template<uint32_t kLCSegment, typename SegmentCommand>
bool AsanProcMaps::NextSegmentLoad(
    uintptr_t *start, uintptr_t *end, uintptr_t *offset,
    char filename[], size_t filename_size) {
  const char* lc = current_load_cmd_addr_;
  current_load_cmd_addr_ += ((const load_command *)lc)->cmdsize;
  if (((const load_command *)lc)->cmd == kLCSegment) {
    const intptr_t dlloff = _dyld_get_image_vmaddr_slide(current_image_);
    const SegmentCommand* sc = (const SegmentCommand *)lc;
    if (start) *start = sc->vmaddr + dlloff;
    if (end) *end = sc->vmaddr + sc->vmsize + dlloff;
    if (offset) *offset = sc->fileoff;
    if (filename) {
      REAL(strncpy)(filename, _dyld_get_image_name(current_image_),
                    filename_size);
    }
    if (FLAG_v >= 4)
      Report("LC_SEGMENT: %p--%p %s+%p\n", *start, *end, filename, *offset);
    return true;
  }
  return false;
}

bool AsanProcMaps::Next(uintptr_t *start, uintptr_t *end,
                        uintptr_t *offset, char filename[],
                        size_t filename_size) {
  for (; current_image_ >= 0; current_image_--) {
    const mach_header* hdr = _dyld_get_image_header(current_image_);
    if (!hdr) continue;
    if (current_load_cmd_count_ < 0) {
      // Set up for this image;
      current_load_cmd_count_ = hdr->ncmds;
      current_magic_ = hdr->magic;
      switch (current_magic_) {
#ifdef MH_MAGIC_64
        case MH_MAGIC_64: {
          current_load_cmd_addr_ = (char*)hdr + sizeof(mach_header_64);
          break;
        }
#endif
        case MH_MAGIC: {
          current_load_cmd_addr_ = (char*)hdr + sizeof(mach_header);
          break;
        }
        default: {
          continue;
        }
      }
    }

    for (; current_load_cmd_count_ >= 0; current_load_cmd_count_--) {
      switch (current_magic_) {
        // current_magic_ may be only one of MH_MAGIC, MH_MAGIC_64.
#ifdef MH_MAGIC_64
        case MH_MAGIC_64: {
          if (NextSegmentLoad<LC_SEGMENT_64, struct segment_command_64>(
                  start, end, offset, filename, filename_size))
            return true;
          break;
        }
#endif
        case MH_MAGIC: {
          if (NextSegmentLoad<LC_SEGMENT, struct segment_command>(
                  start, end, offset, filename, filename_size))
            return true;
          break;
        }
      }
    }
    // If we get here, no more load_cmd's in this image talk about
    // segments.  Go on to the next image.
  }
  return false;
}

bool AsanProcMaps::GetObjectNameAndOffset(uintptr_t addr, uintptr_t *offset,
                                          char filename[],
                                          size_t filename_size) {
  return IterateForObjectNameAndOffset(addr, offset, filename, filename_size);
}

void AsanThread::SetThreadStackTopAndBottom() {
  size_t stacksize = pthread_get_stacksize_np(pthread_self());
  void *stackaddr = pthread_get_stackaddr_np(pthread_self());
  stack_top_ = (uintptr_t)stackaddr;
  stack_bottom_ = stack_top_ - stacksize;
  int local;
  CHECK(AddrIsInStack((uintptr_t)&local));
}

AsanLock::AsanLock(LinkerInitialized) {
  // We assume that OS_SPINLOCK_INIT is zero
}

void AsanLock::Lock() {
  CHECK(sizeof(OSSpinLock) <= sizeof(opaque_storage_));
  CHECK(OS_SPINLOCK_INIT == 0);
  CHECK(owner_ != (uintptr_t)pthread_self());
  OSSpinLockLock((OSSpinLock*)&opaque_storage_);
  CHECK(!owner_);
  owner_ = (uintptr_t)pthread_self();
}

void AsanLock::Unlock() {
  CHECK(owner_ == (uintptr_t)pthread_self());
  owner_ = 0;
  OSSpinLockUnlock((OSSpinLock*)&opaque_storage_);
}

void AsanStackTrace::GetStackTrace(size_t max_s, uintptr_t pc, uintptr_t bp) {
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
static void *island_allocator_pos = NULL;

#if __WORDSIZE == 32
# define kIslandEnd (0xffdf0000 - kPageSize)
# define kIslandBeg (kIslandEnd - 256 * kPageSize)
#else
# define kIslandEnd (0x7fffffdf0000 - kPageSize)
# define kIslandBeg (kIslandEnd - 256 * kPageSize)
#endif

extern "C"
mach_error_t __interception_allocate_island(void **ptr,
                                            size_t unused_size,
                                            void *unused_hint) {
  if (!island_allocator_pos) {
    island_allocator_pos =
        asan_mmap((void*)kIslandBeg, kIslandEnd - kIslandBeg,
                  PROT_READ | PROT_WRITE | PROT_EXEC,
                  MAP_PRIVATE | MAP_ANON | MAP_FIXED,
                 -1, 0);
    if (island_allocator_pos != (void*)kIslandBeg) {
      return KERN_NO_SPACE;
    }
    if (FLAG_v) {
      Report("Mapped pages %p--%p for branch islands.\n",
             kIslandBeg, kIslandEnd);
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
typedef uint64_t dispatch_time_t;
typedef void (*dispatch_function_t)(void *block);
typedef void* (*worker_t)(void *block);

// A wrapper for the ObjC blocks used to support libdispatch.
typedef struct {
  void *block;
  dispatch_function_t func;
  int parent_tid;
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
    t = AsanThread::Create(context->parent_tid, NULL, NULL, &stack);
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
  asan_ctxt->parent_tid = asanThreadRegistry().GetCurrentTidOrMinusOne();
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
  asan_ctxt->parent_tid = asanThreadRegistry().GetCurrentTidOrMinusOne();
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
  uintptr_t _cfisa;
  uint8_t _cfinfo[4];
#if __LP64__
  uint32_t _rc;
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
