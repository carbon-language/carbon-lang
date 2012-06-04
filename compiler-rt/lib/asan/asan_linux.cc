//===-- asan_linux.cc -----------------------------------------------------===//
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
// Linux-specific details.
//===----------------------------------------------------------------------===//
#ifdef __linux__

#include "asan_interceptors.h"
#include "asan_internal.h"
#include "asan_lock.h"
#include "asan_procmaps.h"
#include "asan_thread.h"
#include "sanitizer_common/sanitizer_libc.h"

#include <sys/time.h>
#include <sys/resource.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <unwind.h>

#ifndef ANDROID
// FIXME: where to get ucontext on Android?
#include <sys/ucontext.h>
#endif

using namespace __sanitizer;  // NOLINT

extern "C" void* _DYNAMIC;

namespace __asan {

const uptr kMaxThreadStackSize = 256 * (1 << 20);  // 256M

void *AsanDoesNotSupportStaticLinkage() {
  // This will fail to link with -static.
  return &_DYNAMIC;  // defined in link.h
}

void GetPcSpBp(void *context, uptr *pc, uptr *sp, uptr *bp) {
#ifdef ANDROID
  *pc = *sp = *bp = 0;
#elif defined(__arm__)
  ucontext_t *ucontext = (ucontext_t*)context;
  *pc = ucontext->uc_mcontext.arm_pc;
  *bp = ucontext->uc_mcontext.arm_fp;
  *sp = ucontext->uc_mcontext.arm_sp;
# elif defined(__x86_64__)
  ucontext_t *ucontext = (ucontext_t*)context;
  *pc = ucontext->uc_mcontext.gregs[REG_RIP];
  *bp = ucontext->uc_mcontext.gregs[REG_RBP];
  *sp = ucontext->uc_mcontext.gregs[REG_RSP];
# elif defined(__i386__)
  ucontext_t *ucontext = (ucontext_t*)context;
  *pc = ucontext->uc_mcontext.gregs[REG_EIP];
  *bp = ucontext->uc_mcontext.gregs[REG_EBP];
  *sp = ucontext->uc_mcontext.gregs[REG_ESP];
#else
# error "Unsupported arch"
#endif
}

bool AsanInterceptsSignal(int signum) {
  return signum == SIGSEGV && FLAG_handle_segv;
}

void *AsanMmapSomewhereOrDie(uptr size, const char *mem_type) {
  size = RoundUpTo(size, kPageSize);
  void *res = internal_mmap(0, size,
                            PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANON, -1, 0);
  if (res == (void*)-1) {
    OutOfMemoryMessageAndDie(mem_type, size);
  }
  return res;
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

void AsanUnmapOrDie(void *addr, uptr size) {
  if (!addr || !size) return;
  int res = syscall(__NR_munmap, addr, size);
  if (res != 0) {
    Report("Failed to unmap\n");
    AsanDie();
  }
}

uptr AsanWrite(int fd, const void *buf, uptr count) {
  return (uptr)syscall(__NR_write, fd, buf, count);
}

int AsanOpenReadonly(const char* filename) {
  return syscall(__NR_open, filename, O_RDONLY);
}

// Like getenv, but reads env directly from /proc and does not use libc.
// This function should be called first inside __asan_init.
const char* AsanGetEnv(const char* name) {
  static char *environ;
  static uptr len;
  static bool inited;
  if (!inited) {
    inited = true;
    uptr environ_size;
    len = ReadFileToBuffer("/proc/self/environ",
                           &environ, &environ_size, 1 << 26);
  }
  if (!environ || len == 0) return 0;
  uptr namelen = internal_strlen(name);
  const char *p = environ;
  while (*p != '\0') {  // will happen at the \0\0 that terminates the buffer
    // proc file has the format NAME=value\0NAME=value\0NAME=value\0...
    const char* endp =
        (char*)internal_memchr(p, '\0', len - (p - environ));
    if (endp == 0)  // this entry isn't NUL terminated
      return 0;
    else if (!internal_memcmp(p, name, namelen) && p[namelen] == '=')  // Match.
      return p + namelen + 1;  // point after =
    p = endp + 1;
  }
  return 0;  // Not found.
}

uptr AsanRead(int fd, void *buf, uptr count) {
  return (uptr)syscall(__NR_read, fd, buf, count);
}

int AsanClose(int fd) {
  return syscall(__NR_close, fd);
}

AsanProcMaps::AsanProcMaps() {
  proc_self_maps_buff_len_ =
      ReadFileToBuffer("/proc/self/maps", &proc_self_maps_buff_,
                       &proc_self_maps_buff_mmaped_size_, 1 << 26);
  CHECK(proc_self_maps_buff_len_ > 0);
  // AsanWrite(2, proc_self_maps_buff_, proc_self_maps_buff_len_);
  Reset();
}

AsanProcMaps::~AsanProcMaps() {
  AsanUnmapOrDie(proc_self_maps_buff_, proc_self_maps_buff_mmaped_size_);
}

void AsanProcMaps::Reset() {
  current_ = proc_self_maps_buff_;
}

bool AsanProcMaps::Next(uptr *start, uptr *end,
                        uptr *offset, char filename[],
                        uptr filename_size) {
  char *last = proc_self_maps_buff_ + proc_self_maps_buff_len_;
  if (current_ >= last) return false;
  int consumed = 0;
  char flags[10];
  int major, minor;
  uptr inode;
  uptr dummy;
  if (!start) start = &dummy;
  if (!end) end = &dummy;
  if (!offset) offset = &dummy;
  char *next_line = (char*)internal_memchr(current_, '\n', last - current_);
  if (next_line == 0)
    next_line = last;
  if (SScanf(current_,
             "%lx-%lx %4s %lx %x:%x %ld %n",
             start, end, flags, offset, &major, &minor,
             &inode, &consumed) != 7)
    return false;
  current_ += consumed;
  // Skip spaces.
  while (current_ < next_line && *current_ == ' ')
    current_++;
  // Fill in the filename.
  uptr i = 0;
  while (current_ < next_line) {
    if (filename && i < filename_size - 1)
      filename[i++] = *current_;
    current_++;
  }
  if (filename && i < filename_size)
    filename[i] = 0;
  current_ = next_line + 1;
  return true;
}

// Gets the object name and the offset by walking AsanProcMaps.
bool AsanProcMaps::GetObjectNameAndOffset(uptr addr, uptr *offset,
                                          char filename[],
                                          uptr filename_size) {
  return IterateForObjectNameAndOffset(addr, offset, filename, filename_size);
}

void AsanThread::SetThreadStackTopAndBottom() {
  if (tid() == 0) {
    // This is the main thread. Libpthread may not be initialized yet.
    struct rlimit rl;
    CHECK(getrlimit(RLIMIT_STACK, &rl) == 0);

    // Find the mapping that contains a stack variable.
    AsanProcMaps proc_maps;
    uptr start, end, offset;
    uptr prev_end = 0;
    while (proc_maps.Next(&start, &end, &offset, 0, 0)) {
      if ((uptr)&rl < end)
        break;
      prev_end = end;
    }
    CHECK((uptr)&rl >= start && (uptr)&rl < end);

    // Get stacksize from rlimit, but clip it so that it does not overlap
    // with other mappings.
    uptr stacksize = rl.rlim_cur;
    if (stacksize > end - prev_end)
      stacksize = end - prev_end;
    // When running with unlimited stack size, we still want to set some limit.
    // The unlimited stack size is caused by 'ulimit -s unlimited'.
    // Also, for some reason, GNU make spawns subprocesses with unlimited stack.
    if (stacksize > kMaxThreadStackSize)
      stacksize = kMaxThreadStackSize;
    stack_top_ = end;
    stack_bottom_ = end - stacksize;
    CHECK(AddrIsInStack((uptr)&rl));
    return;
  }
  pthread_attr_t attr;
  CHECK(pthread_getattr_np(pthread_self(), &attr) == 0);
  uptr stacksize = 0;
  void *stackaddr = 0;
  pthread_attr_getstack(&attr, &stackaddr, (size_t*)&stacksize);
  pthread_attr_destroy(&attr);

  stack_top_ = (uptr)stackaddr + stacksize;
  stack_bottom_ = (uptr)stackaddr;
  CHECK(stacksize < kMaxThreadStackSize);  // Sanity check.
  CHECK(AddrIsInStack((uptr)&attr));
}

AsanLock::AsanLock(LinkerInitialized) {
  // We assume that pthread_mutex_t initialized to all zeroes is a valid
  // unlocked mutex. We can not use PTHREAD_MUTEX_INITIALIZER as it triggers
  // a gcc warning:
  // extended initializer lists only available with -std=c++0x or -std=gnu++0x
}

void AsanLock::Lock() {
  CHECK(sizeof(pthread_mutex_t) <= sizeof(opaque_storage_));
  pthread_mutex_lock((pthread_mutex_t*)&opaque_storage_);
  CHECK(!owner_);
  owner_ = (uptr)pthread_self();
}

void AsanLock::Unlock() {
  CHECK(owner_ == (uptr)pthread_self());
  owner_ = 0;
  pthread_mutex_unlock((pthread_mutex_t*)&opaque_storage_);
}

#ifdef __arm__
#define UNWIND_STOP _URC_END_OF_STACK
#define UNWIND_CONTINUE _URC_NO_REASON
#else
#define UNWIND_STOP _URC_NORMAL_STOP
#define UNWIND_CONTINUE _URC_NO_REASON
#endif

uptr Unwind_GetIP(struct _Unwind_Context *ctx) {
#ifdef __arm__
  uptr val;
  _Unwind_VRS_Result res = _Unwind_VRS_Get(ctx, _UVRSC_CORE,
      15 /* r15 = PC */, _UVRSD_UINT32, &val);
  CHECK(res == _UVRSR_OK && "_Unwind_VRS_Get failed");
  // Clear the Thumb bit.
  return val & ~(uptr)1;
#else
  return _Unwind_GetIP(ctx);
#endif
}

_Unwind_Reason_Code Unwind_Trace(struct _Unwind_Context *ctx,
    void *param) {
  AsanStackTrace *b = (AsanStackTrace*)param;
  CHECK(b->size < b->max_size);
  uptr pc = Unwind_GetIP(ctx);
  b->trace[b->size++] = pc;
  if (b->size == b->max_size) return UNWIND_STOP;
  return UNWIND_CONTINUE;
}

void AsanStackTrace::GetStackTrace(uptr max_s, uptr pc, uptr bp) {
  size = 0;
  trace[0] = pc;
  if ((max_s) > 1) {
    max_size = max_s;
#ifdef __arm__
    _Unwind_Backtrace(Unwind_Trace, this);
#else
     FastUnwindStack(pc, bp);
#endif
  }
}

}  // namespace __asan

#endif  // __linux__
