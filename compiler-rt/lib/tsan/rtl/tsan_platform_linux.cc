//===-- tsan_platform_linux.cc --------------------------------------------===//
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
// Linux-specific code.
//===----------------------------------------------------------------------===//

#ifdef __linux__

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_procmaps.h"
#include "tsan_platform.h"
#include "tsan_rtl.h"
#include "tsan_flags.h"

#include <asm/prctl.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <sched.h>
#include <dlfcn.h>
#define __need_res_state
#include <resolv.h>

extern "C" int arch_prctl(int code, __sanitizer::uptr *addr);

namespace __tsan {

#ifndef TSAN_GO
ScopedInRtl::ScopedInRtl()
    : thr_(cur_thread()) {
  in_rtl_ = thr_->in_rtl;
  thr_->in_rtl++;
  errno_ = errno;
}

ScopedInRtl::~ScopedInRtl() {
  thr_->in_rtl--;
  errno = errno_;
  CHECK_EQ(in_rtl_, thr_->in_rtl);
}
#else
ScopedInRtl::ScopedInRtl() {
}

ScopedInRtl::~ScopedInRtl() {
}
#endif

uptr GetShadowMemoryConsumption() {
  return 0;
}

void FlushShadowMemory() {
  FlushUnneededShadowMemory(kLinuxShadowBeg, kLinuxShadowEnd - kLinuxShadowBeg);
}

#ifndef TSAN_GO
static void ProtectRange(uptr beg, uptr end) {
  ScopedInRtl in_rtl;
  CHECK_LE(beg, end);
  if (beg == end)
    return;
  if (beg != (uptr)Mprotect(beg, end - beg)) {
    Printf("FATAL: ThreadSanitizer can not protect [%zx,%zx]\n", beg, end);
    Printf("FATAL: Make sure you are not using unlimited stack\n");
    Die();
  }
}
#endif

#ifndef TSAN_GO
void InitializeShadowMemory() {
  uptr shadow = (uptr)MmapFixedNoReserve(kLinuxShadowBeg,
    kLinuxShadowEnd - kLinuxShadowBeg);
  if (shadow != kLinuxShadowBeg) {
    Printf("FATAL: ThreadSanitizer can not mmap the shadow memory\n");
    Printf("FATAL: Make sure to compile with -fPIE and "
               "to link with -pie (%p, %p).\n", shadow, kLinuxShadowBeg);
    Die();
  }
  const uptr kClosedLowBeg  = 0x200000;
  const uptr kClosedLowEnd  = kLinuxShadowBeg - 1;
  const uptr kClosedMidBeg = kLinuxShadowEnd + 1;
  const uptr kClosedMidEnd = min(kLinuxAppMemBeg, kTraceMemBegin);
  ProtectRange(kClosedLowBeg, kClosedLowEnd);
  ProtectRange(kClosedMidBeg, kClosedMidEnd);
  DPrintf("kClosedLow   %zx-%zx (%zuGB)\n",
      kClosedLowBeg, kClosedLowEnd, (kClosedLowEnd - kClosedLowBeg) >> 30);
  DPrintf("kLinuxShadow %zx-%zx (%zuGB)\n",
      kLinuxShadowBeg, kLinuxShadowEnd,
      (kLinuxShadowEnd - kLinuxShadowBeg) >> 30);
  DPrintf("kClosedMid   %zx-%zx (%zuGB)\n",
      kClosedMidBeg, kClosedMidEnd, (kClosedMidEnd - kClosedMidBeg) >> 30);
  DPrintf("kLinuxAppMem %zx-%zx (%zuGB)\n",
      kLinuxAppMemBeg, kLinuxAppMemEnd,
      (kLinuxAppMemEnd - kLinuxAppMemBeg) >> 30);
  DPrintf("stack        %zx\n", (uptr)&shadow);
}
#endif

static uptr g_data_start;
static uptr g_data_end;

#ifndef TSAN_GO
static void CheckPIE() {
  // Ensure that the binary is indeed compiled with -pie.
  MemoryMappingLayout proc_maps;
  uptr start, end;
  if (proc_maps.Next(&start, &end,
                     /*offset*/0, /*filename*/0, /*filename_size*/0,
                     /*protection*/0)) {
    if ((u64)start < kLinuxAppMemBeg) {
      Printf("FATAL: ThreadSanitizer can not mmap the shadow memory ("
             "something is mapped at 0x%zx < 0x%zx)\n",
             start, kLinuxAppMemBeg);
      Printf("FATAL: Make sure to compile with -fPIE"
             " and to link with -pie.\n");
      Die();
    }
  }
}

static void InitDataSeg() {
  MemoryMappingLayout proc_maps;
  uptr start, end, offset;
  char name[128];
  bool prev_is_data = false;
  while (proc_maps.Next(&start, &end, &offset, name, ARRAY_SIZE(name),
                        /*protection*/ 0)) {
    DPrintf("%p-%p %p %s\n", start, end, offset, name);
    bool is_data = offset != 0 && name[0] != 0;
    // BSS may get merged with [heap] in /proc/self/maps. This is not very
    // reliable.
    bool is_bss = offset == 0 &&
      (name[0] == 0 || internal_strcmp(name, "[heap]") == 0) && prev_is_data;
    if (g_data_start == 0 && is_data)
      g_data_start = start;
    if (is_bss)
      g_data_end = end;
    prev_is_data = is_data;
  }
  DPrintf("guessed data_start=%p data_end=%p\n",  g_data_start, g_data_end);
  CHECK_LT(g_data_start, g_data_end);
  CHECK_GE((uptr)&g_data_start, g_data_start);
  CHECK_LT((uptr)&g_data_start, g_data_end);
}

static uptr g_tls_size;

#ifdef __i386__
# define INTERNAL_FUNCTION __attribute__((regparm(3), stdcall))
#else
# define INTERNAL_FUNCTION
#endif

static int InitTlsSize() {
  typedef void (*get_tls_func)(size_t*, size_t*) INTERNAL_FUNCTION;
  get_tls_func get_tls;
  void *get_tls_static_info_ptr = dlsym(RTLD_NEXT, "_dl_get_tls_static_info");
  CHECK_EQ(sizeof(get_tls), sizeof(get_tls_static_info_ptr));
  internal_memcpy(&get_tls, &get_tls_static_info_ptr,
                  sizeof(get_tls_static_info_ptr));
  CHECK_NE(get_tls, 0);
  size_t tls_size = 0;
  size_t tls_align = 0;
  get_tls(&tls_size, &tls_align);
  return tls_size;
}
#endif  // #ifndef TSAN_GO

static rlim_t getlim(int res) {
  rlimit rlim;
  CHECK_EQ(0, getrlimit(res, &rlim));
  return rlim.rlim_cur;
}

static void setlim(int res, rlim_t lim) {
  // The following magic is to prevent clang from replacing it with memset.
  volatile rlimit rlim;
  rlim.rlim_cur = lim;
  rlim.rlim_max = lim;
  setrlimit(res, (rlimit*)&rlim);
}

const char *InitializePlatform() {
  void *p = 0;
  if (sizeof(p) == 8) {
    // Disable core dumps, dumping of 16TB usually takes a bit long.
    setlim(RLIMIT_CORE, 0);
  }

  // Go maps shadow memory lazily and works fine with limited address space.
  // Unlimited stack is not a problem as well, because the executable
  // is not compiled with -pie.
  if (kCppMode) {
    bool reexec = false;
    // TSan doesn't play well with unlimited stack size (as stack
    // overlaps with shadow memory). If we detect unlimited stack size,
    // we re-exec the program with limited stack size as a best effort.
    if (getlim(RLIMIT_STACK) == (rlim_t)-1) {
      const uptr kMaxStackSize = 32 * 1024 * 1024;
      Report("WARNING: Program is run with unlimited stack size, which "
             "wouldn't work with ThreadSanitizer.\n");
      Report("Re-execing with stack size limited to %zd bytes.\n",
             kMaxStackSize);
      SetStackSizeLimitInBytes(kMaxStackSize);
      reexec = true;
    }

    if (getlim(RLIMIT_AS) != (rlim_t)-1) {
      Report("WARNING: Program is run with limited virtual address space,"
             " which wouldn't work with ThreadSanitizer.\n");
      Report("Re-execing with unlimited virtual address space.\n");
      setlim(RLIMIT_AS, -1);
      reexec = true;
    }
    if (reexec)
      ReExec();
  }

#ifndef TSAN_GO
  CheckPIE();
  g_tls_size = (uptr)InitTlsSize();
  InitDataSeg();
#endif
  return GetEnv(kTsanOptionsEnv);
}

void FinalizePlatform() {
  fflush(0);
}

uptr GetTlsSize() {
#ifndef TSAN_GO
  return g_tls_size;
#else
  return 0;
#endif
}

void GetThreadStackAndTls(bool main, uptr *stk_addr, uptr *stk_size,
                          uptr *tls_addr, uptr *tls_size) {
#ifndef TSAN_GO
  arch_prctl(ARCH_GET_FS, tls_addr);
  *tls_addr -= g_tls_size;
  *tls_size = g_tls_size;

  uptr stack_top, stack_bottom;
  GetThreadStackTopAndBottom(main, &stack_top, &stack_bottom);
  *stk_addr = stack_bottom;
  *stk_size = stack_top - stack_bottom;

  if (!main) {
    // If stack and tls intersect, make them non-intersecting.
    if (*tls_addr > *stk_addr && *tls_addr < *stk_addr + *stk_size) {
      CHECK_GT(*tls_addr + *tls_size, *stk_addr);
      CHECK_LE(*tls_addr + *tls_size, *stk_addr + *stk_size);
      *stk_size -= *tls_size;
      *tls_addr = *stk_addr + *stk_size;
    }
  }
#else
  *stk_addr = 0;
  *stk_size = 0;
  *tls_addr = 0;
  *tls_size = 0;
#endif
}

bool IsGlobalVar(uptr addr) {
  return g_data_start && addr >= g_data_start && addr < g_data_end;
}

#ifndef TSAN_GO
int ExtractResolvFDs(void *state, int *fds, int nfd) {
  int cnt = 0;
  __res_state *statp = (__res_state*)state;
  for (int i = 0; i < MAXNS && cnt < nfd; i++) {
    if (statp->_u._ext.nsaddrs[i] && statp->_u._ext.nssocks[i] != -1)
      fds[cnt++] = statp->_u._ext.nssocks[i];
  }
  return cnt;
}
#endif


}  // namespace __tsan

#endif  // #ifdef __linux__
