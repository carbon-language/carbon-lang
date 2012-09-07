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

extern "C" int arch_prctl(int code, __sanitizer::uptr *addr);

namespace __sanitizer {

void Die() {
  _exit(1);
}

}  // namespace __sanitizer

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
  madvise((void*)kLinuxShadowBeg,
          kLinuxShadowEnd - kLinuxShadowBeg,
          MADV_DONTNEED);
}

#ifndef TSAN_GO
static void ProtectRange(uptr beg, uptr end) {
  ScopedInRtl in_rtl;
  CHECK_LE(beg, end);
  if (beg == end)
    return;
  if (beg != (uptr)Mprotect(beg, end - beg)) {
    TsanPrintf("FATAL: ThreadSanitizer can not protect [%zx,%zx]\n", beg, end);
    TsanPrintf("FATAL: Make sure you are not using unlimited stack\n");
    Die();
  }
}
#endif

void InitializeShadowMemory() {
  uptr shadow = (uptr)MmapFixedNoReserve(kLinuxShadowBeg,
    kLinuxShadowEnd - kLinuxShadowBeg);
  if (shadow != kLinuxShadowBeg) {
    TsanPrintf("FATAL: ThreadSanitizer can not mmap the shadow memory\n");
    TsanPrintf("FATAL: Make sure to compile with -fPIE and "
               "to link with -pie (%p, %p).\n", shadow, kLinuxShadowBeg);
    Die();
  }
#ifndef TSAN_GO
  const uptr kClosedLowBeg  = 0x200000;
  const uptr kClosedLowEnd  = kLinuxShadowBeg - 1;
  const uptr kClosedMidBeg = kLinuxShadowEnd + 1;
  const uptr kClosedMidEnd = kLinuxAppMemBeg - 1;
  ProtectRange(kClosedLowBeg, kClosedLowEnd);
  ProtectRange(kClosedMidBeg, kClosedMidEnd);
#endif
#ifndef TSAN_GO
  DPrintf("kClosedLow   %zx-%zx (%zuGB)\n",
      kClosedLowBeg, kClosedLowEnd, (kClosedLowEnd - kClosedLowBeg) >> 30);
#endif
  DPrintf("kLinuxShadow %zx-%zx (%zuGB)\n",
      kLinuxShadowBeg, kLinuxShadowEnd,
      (kLinuxShadowEnd - kLinuxShadowBeg) >> 30);
#ifndef TSAN_GO
  DPrintf("kClosedMid   %zx-%zx (%zuGB)\n",
      kClosedMidBeg, kClosedMidEnd, (kClosedMidEnd - kClosedMidBeg) >> 30);
#endif
  DPrintf("kLinuxAppMem %zx-%zx (%zuGB)\n",
      kLinuxAppMemBeg, kLinuxAppMemEnd,
      (kLinuxAppMemEnd - kLinuxAppMemBeg) >> 30);
  DPrintf("stack        %zx\n", (uptr)&shadow);
}

static uptr g_tls_size;
static uptr g_data_start;
static uptr g_data_end;

#ifndef TSAN_GO
static void CheckPIE() {
  // Ensure that the binary is indeed compiled with -pie.
  MemoryMappingLayout proc_maps;
  uptr start, end;
  if (proc_maps.Next(&start, &end,
                     /*offset*/0, /*filename*/0, /*filename_size*/0)) {
    if ((u64)start < kLinuxAppMemBeg) {
      TsanPrintf("FATAL: ThreadSanitizer can not mmap the shadow memory ("
             "something is mapped at 0x%zx < 0x%zx)\n",
             start, kLinuxAppMemBeg);
      TsanPrintf("FATAL: Make sure to compile with -fPIE"
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
  while (proc_maps.Next(&start, &end, &offset, name, ARRAY_SIZE(name))) {
    DPrintf("%p-%p %p %s\n", start, end, offset, name);
    bool is_data = offset != 0 && name[0] != 0;
    bool is_bss = offset == 0 && name[0] == 0 && prev_is_data;
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

#ifdef __i386__
# define INTERNAL_FUNCTION __attribute__((regparm(3), stdcall))
#else
# define INTERNAL_FUNCTION
#endif
extern "C" void _dl_get_tls_static_info(size_t*, size_t*)
    __attribute__((weak)) INTERNAL_FUNCTION;

static int InitTlsSize() {
  typedef void (*get_tls_func)(size_t*, size_t*) INTERNAL_FUNCTION;
  get_tls_func get_tls = &_dl_get_tls_static_info;
  if (get_tls == 0)
    get_tls = (get_tls_func)dlsym(RTLD_NEXT, "_dl_get_tls_static_info");
  CHECK_NE(get_tls, 0);
  size_t tls_size = 0;
  size_t tls_align = 0;
  get_tls(&tls_size, &tls_align);
  return tls_size;
}
#endif  // #ifndef TSAN_GO

const char *InitializePlatform() {
  void *p = 0;
  if (sizeof(p) == 8) {
    // Disable core dumps, dumping of 16TB usually takes a bit long.
    // The following magic is to prevent clang from replacing it with memset.
    volatile rlimit lim;
    lim.rlim_cur = 0;
    lim.rlim_max = 0;
    setrlimit(RLIMIT_CORE, (rlimit*)&lim);
  }

#ifndef TSAN_GO
  CheckPIE();
  g_tls_size = (uptr)InitTlsSize();
  InitDataSeg();
#endif
  return getenv("TSAN_OPTIONS");
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

}  // namespace __tsan

#endif  // #ifdef __linux__
