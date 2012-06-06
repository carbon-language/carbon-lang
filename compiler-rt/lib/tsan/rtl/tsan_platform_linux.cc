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

#include "sanitizer_common/sanitizer_libc.h"
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

static uptr g_tls_size;

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

uptr GetShadowMemoryConsumption() {
  return 0;
}

void FlushShadowMemory() {
  madvise((void*)kLinuxShadowBeg,
          kLinuxShadowEnd - kLinuxShadowBeg,
          MADV_DONTNEED);
}

void internal_yield() {
  ScopedInRtl in_rtl;
  syscall(__NR_sched_yield);
}

void internal_sleep_ms(u32 ms) {
  usleep(ms * 1000);
}

uptr internal_filesize(fd_t fd) {
  struct stat st = {};
  if (syscall(__NR_fstat, fd, &st))
    return -1;
  return (uptr)st.st_size;
}

int internal_dup2(int oldfd, int newfd) {
  ScopedInRtl in_rtl;
  return syscall(__NR_dup2, oldfd, newfd);
}

const char *internal_getpwd() {
  return getenv("PWD");
}

static void ProtectRange(uptr beg, uptr end) {
  ScopedInRtl in_rtl;
  CHECK_LE(beg, end);
  if (beg == end)
    return;
  if (beg != (uptr)internal_mmap((void*)(beg), end - beg,
      PROT_NONE,
      MAP_PRIVATE | MAP_ANON | MAP_FIXED | MAP_NORESERVE,
      -1, 0)) {
    Printf("FATAL: ThreadSanitizer can not protect [%lx,%lx]\n", beg, end);
    Printf("FATAL: Make sure you are not using unlimited stack\n");
    Die();
  }
}

void InitializeShadowMemory() {
  const uptr kClosedLowBeg  = 0x200000;
  const uptr kClosedLowEnd  = kLinuxShadowBeg - 1;
  const uptr kClosedMidBeg = kLinuxShadowEnd + 1;
  const uptr kClosedMidEnd = kLinuxAppMemBeg - 1;
  uptr shadow = (uptr)internal_mmap((void*)kLinuxShadowBeg,
      kLinuxShadowEnd - kLinuxShadowBeg,
      PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANON | MAP_FIXED | MAP_NORESERVE,
      -1, 0);
  if (shadow != kLinuxShadowBeg) {
    Printf("FATAL: ThreadSanitizer can not mmap the shadow memory\n");
    Printf("FATAL: Make sure to compile with -fPIE and to link with -pie.\n");
    Die();
  }
  ProtectRange(kClosedLowBeg, kClosedLowEnd);
  ProtectRange(kClosedMidBeg, kClosedMidEnd);
  DPrintf("kClosedLow   %lx-%lx (%luGB)\n",
      kClosedLowBeg, kClosedLowEnd, (kClosedLowEnd - kClosedLowBeg) >> 30);
  DPrintf("kLinuxShadow %lx-%lx (%luGB)\n",
      kLinuxShadowBeg, kLinuxShadowEnd,
      (kLinuxShadowEnd - kLinuxShadowBeg) >> 30);
  DPrintf("kClosedMid   %lx-%lx (%luGB)\n",
      kClosedMidBeg, kClosedMidEnd, (kClosedMidEnd - kClosedMidBeg) >> 30);
  DPrintf("kLinuxAppMem %lx-%lx (%luGB)\n",
      kLinuxAppMemBeg, kLinuxAppMemEnd,
      (kLinuxAppMemEnd - kLinuxAppMemBeg) >> 30);
  DPrintf("stack        %lx\n", (uptr)&shadow);
}

static void CheckPIE() {
  // Ensure that the binary is indeed compiled with -pie.
  fd_t fmaps = internal_open("/proc/self/maps", false);
  if (fmaps == kInvalidFd)
    return;
  char buf[20];
  if (internal_read(fmaps, buf, sizeof(buf)) == sizeof(buf)) {
    buf[sizeof(buf) - 1] = 0;
    u64 addr = strtoll(buf, 0, 16);
    if ((u64)addr < kLinuxAppMemBeg) {
      Printf("FATAL: ThreadSanitizer can not mmap the shadow memory ("
             "something is mapped at 0x%llx < 0x%lx)\n",
             addr, kLinuxAppMemBeg);
      Printf("FATAL: Make sure to compile with -fPIE"
             " and to link with -pie.\n");
      Die();
    }
  }
  internal_close(fmaps);
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

  CheckPIE();
  g_tls_size = (uptr)InitTlsSize();
  return getenv("TSAN_OPTIONS");
}

void FinalizePlatform() {
  fflush(0);
}

uptr GetTlsSize() {
  return g_tls_size;
}

void GetThreadStackAndTls(bool main, uptr *stk_addr, uptr *stk_size,
                          uptr *tls_addr, uptr *tls_size) {
  arch_prctl(ARCH_GET_FS, tls_addr);
  *tls_addr -= g_tls_size;
  *tls_size = g_tls_size;

  if (main) {
    uptr kBufSize = 1 << 26;
    char *buf = (char*)internal_mmap(0, kBufSize, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANON, -1, 0);
    fd_t maps = internal_open("/proc/self/maps", false);
    if (maps == kInvalidFd) {
      Printf("Failed to open /proc/self/maps\n");
      Die();
    }
    char *end = buf;
    while (end + kPageSize < buf + kBufSize) {
      uptr read = internal_read(maps, end, kPageSize);
      if ((int)read <= 0)
        break;
      end += read;
    }
    end[0] = 0;
    end = (char*)internal_strstr(buf, "[stack]");
    if (end == 0) {
      Printf("Can't find [stack] in /proc/self/maps\n");
      Die();
    }
    end[0] = 0;
    char *pos = (char*)internal_strrchr(buf, '\n');
    if (pos == 0) {
      Printf("Can't find [stack] in /proc/self/maps\n");
      Die();
    }
    pos = (char*)internal_strchr(pos, '-');
    if (pos == 0) {
      Printf("Can't find [stack] in /proc/self/maps\n");
      Die();
    }
    uptr stack = 0;
    for (; pos++;) {
      uptr num = 0;
      if (pos[0] >= '0' && pos[0] <= '9')
        num = pos[0] - '0';
      else if (pos[0] >= 'a' && pos[0] <= 'f')
        num = pos[0] - 'a' + 10;
      else
        break;
      stack = stack * 16 + num;
    }
    internal_close(maps);
    internal_munmap(buf, kBufSize);

    struct rlimit rl;
    CHECK_EQ(getrlimit(RLIMIT_STACK, &rl), 0);
    *stk_addr = stack - rl.rlim_cur;
    *stk_size = rl.rlim_cur;
  } else {
    *stk_addr = 0;
    *stk_size = 0;
    pthread_attr_t attr;
    if (pthread_getattr_np(pthread_self(), &attr) == 0) {
      pthread_attr_getstack(&attr, (void**)stk_addr, (size_t*)stk_size);
      pthread_attr_destroy(&attr);
    }

    // If stack and tls intersect, make them non-intersecting.
    if (*tls_addr > *stk_addr && *tls_addr < *stk_addr + *stk_size) {
      CHECK_GT(*tls_addr + *tls_size, *stk_addr);
      CHECK_LE(*tls_addr + *tls_size, *stk_addr + *stk_size);
      *stk_size -= *tls_size;
      *tls_addr = *stk_addr + *stk_size;
    }
  }
}

int GetPid() {
  return getpid();
}

}  // namespace __tsan
