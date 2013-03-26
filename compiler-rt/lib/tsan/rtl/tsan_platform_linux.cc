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


#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_LINUX

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
#include <malloc.h>

extern "C" int arch_prctl(int code, __sanitizer::uptr *addr);
extern "C" struct mallinfo __libc_mallinfo();

namespace __tsan {

const uptr kPageSize = 4096;

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

static bool ishex(char c) {
  return (c >= '0' && c <= '9')
      || (c >= 'a' && c <= 'f');
}

static uptr readhex(const char *p) {
  uptr v = 0;
  for (; ishex(p[0]); p++) {
    if (p[0] >= '0' && p[0] <= '9')
      v = v * 16 + p[0] - '0';
    else
      v = v * 16 + p[0] - 'a' + 10;
  }
  return v;
}

static uptr readdec(const char *p) {
  uptr v = 0;
  for (; p[0] >= '0' && p[0] <= '9' ; p++)
    v = v * 10 + p[0] - '0';
  return v;
}

void WriteMemoryProfile(char *buf, uptr buf_size) {
  char *smaps = 0;
  uptr smaps_cap = 0;
  uptr smaps_len = ReadFileToBuffer("/proc/self/smaps",
      &smaps, &smaps_cap, 64<<20);
  uptr mem[6] = {};
  uptr total = 0;
  uptr start = 0;
  bool file = false;
  const char *pos = smaps;
  while (pos < smaps + smaps_len) {
    if (ishex(pos[0])) {
      start = readhex(pos);
      for (; *pos != '/' && *pos > '\n'; pos++) {}
      file = *pos == '/';
    } else if (internal_strncmp(pos, "Rss:", 4) == 0) {
      for (; *pos < '0' || *pos > '9'; pos++) {}
      uptr rss = readdec(pos) * 1024;
      total += rss;
      start >>= 40;
      if (start < 0x10)  // shadow
        mem[0] += rss;
      else if (start >= 0x20 && start < 0x30)  // compat modules
        mem[file ? 1 : 2] += rss;
      else if (start >= 0x7e)  // modules
        mem[file ? 1 : 2] += rss;
      else if (start >= 0x60 && start < 0x62)  // traces
        mem[3] += rss;
      else if (start >= 0x7d && start < 0x7e)  // heap
        mem[4] += rss;
      else  // other
        mem[5] += rss;
    }
    while (*pos++ != '\n') {}
  }
  UnmapOrDie(smaps, smaps_cap);
  char *buf_pos = buf;
  char *buf_end = buf + buf_size;
  buf_pos += internal_snprintf(buf_pos, buf_end - buf_pos,
      "RSS %zd MB: shadow:%zd file:%zd mmap:%zd trace:%zd heap:%zd other:%zd\n",
      total >> 20, mem[0] >> 20, mem[1] >> 20, mem[2] >> 20,
      mem[3] >> 20, mem[4] >> 20, mem[5] >> 20);
  struct mallinfo mi = __libc_mallinfo();
  buf_pos += internal_snprintf(buf_pos, buf_end - buf_pos,
      "mallinfo: arena=%d mmap=%d fordblks=%d keepcost=%d\n",
      mi.arena >> 20, mi.hblkhd >> 20, mi.fordblks >> 20, mi.keepcost >> 20);
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
// Mark shadow for .rodata sections with the special kShadowRodata marker.
// Accesses to .rodata can't race, so this saves time, memory and trace space.
static void MapRodata() {
  // First create temp file.
  const char *tmpdir = GetEnv("TMPDIR");
  if (tmpdir == 0)
    tmpdir = GetEnv("TEST_TMPDIR");
#ifdef P_tmpdir
  if (tmpdir == 0)
    tmpdir = P_tmpdir;
#endif
  if (tmpdir == 0)
    return;
  char filename[256];
  internal_snprintf(filename, sizeof(filename), "%s/tsan.rodata.%u",
                    tmpdir, GetPid());
  fd_t fd = internal_open(filename, O_RDWR | O_CREAT | O_EXCL, 0600);
  if (fd == kInvalidFd)
    return;
  // Fill the file with kShadowRodata.
  const uptr kMarkerSize = 512 * 1024 / sizeof(u64);
  InternalScopedBuffer<u64> marker(kMarkerSize);
  for (u64 *p = marker.data(); p < marker.data() + kMarkerSize; p++)
    *p = kShadowRodata;
  internal_write(fd, marker.data(), marker.size());
  // Map the file into memory.
  void *page = internal_mmap(0, kPageSize, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, fd, 0);
  if (page == MAP_FAILED) {
    internal_close(fd);
    internal_unlink(filename);
    return;
  }
  // Map the file into shadow of .rodata sections.
  MemoryMappingLayout proc_maps(/*cache_enabled*/true);
  uptr start, end, offset, prot;
  char name[128];
  while (proc_maps.Next(&start, &end, &offset, name, ARRAY_SIZE(name), &prot)) {
    if (name[0] != 0 && name[0] != '['
        && (prot & MemoryMappingLayout::kProtectionRead)
        && (prot & MemoryMappingLayout::kProtectionExecute)
        && !(prot & MemoryMappingLayout::kProtectionWrite)
        && IsAppMem(start)) {
      // Assume it's .rodata
      char *shadow_start = (char*)MemToShadow(start);
      char *shadow_end = (char*)MemToShadow(end);
      for (char *p = shadow_start; p < shadow_end; p += marker.size()) {
        internal_mmap(p, Min<uptr>(marker.size(), shadow_end - p),
                      PROT_READ, MAP_PRIVATE | MAP_FIXED, fd, 0);
      }
    }
  }
  internal_close(fd);
  internal_unlink(filename);
}

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

  MapRodata();
}
#endif

static uptr g_data_start;
static uptr g_data_end;

#ifndef TSAN_GO
static void CheckPIE() {
  // Ensure that the binary is indeed compiled with -pie.
  MemoryMappingLayout proc_maps(true);
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
  MemoryMappingLayout proc_maps(true);
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
  InitTlsSize();
  InitDataSeg();
#endif
  return GetEnv(kTsanOptionsEnv);
}

void GetThreadStackAndTls(bool main, uptr *stk_addr, uptr *stk_size,
                          uptr *tls_addr, uptr *tls_size) {
#ifndef TSAN_GO
  arch_prctl(ARCH_GET_FS, tls_addr);
  *tls_size = GetTlsSize();
  *tls_addr -= *tls_size;

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

#endif  // SANITIZER_LINUX
