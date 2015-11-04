//===-- tsan_platform.h -----------------------------------------*- C++ -*-===//
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
// Platform-specific code.
//===----------------------------------------------------------------------===//

#ifndef TSAN_PLATFORM_H
#define TSAN_PLATFORM_H

#if !defined(__LP64__) && !defined(_WIN64)
# error "Only 64-bit is supported"
#endif

#include "tsan_defs.h"
#include "tsan_trace.h"

namespace __tsan {

#if !defined(SANITIZER_GO)

#if defined(__x86_64__)
/*
C/C++ on linux/x86_64 and freebsd/x86_64
0000 0000 1000 - 0100 0000 0000: main binary and/or MAP_32BIT mappings
0100 0000 0000 - 0200 0000 0000: -
0200 0000 0000 - 1000 0000 0000: shadow
1000 0000 0000 - 3000 0000 0000: -
3000 0000 0000 - 4000 0000 0000: metainfo (memory blocks and sync objects)
4000 0000 0000 - 6000 0000 0000: -
6000 0000 0000 - 6200 0000 0000: traces
6200 0000 0000 - 7d00 0000 0000: -
7d00 0000 0000 - 7e00 0000 0000: heap
7e00 0000 0000 - 7e80 0000 0000: -
7e80 0000 0000 - 8000 0000 0000: modules and main thread stack
*/
const uptr kMetaShadowBeg = 0x300000000000ull;
const uptr kMetaShadowEnd = 0x400000000000ull;
const uptr kTraceMemBeg   = 0x600000000000ull;
const uptr kTraceMemEnd   = 0x620000000000ull;
const uptr kShadowBeg     = 0x020000000000ull;
const uptr kShadowEnd     = 0x100000000000ull;
const uptr kHeapMemBeg    = 0x7d0000000000ull;
const uptr kHeapMemEnd    = 0x7e0000000000ull;
const uptr kLoAppMemBeg   = 0x000000001000ull;
const uptr kLoAppMemEnd   = 0x010000000000ull;
const uptr kHiAppMemBeg   = 0x7e8000000000ull;
const uptr kHiAppMemEnd   = 0x800000000000ull;
const uptr kAppMemMsk     = 0x7c0000000000ull;
const uptr kAppMemXor     = 0x020000000000ull;
const uptr kVdsoBeg       = 0xf000000000000000ull;
#elif defined(__mips64)
/*
C/C++ on linux/mips64
0100 0000 00 - 0200 0000 00: main binary
0200 0000 00 - 1400 0000 00: -
1400 0000 00 - 2400 0000 00: shadow
2400 0000 00 - 3000 0000 00: -
3000 0000 00 - 4000 0000 00: metainfo (memory blocks and sync objects)
4000 0000 00 - 6000 0000 00: -
6000 0000 00 - 6200 0000 00: traces
6200 0000 00 - fe00 0000 00: -
fe00 0000 00 - ff00 0000 00: heap
ff00 0000 00 - ff80 0000 00: -
ff80 0000 00 - ffff ffff ff: modules and main thread stack
*/
const uptr kMetaShadowBeg = 0x3000000000ull;
const uptr kMetaShadowEnd = 0x4000000000ull;
const uptr kTraceMemBeg   = 0x6000000000ull;
const uptr kTraceMemEnd   = 0x6200000000ull;
const uptr kShadowBeg     = 0x1400000000ull;
const uptr kShadowEnd     = 0x2400000000ull;
const uptr kHeapMemBeg    = 0xfe00000000ull;
const uptr kHeapMemEnd    = 0xff00000000ull;
const uptr kLoAppMemBeg   = 0x0100000000ull;
const uptr kLoAppMemEnd   = 0x0200000000ull;
const uptr kHiAppMemBeg   = 0xff80000000ull;
const uptr kHiAppMemEnd   = 0xffffffffffull;
const uptr kAppMemMsk     = 0xfc00000000ull;
const uptr kAppMemXor     = 0x0400000000ull;
const uptr kVdsoBeg       = 0xfffff00000ull;
#elif defined(__aarch64__)
# if SANITIZER_AARCH64_VMA == 39
/*
C/C++ on linux/aarch64 (39-bit VMA)
0000 4000 00 - 0200 0000 00: main binary
2000 0000 00 - 4000 0000 00: shadow memory
4000 0000 00 - 5000 0000 00: metainfo
5000 0000 00 - 6000 0000 00: -
6000 0000 00 - 6200 0000 00: traces
6200 0000 00 - 7d00 0000 00: -
7d00 0000 00 - 7e00 0000 00: heap
7e00 0000 00 - 7fff ffff ff: modules and main thread stack
*/
const uptr kLoAppMemBeg   = 0x0000400000ull;
const uptr kLoAppMemEnd   = 0x0200000000ull;
const uptr kShadowBeg     = 0x2000000000ull;
const uptr kShadowEnd     = 0x4000000000ull;
const uptr kMetaShadowBeg = 0x4000000000ull;
const uptr kMetaShadowEnd = 0x5000000000ull;
const uptr kTraceMemBeg   = 0x6000000000ull;
const uptr kTraceMemEnd   = 0x6200000000ull;
const uptr kHeapMemBeg    = 0x7d00000000ull;
const uptr kHeapMemEnd    = 0x7e00000000ull;
const uptr kHiAppMemBeg   = 0x7e00000000ull;
const uptr kHiAppMemEnd   = 0x7fffffffffull;
const uptr kAppMemMsk     = 0x7800000000ull;
const uptr kAppMemXor     = 0x0800000000ull;
const uptr kVdsoBeg       = 0x7f00000000ull;
# elif SANITIZER_AARCH64_VMA == 42
/*
C/C++ on linux/aarch64 (42-bit VMA)
00000 4000 00 - 01000 0000 00: main binary
01000 0000 00 - 10000 0000 00: -
10000 0000 00 - 20000 0000 00: shadow memory
20000 0000 00 - 26000 0000 00: -
26000 0000 00 - 28000 0000 00: metainfo
28000 0000 00 - 36200 0000 00: -
36200 0000 00 - 36240 0000 00: traces
36240 0000 00 - 3e000 0000 00: -
3e000 0000 00 - 3f000 0000 00: heap
3c000 0000 00 - 3ff00 0000 00: -
3ff00 0000 00 - 3ffff f000 00: modules and main thread stack
*/
const uptr kLoAppMemBeg   = 0x00000400000ull;
const uptr kLoAppMemEnd   = 0x01000000000ull;
const uptr kShadowBeg     = 0x10000000000ull;
const uptr kShadowEnd     = 0x20000000000ull;
const uptr kMetaShadowBeg = 0x26000000000ull;
const uptr kMetaShadowEnd = 0x28000000000ull;
const uptr kTraceMemBeg   = 0x36200000000ull;
const uptr kTraceMemEnd   = 0x36400000000ull;
const uptr kHeapMemBeg    = 0x3e000000000ull;
const uptr kHeapMemEnd    = 0x3f000000000ull;
const uptr kHiAppMemBeg   = 0x3ff00000000ull;
const uptr kHiAppMemEnd   = 0x3fffff00000ull;
const uptr kAppMemMsk     = 0x3c000000000ull;
const uptr kAppMemXor     = 0x04000000000ull;
const uptr kVdsoBeg       = 0x37f00000000ull;
# endif
#endif

ALWAYS_INLINE
bool IsAppMem(uptr mem) {
  return (mem >= kHeapMemBeg && mem < kHeapMemEnd) ||
         (mem >= kLoAppMemBeg && mem < kLoAppMemEnd) ||
         (mem >= kHiAppMemBeg && mem < kHiAppMemEnd);
}

ALWAYS_INLINE
bool IsShadowMem(uptr mem) {
  return mem >= kShadowBeg && mem <= kShadowEnd;
}

ALWAYS_INLINE
bool IsMetaMem(uptr mem) {
  return mem >= kMetaShadowBeg && mem <= kMetaShadowEnd;
}

ALWAYS_INLINE
uptr MemToShadow(uptr x) {
  DCHECK(IsAppMem(x));
  return (((x) & ~(kAppMemMsk | (kShadowCell - 1)))
      ^ kAppMemXor) * kShadowCnt;
}

ALWAYS_INLINE
u32 *MemToMeta(uptr x) {
  DCHECK(IsAppMem(x));
  return (u32*)(((((x) & ~(kAppMemMsk | (kMetaShadowCell - 1)))
      ^ kAppMemXor) / kMetaShadowCell * kMetaShadowSize) | kMetaShadowBeg);
}

ALWAYS_INLINE
uptr ShadowToMem(uptr s) {
  CHECK(IsShadowMem(s));
  if (s >= MemToShadow(kLoAppMemBeg) && s <= MemToShadow(kLoAppMemEnd - 1))
    return (s / kShadowCnt) ^ kAppMemXor;
  else
    return ((s / kShadowCnt) ^ kAppMemXor) | kAppMemMsk;
}

static USED uptr UserRegions[] = {
  kLoAppMemBeg, kLoAppMemEnd,
  kHiAppMemBeg, kHiAppMemEnd,
  kHeapMemBeg,  kHeapMemEnd,
};

#elif defined(SANITIZER_GO) && !SANITIZER_WINDOWS

/* Go on linux, darwin and freebsd
0000 0000 1000 - 0000 1000 0000: executable
0000 1000 0000 - 00c0 0000 0000: -
00c0 0000 0000 - 00e0 0000 0000: heap
00e0 0000 0000 - 2000 0000 0000: -
2000 0000 0000 - 2380 0000 0000: shadow
2380 0000 0000 - 3000 0000 0000: -
3000 0000 0000 - 4000 0000 0000: metainfo (memory blocks and sync objects)
4000 0000 0000 - 6000 0000 0000: -
6000 0000 0000 - 6200 0000 0000: traces
6200 0000 0000 - 8000 0000 0000: -
*/

const uptr kMetaShadowBeg = 0x300000000000ull;
const uptr kMetaShadowEnd = 0x400000000000ull;
const uptr kTraceMemBeg   = 0x600000000000ull;
const uptr kTraceMemEnd   = 0x620000000000ull;
const uptr kShadowBeg     = 0x200000000000ull;
const uptr kShadowEnd     = 0x238000000000ull;
const uptr kAppMemBeg     = 0x000000001000ull;
const uptr kAppMemEnd     = 0x00e000000000ull;

ALWAYS_INLINE
bool IsAppMem(uptr mem) {
  return mem >= kAppMemBeg && mem < kAppMemEnd;
}

ALWAYS_INLINE
bool IsShadowMem(uptr mem) {
  return mem >= kShadowBeg && mem <= kShadowEnd;
}

ALWAYS_INLINE
bool IsMetaMem(uptr mem) {
  return mem >= kMetaShadowBeg && mem <= kMetaShadowEnd;
}

ALWAYS_INLINE
uptr MemToShadow(uptr x) {
  DCHECK(IsAppMem(x));
  return ((x & ~(kShadowCell - 1)) * kShadowCnt) | kShadowBeg;
}

ALWAYS_INLINE
u32 *MemToMeta(uptr x) {
  DCHECK(IsAppMem(x));
  return (u32*)(((x & ~(kMetaShadowCell - 1)) / \
      kMetaShadowCell * kMetaShadowSize) | kMetaShadowBeg);
}

ALWAYS_INLINE
uptr ShadowToMem(uptr s) {
  CHECK(IsShadowMem(s));
  return (s & ~kShadowBeg) / kShadowCnt;
}

static USED uptr UserRegions[] = {
  kAppMemBeg, kAppMemEnd,
};

#elif defined(SANITIZER_GO) && SANITIZER_WINDOWS

/* Go on windows
0000 0000 1000 - 0000 1000 0000: executable
0000 1000 0000 - 00f8 0000 0000: -
00c0 0000 0000 - 00e0 0000 0000: heap
00e0 0000 0000 - 0100 0000 0000: -
0100 0000 0000 - 0500 0000 0000: shadow
0500 0000 0000 - 0560 0000 0000: -
0560 0000 0000 - 0760 0000 0000: traces
0760 0000 0000 - 07d0 0000 0000: metainfo (memory blocks and sync objects)
07d0 0000 0000 - 8000 0000 0000: -
*/

const uptr kMetaShadowBeg = 0x076000000000ull;
const uptr kMetaShadowEnd = 0x07d000000000ull;
const uptr kTraceMemBeg   = 0x056000000000ull;
const uptr kTraceMemEnd   = 0x076000000000ull;
const uptr kShadowBeg     = 0x010000000000ull;
const uptr kShadowEnd     = 0x050000000000ull;
const uptr kAppMemBeg     = 0x000000001000ull;
const uptr kAppMemEnd     = 0x00e000000000ull;

ALWAYS_INLINE
bool IsAppMem(uptr mem) {
  return mem >= kAppMemBeg && mem < kAppMemEnd;
}

ALWAYS_INLINE
bool IsShadowMem(uptr mem) {
  return mem >= kShadowBeg && mem <= kShadowEnd;
}

ALWAYS_INLINE
bool IsMetaMem(uptr mem) {
  return mem >= kMetaShadowBeg && mem <= kMetaShadowEnd;
}

ALWAYS_INLINE
uptr MemToShadow(uptr x) {
  DCHECK(IsAppMem(x));
  return ((x & ~(kShadowCell - 1)) * kShadowCnt) + kShadowBeg;
}

ALWAYS_INLINE
u32 *MemToMeta(uptr x) {
  DCHECK(IsAppMem(x));
  return (u32*)(((x & ~(kMetaShadowCell - 1)) / \
      kMetaShadowCell * kMetaShadowSize) | kMetaShadowBeg);
}

ALWAYS_INLINE
uptr ShadowToMem(uptr s) {
  CHECK(IsShadowMem(s));
  // FIXME(dvyukov): this is most likely wrong as the mapping is not bijection.
  return (s - kShadowBeg) / kShadowCnt;
}

static USED uptr UserRegions[] = {
  kAppMemBeg, kAppMemEnd,
};

#else
# error "Unknown platform"
#endif

// The additional page is to catch shadow stack overflow as paging fault.
// Windows wants 64K alignment for mmaps.
const uptr kTotalTraceSize = (kTraceSize * sizeof(Event) + sizeof(Trace)
    + (64 << 10) + (64 << 10) - 1) & ~((64 << 10) - 1);

uptr ALWAYS_INLINE GetThreadTrace(int tid) {
  uptr p = kTraceMemBeg + (uptr)tid * kTotalTraceSize;
  DCHECK_LT(p, kTraceMemEnd);
  return p;
}

uptr ALWAYS_INLINE GetThreadTraceHeader(int tid) {
  uptr p = kTraceMemBeg + (uptr)tid * kTotalTraceSize
      + kTraceSize * sizeof(Event);
  DCHECK_LT(p, kTraceMemEnd);
  return p;
}

void InitializePlatform();
void CheckAndProtect();
void InitializeShadowMemoryPlatform();
void FlushShadowMemory();
void WriteMemoryProfile(char *buf, uptr buf_size, uptr nthread, uptr nlive);

// Says whether the addr relates to a global var.
// Guesses with high probability, may yield both false positives and negatives.
bool IsGlobalVar(uptr addr);
int ExtractResolvFDs(void *state, int *fds, int nfd);
int ExtractRecvmsgFDs(void *msg, int *fds, int nfd);

int call_pthread_cancel_with_cleanup(int(*fn)(void *c, void *m,
    void *abstime), void *c, void *m, void *abstime,
    void(*cleanup)(void *arg), void *arg);

}  // namespace __tsan

#endif  // TSAN_PLATFORM_H
