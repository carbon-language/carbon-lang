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

/*
C/C++ on linux and freebsd
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
0100 0000 0000 - 0380 0000 0000: shadow
0380 0000 0000 - 0560 0000 0000: -
0560 0000 0000 - 0760 0000 0000: traces
0760 0000 0000 - 07d0 0000 0000: metainfo (memory blocks and sync objects)
07d0 0000 0000 - 8000 0000 0000: -
*/

const uptr kMetaShadowBeg = 0x076000000000ull;
const uptr kMetaShadowEnd = 0x07d000000000ull;
const uptr kTraceMemBeg   = 0x056000000000ull;
const uptr kTraceMemEnd   = 0x076000000000ull;
const uptr kShadowBeg     = 0x010000000000ull;
const uptr kShadowEnd     = 0x038000000000ull;
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
      kMetaShadowCell * kMetaShadowSize) | kMetaShadowEnd);
}

ALWAYS_INLINE
uptr ShadowToMem(uptr s) {
  CHECK(IsShadowMem(s));
  // FIXME(dvyukov): this is most likely wrong as the mapping is not bijection.
  return (x & ~kShadowBeg) / kShadowCnt;
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
