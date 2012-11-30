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

/*
C++ linux memory layout:
0000 0000 0000 - 03c0 0000 0000: protected
03c0 0000 0000 - 1000 0000 0000: shadow
1000 0000 0000 - 6000 0000 0000: protected
6000 0000 0000 - 6200 0000 0000: traces
6200 0000 0000 - 7d00 0000 0000: -
7d00 0000 0000 - 7e00 0000 0000: heap
7e00 0000 0000 - 7fff ffff ffff: modules and main thread stack

C++ COMPAT linux memory layout:
0000 0000 0000 - 0400 0000 0000: protected
0400 0000 0000 - 1000 0000 0000: shadow
1000 0000 0000 - 2900 0000 0000: protected
2900 0000 0000 - 2c00 0000 0000: modules
2c00 0000 0000 - 6000 0000 0000: -
6000 0000 0000 - 6200 0000 0000: traces
6200 0000 0000 - 7d00 0000 0000: -
7d00 0000 0000 - 7e00 0000 0000: heap
7e00 0000 0000 - 7f00 0000 0000: -
7f00 0000 0000 - 7fff ffff ffff: main thread stack

Go linux and darwin memory layout:
0000 0000 0000 - 0000 1000 0000: executable
0000 1000 0000 - 00f8 0000 0000: -
00f8 0000 0000 - 0118 0000 0000: heap
0118 0000 0000 - 1000 0000 0000: -
1000 0000 0000 - 1460 0000 0000: shadow
1460 0000 0000 - 6000 0000 0000: -
6000 0000 0000 - 6200 0000 0000: traces
6200 0000 0000 - 7fff ffff ffff: -

Go windows memory layout:
0000 0000 0000 - 0000 1000 0000: executable
0000 1000 0000 - 00f8 0000 0000: -
00f8 0000 0000 - 0118 0000 0000: heap
0118 0000 0000 - 0100 0000 0000: -
0100 0000 0000 - 0560 0000 0000: shadow
0560 0000 0000 - 0760 0000 0000: traces
0760 0000 0000 - 07ff ffff ffff: -
*/

#ifndef TSAN_PLATFORM_H
#define TSAN_PLATFORM_H

#include "tsan_defs.h"
#include "tsan_trace.h"

#if defined(__LP64__) || defined(_WIN64)
namespace __tsan {

#if defined(TSAN_GO)
static const uptr kLinuxAppMemBeg = 0x000000000000ULL;
static const uptr kLinuxAppMemEnd = 0x00fcffffffffULL;
# if defined(_WIN32)
static const uptr kLinuxShadowMsk = 0x010000000000ULL;
# else
static const uptr kLinuxShadowMsk = 0x100000000000ULL;
# endif
// TSAN_COMPAT_SHADOW is intended for COMPAT virtual memory layout,
// when memory addresses are of the 0x2axxxxxxxxxx form.
// The option is enabled with 'setarch x86_64 -L'.
#elif defined(TSAN_COMPAT_SHADOW) && TSAN_COMPAT_SHADOW
static const uptr kLinuxAppMemBeg = 0x290000000000ULL;
static const uptr kLinuxAppMemEnd = 0x7fffffffffffULL;
#else
static const uptr kLinuxAppMemBeg = 0x7cf000000000ULL;
static const uptr kLinuxAppMemEnd = 0x7fffffffffffULL;
#endif

static const uptr kLinuxAppMemMsk = 0x7c0000000000ULL;

#if defined(_WIN32)
const uptr kTraceMemBegin = 0x056000000000ULL;
#else
const uptr kTraceMemBegin = 0x600000000000ULL;
#endif
const uptr kTraceMemSize = 0x020000000000ULL;

// This has to be a macro to allow constant initialization of constants below.
#ifndef TSAN_GO
#define MemToShadow(addr) \
    (((addr) & ~(kLinuxAppMemMsk | (kShadowCell - 1))) * kShadowCnt)
#else
#define MemToShadow(addr) \
    ((((addr) & ~(kShadowCell - 1)) * kShadowCnt) | kLinuxShadowMsk)
#endif

static const uptr kLinuxShadowBeg = MemToShadow(kLinuxAppMemBeg);
static const uptr kLinuxShadowEnd =
    MemToShadow(kLinuxAppMemEnd) | 0xff;

static inline bool IsAppMem(uptr mem) {
  return mem >= kLinuxAppMemBeg && mem <= kLinuxAppMemEnd;
}

static inline bool IsShadowMem(uptr mem) {
  return mem >= kLinuxShadowBeg && mem <= kLinuxShadowEnd;
}

static inline uptr ShadowToMem(uptr shadow) {
  CHECK(IsShadowMem(shadow));
#ifdef TSAN_GO
  return (shadow & ~kLinuxShadowMsk) / kShadowCnt;
#else
  return (shadow / kShadowCnt) | kLinuxAppMemMsk;
#endif
}

// For COMPAT mapping returns an alternative address
// that mapped to the same shadow address.
// COMPAT mapping is not quite one-to-one.
static inline uptr AlternativeAddress(uptr addr) {
#if defined(TSAN_COMPAT_SHADOW) && TSAN_COMPAT_SHADOW
  return (addr & ~kLinuxAppMemMsk) | 0x280000000000ULL;
#else
  return 0;
#endif
}

uptr GetShadowMemoryConsumption();
void FlushShadowMemory();

const char *InitializePlatform();
void FinalizePlatform();
void MapThreadTrace(uptr addr, uptr size);
uptr ALWAYS_INLINE INLINE GetThreadTrace(int tid) {
  uptr p = kTraceMemBegin + (uptr)tid * kTraceSize * sizeof(Event);
  DCHECK_LT(p, kTraceMemBegin + kTraceMemSize);
  return p;
}

void internal_start_thread(void(*func)(void*), void *arg);

// Says whether the addr relates to a global var.
// Guesses with high probability, may yield both false positives and negatives.
bool IsGlobalVar(uptr addr);
uptr GetTlsSize();
void GetThreadStackAndTls(bool main, uptr *stk_addr, uptr *stk_size,
                          uptr *tls_addr, uptr *tls_size);

}  // namespace __tsan

#else  // defined(__LP64__) || defined(_WIN64)
# error "Only 64-bit is supported"
#endif

#endif  // TSAN_PLATFORM_H
