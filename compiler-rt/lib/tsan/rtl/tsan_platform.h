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

#include "tsan_rtl.h"

#if __LP64__
namespace __tsan {

#if defined(TSAN_GO)
static const uptr kLinuxAppMemBeg = 0x000000000000ULL;
static const uptr kLinuxAppMemEnd = 0x00fcffffffffULL;
static const uptr kLinuxShadowMsk = 0x100000000000ULL;
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
  MemToShadow(kLinuxAppMemEnd) | (kPageSize - 1);

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

void internal_start_thread(void(*func)(void*), void *arg);

// Says whether the addr relates to a global var.
// Guesses with high probability, may yield both false positives and negatives.
bool IsGlobalVar(uptr addr);
uptr GetTlsSize();
void GetThreadStackAndTls(bool main, uptr *stk_addr, uptr *stk_size,
                          uptr *tls_addr, uptr *tls_size);

}  // namespace __tsan

#else  // __LP64__
# error "Only 64-bit is supported"
#endif

#endif  // TSAN_PLATFORM_H
