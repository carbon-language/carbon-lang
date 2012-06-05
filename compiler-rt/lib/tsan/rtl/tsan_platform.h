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

#ifndef TSAN_LINUX_H
#define TSAN_LINUX_H
#ifdef __linux__

#include "tsan_rtl.h"

#if __LP64__
namespace __tsan {

// TSAN_COMPAT_SHADOW is intended for COMPAT virtual memory layout,
// when memory addresses are of the 0x2axxxxxxxxxx form.
// The option is enabled with 'setarch x86_64 -L'.
#if defined(TSAN_COMPAT_SHADOW) && TSAN_COMPAT_SHADOW

static const uptr kLinuxAppMemBeg = 0x2a0000000000ULL;
static const uptr kLinuxAppMemEnd = 0x7fffffffffffULL;

#else

static const uptr kLinuxAppMemBeg = 0x7ef000000000ULL;
static const uptr kLinuxAppMemEnd = 0x7fffffffffffULL;

#endif

static const uptr kLinuxAppMemMsk = 0x7c0000000000ULL;

// This has to be a macro to allow constant initialization of constants below.
#define MemToShadow(addr) \
    (((addr) & ~(kLinuxAppMemMsk | (kShadowCell - 1))) * kShadowCnt)

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
#if defined(TSAN_COMPAT_SHADOW) && TSAN_COMPAT_SHADOW
  // COMPAT mapping is not quite one-to-one.
  return (shadow / kShadowCnt) | 0x280000000000ULL;
#else
  return (shadow / kShadowCnt) | kLinuxAppMemMsk;
#endif
}

uptr GetShadowMemoryConsumption();
void FlushShadowMemory();

const char *InitializePlatform();
void FinalizePlatform();
int GetPid();

void internal_yield();
void internal_sleep_ms(u32 ms);

void internal_start_thread(void(*func)(void*), void *arg);

typedef int fd_t;
const fd_t kInvalidFd = -1;
void internal_close(fd_t fd);
uptr internal_filesize(fd_t fd);  // -1 on error.
uptr internal_read(fd_t fd, void *p, uptr size);
uptr internal_write(fd_t fd, const void *p, uptr size);
int internal_dup2(int oldfd, int newfd);
const char *internal_getpwd();

uptr GetTlsSize();
void GetThreadStackAndTls(bool main, uptr *stk_addr, uptr *stk_size,
                          uptr *tls_addr, uptr *tls_size);

}  // namespace __tsan

#else  // __LP64__
# error "Only 64-bit is supported"
#endif

#endif  // __linux__
#endif  // TSAN_LINUX_H
