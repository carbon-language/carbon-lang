//===-- tsan_platform_windows.cc ------------------------------------------===//
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
// Windows-specific code.
//===----------------------------------------------------------------------===//

#ifdef _WIN32

#include "tsan_platform.h"

#include <stdlib.h>

namespace __tsan {

ScopedInRtl::ScopedInRtl() {
}

ScopedInRtl::~ScopedInRtl() {
}

uptr GetShadowMemoryConsumption() {
  return 0;
}

void FlushShadowMemory() {
}

void InitializeShadowMemory() {
/*
  uptr shadow = (uptr)MmapFixedNoReserve(kLinuxShadowBeg,
    kLinuxShadowEnd - kLinuxShadowBeg);
  if (shadow != kLinuxShadowBeg) {
    Printf("FATAL: ThreadSanitizer can not mmap the shadow memory\n");
    Printf("FATAL: Make sure to compile with -fPIE and "
           "to link with -pie.\n");
    Die();
  }
*/

  MmapFixedNoReserve(MemToShadow(kLinuxAppMemBeg), (1ull<<20) * 16 * 4);
  MmapCommit(MemToShadow(kLinuxAppMemBeg), (1ull<<20) * 16 * 4);
  MmapFixedNoReserve(MemToShadow(0xf840000000ull), (1ull<<20) * 4096 * 4);
  MmapCommit(MemToShadow(0xf840000000ull), (1ull<<20) * 256 * 4);
  DPrintf("kLinuxShadow %zx-%zx (%zuGB)\n",
      kLinuxShadowBeg, kLinuxShadowEnd,
      (kLinuxShadowEnd - kLinuxShadowBeg) >> 30);
  DPrintf("kLinuxAppMem %zx-%zx (%zuGB)\n",
      kLinuxAppMemBeg, kLinuxAppMemEnd,
      (kLinuxAppMemEnd - kLinuxAppMemBeg) >> 30);
}

const char *InitializePlatform() {
  return getenv("TSAN_OPTIONS");
}

void FinalizePlatform() {
  fflush(0);
}

uptr GetTlsSize() {
  return 0;
}

void GetThreadStackAndTls(bool main, uptr *stk_addr, uptr *stk_size,
                          uptr *tls_addr, uptr *tls_size) {
  *stk_addr = 0;
  *stk_size = 0;
  *tls_addr = 0;
  *tls_size = 0;
}

}  // namespace __tsan

#endif  // #ifdef _WIN32
