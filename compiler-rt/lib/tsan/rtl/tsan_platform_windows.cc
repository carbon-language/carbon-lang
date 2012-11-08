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

const char *InitializePlatform() {
  return getenv(kTsanOptionsEnv);
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
