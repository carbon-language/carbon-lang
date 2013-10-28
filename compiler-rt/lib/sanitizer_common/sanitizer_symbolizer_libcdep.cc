//===-- sanitizer_symbolizer_libcdep.cc -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_symbolizer.h"

namespace __sanitizer {

Symbolizer *Symbolizer::CreateAndStore(const char *path_to_external) {
  Symbolizer *platform_symbolizer = PlatformInit(path_to_external);
  if (!platform_symbolizer)
    return Disable();
  atomic_store(&symbolizer_, reinterpret_cast<uptr>(platform_symbolizer),
               memory_order_release);
  return platform_symbolizer;
}

Symbolizer *Symbolizer::Init(const char *path_to_external) {
  CHECK_EQ(0, atomic_load(&symbolizer_, memory_order_acquire));
  return CreateAndStore(path_to_external);
}

Symbolizer *Symbolizer::GetOrInit() {
  static StaticSpinMutex init_mu;

  uptr sym = atomic_load(&symbolizer_, memory_order_acquire);
  if (!sym) {
    SpinMutexLock l(&init_mu);
    sym = atomic_load(&symbolizer_, memory_order_relaxed);
    if (!sym) return CreateAndStore(0);
  }

  return reinterpret_cast<Symbolizer *>(sym);
}

}  // namespace __sanitizer
