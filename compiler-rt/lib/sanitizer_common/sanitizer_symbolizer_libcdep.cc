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

#include "sanitizer_internal_defs.h"
#include "sanitizer_symbolizer.h"

namespace __sanitizer {

Symbolizer *Symbolizer::CreateAndStore(const char *path_to_external) {
  Symbolizer *platform_symbolizer = PlatformInit(path_to_external);
  if (!platform_symbolizer)
    return Disable();
  symbolizer_ = platform_symbolizer;
  return platform_symbolizer;
}

Symbolizer *Symbolizer::Init(const char *path_to_external) {
  CHECK_EQ(0, symbolizer_);
  return CreateAndStore(path_to_external);
}

Symbolizer *Symbolizer::GetOrInit() {
  SpinMutexLock l(&init_mu_);
  if (symbolizer_ == 0)
    return CreateAndStore(0);
  return symbolizer_;
}

}  // namespace __sanitizer
