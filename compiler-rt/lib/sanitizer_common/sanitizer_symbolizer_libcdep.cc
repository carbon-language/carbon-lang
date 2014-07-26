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

Symbolizer *Symbolizer::GetOrInit() {
  SpinMutexLock l(&init_mu_);
  if (symbolizer_)
    return symbolizer_;
  if ((symbolizer_ = PlatformInit()))
    return symbolizer_;
  return Disable();
}

}  // namespace __sanitizer
