//===-- sanitizer_symbolizer.cc -------------------------------------------===//
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

Symbolizer *Symbolizer::symbolizer_;
StaticSpinMutex Symbolizer::init_mu_;
LowLevelAllocator Symbolizer::symbolizer_allocator_;

Symbolizer *Symbolizer::GetOrNull() {
  SpinMutexLock l(&init_mu_);
  return symbolizer_;
}

Symbolizer *Symbolizer::Get() {
  SpinMutexLock l(&init_mu_);
  RAW_CHECK_MSG(symbolizer_ != 0, "Using uninitialized symbolizer!");
  return symbolizer_;
}

Symbolizer *Symbolizer::Disable() {
  CHECK_EQ(0, symbolizer_);
  // Initialize a dummy symbolizer.
  symbolizer_ = new(symbolizer_allocator_) Symbolizer;
  return symbolizer_;
}

}  // namespace __sanitizer
