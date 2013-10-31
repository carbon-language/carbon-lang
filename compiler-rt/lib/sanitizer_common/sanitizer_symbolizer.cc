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

void Symbolizer::AddHooks(Symbolizer::StartSymbolizationHook start_hook,
                          Symbolizer::EndSymbolizationHook end_hook) {
  CHECK(start_hook_ == 0 && end_hook_ == 0);
  start_hook_ = start_hook;
  end_hook_ = end_hook;
}

Symbolizer::Symbolizer() : start_hook_(0), end_hook_(0) {}

Symbolizer::SymbolizerScope::SymbolizerScope(const Symbolizer *sym)
    : sym_(sym) {
  if (sym_->start_hook_)
    sym_->start_hook_();
}

Symbolizer::SymbolizerScope::~SymbolizerScope() {
  if (sym_->end_hook_)
    sym_->end_hook_();
}

}  // namespace __sanitizer
