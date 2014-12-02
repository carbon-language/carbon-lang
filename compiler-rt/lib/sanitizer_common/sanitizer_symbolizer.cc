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

#include "sanitizer_allocator_internal.h"
#include "sanitizer_platform.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_libc.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_symbolizer.h"

namespace __sanitizer {

AddressInfo::AddressInfo() {
  internal_memset(this, 0, sizeof(AddressInfo));
  function_offset = kUnknown;
}

void AddressInfo::Clear() {
  InternalFree(module);
  InternalFree(function);
  InternalFree(file);
  internal_memset(this, 0, sizeof(AddressInfo));
  function_offset = kUnknown;
}

void AddressInfo::FillAddressAndModuleInfo(uptr addr, const char *mod_name,
                                           uptr mod_offset) {
  address = addr;
  module = internal_strdup(mod_name);
  module_offset = mod_offset;
}

SymbolizedStack::SymbolizedStack() : next(nullptr), info() {}

SymbolizedStack *SymbolizedStack::New(uptr addr) {
  void *mem = InternalAlloc(sizeof(SymbolizedStack));
  SymbolizedStack *res = new(mem) SymbolizedStack();
  res->info.address = addr;
  return res;
}

void SymbolizedStack::ClearAll() {
  info.Clear();
  if (next)
    next->ClearAll();
  InternalFree(this);
}

DataInfo::DataInfo() {
  internal_memset(this, 0, sizeof(DataInfo));
}

void DataInfo::Clear() {
  InternalFree(module);
  InternalFree(name);
  internal_memset(this, 0, sizeof(DataInfo));
}

Symbolizer *Symbolizer::symbolizer_;
StaticSpinMutex Symbolizer::init_mu_;
LowLevelAllocator Symbolizer::symbolizer_allocator_;

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
