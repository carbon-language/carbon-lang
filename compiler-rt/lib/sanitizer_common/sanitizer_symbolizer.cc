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
#include "sanitizer_symbolizer_internal.h"

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

void AddressInfo::FillModuleInfo(const char *mod_name, uptr mod_offset) {
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

void Symbolizer::AddHooks(Symbolizer::StartSymbolizationHook start_hook,
                          Symbolizer::EndSymbolizationHook end_hook) {
  CHECK(start_hook_ == 0 && end_hook_ == 0);
  start_hook_ = start_hook;
  end_hook_ = end_hook;
}

Symbolizer::Symbolizer(IntrusiveList<SymbolizerTool> tools)
    : tools_(tools), start_hook_(0), end_hook_(0) {}

Symbolizer::SymbolizerScope::SymbolizerScope(const Symbolizer *sym)
    : sym_(sym) {
  if (sym_->start_hook_)
    sym_->start_hook_();
}

Symbolizer::SymbolizerScope::~SymbolizerScope() {
  if (sym_->end_hook_)
    sym_->end_hook_();
}

SymbolizedStack *Symbolizer::SymbolizePC(uptr addr) {
  BlockingMutexLock l(&mu_);
  const char *module_name;
  uptr module_offset;
  SymbolizedStack *res = SymbolizedStack::New(addr);
  if (!PlatformFindModuleNameAndOffsetForAddress(addr, &module_name,
                                                 &module_offset))
    return res;
  // Always fill data about module name and offset.
  res->info.FillModuleInfo(module_name, module_offset);
  for (auto iter = Iterator(&tools_); iter.hasNext();) {
    auto *tool = iter.next();
    SymbolizerScope sym_scope(this);
    if (tool->SymbolizePC(addr, res)) {
      return res;
    }
  }
  return res;
}

bool Symbolizer::SymbolizeData(uptr addr, DataInfo *info) {
  BlockingMutexLock l(&mu_);
  const char *module_name;
  uptr module_offset;
  if (!PlatformFindModuleNameAndOffsetForAddress(addr, &module_name,
                                                 &module_offset))
    return false;
  info->Clear();
  info->module = internal_strdup(module_name);
  info->module_offset = module_offset;
  for (auto iter = Iterator(&tools_); iter.hasNext();) {
    auto *tool = iter.next();
    SymbolizerScope sym_scope(this);
    if (tool->SymbolizeData(addr, info)) {
      return true;
    }
  }
  return true;
}

bool Symbolizer::GetModuleNameAndOffsetForPC(uptr pc, const char **module_name,
                                 uptr *module_address) {
  BlockingMutexLock l(&mu_);
  return PlatformFindModuleNameAndOffsetForAddress(pc, module_name,
                                                   module_address);
}

void Symbolizer::Flush() {
  BlockingMutexLock l(&mu_);
  for (auto iter = Iterator(&tools_); iter.hasNext();) {
    auto *tool = iter.next();
    SymbolizerScope sym_scope(this);
    tool->Flush();
  }
}

const char *Symbolizer::Demangle(const char *name) {
  BlockingMutexLock l(&mu_);
  for (auto iter = Iterator(&tools_); iter.hasNext();) {
    auto *tool = iter.next();
    SymbolizerScope sym_scope(this);
    if (const char *demangled = tool->Demangle(name))
      return demangled;
  }
  return PlatformDemangle(name);
}

void Symbolizer::PrepareForSandboxing() {
  BlockingMutexLock l(&mu_);
  PlatformPrepareForSandboxing();
}

}  // namespace __sanitizer
