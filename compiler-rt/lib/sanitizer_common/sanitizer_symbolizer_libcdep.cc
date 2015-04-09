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

#include "sanitizer_allocator_internal.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_symbolizer_internal.h"

namespace __sanitizer {

const char *ExtractToken(const char *str, const char *delims, char **result) {
  uptr prefix_len = internal_strcspn(str, delims);
  *result = (char*)InternalAlloc(prefix_len + 1);
  internal_memcpy(*result, str, prefix_len);
  (*result)[prefix_len] = '\0';
  const char *prefix_end = str + prefix_len;
  if (*prefix_end != '\0') prefix_end++;
  return prefix_end;
}

const char *ExtractInt(const char *str, const char *delims, int *result) {
  char *buff;
  const char *ret = ExtractToken(str, delims, &buff);
  if (buff != 0) {
    *result = (int)internal_atoll(buff);
  }
  InternalFree(buff);
  return ret;
}

const char *ExtractUptr(const char *str, const char *delims, uptr *result) {
  char *buff;
  const char *ret = ExtractToken(str, delims, &buff);
  if (buff != 0) {
    *result = (uptr)internal_atoll(buff);
  }
  InternalFree(buff);
  return ret;
}

const char *ExtractTokenUpToDelimiter(const char *str, const char *delimiter,
                                      char **result) {
  const char *found_delimiter = internal_strstr(str, delimiter);
  uptr prefix_len =
      found_delimiter ? found_delimiter - str : internal_strlen(str);
  *result = (char *)InternalAlloc(prefix_len + 1);
  internal_memcpy(*result, str, prefix_len);
  (*result)[prefix_len] = '\0';
  const char *prefix_end = str + prefix_len;
  if (*prefix_end != '\0') prefix_end += internal_strlen(delimiter);
  return prefix_end;
}

SymbolizedStack *Symbolizer::SymbolizePC(uptr addr) {
  BlockingMutexLock l(&mu_);
  const char *module_name;
  uptr module_offset;
  SymbolizedStack *res = SymbolizedStack::New(addr);
  if (!FindModuleNameAndOffsetForAddress(addr, &module_name, &module_offset))
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
  if (!FindModuleNameAndOffsetForAddress(addr, &module_name, &module_offset))
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
  const char *internal_module_name = nullptr;
  if (!FindModuleNameAndOffsetForAddress(pc, &internal_module_name,
                                         module_address))
    return false;

  if (module_name)
    *module_name = module_names_.GetOwnedCopy(internal_module_name);
  return true;
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

bool Symbolizer::FindModuleNameAndOffsetForAddress(uptr address,
                                                   const char **module_name,
                                                   uptr *module_offset) {
  LoadedModule *module = FindModuleForAddress(address);
  if (module == 0)
    return false;
  *module_name = module->full_name();
  *module_offset = address - module->base_address();
  return true;
}

LoadedModule *Symbolizer::FindModuleForAddress(uptr address) {
  bool modules_were_reloaded = false;
  if (!modules_fresh_) {
    for (uptr i = 0; i < n_modules_; i++)
      modules_[i].clear();
    n_modules_ =
        GetListOfModules(modules_, kMaxNumberOfModules, /* filter */ nullptr);
    CHECK_GT(n_modules_, 0);
    CHECK_LT(n_modules_, kMaxNumberOfModules);
    modules_fresh_ = true;
    modules_were_reloaded = true;
  }
  for (uptr i = 0; i < n_modules_; i++) {
    if (modules_[i].containsAddress(address)) {
      return &modules_[i];
    }
  }
  // Reload the modules and look up again, if we haven't tried it yet.
  if (!modules_were_reloaded) {
    // FIXME: set modules_fresh_ from dlopen()/dlclose() interceptors.
    // It's too aggressive to reload the list of modules each time we fail
    // to find a module for a given address.
    modules_fresh_ = false;
    return FindModuleForAddress(address);
  }
  return 0;
}

Symbolizer *Symbolizer::GetOrInit() {
  SpinMutexLock l(&init_mu_);
  if (symbolizer_)
    return symbolizer_;
  symbolizer_ = PlatformInit();
  CHECK(symbolizer_);
  return symbolizer_;
}

}  // namespace __sanitizer
