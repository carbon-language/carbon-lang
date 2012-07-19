//===-- sanitizer_symbolizer.cc -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a stub for LLVM-based symbolizer.
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries. See sanitizer.h for details.
//===----------------------------------------------------------------------===//

#include "sanitizer_common.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_procmaps.h"
#include "sanitizer_symbolizer.h"

namespace __sanitizer {

bool IsFullNameOfDWARFSection(const char *full_name, const char *short_name) {
  // Skip "__DWARF," prefix.
  if (0 == internal_strncmp(full_name, "__DWARF,", 8)) {
    full_name += 8;
  }
  // Skip . and _ prefices.
  while (*full_name == '.' || *full_name == '_') {
    full_name++;
  }
  return 0 == internal_strcmp(full_name, short_name);
}

void AddressInfo::Clear() {
  InternalFree(module);
  InternalFree(function);
  InternalFree(file);
  internal_memset(this, 0, sizeof(AddressInfo));
}

ModuleDIContext::ModuleDIContext(const char *module_name, uptr base_address) {
  full_name_ = internal_strdup(module_name);
  short_name_ = internal_strrchr(module_name, '/');
  if (short_name_ == 0) {
    short_name_ = full_name_;
  } else {
    short_name_++;
  }
  base_address_ = base_address;
  n_ranges_ = 0;
  mapped_addr_ = 0;
  mapped_size_ = 0;
}

void ModuleDIContext::addAddressRange(uptr beg, uptr end) {
  CHECK_LT(n_ranges_, kMaxNumberOfAddressRanges);
  ranges_[n_ranges_].beg = beg;
  ranges_[n_ranges_].end = end;
  n_ranges_++;
}

bool ModuleDIContext::containsAddress(uptr address) const {
  for (uptr i = 0; i < n_ranges_; i++) {
    if (ranges_[i].beg <= address && address < ranges_[i].end)
      return true;
  }
  return false;
}

void ModuleDIContext::getAddressInfo(AddressInfo *info) {
  info->module = internal_strdup(full_name_);
  info->module_offset = info->address - base_address_;
  if (mapped_addr_ == 0)
    CreateDIContext();
  // FIXME: Use the actual debug info context here.
  info->function = 0;
  info->file = 0;
  info->line = 0;
  info->column = 0;
}

void ModuleDIContext::CreateDIContext() {
  mapped_addr_ = (uptr)MapFileToMemory(full_name_, &mapped_size_);
  CHECK(mapped_addr_);
  DWARFSection debug_info;
  DWARFSection debug_abbrev;
  DWARFSection debug_line;
  DWARFSection debug_aranges;
  DWARFSection debug_str;
  FindDWARFSection(mapped_addr_, "debug_info", &debug_info);
  FindDWARFSection(mapped_addr_, "debug_abbrev", &debug_abbrev);
  FindDWARFSection(mapped_addr_, "debug_line", &debug_line);
  FindDWARFSection(mapped_addr_, "debug_aranges", &debug_aranges);
  FindDWARFSection(mapped_addr_, "debug_str", &debug_str);
  // FIXME: Construct actual debug info context using mapped_addr,
  // mapped_size and pointers to DWARF sections in memory.
}

class Symbolizer {
 public:
  uptr SymbolizeCode(uptr addr, AddressInfo *frames, uptr max_frames) {
    if (max_frames == 0)
      return 0;
    AddressInfo *info = &frames[0];
    info->Clear();
    info->address = addr;
    ModuleDIContext *module = FindModuleForAddress(addr);
    if (module) {
      module->getAddressInfo(info);
      return 1;
    }
    return 0;
  }

 private:
  ModuleDIContext *FindModuleForAddress(uptr address) {
    if (modules_ == 0) {
      modules_ = (ModuleDIContext*)InternalAlloc(
          kMaxNumberOfModuleContexts * sizeof(ModuleDIContext));
      CHECK(modules_);
      n_modules_ = GetListOfModules(modules_, kMaxNumberOfModuleContexts);
      CHECK_GT(n_modules_, 0);
      CHECK_LT(n_modules_, kMaxNumberOfModuleContexts);
    }
    for (uptr i = 0; i < n_modules_; i++) {
      if (modules_[i].containsAddress(address)) {
        return &modules_[i];
      }
    }
    return 0;
  }
  static const uptr kMaxNumberOfModuleContexts = 4096;
  // Array of module debug info contexts is leaked.
  ModuleDIContext *modules_;
  uptr n_modules_;
};

static Symbolizer symbolizer;  // Linker initialized.

uptr SymbolizeCode(uptr address, AddressInfo *frames, uptr max_frames) {
  return symbolizer.SymbolizeCode(address, frames, max_frames);
}

}  // namespace __sanitizer
