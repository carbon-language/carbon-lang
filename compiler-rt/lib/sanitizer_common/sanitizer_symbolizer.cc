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

void AddressInfo::Clear() {
  InternalFree(module);
  InternalFree(function);
  InternalFree(file);
  internal_memset(this, 0, sizeof(AddressInfo));
}

static const int kMaxModuleNameLength = 4096;

struct ModuleDesc {
  ModuleDesc *next;
  uptr start;
  uptr end;
  uptr offset;
  char *full_name;
  char *name;

  ModuleDesc(uptr _start, uptr _end, uptr _offset, const char *module_name) {
    next = 0;
    start = _start;
    end = _end;
    offset = _offset;
    full_name = internal_strdup(module_name);
    name = internal_strrchr(module_name, '/');
    if (name == 0) {
      name = full_name;
    } else {
      name++;
    }
  }
};

class Symbolizer {
 public:
  void GetModuleDescriptions() {
    ProcessMaps proc_maps;
    uptr start, end, offset;
    char *module_name = (char*)InternalAlloc(kMaxModuleNameLength);
    ModuleDesc *prev_module = 0;
    while (proc_maps.Next(&start, &end, &offset, module_name,
                          kMaxModuleNameLength)) {
      void *mem = InternalAlloc(sizeof(ModuleDesc));
      ModuleDesc *cur_module = new(mem) ModuleDesc(start, end, offset,
                                                   module_name);
      if (!prev_module) {
        modules_ = cur_module;
      } else {
        prev_module->next = cur_module;
      }
      prev_module = cur_module;
    }
    InternalFree(module_name);
  }

  uptr SymbolizeCode(uptr addr, AddressInfo *frames, uptr max_frames) {
    if (max_frames == 0)
      return 0;
    AddressInfo *info = &frames[0];
    info->Clear();
    info->address = addr;
    if (modules_ == 0) {
      GetModuleDescriptions();
    }
    bool first = true;
    for (ModuleDesc *module = modules_; module; module = module->next) {
      if (addr >= module->start && addr < module->end) {
        info->module = internal_strdup(module->full_name);
        // Don't subtract 'start' for the first entry:
        // * If a binary is compiled w/o -pie, then the first entry in
        //   process maps is likely the binary itself (all dynamic libs
        //   are mapped higher in address space). For such a binary,
        //   instruction offset in binary coincides with the actual
        //   instruction address in virtual memory (as code section
        //   is mapped to a fixed memory range).
        // * If a binary is compiled with -pie, all the modules are
        //   mapped high at address space (in particular, higher than
        //   shadow memory of the tool), so the module can't be the
        //   first entry.
        info->module_offset = (addr - (first ? 0 : module->start)) +
                              module->offset;
        // FIXME: Fill other fields here as well: create debug
        // context for a given module and fetch file/line info from it.
        info->function = 0;
        info->file = 0;
        info->line = 0;
        info->column = 0;
        return 1;
      }
      first = false;
    }
    return 0;
  }

 private:
  ModuleDesc *modules_;  // List of module descriptions is leaked.
};

static Symbolizer symbolizer;

uptr SymbolizeCode(uptr address, AddressInfo *frames, uptr max_frames) {
  return symbolizer.SymbolizeCode(address, frames, max_frames);
}

}  // namespace __sanitizer
