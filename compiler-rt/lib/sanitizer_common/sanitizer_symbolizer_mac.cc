//===-- sanitizer_symbolizer_mac.cc ---------------------------------------===//
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
// Mac-specific implementation of symbolizer parts.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"
#if SANITIZER_MAC
#include "sanitizer_common.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_procmaps.h"
#include "sanitizer_symbolizer.h"

namespace __sanitizer {

uptr GetListOfModules(LoadedModule *modules, uptr max_modules,
                      string_predicate_t filter) {
  MemoryMappingLayout memory_mapping(false);
  memory_mapping.Reset();
  uptr cur_beg, cur_end, cur_offset;
  InternalScopedBuffer<char> module_name(kMaxPathLength);
  uptr n_modules = 0;
  for (uptr i = 0;
       n_modules < max_modules &&
           memory_mapping.Next(&cur_beg, &cur_end, &cur_offset,
                               module_name.data(), module_name.size(), 0);
       i++) {
    const char *cur_name = module_name.data();
    if (cur_name[0] == '\0')
      continue;
    if (filter && !filter(cur_name))
      continue;
    LoadedModule *cur_module = 0;
    if (n_modules > 0 &&
        0 == internal_strcmp(cur_name, modules[n_modules - 1].full_name())) {
      cur_module = &modules[n_modules - 1];
    } else {
      void *mem = &modules[n_modules];
      cur_module = new(mem) LoadedModule(cur_name, cur_beg);
      n_modules++;
    }
    cur_module->addAddressRange(cur_beg, cur_end);
  }
  return n_modules;
}

void SymbolizerPrepareForSandboxing() {
  // Do nothing on Mac.
}

}  // namespace __sanitizer

#endif  // SANITIZER_MAC
