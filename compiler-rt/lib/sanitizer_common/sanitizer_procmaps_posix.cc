//===-- sanitizer_procmaps_posix.cc ---------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Information about the process mappings.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"
#if SANITIZER_POSIX
#include "sanitizer_common.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_procmaps.h"

namespace __sanitizer {

bool MemoryMappingLayout::IterateForObjectNameAndOffset(uptr addr, uptr *offset,
                                                        char filename[],
                                                        uptr filename_size,
                                                        uptr *protection) {
  Reset();
  uptr start, end, file_offset;
  for (int i = 0; Next(&start, &end, &file_offset, filename, filename_size,
                       protection);
       i++) {
    if (addr >= start && addr < end) {
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
      *offset = (addr - (i ? start : 0)) + file_offset;
      return true;
    }
  }
  if (filename_size)
    filename[0] = '\0';
  return false;
}

uptr MemoryMappingLayout::DumpListOfModules(LoadedModule *modules,
                                            uptr max_modules,
                                            string_predicate_t filter) {
  Reset();
  uptr cur_beg, cur_end, cur_offset;
  InternalScopedBuffer<char> module_name(kMaxPathLength);
  uptr n_modules = 0;
  for (uptr i = 0; n_modules < max_modules &&
                       Next(&cur_beg, &cur_end, &cur_offset, module_name.data(),
                            module_name.size(), 0);
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

}  // namespace __sanitizer

#endif  // SANITIZER_POSIX
