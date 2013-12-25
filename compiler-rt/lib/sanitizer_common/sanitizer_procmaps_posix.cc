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

}  // namespace __sanitizer

#endif  // SANITIZER_POSIX
