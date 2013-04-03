//===-- sanitizer_procmaps.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer.
//
// Information about the process mappings.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_PROCMAPS_H
#define SANITIZER_PROCMAPS_H

#include "sanitizer_internal_defs.h"
#include "sanitizer_mutex.h"

namespace __sanitizer {

#if SANITIZER_WINDOWS
class MemoryMappingLayout {
 public:
  explicit MemoryMappingLayout(bool cache_enabled) {
    (void)cache_enabled;
  }
  bool GetObjectNameAndOffset(uptr addr, uptr *offset,
                              char filename[], uptr filename_size,
                              uptr *protection) {
    UNIMPLEMENTED();
  }
};

#else  // _WIN32
#if SANITIZER_LINUX
struct ProcSelfMapsBuff {
  char *data;
  uptr mmaped_size;
  uptr len;
};
#endif  // SANITIZER_LINUX

class MemoryMappingLayout {
 public:
  explicit MemoryMappingLayout(bool cache_enabled);
  bool Next(uptr *start, uptr *end, uptr *offset,
            char filename[], uptr filename_size, uptr *protection);
  void Reset();
  // Gets the object file name and the offset in that object for a given
  // address 'addr'. Returns true on success.
  bool GetObjectNameAndOffset(uptr addr, uptr *offset,
                              char filename[], uptr filename_size,
                              uptr *protection);
  // In some cases, e.g. when running under a sandbox on Linux, ASan is unable
  // to obtain the memory mappings. It should fall back to pre-cached data
  // instead of aborting.
  static void CacheMemoryMappings();
  ~MemoryMappingLayout();

  // Memory protection masks.
  static const uptr kProtectionRead = 1;
  static const uptr kProtectionWrite = 2;
  static const uptr kProtectionExecute = 4;
  static const uptr kProtectionShared = 8;

 private:
  void LoadFromCache();
  // Default implementation of GetObjectNameAndOffset.
  // Quite slow, because it iterates through the whole process map for each
  // lookup.
  bool IterateForObjectNameAndOffset(uptr addr, uptr *offset,
                                     char filename[], uptr filename_size,
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

# if SANITIZER_LINUX
  ProcSelfMapsBuff proc_self_maps_;
  char *current_;

  // Static mappings cache.
  static ProcSelfMapsBuff cached_proc_self_maps_;
  static StaticSpinMutex cache_lock_;  // protects cached_proc_self_maps_.
# elif SANITIZER_MAC
  template<u32 kLCSegment, typename SegmentCommand>
  bool NextSegmentLoad(uptr *start, uptr *end, uptr *offset,
                       char filename[], uptr filename_size,
                       uptr *protection);
  int current_image_;
  u32 current_magic_;
  u32 current_filetype_;
  int current_load_cmd_count_;
  char *current_load_cmd_addr_;
# endif
};

#endif  // _WIN32

}  // namespace __sanitizer

#endif  // SANITIZER_PROCMAPS_H
