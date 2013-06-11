//===-- sanitizer_symbolizer_linux_libcdep.cc -----------------------------===//
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
// Linux-specific implementation of symbolizer parts.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"
#if SANITIZER_LINUX
#include "sanitizer_common.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_libc.h"
#include "sanitizer_linux.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_symbolizer.h"

// Android NDK r8e elf.h depends on stdint.h without including the latter.
#include <stdint.h>

#include <elf.h>
#include <errno.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#if !SANITIZER_ANDROID
#include <link.h>
#endif

namespace __sanitizer {

#if SANITIZER_ANDROID
uptr GetListOfModules(LoadedModule *modules, uptr max_modules,
                      string_predicate_t filter) {
  return 0;
}
#else  // SANITIZER_ANDROID
typedef ElfW(Phdr) Elf_Phdr;

struct DlIteratePhdrData {
  LoadedModule *modules;
  uptr current_n;
  bool first;
  uptr max_n;
  string_predicate_t filter;
};

static int dl_iterate_phdr_cb(dl_phdr_info *info, size_t size, void *arg) {
  DlIteratePhdrData *data = (DlIteratePhdrData*)arg;
  if (data->current_n == data->max_n)
    return 0;
  InternalScopedBuffer<char> module_name(kMaxPathLength);
  module_name.data()[0] = '\0';
  if (data->first) {
    data->first = false;
    // First module is the binary itself.
    ReadBinaryName(module_name.data(), module_name.size());
  } else if (info->dlpi_name) {
    internal_strncpy(module_name.data(), info->dlpi_name, module_name.size());
  }
  if (module_name.data()[0] == '\0')
    return 0;
  if (data->filter && !data->filter(module_name.data()))
    return 0;
  void *mem = &data->modules[data->current_n];
  LoadedModule *cur_module = new(mem) LoadedModule(module_name.data(),
                                                   info->dlpi_addr);
  data->current_n++;
  for (int i = 0; i < info->dlpi_phnum; i++) {
    const Elf_Phdr *phdr = &info->dlpi_phdr[i];
    if (phdr->p_type == PT_LOAD) {
      uptr cur_beg = info->dlpi_addr + phdr->p_vaddr;
      uptr cur_end = cur_beg + phdr->p_memsz;
      cur_module->addAddressRange(cur_beg, cur_end);
    }
  }
  return 0;
}

uptr GetListOfModules(LoadedModule *modules, uptr max_modules,
                      string_predicate_t filter) {
  CHECK(modules);
  DlIteratePhdrData data = {modules, 0, true, max_modules, filter};
  dl_iterate_phdr(dl_iterate_phdr_cb, &data);
  return data.current_n;
}
#endif  // SANITIZER_ANDROID

}  // namespace __sanitizer

#endif  // SANITIZER_LINUX
