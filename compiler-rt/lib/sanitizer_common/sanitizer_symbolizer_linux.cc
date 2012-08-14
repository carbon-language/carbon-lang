//===-- sanitizer_symbolizer_linux.cc -------------------------------------===//
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
#ifdef __linux__
#include "sanitizer_common.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_symbolizer.h"

#include <elf.h>
#include <link.h>
#include <unistd.h>

namespace __sanitizer {

typedef ElfW(Ehdr) Elf_Ehdr;
typedef ElfW(Shdr) Elf_Shdr;
typedef ElfW(Phdr) Elf_Phdr;

bool FindDWARFSection(uptr object_file_addr, const char *section_name,
                      DWARFSection *section) {
  Elf_Ehdr *exe = (Elf_Ehdr*)object_file_addr;
  Elf_Shdr *sections = (Elf_Shdr*)(object_file_addr + exe->e_shoff);
  uptr section_names = object_file_addr +
                       sections[exe->e_shstrndx].sh_offset;
  for (int i = 0; i < exe->e_shnum; i++) {
    Elf_Shdr *current_section = &sections[i];
    const char *current_name = (const char*)section_names +
                               current_section->sh_name;
    if (IsFullNameOfDWARFSection(current_name, section_name)) {
      section->data = (const char*)object_file_addr +
                      current_section->sh_offset;
      section->size = current_section->sh_size;
      return true;
    }
  }
  return false;
}

#ifdef ANDROID
uptr GetListOfModules(ModuleDIContext *modules, uptr max_modules) {
  UNIMPLEMENTED();
}
#else  // ANDROID
struct DlIteratePhdrData {
  ModuleDIContext *modules;
  uptr current_n;
  uptr max_n;
};

static const uptr kMaxPathLength = 512;

static int dl_iterate_phdr_cb(dl_phdr_info *info, size_t size, void *arg) {
  DlIteratePhdrData *data = (DlIteratePhdrData*)arg;
  if (data->current_n == data->max_n)
    return 0;
  char *module_name = 0;
  if (data->current_n == 0) {
    // First module is the binary itself.
    module_name = (char*)InternalAlloc(kMaxPathLength);
    uptr module_name_len = readlink("/proc/self/exe",
                                    module_name, kMaxPathLength);
    CHECK_NE(module_name_len, (uptr)-1);
    CHECK_LT(module_name_len, kMaxPathLength);
    module_name[module_name_len] = '\0';
  } else if (info->dlpi_name) {
    module_name = internal_strdup(info->dlpi_name);
  }
  if (module_name == 0 || module_name[0] == '\0')
    return 0;
  void *mem = &data->modules[data->current_n];
  ModuleDIContext *cur_module = new(mem) ModuleDIContext(module_name,
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
  InternalFree(module_name);
  return 0;
}

uptr GetListOfModules(ModuleDIContext *modules, uptr max_modules) {
  CHECK(modules);
  DlIteratePhdrData data = {modules, 0, max_modules};
  dl_iterate_phdr(dl_iterate_phdr_cb, &data);
  return data.current_n;
}
#endif  // ANDROID

}  // namespace __sanitizer

#endif  // __linux__
