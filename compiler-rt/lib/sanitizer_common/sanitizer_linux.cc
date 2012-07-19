//===-- sanitizer_linux.cc ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries and implements linux-specific functions from
// sanitizer_libc.h.
//===----------------------------------------------------------------------===//
#ifdef __linux__

#include "sanitizer_common.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_libc.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_procmaps.h"
#include "sanitizer_symbolizer.h"

#include <elf.h>
#include <fcntl.h>
#include <link.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

namespace __sanitizer {

// --------------- sanitizer_libc.h
void *internal_mmap(void *addr, uptr length, int prot, int flags,
                    int fd, u64 offset) {
#if __WORDSIZE == 64
  return (void *)syscall(__NR_mmap, addr, length, prot, flags, fd, offset);
#else
  return (void *)syscall(__NR_mmap2, addr, length, prot, flags, fd, offset);
#endif
}

int internal_munmap(void *addr, uptr length) {
  return syscall(__NR_munmap, addr, length);
}

int internal_close(fd_t fd) {
  return syscall(__NR_close, fd);
}

fd_t internal_open(const char *filename, bool write) {
  return syscall(__NR_open, filename,
      write ? O_WRONLY | O_CREAT /*| O_CLOEXEC*/ : O_RDONLY, 0660);
}

uptr internal_read(fd_t fd, void *buf, uptr count) {
  return (uptr)syscall(__NR_read, fd, buf, count);
}

uptr internal_write(fd_t fd, const void *buf, uptr count) {
  return (uptr)syscall(__NR_write, fd, buf, count);
}

uptr internal_filesize(fd_t fd) {
#if __WORDSIZE == 64
  struct stat st;
  if (syscall(__NR_fstat, fd, &st))
    return -1;
#else
  struct stat64 st;
  if (syscall(__NR_fstat64, fd, &st))
    return -1;
#endif
  return (uptr)st.st_size;
}

int internal_dup2(int oldfd, int newfd) {
  return syscall(__NR_dup2, oldfd, newfd);
}

int internal_sched_yield() {
  return syscall(__NR_sched_yield);
}

// ----------------- sanitizer_common.h
void GetThreadStackTopAndBottom(bool at_initialization, uptr *stack_top,
                                uptr *stack_bottom) {
  static const uptr kMaxThreadStackSize = 256 * (1 << 20);  // 256M
  CHECK(stack_top);
  CHECK(stack_bottom);
  if (at_initialization) {
    // This is the main thread. Libpthread may not be initialized yet.
    struct rlimit rl;
    CHECK_EQ(getrlimit(RLIMIT_STACK, &rl), 0);

    // Find the mapping that contains a stack variable.
    ProcessMaps proc_maps;
    uptr start, end, offset;
    uptr prev_end = 0;
    while (proc_maps.Next(&start, &end, &offset, 0, 0)) {
      if ((uptr)&rl < end)
        break;
      prev_end = end;
    }
    CHECK((uptr)&rl >= start && (uptr)&rl < end);

    // Get stacksize from rlimit, but clip it so that it does not overlap
    // with other mappings.
    uptr stacksize = rl.rlim_cur;
    if (stacksize > end - prev_end)
      stacksize = end - prev_end;
    // When running with unlimited stack size, we still want to set some limit.
    // The unlimited stack size is caused by 'ulimit -s unlimited'.
    // Also, for some reason, GNU make spawns subprocesses with unlimited stack.
    if (stacksize > kMaxThreadStackSize)
      stacksize = kMaxThreadStackSize;
    *stack_top = end;
    *stack_bottom = end - stacksize;
    return;
  }
  pthread_attr_t attr;
  CHECK_EQ(pthread_getattr_np(pthread_self(), &attr), 0);
  uptr stacksize = 0;
  void *stackaddr = 0;
  pthread_attr_getstack(&attr, &stackaddr, (size_t*)&stacksize);
  pthread_attr_destroy(&attr);

  *stack_top = (uptr)stackaddr + stacksize;
  *stack_bottom = (uptr)stackaddr;
  CHECK(stacksize < kMaxThreadStackSize);  // Sanity check.
}

// Like getenv, but reads env directly from /proc and does not use libc.
// This function should be called first inside __asan_init.
const char *GetEnv(const char *name) {
  static char *environ;
  static uptr len;
  static bool inited;
  if (!inited) {
    inited = true;
    uptr environ_size;
    len = ReadFileToBuffer("/proc/self/environ",
                           &environ, &environ_size, 1 << 26);
  }
  if (!environ || len == 0) return 0;
  uptr namelen = internal_strlen(name);
  const char *p = environ;
  while (*p != '\0') {  // will happen at the \0\0 that terminates the buffer
    // proc file has the format NAME=value\0NAME=value\0NAME=value\0...
    const char* endp =
        (char*)internal_memchr(p, '\0', len - (p - environ));
    if (endp == 0)  // this entry isn't NUL terminated
      return 0;
    else if (!internal_memcmp(p, name, namelen) && p[namelen] == '=')  // Match.
      return p + namelen + 1;  // point after =
    p = endp + 1;
  }
  return 0;  // Not found.
}

// ------------------ sanitizer_symbolizer.h
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

// ----------------- sanitizer_procmaps.h
ProcessMaps::ProcessMaps() {
  proc_self_maps_buff_len_ =
      ReadFileToBuffer("/proc/self/maps", &proc_self_maps_buff_,
                       &proc_self_maps_buff_mmaped_size_, 1 << 26);
  CHECK_GT(proc_self_maps_buff_len_, 0);
  // internal_write(2, proc_self_maps_buff_, proc_self_maps_buff_len_);
  Reset();
}

ProcessMaps::~ProcessMaps() {
  UnmapOrDie(proc_self_maps_buff_, proc_self_maps_buff_mmaped_size_);
}

void ProcessMaps::Reset() {
  current_ = proc_self_maps_buff_;
}

// Parse a hex value in str and update str.
static uptr ParseHex(char **str) {
  uptr x = 0;
  char *s;
  for (s = *str; ; s++) {
    char c = *s;
    uptr v = 0;
    if (c >= '0' && c <= '9')
      v = c - '0';
    else if (c >= 'a' && c <= 'f')
      v = c - 'a' + 10;
    else if (c >= 'A' && c <= 'F')
      v = c - 'A' + 10;
    else
      break;
    x = x * 16 + v;
  }
  *str = s;
  return x;
}

static bool IsOnOf(char c, char c1, char c2) {
  return c == c1 || c == c2;
}

static bool IsDecimal(char c) {
  return c >= '0' && c <= '9';
}

bool ProcessMaps::Next(uptr *start, uptr *end, uptr *offset,
                       char filename[], uptr filename_size) {
  char *last = proc_self_maps_buff_ + proc_self_maps_buff_len_;
  if (current_ >= last) return false;
  uptr dummy;
  if (!start) start = &dummy;
  if (!end) end = &dummy;
  if (!offset) offset = &dummy;
  char *next_line = (char*)internal_memchr(current_, '\n', last - current_);
  if (next_line == 0)
    next_line = last;
  // Example: 08048000-08056000 r-xp 00000000 03:0c 64593   /foo/bar
  *start = ParseHex(&current_);
  CHECK_EQ(*current_++, '-');
  *end = ParseHex(&current_);
  CHECK_EQ(*current_++, ' ');
  CHECK(IsOnOf(*current_++, '-', 'r'));
  CHECK(IsOnOf(*current_++, '-', 'w'));
  CHECK(IsOnOf(*current_++, '-', 'x'));
  CHECK(IsOnOf(*current_++, 's', 'p'));
  CHECK_EQ(*current_++, ' ');
  *offset = ParseHex(&current_);
  CHECK_EQ(*current_++, ' ');
  ParseHex(&current_);
  CHECK_EQ(*current_++, ':');
  ParseHex(&current_);
  CHECK_EQ(*current_++, ' ');
  while (IsDecimal(*current_))
    current_++;
  CHECK_EQ(*current_++, ' ');
  // Skip spaces.
  while (current_ < next_line && *current_ == ' ')
    current_++;
  // Fill in the filename.
  uptr i = 0;
  while (current_ < next_line) {
    if (filename && i < filename_size - 1)
      filename[i++] = *current_;
    current_++;
  }
  if (filename && i < filename_size)
    filename[i] = 0;
  current_ = next_line + 1;
  return true;
}

// Gets the object name and the offset by walking ProcessMaps.
bool ProcessMaps::GetObjectNameAndOffset(uptr addr, uptr *offset,
                                         char filename[],
                                         uptr filename_size) {
  return IterateForObjectNameAndOffset(addr, offset, filename, filename_size);
}

}  // namespace __sanitizer

#endif  // __linux__
