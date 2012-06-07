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
#include "sanitizer_procmaps.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

namespace __sanitizer {

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
  struct stat st = {};
  if (syscall(__NR_fstat, fd, &st))
    return -1;
  return (uptr)st.st_size;
}

int internal_dup2(int oldfd, int newfd) {
  return syscall(__NR_dup2, oldfd, newfd);
}

// ----------------- ProcessMaps implementation.
ProcessMaps::ProcessMaps() {
  proc_self_maps_buff_len_ =
      ReadFileToBuffer("/proc/self/maps", &proc_self_maps_buff_,
                       &proc_self_maps_buff_mmaped_size_, 1 << 26);
  CHECK(proc_self_maps_buff_len_ > 0);
  // internal_write(2, proc_self_maps_buff_, proc_self_maps_buff_len_);
  Reset();
}

ProcessMaps::~ProcessMaps() {
  UnmapOrDie(proc_self_maps_buff_, proc_self_maps_buff_mmaped_size_);
}

void ProcessMaps::Reset() {
  current_ = proc_self_maps_buff_;
}

bool ProcessMaps::Next(uptr *start, uptr *end, uptr *offset,
                       char filename[], uptr filename_size) {
  char *last = proc_self_maps_buff_ + proc_self_maps_buff_len_;
  if (current_ >= last) return false;
  int consumed = 0;
  char flags[10];
  int major, minor;
  uptr inode;
  uptr dummy;
  if (!start) start = &dummy;
  if (!end) end = &dummy;
  if (!offset) offset = &dummy;
  char *next_line = (char*)internal_memchr(current_, '\n', last - current_);
  if (next_line == 0)
    next_line = last;
  if (internal_sscanf(current_, "%lx-%lx %4s %lx %x:%x %ld %n",
                      start, end, flags, offset, &major, &minor,
                      &inode, &consumed) != 7)
    return false;
  current_ += consumed;
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
