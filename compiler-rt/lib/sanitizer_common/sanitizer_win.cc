//===-- sanitizer_win.cc --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries and implements windows-specific functions from
// sanitizer_libc.h.
//===----------------------------------------------------------------------===//
#ifdef _WIN32
#include <windows.h>

#include "sanitizer_common.h"
#include "sanitizer_libc.h"

namespace __sanitizer {

int GetPid() {
  return GetProcessId(GetCurrentProcess());
}

void *MmapOrDie(uptr size) {
  void *rv = VirtualAlloc(0, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
  if (rv == 0)
    RawWrite("Failed to map!\n");
    Die();
  return rv;
}

void UnmapOrDie(void *addr, uptr size) {
  if (VirtualFree(addr, size, MEM_DECOMMIT) == 0) {
    RawWrite("Failed to unmap!\n");
    Die();
  }
}

void *internal_mmap(void *addr, uptr length, int prot, int flags,
                    int fd, u64 offset) {
  UNIMPLEMENTED();
}

int internal_munmap(void *addr, uptr length) {
  UNIMPLEMENTED();
}

int internal_close(fd_t fd) {
  UNIMPLEMENTED();
}

fd_t internal_open(const char *filename, bool write) {
  UNIMPLEMENTED();
}

uptr internal_read(fd_t fd, void *buf, uptr count) {
  UNIMPLEMENTED();
}

uptr internal_write(fd_t fd, const void *buf, uptr count) {
  if (fd != 2)
    UNIMPLEMENTED();
  HANDLE err = GetStdHandle(STD_ERROR_HANDLE);
  if (err == 0)
    return 0;  // FIXME: this might not work on some apps.
  DWORD ret;
  if (!WriteFile(err, buf, count, &ret, 0))
    return 0;
  return ret;
}

uptr internal_filesize(fd_t fd) {
  UNIMPLEMENTED();
}

int internal_dup2(int oldfd, int newfd) {
  UNIMPLEMENTED();
}

int internal_sscanf(const char *str, const char *format, ...) {
  UNIMPLEMENTED();
}

}  // namespace __sanitizer

#endif  // _WIN32
