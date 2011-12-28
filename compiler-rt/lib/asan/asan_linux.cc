//===-- asan_linux.cc -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// Linux-specific details.
//===----------------------------------------------------------------------===//
#ifdef __linux__

#include "asan_internal.h"

#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>

extern char _DYNAMIC[];

namespace __asan {

void *AsanDoesNotSupportStaticLinkage() {
  // This will fail to link with -static.
  return &_DYNAMIC;
}

static void *asan_mmap(void *addr, size_t length, int prot, int flags,
                int fd, uint64_t offset) {
# if __WORDSIZE == 64
  return (void *)syscall(__NR_mmap, addr, length, prot, flags, fd, offset);
# else
  return (void *)syscall(__NR_mmap2, addr, length, prot, flags, fd, offset);
# endif
}

void *AsanMmapSomewhereOrDie(size_t size, const char *mem_type) {
  size = RoundUpTo(size, kPageSize);
  void *res = asan_mmap(0, size,
                        PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANON, -1, 0);
  if (res == (void*)-1) {
    OutOfMemoryMessageAndDie(mem_type, size);
  }
  return res;
}

void *AsanMmapFixedNoReserve(uintptr_t fixed_addr, size_t size) {
  return asan_mmap((void*)fixed_addr, size,
                   PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANON | MAP_FIXED | MAP_NORESERVE,
                   0, 0);
}

void *AsanMmapFixedReserve(uintptr_t fixed_addr, size_t size) {
  return asan_mmap((void*)fixed_addr, size,
                   PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANON | MAP_FIXED,
                   0, 0);
}

void *AsanMprotect(uintptr_t fixed_addr, size_t size) {
  return asan_mmap((void*)fixed_addr, size,
                   PROT_NONE,
                   MAP_PRIVATE | MAP_ANON | MAP_FIXED | MAP_NORESERVE,
                   0, 0);
}

void AsanUnmapOrDie(void *addr, size_t size) {
  if (!addr || !size) return;
  int res = syscall(__NR_munmap, addr, size);
  if (res != 0) {
    Report("Failed to unmap\n");
    ASAN_DIE;
  }
}

ssize_t AsanWrite(int fd, const void *buf, size_t count) {
  return (ssize_t)syscall(__NR_write, fd, buf, count);
}

int AsanOpenReadonly(const char* filename) {
  return open(filename, O_RDONLY);
}

ssize_t AsanRead(int fd, void *buf, size_t count) {
  return (ssize_t)syscall(__NR_read, fd, buf, count);
}

int AsanClose(int fd) {
  return close(fd);
}

}  // namespace __asan

#endif  // __linux__
