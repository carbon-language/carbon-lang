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

#include "asan_internal.h"

#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

extern char _DYNAMIC[];

namespace __asan {

void *AsanDoesNotSupportStaticLinkage() {
  // This will fail to link with -static.
  return &_DYNAMIC;
}

#ifdef ANDROID
#define SYS_mmap2 __NR_mmap2
#define SYS_write __NR_write
#endif

void *asan_mmap(void *addr, size_t length, int prot, int flags,
                int fd, uint64_t offset) {
# if __WORDSIZE == 64
  return (void *)syscall(SYS_mmap, addr, length, prot, flags, fd, offset);
# else
  return (void *)syscall(SYS_mmap2, addr, length, prot, flags, fd, offset);
# endif
}

ssize_t asan_write(int fd, const void *buf, size_t count) {
  return (ssize_t)syscall(SYS_write, fd, buf, count);
}

}  // namespace __asan
