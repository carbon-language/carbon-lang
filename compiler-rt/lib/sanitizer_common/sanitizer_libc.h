//===-- sanitizer_libc.h ----------------------------------------*- C++ -*-===//
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
// These tools can not use some of the libc functions directly because those
// functions are intercepted. Instead, we implement a tiny subset of libc here.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_LIBC_H
#define SANITIZER_LIBC_H

#include "sanitizer_internal_defs.h"

namespace __sanitizer {

void MiniLibcStub();

// internal_X() is a custom implementation of X() for use in RTL.

// String functions
void *internal_memchr(const void *s, int c, uptr n);
void *internal_memcpy(void *dest, const void *src, uptr n);
int internal_strcmp(const char *s1, const char *s2);
char *internal_strdup(const char *s);
uptr internal_strlen(const char *s);
char *internal_strncpy(char *dst, const char *src, uptr n);

// Memory
void *internal_mmap(void *addr, uptr length, int prot, int flags,
                    int fd, u64 offset);
int internal_munmap(void *addr, uptr length);

// I/O
typedef int fd_t;
const fd_t kInvalidFd = -1;
int internal_close(fd_t fd);
fd_t internal_open(const char *filename, bool write);
uptr internal_read(fd_t fd, void *buf, uptr count);
uptr internal_write(fd_t fd, const void *buf, uptr count);
uptr internal_filesize(fd_t fd);  // -1 on error.
int internal_dup2(int oldfd, int newfd);
int internal_sscanf(const char *str, const char *format, ...);

}  // namespace __sanitizer

#endif  // SANITIZER_LIBC_H
