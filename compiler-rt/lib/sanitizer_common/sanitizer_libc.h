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
// NOTE: This file may be included into user code.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_LIBC_H
#define SANITIZER_LIBC_H

// ----------- ATTENTION -------------
// This header should NOT include any other headers from sanitizer runtime.
#include "sanitizer/common_interface_defs.h"

namespace __sanitizer {

// internal_X() is a custom implementation of X() for use in RTL.

// String functions
s64 internal_atoll(const char *nptr);
void *internal_memchr(const void *s, int c, uptr n);
int internal_memcmp(const void* s1, const void* s2, uptr n);
void *internal_memcpy(void *dest, const void *src, uptr n);
void *internal_memmove(void *dest, const void *src, uptr n);
// Should not be used in performance-critical places.
void *internal_memset(void *s, int c, uptr n);
char* internal_strchr(const char *s, int c);
int internal_strcmp(const char *s1, const char *s2);
uptr internal_strcspn(const char *s, const char *reject);
char *internal_strdup(const char *s);
uptr internal_strlen(const char *s);
char *internal_strncat(char *dst, const char *src, uptr n);
int internal_strncmp(const char *s1, const char *s2, uptr n);
char *internal_strncpy(char *dst, const char *src, uptr n);
uptr internal_strnlen(const char *s, uptr maxlen);
char *internal_strrchr(const char *s, int c);
// This is O(N^2), but we are not using it in hot places.
char *internal_strstr(const char *haystack, const char *needle);
// Works only for base=10 and doesn't set errno.
s64 internal_simple_strtoll(const char *nptr, char **endptr, int base);

// Memory
void *internal_mmap(void *addr, uptr length, int prot, int flags,
                    int fd, u64 offset);
int internal_munmap(void *addr, uptr length);

// I/O
typedef int fd_t;
const fd_t kInvalidFd = -1;
const fd_t kStdinFd = 0;
const fd_t kStdoutFd = 1;
const fd_t kStderrFd = 2;
int internal_close(fd_t fd);
fd_t internal_open(const char *filename, bool write);
uptr internal_read(fd_t fd, void *buf, uptr count);
uptr internal_write(fd_t fd, const void *buf, uptr count);
uptr internal_filesize(fd_t fd);  // -1 on error.
int internal_dup2(int oldfd, int newfd);
uptr internal_readlink(const char *path, char *buf, uptr bufsize);
int internal_snprintf(char *buffer, uptr length, const char *format, ...);

// Threading
int internal_sched_yield();

}  // namespace __sanitizer

#endif  // SANITIZER_LIBC_H
