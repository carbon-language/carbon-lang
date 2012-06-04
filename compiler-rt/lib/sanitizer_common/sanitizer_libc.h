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
//
// We also define several basic types here to avoid using system headers
// as the latter complicate portability of this low-level code.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_LIBC_H
#define SANITIZER_LIBC_H

#include "sanitizer_defs.h"

// No code here yet. Will move more code in the next changes.
namespace __sanitizer {

void MiniLibcStub();

// internal_X() is a custom implementation of X() for use in RTL.
int internal_strcmp(const char *s1, const char *s2);
char *internal_strncpy(char *dst, const char *src, uptr n);

#ifndef _WIN32
void *internal_mmap(void *addr, uptr length, int prot, int flags,
                    int fd, u64 offset);
#endif  // _WIN32

}  // namespace __sanitizer

#endif  // SANITIZER_LIBC_H
