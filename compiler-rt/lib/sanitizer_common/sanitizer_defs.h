//===-- sanitizer_defs.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_DEFS_H
#define SANITIZER_DEFS_H

// ----------- ATTENTION -------------
// This header should NOT include any other headers to avoid portability issues.

#if defined(_WIN32)
// FIXME find out what we need on Windows. __declspec(dllexport) ?
#define SANITIZER_INTERFACE_FUNCTION_ATTRIBUTE
#define SANITIZER_WEAK_ATTRIBUTE
#else
#define SANITIZER_INTERFACE_FUNCTION_ATTRIBUTE \
  __attribute__((visibility("default")))
#define SANITIZER_WEAK_ATTRIBUTE __attribute__((weak));
#endif

// For portability reasons we do not include stddef.h, stdint.h or any other
// system header, but we do need some basic types that are not defined
// in a portable way by the language itself.
typedef unsigned long uptr;  // Unsigned integer of the same size as a pointer.
// FIXME: add u64, u32, etc.

#endif  // SANITIZER_DEFS_H
