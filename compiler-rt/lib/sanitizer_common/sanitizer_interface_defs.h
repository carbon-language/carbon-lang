//===-- sanitizer_interface_defs.h -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer.
// It contains basic macro and types.
// NOTE: This file may be included into user code.
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_INTERFACE_DEFS_H
#define SANITIZER_INTERFACE_DEFS_H

// ----------- ATTENTION -------------
// This header should NOT include any other headers to avoid portability issues.

#if defined(_WIN32)
// FIXME find out what we need on Windows. __declspec(dllexport) ?
# define SANITIZER_INTERFACE_ATTRIBUTE
# define SANITIZER_WEAK_ATTRIBUTE
#elif defined(SANITIZER_GO)
# define SANITIZER_INTERFACE_ATTRIBUTE
# define SANITIZER_WEAK_ATTRIBUTE
#else
# define SANITIZER_INTERFACE_ATTRIBUTE __attribute__((visibility("default")))
# define SANITIZER_WEAK_ATTRIBUTE  __attribute__((weak))
#endif

// __has_feature
#if !defined(__has_feature)
# define __has_feature(x) 0
#endif

// For portability reasons we do not include stddef.h, stdint.h or any other
// system header, but we do need some basic types that are not defined
// in a portable way by the language itself.
namespace __sanitizer {

typedef unsigned long uptr;  // NOLINT
typedef signed   long sptr;  // NOLINT
typedef unsigned char u8;
typedef unsigned short u16;  // NOLINT
typedef unsigned int u32;
typedef unsigned long long u64;  // NOLINT
typedef signed   char s8;
typedef signed   short s16;  // NOLINT
typedef signed   int s32;
typedef signed   long long s64;  // NOLINT

}  // namespace __sanitizer

#endif  // SANITIZER_INTERFACE_DEFS_H
