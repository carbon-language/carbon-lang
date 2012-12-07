//===-- sanitizer/common_interface_defs.h -----------------------*- C++ -*-===//
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

#ifndef SANITIZER_COMMON_INTERFACE_DEFS_H
#define SANITIZER_COMMON_INTERFACE_DEFS_H

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

#ifdef __linux__
# define SANITIZER_SUPPORTS_WEAK_HOOKS 1
#else
# define SANITIZER_SUPPORTS_WEAK_HOOKS 0
#endif

// __has_feature
#if !defined(__has_feature)
# define __has_feature(x) 0
#endif

// For portability reasons we do not include stddef.h, stdint.h or any other
// system header, but we do need some basic types that are not defined
// in a portable way by the language itself.
namespace __sanitizer {

#if defined(_WIN64)
// 64-bit Windows uses LLP64 data model.
typedef unsigned long long uptr;  // NOLINT
typedef signed   long long sptr;  // NOLINT
#else
typedef unsigned long uptr;  // NOLINT
typedef signed   long sptr;  // NOLINT
#endif  // defined(_WIN64)
#if defined(__x86_64__)
// Since x32 uses ILP32 data model in 64-bit hardware mode,  we must use
// 64-bit pointer to unwind stack frame.
typedef unsigned long long uhwptr;  // NOLINT
#else
typedef uptr uhwptr;   // NOLINT
#endif
typedef unsigned char u8;
typedef unsigned short u16;  // NOLINT
typedef unsigned int u32;
typedef unsigned long long u64;  // NOLINT
typedef signed   char s8;
typedef signed   short s16;  // NOLINT
typedef signed   int s32;
typedef signed   long long s64;  // NOLINT

}  // namespace __sanitizer

extern "C" {
  // Tell the tools to write their reports to "path.<pid>" instead of stderr.
  void __sanitizer_set_report_path(const char *path)
      SANITIZER_INTERFACE_ATTRIBUTE;

  // Tell the tools to write their reports to given file descriptor instead of
  // stderr.
  void __sanitizer_set_report_fd(int fd)
      SANITIZER_INTERFACE_ATTRIBUTE;
}  // extern "C"

#endif  // SANITIZER_COMMON_INTERFACE_DEFS_H
