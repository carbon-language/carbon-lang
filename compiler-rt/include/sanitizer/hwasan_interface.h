//===-- sanitizer/asan_interface.h ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of HWAddressSanitizer.
//
// Public interface header.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_HWASAN_INTERFACE_H
#define SANITIZER_HWASAN_INTERFACE_H

#include <sanitizer/common_interface_defs.h>

#ifdef __cplusplus
extern "C" {
#endif
  // Initialize shadow but not the rest of the runtime.
  // Does not call libc unless there is an error.
  // Can be called multiple times, or not at all (in which case shadow will
  // be initialized in compiler-inserted __hwasan_init() call).
  void __hwasan_shadow_init(void);

  // This function may be optionally provided by user and should return
  // a string containing HWASan runtime options. See asan_flags.h for details.
  const char* __hwasan_default_options(void);

  void __hwasan_enable_allocator_tagging(void);
  void __hwasan_disable_allocator_tagging(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SANITIZER_HWASAN_INTERFACE_H
