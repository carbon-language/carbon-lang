//===-- msan_interface.h --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of MemorySanitizer.
//
// Public interface header.
//===----------------------------------------------------------------------===//
#ifndef MSAN_INTERFACE_H
#define MSAN_INTERFACE_H

#include <sanitizer/common_interface_defs.h>

using __sanitizer::uptr;
using __sanitizer::sptr;
using __sanitizer::u32;

#ifdef __cplusplus
extern "C" {
#endif

#if __has_feature(memory_sanitizer)
  /* Returns a string describing a stack origin.
     Return NULL if the origin is invalid, or is not a stack origin. */
  SANITIZER_INTERFACE_ATTRIBUTE
  const char *__msan_get_origin_descr_if_stack(u32 id);


  /* Set raw origin for the memory range. */
  SANITIZER_INTERFACE_ATTRIBUTE
  void __msan_set_origin(void *a, uptr size, u32 origin);

  /* Get raw origin for an address. */
  SANITIZER_INTERFACE_ATTRIBUTE
  u32 __msan_get_origin(void *a);

  /* Returns non-zero if tracking origins. */
  SANITIZER_INTERFACE_ATTRIBUTE
  int __msan_get_track_origins();

  /* Returns the origin id of the latest UMR in the calling thread. */
  SANITIZER_INTERFACE_ATTRIBUTE
  u32 __msan_get_umr_origin();

  /* Make memory region fully initialized (without changing its contents). */
  SANITIZER_INTERFACE_ATTRIBUTE
  void __msan_unpoison(void *a, uptr size);

  /* Make memory region fully uninitialized (without changing its contents). */
  SANITIZER_INTERFACE_ATTRIBUTE
  void __msan_poison(void *a, uptr size);

  /* Make memory region partially uninitialized (without changing its contents).
   */
  SANITIZER_INTERFACE_ATTRIBUTE
  void __msan_partial_poison(void* data, void* shadow, uptr size);

  /* Returns the offset of the first (at least partially) poisoned byte in the
     memory range, or -1 if the whole range is good. */
  SANITIZER_INTERFACE_ATTRIBUTE
  sptr __msan_test_shadow(const void *x, uptr size);

  /* Set exit code when error(s) were detected.
     Value of 0 means don't change the program exit code. */
  SANITIZER_INTERFACE_ATTRIBUTE
  void __msan_set_exit_code(int exit_code);

  /* For testing:
     __msan_set_expect_umr(1);
     ... some buggy code ...
     __msan_set_expect_umr(0);
     The last line will verify that a UMR happened. */
  SANITIZER_INTERFACE_ATTRIBUTE
  void __msan_set_expect_umr(int expect_umr);

  /* Print shadow and origin for the memory range to stdout in a human-readable
     format. */
  SANITIZER_INTERFACE_ATTRIBUTE
  void __msan_print_shadow(const void *x, uptr size);

  /* Print current function arguments shadow and origin to stdout in a
     human-readable format. */
  SANITIZER_INTERFACE_ATTRIBUTE
  void __msan_print_param_shadow();

  /* Returns true if running under a dynamic tool (DynamoRio-based). */
  SANITIZER_INTERFACE_ATTRIBUTE
  int  __msan_has_dynamic_component();

  /* Tell MSan about newly allocated memory (ex.: custom allocator).
     Memory will be marked uninitialized, with origin at the call site. */
  SANITIZER_INTERFACE_ATTRIBUTE
  void __msan_allocated_memory(void* data, uptr size);

#else  // __has_feature(memory_sanitizer)

#define __msan_get_origin_descr_if_stack(u32 id) ((const char*)0)
#define __msan_set_origin(void *a, uptr size, u32 origin)
#define __msan_get_origin(void *a) ((u32)-1)
#define __msan_get_track_origins() (0)
#define __msan_get_umr_origin() ((u32)-1)
#define __msan_unpoison(void *a, uptr size)
#define __msan_poison(void *a, uptr size)
#define __msan_partial_poison(void* data, void* shadow, uptr size)
#define __msan_test_shadow(const void *x, uptr size) ((sptr)-1)
#define __msan_set_exit_code(int exit_code)
#define __msan_set_expect_umr(int expect_umr)
#define __msan_print_shadow(const void *x, uptr size)
#define __msan_print_param_shadow()
#define __msan_has_dynamic_component() (0)
#define __msan_allocated_memory(data, size)

#endif   // __has_feature(memory_sanitizer)

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
