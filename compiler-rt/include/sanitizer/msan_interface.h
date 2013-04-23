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

#ifdef __cplusplus
extern "C" {
#endif

#if __has_feature(memory_sanitizer)
  /* Returns a string describing a stack origin.
     Return NULL if the origin is invalid, or is not a stack origin. */
  const char *__msan_get_origin_descr_if_stack(uint32_t id);


  /* Set raw origin for the memory range. */
  void __msan_set_origin(const void *a, size_t size, uint32_t origin);

  /* Get raw origin for an address. */
  uint32_t __msan_get_origin(const void *a);

  /* Returns non-zero if tracking origins. */
  int __msan_get_track_origins();

  /* Returns the origin id of the latest UMR in the calling thread. */
  uint32_t __msan_get_umr_origin();

  /* Make memory region fully initialized (without changing its contents). */
  void __msan_unpoison(const void *a, size_t size);

  /* Make memory region fully uninitialized (without changing its contents). */
  void __msan_poison(const void *a, size_t size);

  /* Make memory region partially uninitialized (without changing its contents).
   */
  void __msan_partial_poison(const void* data, void* shadow, size_t size);

  /* Returns the offset of the first (at least partially) poisoned byte in the
     memory range, or -1 if the whole range is good. */
  intptr_t __msan_test_shadow(const void *x, size_t size);

  /* Set exit code when error(s) were detected.
     Value of 0 means don't change the program exit code. */
  void __msan_set_exit_code(int exit_code);

  /* For testing:
     __msan_set_expect_umr(1);
     ... some buggy code ...
     __msan_set_expect_umr(0);
     The last line will verify that a UMR happened. */
  void __msan_set_expect_umr(int expect_umr);

  /* Print shadow and origin for the memory range to stdout in a human-readable
     format. */
  void __msan_print_shadow(const void *x, size_t size);

  /* Print current function arguments shadow and origin to stdout in a
     human-readable format. */
  void __msan_print_param_shadow();

  /* Returns true if running under a dynamic tool (DynamoRio-based). */
  int  __msan_has_dynamic_component();

  /* Tell MSan about newly allocated memory (ex.: custom allocator).
     Memory will be marked uninitialized, with origin at the call site. */
  void __msan_allocated_memory(const void* data, size_t size);

#else  // __has_feature(memory_sanitizer)

#define __msan_get_origin_descr_if_stack(id) ((const char*)0)
#define __msan_set_origin(a, size, origin)
#define __msan_get_origin(a) ((uint32_t)-1)
#define __msan_get_track_origins() (0)
#define __msan_get_umr_origin() ((uint32_t)-1)
#define __msan_unpoison(a, size)
#define __msan_poison(a, size)
#define __msan_partial_poison(data, shadow, size)
#define __msan_test_shadow(x, size) ((intptr_t)-1)
#define __msan_set_exit_code(exit_code)
#define __msan_set_expect_umr(expect_umr)
#define __msan_print_shadow(x, size)
#define __msan_print_param_shadow()
#define __msan_has_dynamic_component() (0)
#define __msan_allocated_memory(data, size)

#endif   // __has_feature(memory_sanitizer)

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
