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
  /* Returns a string describing a stack origin.
     Return NULL if the origin is invalid, or is not a stack origin. */
  const char *__msan_get_origin_descr_if_stack(uint32_t id);

  /* Set raw origin for the memory range. */
  void __msan_set_origin(const volatile void *a, size_t size, uint32_t origin);

  /* Get raw origin for an address. */
  uint32_t __msan_get_origin(const volatile void *a);

  /* Returns non-zero if tracking origins. */
  int __msan_get_track_origins();

  /* Returns the origin id of the latest UMR in the calling thread. */
  uint32_t __msan_get_umr_origin();

  /* Make memory region fully initialized (without changing its contents). */
  void __msan_unpoison(const volatile void *a, size_t size);

  /* Make a null-terminated string fully initialized (without changing its
     contents). */
  void __msan_unpoison_string(const volatile char *a);

  /* Make memory region fully uninitialized (without changing its contents). */
  void __msan_poison(const volatile void *a, size_t size);

  /* Make memory region partially uninitialized (without changing its contents).
   */
  void __msan_partial_poison(const volatile void *data, void *shadow,
                             size_t size);

  /* Returns the offset of the first (at least partially) poisoned byte in the
     memory range, or -1 if the whole range is good. */
  intptr_t __msan_test_shadow(const volatile void *x, size_t size);

  /* Checks that memory range is fully initialized, and reports an error if it
   * is not. */
  void __msan_check_mem_is_initialized(const volatile void *x, size_t size);

  /* Set exit code when error(s) were detected.
     Value of 0 means don't change the program exit code. */
  void __msan_set_exit_code(int exit_code);

  /* For testing:
     __msan_set_expect_umr(1);
     ... some buggy code ...
     __msan_set_expect_umr(0);
     The last line will verify that a UMR happened. */
  void __msan_set_expect_umr(int expect_umr);

  /* Change the value of keep_going flag. Non-zero value means don't terminate
     program execution when an error is detected. This will not affect error in
     modules that were compiled without the corresponding compiler flag. */
  void __msan_set_keep_going(int keep_going);

  /* Print shadow and origin for the memory range to stderr in a human-readable
     format. */
  void __msan_print_shadow(const volatile void *x, size_t size);

  /* Print shadow for the memory range to stderr in a minimalistic
     human-readable format. */
  void __msan_dump_shadow(const volatile void *x, size_t size);

  /* Returns true if running under a dynamic tool (DynamoRio-based). */
  int  __msan_has_dynamic_component();

  /* Tell MSan about newly allocated memory (ex.: custom allocator).
     Memory will be marked uninitialized, with origin at the call site. */
  void __msan_allocated_memory(const volatile void* data, size_t size);

  /* This function may be optionally provided by user and should return
     a string containing Msan runtime options. See msan_flags.h for details. */
  const char* __msan_default_options();

  // Sets the callback to be called right before death on error.
  // Passing 0 will unset the callback.
  void __msan_set_death_callback(void (*callback)(void));

  /***********************************/
  /* Allocator statistics interface. */

  /* Returns the estimated number of bytes that will be reserved by allocator
     for request of "size" bytes. If Msan allocator can't allocate that much
     memory, returns the maximal possible allocation size, otherwise returns
     "size". */
  size_t __msan_get_estimated_allocated_size(size_t size);

  /* Returns true if p was returned by the Msan allocator and
     is not yet freed. */
  int __msan_get_ownership(const volatile void *p);

  /* Returns the number of bytes reserved for the pointer p.
     Requires (get_ownership(p) == true) or (p == 0). */
  size_t __msan_get_allocated_size(const volatile void *p);

  /* Number of bytes, allocated and not yet freed by the application. */
  size_t __msan_get_current_allocated_bytes();

  /* Number of bytes, mmaped by msan allocator to fulfill allocation requests.
     Generally, for request of X bytes, allocator can reserve and add to free
     lists a large number of chunks of size X to use them for future requests.
     All these chunks count toward the heap size. Currently, allocator never
     releases memory to OS (instead, it just puts freed chunks to free
     lists). */
  size_t __msan_get_heap_size();

  /* Number of bytes, mmaped by msan allocator, which can be used to fulfill
     allocation requests. When a user program frees memory chunk, it can first
     fall into quarantine and will count toward __msan_get_free_bytes()
     later. */
  size_t __msan_get_free_bytes();

  /* Number of bytes in unmapped pages, that are released to OS. Currently,
     always returns 0. */
  size_t __msan_get_unmapped_bytes();

  /* Malloc hooks that may be optionally provided by user.
     __msan_malloc_hook(ptr, size) is called immediately after
       allocation of "size" bytes, which returned "ptr".
     __msan_free_hook(ptr) is called immediately before
       deallocation of "ptr". */
  void __msan_malloc_hook(const volatile void *ptr, size_t size);
  void __msan_free_hook(const volatile void *ptr);
#ifdef __cplusplus
}  // extern "C"
#endif

#endif
