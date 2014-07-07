//===-- allocator_interface.h ---------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Public interface header for allocator used in sanitizers (ASan/TSan/MSan).
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_ALLOCATOR_INTERFACE_H
#define SANITIZER_ALLOCATOR_INTERFACE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif
  /* Returns the estimated number of bytes that will be reserved by allocator
     for request of "size" bytes. If allocator can't allocate that much
     memory, returns the maximal possible allocation size, otherwise returns
     "size". */
  size_t __sanitizer_get_estimated_allocated_size(size_t size);

  /* Returns true if p was returned by the allocator and
     is not yet freed. */
  int __sanitizer_get_ownership(const volatile void *p);

  /* Returns the number of bytes reserved for the pointer p.
     Requires (get_ownership(p) == true) or (p == 0). */
  size_t __sanitizer_get_allocated_size(const volatile void *p);

  /* Number of bytes, allocated and not yet freed by the application. */
  size_t __sanitizer_get_current_allocated_bytes();

  /* Number of bytes, mmaped by the allocator to fulfill allocation requests.
     Generally, for request of X bytes, allocator can reserve and add to free
     lists a large number of chunks of size X to use them for future requests.
     All these chunks count toward the heap size. Currently, allocator never
     releases memory to OS (instead, it just puts freed chunks to free
     lists). */
  size_t __sanitizer_get_heap_size();

  /* Number of bytes, mmaped by the allocator, which can be used to fulfill
     allocation requests. When a user program frees memory chunk, it can first
     fall into quarantine and will count toward __sanitizer_get_free_bytes()
     later. */
  size_t __sanitizer_get_free_bytes();

  /* Number of bytes in unmapped pages, that are released to OS. Currently,
     always returns 0. */
  size_t __sanitizer_get_unmapped_bytes();

  /* Malloc hooks that may be optionally provided by user.
     __sanitizer_malloc_hook(ptr, size) is called immediately after
       allocation of "size" bytes, which returned "ptr".
     __sanitizer_free_hook(ptr) is called immediately before
       deallocation of "ptr". */
  void __sanitizer_malloc_hook(const volatile void *ptr, size_t size);
  void __sanitizer_free_hook(const volatile void *ptr);
#ifdef __cplusplus
}  // extern "C"
#endif

#endif
