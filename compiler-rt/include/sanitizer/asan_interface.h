//===-- sanitizer/asan_interface.h ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer.
//
// Public interface header.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_ASAN_INTERFACE_H
#define SANITIZER_ASAN_INTERFACE_H

#include <sanitizer/common_interface_defs.h>

#ifdef __cplusplus
extern "C" {
#endif
  // Marks memory region [addr, addr+size) as unaddressable.
  // This memory must be previously allocated by the user program. Accessing
  // addresses in this region from instrumented code is forbidden until
  // this region is unpoisoned. This function is not guaranteed to poison
  // the whole region - it may poison only subregion of [addr, addr+size) due
  // to ASan alignment restrictions.
  // Method is NOT thread-safe in the sense that no two threads can
  // (un)poison memory in the same memory region simultaneously.
  void __asan_poison_memory_region(void const volatile *addr, size_t size);
  // Marks memory region [addr, addr+size) as addressable.
  // This memory must be previously allocated by the user program. Accessing
  // addresses in this region is allowed until this region is poisoned again.
  // This function may unpoison a superregion of [addr, addr+size) due to
  // ASan alignment restrictions.
  // Method is NOT thread-safe in the sense that no two threads can
  // (un)poison memory in the same memory region simultaneously.
  void __asan_unpoison_memory_region(void const volatile *addr, size_t size);

// User code should use macros instead of functions.
#if __has_feature(address_sanitizer) || defined(__SANITIZE_ADDRESS__)
#define ASAN_POISON_MEMORY_REGION(addr, size) \
  __asan_poison_memory_region((addr), (size))
#define ASAN_UNPOISON_MEMORY_REGION(addr, size) \
  __asan_unpoison_memory_region((addr), (size))
#else
#define ASAN_POISON_MEMORY_REGION(addr, size) \
  ((void)(addr), (void)(size))
#define ASAN_UNPOISON_MEMORY_REGION(addr, size) \
  ((void)(addr), (void)(size))
#endif

  // Returns 1 if addr is poisoned (i.e. 1-byte read/write access to this
  // address will result in error report from AddressSanitizer).
  // Otherwise returns 0.
  int __asan_address_is_poisoned(void const volatile *addr);

  // If at least on byte in [beg, beg+size) is poisoned, return the address
  // of the first such byte. Otherwise return 0.
  void *__asan_region_is_poisoned(void *beg, size_t size);

  // Print the description of addr (useful when debugging in gdb).
  void __asan_describe_address(void *addr);

  // This is an internal function that is called to report an error.
  // However it is still a part of the interface because users may want to
  // set a breakpoint on this function in a debugger.
  void __asan_report_error(void *pc, void *bp, void *sp,
                           void *addr, int is_write, size_t access_size);

  // Sets the exit code to use when reporting an error.
  // Returns the old value.
  int __asan_set_error_exit_code(int exit_code);

  // Sets the callback to be called right before death on error.
  // Passing 0 will unset the callback.
  void __asan_set_death_callback(void (*callback)(void));

  void __asan_set_error_report_callback(void (*callback)(const char*));

  // User may provide function that would be called right when ASan detects
  // an error. This can be used to notice cases when ASan detects an error, but
  // the program crashes before ASan report is printed.
  void __asan_on_error();

  // Returns the estimated number of bytes that will be reserved by allocator
  // for request of "size" bytes. If ASan allocator can't allocate that much
  // memory, returns the maximal possible allocation size, otherwise returns
  // "size".
  /* DEPRECATED: Use __sanitizer_get_estimated_allocated_size instead. */
  size_t __asan_get_estimated_allocated_size(size_t size);

  // Returns 1 if p was returned by the ASan allocator and is not yet freed.
  // Otherwise returns 0.
  /* DEPRECATED: Use __sanitizer_get_ownership instead. */
  int __asan_get_ownership(const void *p);

  // Returns the number of bytes reserved for the pointer p.
  // Requires (get_ownership(p) == true) or (p == 0).
  /* DEPRECATED: Use __sanitizer_get_allocated_size instead. */
  size_t __asan_get_allocated_size(const void *p);

  // Number of bytes, allocated and not yet freed by the application.
  /* DEPRECATED: Use __sanitizer_get_current_allocated_bytes instead. */
  size_t __asan_get_current_allocated_bytes();

  // Number of bytes, mmaped by asan allocator to fulfill allocation requests.
  // Generally, for request of X bytes, allocator can reserve and add to free
  // lists a large number of chunks of size X to use them for future requests.
  // All these chunks count toward the heap size. Currently, allocator never
  // releases memory to OS (instead, it just puts freed chunks to free lists).
  /* DEPRECATED: Use __sanitizer_get_heap_size instead. */
  size_t __asan_get_heap_size();

  // Number of bytes, mmaped by asan allocator, which can be used to fulfill
  // allocation requests. When a user program frees memory chunk, it can first
  // fall into quarantine and will count toward __asan_get_free_bytes() later.
  /* DEPRECATED: Use __sanitizer_get_free_bytes instead. */
  size_t __asan_get_free_bytes();

  // Number of bytes in unmapped pages, that are released to OS. Currently,
  // always returns 0.
  /* DEPRECATED: Use __sanitizer_get_unmapped_bytes instead. */
  size_t __asan_get_unmapped_bytes();

  // Prints accumulated stats to stderr. Used for debugging.
  void __asan_print_accumulated_stats();

  // This function may be optionally provided by user and should return
  // a string containing ASan runtime options. See asan_flags.h for details.
  const char* __asan_default_options();

  // Malloc hooks that may be optionally provided by user.
  // __asan_malloc_hook(ptr, size) is called immediately after
  //   allocation of "size" bytes, which returned "ptr".
  // __asan_free_hook(ptr) is called immediately before
  //   deallocation of "ptr".
  /* DEPRECATED: Use __sanitizer_malloc_hook / __sanitizer_free_hook instead. */
  void __asan_malloc_hook(void *ptr, size_t size);
  void __asan_free_hook(void *ptr);

  // The following 2 functions facilitate garbage collection in presence of
  // asan's fake stack.

  // Returns an opaque handler to be used later in __asan_addr_is_in_fake_stack.
  // Returns NULL if the current thread does not have a fake stack.
  void *__asan_get_current_fake_stack();

  // If fake_stack is non-NULL and addr belongs to a fake frame in
  // fake_stack, returns the address on real stack that corresponds to
  // the fake frame and sets beg/end to the boundaries of this fake frame.
  // Otherwise returns NULL and does not touch beg/end.
  // If beg/end are NULL, they are not touched.
  // This function may be called from a thread other than the owner of
  // fake_stack, but the owner thread need to be alive.
  void *__asan_addr_is_in_fake_stack(void *fake_stack, void *addr, void **beg,
                                     void **end);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SANITIZER_ASAN_INTERFACE_H
