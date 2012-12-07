//===-- sanitizer/asan_interface.h ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// This header can be included by the instrumented program to fetch
// data (mostly allocator statistics) from ASan runtime library.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_ASAN_INTERFACE_H
#define SANITIZER_ASAN_INTERFACE_H

#include <sanitizer/common_interface_defs.h>

// ----------- ATTENTION -------------
// This header should NOT include any other headers from ASan runtime.
// All functions in this header are extern "C" and start with __asan_.

using __sanitizer::uptr;

extern "C" {
  // This function should be called at the very beginning of the process,
  // before any instrumented code is executed and before any call to malloc.
  void __asan_init() SANITIZER_INTERFACE_ATTRIBUTE;

  // This function should be called by the instrumented code.
  // 'addr' is the address of a global variable called 'name' of 'size' bytes.
  void __asan_register_global(uptr addr, uptr size, const char *name)
      SANITIZER_INTERFACE_ATTRIBUTE;

  // This structure describes an instrumented global variable.
  struct __asan_global {
    uptr beg;                // The address of the global.
    uptr size;               // The original size of the global.
    uptr size_with_redzone;  // The size with the redzone.
    const char *name;        // Name as a C string.
    uptr has_dynamic_init;   // Non-zero if the global has dynamic initializer.
  };

  // These two functions should be called by the instrumented code.
  // 'globals' is an array of structures describing 'n' globals.
  void __asan_register_globals(__asan_global *globals, uptr n)
      SANITIZER_INTERFACE_ATTRIBUTE;
  void __asan_unregister_globals(__asan_global *globals, uptr n)
      SANITIZER_INTERFACE_ATTRIBUTE;

  // These two functions should be called before and after dynamic initializers
  // run, respectively.  They should be called with parameters describing all
  // dynamically initialized globals defined in the calling TU.
  void __asan_before_dynamic_init(uptr first_addr, uptr last_addr)
      SANITIZER_INTERFACE_ATTRIBUTE;
  void __asan_after_dynamic_init()
      SANITIZER_INTERFACE_ATTRIBUTE;

  // These two functions are used by the instrumented code in the
  // use-after-return mode. __asan_stack_malloc allocates size bytes of
  // fake stack and __asan_stack_free poisons it. real_stack is a pointer to
  // the real stack region.
  uptr __asan_stack_malloc(uptr size, uptr real_stack)
      SANITIZER_INTERFACE_ATTRIBUTE;
  void __asan_stack_free(uptr ptr, uptr size, uptr real_stack)
      SANITIZER_INTERFACE_ATTRIBUTE;

  // These two functions are used by instrumented code in the
  // use-after-scope mode. They mark memory for local variables as
  // unaddressable when they leave scope and addressable before the
  // function exits.
  void __asan_poison_stack_memory(uptr addr, uptr size)
      SANITIZER_INTERFACE_ATTRIBUTE;
  void __asan_unpoison_stack_memory(uptr addr, uptr size)
      SANITIZER_INTERFACE_ATTRIBUTE;

  // Marks memory region [addr, addr+size) as unaddressable.
  // This memory must be previously allocated by the user program. Accessing
  // addresses in this region from instrumented code is forbidden until
  // this region is unpoisoned. This function is not guaranteed to poison
  // the whole region - it may poison only subregion of [addr, addr+size) due
  // to ASan alignment restrictions.
  // Method is NOT thread-safe in the sense that no two threads can
  // (un)poison memory in the same memory region simultaneously.
  void __asan_poison_memory_region(void const volatile *addr, uptr size)
      SANITIZER_INTERFACE_ATTRIBUTE;
  // Marks memory region [addr, addr+size) as addressable.
  // This memory must be previously allocated by the user program. Accessing
  // addresses in this region is allowed until this region is poisoned again.
  // This function may unpoison a superregion of [addr, addr+size) due to
  // ASan alignment restrictions.
  // Method is NOT thread-safe in the sense that no two threads can
  // (un)poison memory in the same memory region simultaneously.
  void __asan_unpoison_memory_region(void const volatile *addr, uptr size)
      SANITIZER_INTERFACE_ATTRIBUTE;

  // Performs cleanup before a NoReturn function. Must be called before things
  // like _exit and execl to avoid false positives on stack.
  void __asan_handle_no_return() SANITIZER_INTERFACE_ATTRIBUTE;

// User code should use macro instead of functions.
#if __has_feature(address_sanitizer)
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

  // Returns true iff addr is poisoned (i.e. 1-byte read/write access to this
  // address will result in error report from AddressSanitizer).
  bool __asan_address_is_poisoned(void const volatile *addr)
      SANITIZER_INTERFACE_ATTRIBUTE;

  // This is an internal function that is called to report an error.
  // However it is still a part of the interface because users may want to
  // set a breakpoint on this function in a debugger.
  void __asan_report_error(uptr pc, uptr bp, uptr sp,
                           uptr addr, bool is_write, uptr access_size)
    SANITIZER_INTERFACE_ATTRIBUTE;

  // Sets the exit code to use when reporting an error.
  // Returns the old value.
  int __asan_set_error_exit_code(int exit_code)
      SANITIZER_INTERFACE_ATTRIBUTE;

  // Sets the callback to be called right before death on error.
  // Passing 0 will unset the callback.
  void __asan_set_death_callback(void (*callback)(void))
      SANITIZER_INTERFACE_ATTRIBUTE;

  void __asan_set_error_report_callback(void (*callback)(const char*))
      SANITIZER_INTERFACE_ATTRIBUTE;

  // User may provide function that would be called right when ASan detects
  // an error. This can be used to notice cases when ASan detects an error, but
  // the program crashes before ASan report is printed.
  /* OPTIONAL */ void __asan_on_error()
      SANITIZER_WEAK_ATTRIBUTE SANITIZER_INTERFACE_ATTRIBUTE;

  // User may provide its own implementation for symbolization function.
  // It should print the description of instruction at address "pc" to
  // "out_buffer". Description should be at most "out_size" bytes long.
  // User-specified function should return true if symbolization was
  // successful.
  /* OPTIONAL */ bool __asan_symbolize(const void *pc, char *out_buffer,
                                       int out_size)
      SANITIZER_WEAK_ATTRIBUTE SANITIZER_INTERFACE_ATTRIBUTE;

  // Returns the estimated number of bytes that will be reserved by allocator
  // for request of "size" bytes. If ASan allocator can't allocate that much
  // memory, returns the maximal possible allocation size, otherwise returns
  // "size".
  uptr __asan_get_estimated_allocated_size(uptr size)
      SANITIZER_INTERFACE_ATTRIBUTE;
  // Returns true if p was returned by the ASan allocator and
  // is not yet freed.
  bool __asan_get_ownership(const void *p)
      SANITIZER_INTERFACE_ATTRIBUTE;
  // Returns the number of bytes reserved for the pointer p.
  // Requires (get_ownership(p) == true) or (p == 0).
  uptr __asan_get_allocated_size(const void *p)
      SANITIZER_INTERFACE_ATTRIBUTE;
  // Number of bytes, allocated and not yet freed by the application.
  uptr __asan_get_current_allocated_bytes()
      SANITIZER_INTERFACE_ATTRIBUTE;
  // Number of bytes, mmaped by asan allocator to fulfill allocation requests.
  // Generally, for request of X bytes, allocator can reserve and add to free
  // lists a large number of chunks of size X to use them for future requests.
  // All these chunks count toward the heap size. Currently, allocator never
  // releases memory to OS (instead, it just puts freed chunks to free lists).
  uptr __asan_get_heap_size()
      SANITIZER_INTERFACE_ATTRIBUTE;
  // Number of bytes, mmaped by asan allocator, which can be used to fulfill
  // allocation requests. When a user program frees memory chunk, it can first
  // fall into quarantine and will count toward __asan_get_free_bytes() later.
  uptr __asan_get_free_bytes()
      SANITIZER_INTERFACE_ATTRIBUTE;
  // Number of bytes in unmapped pages, that are released to OS. Currently,
  // always returns 0.
  uptr __asan_get_unmapped_bytes()
      SANITIZER_INTERFACE_ATTRIBUTE;
  // Prints accumulated stats to stderr. Used for debugging.
  void __asan_print_accumulated_stats()
      SANITIZER_INTERFACE_ATTRIBUTE;

  // This function may be optionally provided by user and should return
  // a string containing ASan runtime options. See asan_flags.h for details.
  /* OPTIONAL */ const char* __asan_default_options()
      SANITIZER_WEAK_ATTRIBUTE SANITIZER_INTERFACE_ATTRIBUTE;

  // Malloc hooks that may be optionally provided by user.
  // __asan_malloc_hook(ptr, size) is called immediately after
  //   allocation of "size" bytes, which returned "ptr".
  // __asan_free_hook(ptr) is called immediately before
  //   deallocation of "ptr".
  /* OPTIONAL */ void __asan_malloc_hook(void *ptr, uptr size)
      SANITIZER_WEAK_ATTRIBUTE SANITIZER_INTERFACE_ATTRIBUTE;
  /* OPTIONAL */ void __asan_free_hook(void *ptr)
      SANITIZER_WEAK_ATTRIBUTE SANITIZER_INTERFACE_ATTRIBUTE;
}  // extern "C"

#endif  // SANITIZER_ASAN_INTERFACE_H
