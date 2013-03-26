//===-- asan_interface_internal.h -------------------------------*- C++ -*-===//
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
#ifndef ASAN_INTERFACE_INTERNAL_H
#define ASAN_INTERFACE_INTERNAL_H

#include "sanitizer_common/sanitizer_internal_defs.h"

using __sanitizer::uptr;

extern "C" {
  // This function should be called at the very beginning of the process,
  // before any instrumented code is executed and before any call to malloc.
  // Everytime the asan ABI changes we also change the version number in this
  // name. Objects build with incompatible asan ABI version
  // will not link with run-time.
  // Changes between ABI versions:
  // v1=>v2: added 'module_name' to __asan_global
  // v2=>v3: stack frame description (created by the compiler)
  //         contains the function PC as the 3-rd field (see
  //         DescribeAddressIfStack).
  void __asan_init_v3() SANITIZER_INTERFACE_ATTRIBUTE;
  #define __asan_init __asan_init_v3

  // This structure describes an instrumented global variable.
  struct __asan_global {
    uptr beg;                // The address of the global.
    uptr size;               // The original size of the global.
    uptr size_with_redzone;  // The size with the redzone.
    const char *name;        // Name as a C string.
    const char *module_name; // Module name as a C string. This pointer is a
                             // unique identifier of a module.
    uptr has_dynamic_init;   // Non-zero if the global has dynamic initializer.
  };

  // These two functions should be called by the instrumented code.
  // 'globals' is an array of structures describing 'n' globals.
  void __asan_register_globals(__asan_global *globals, uptr n)
      SANITIZER_INTERFACE_ATTRIBUTE;
  void __asan_unregister_globals(__asan_global *globals, uptr n)
      SANITIZER_INTERFACE_ATTRIBUTE;

  // These two functions should be called before and after dynamic initializers
  // of a single module run, respectively.
  void __asan_before_dynamic_init(const char *module_name)
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

  // Performs cleanup before a NoReturn function. Must be called before things
  // like _exit and execl to avoid false positives on stack.
  void __asan_handle_no_return() SANITIZER_INTERFACE_ATTRIBUTE;

  void __asan_poison_memory_region(void const volatile *addr, uptr size)
      SANITIZER_INTERFACE_ATTRIBUTE;
  void __asan_unpoison_memory_region(void const volatile *addr, uptr size)
      SANITIZER_INTERFACE_ATTRIBUTE;

  bool __asan_address_is_poisoned(void const volatile *addr)
      SANITIZER_INTERFACE_ATTRIBUTE;

  uptr __asan_region_is_poisoned(uptr beg, uptr size)
      SANITIZER_INTERFACE_ATTRIBUTE;

  void __asan_describe_address(uptr addr)
      SANITIZER_INTERFACE_ATTRIBUTE;

  void __asan_report_error(uptr pc, uptr bp, uptr sp,
                           uptr addr, bool is_write, uptr access_size)
    SANITIZER_INTERFACE_ATTRIBUTE;

  int __asan_set_error_exit_code(int exit_code)
      SANITIZER_INTERFACE_ATTRIBUTE;
  void __asan_set_death_callback(void (*callback)(void))
      SANITIZER_INTERFACE_ATTRIBUTE;
  void __asan_set_error_report_callback(void (*callback)(const char*))
      SANITIZER_INTERFACE_ATTRIBUTE;

  /* OPTIONAL */ void __asan_on_error()
      SANITIZER_WEAK_ATTRIBUTE SANITIZER_INTERFACE_ATTRIBUTE;

  /* OPTIONAL */ bool __asan_symbolize(const void *pc, char *out_buffer,
                                       int out_size)
      SANITIZER_WEAK_ATTRIBUTE SANITIZER_INTERFACE_ATTRIBUTE;

  uptr __asan_get_estimated_allocated_size(uptr size)
      SANITIZER_INTERFACE_ATTRIBUTE;
  bool __asan_get_ownership(const void *p)
      SANITIZER_INTERFACE_ATTRIBUTE;
  uptr __asan_get_allocated_size(const void *p)
      SANITIZER_INTERFACE_ATTRIBUTE;
  uptr __asan_get_current_allocated_bytes()
      SANITIZER_INTERFACE_ATTRIBUTE;
  uptr __asan_get_heap_size()
      SANITIZER_INTERFACE_ATTRIBUTE;
  uptr __asan_get_free_bytes()
      SANITIZER_INTERFACE_ATTRIBUTE;
  uptr __asan_get_unmapped_bytes()
      SANITIZER_INTERFACE_ATTRIBUTE;
  void __asan_print_accumulated_stats()
      SANITIZER_INTERFACE_ATTRIBUTE;

  /* OPTIONAL */ const char* __asan_default_options()
      SANITIZER_WEAK_ATTRIBUTE SANITIZER_INTERFACE_ATTRIBUTE;

  /* OPTIONAL */ void __asan_malloc_hook(void *ptr, uptr size)
      SANITIZER_WEAK_ATTRIBUTE SANITIZER_INTERFACE_ATTRIBUTE;
  /* OPTIONAL */ void __asan_free_hook(void *ptr)
      SANITIZER_WEAK_ATTRIBUTE SANITIZER_INTERFACE_ATTRIBUTE;
}  // extern "C"

#endif  // ASAN_INTERFACE_INTERNAL_H
