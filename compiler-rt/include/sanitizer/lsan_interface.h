//===-- sanitizer/lsan_interface.h ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of LeakSanitizer.
//
// Public interface header.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_LSAN_INTERFACE_H
#define SANITIZER_LSAN_INTERFACE_H

#include <sanitizer/common_interface_defs.h>

#ifdef __cplusplus
extern "C" {
#endif
  // Allocations made between calls to __lsan_disable() and __lsan_enable() will
  // be treated as non-leaks. Disable/enable pairs may be nested.
  void __lsan_disable();
  void __lsan_enable();

  // The heap object into which p points will be treated as a non-leak.
  void __lsan_ignore_object(const void *p);

  // Memory regions registered through this interface will be treated as sources
  // of live pointers during leak checking. Useful if you store pointers in
  // mapped memory.
  // Points of note:
  // - __lsan_unregister_root_region() must be called with the same pointer and
  // size that have earlier been passed to __lsan_register_root_region()
  // - LSan will skip any inaccessible memory when scanning a root region. E.g.,
  // if you map memory within a larger region that you have mprotect'ed, you can
  // register the entire large region.
  // - the implementation is not optimized for performance. This interface is
  // intended to be used for a small number of relatively static regions.
  void __lsan_register_root_region(const void *p, size_t size);
  void __lsan_unregister_root_region(const void *p, size_t size);

  // Calling this function makes LSan enter the leak checking phase immediately.
  // Use this if normal end-of-process leak checking happens too late (e.g. if
  // you have intentional memory leaks in your shutdown code). Calling this
  // function overrides end-of-process leak checking; it must be called at
  // most once per process. This function will terminate the process if there
  // are memory leaks and the exit_code flag is non-zero.
  void __lsan_do_leak_check();

  // The user may optionally provide this function to disallow leak checking
  // for the program it is linked into (if the return value is non-zero). This
  // function must be defined as returning a constant value; any behavior beyond
  // that is unsupported.
  int __lsan_is_turned_off();

  // This function may be optionally provided by the user and should return
  // a string containing LSan suppressions.
  const char *__lsan_default_suppressions();
#ifdef __cplusplus
}  // extern "C"

namespace __lsan {
class ScopedDisabler {
 public:
  ScopedDisabler() { __lsan_disable(); }
  ~ScopedDisabler() { __lsan_enable(); }
};
}  // namespace __lsan
#endif

#endif  // SANITIZER_LSAN_INTERFACE_H
