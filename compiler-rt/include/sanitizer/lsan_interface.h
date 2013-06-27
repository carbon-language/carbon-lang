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
  // be treated as non-leaks. Disable/enable pairs can be nested.
  void __lsan_disable();
  void __lsan_enable();
  // The heap object into which p points will be treated as a non-leak.
  void __lsan_ignore_object(const void *p);
  // The user may optionally provide this function to disallow leak checking
  // for the program it is linked into. Note: this function may be called late,
  // after all the global destructors.
  int __lsan_is_turned_off();
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
