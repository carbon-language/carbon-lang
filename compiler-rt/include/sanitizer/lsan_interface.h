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
