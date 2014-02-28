//===-- sanitizer_symbolizer_libbacktrace.h ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries.
// Header for libbacktrace symbolizer.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_SYMBOLIZER_LIBBACKTRACE_H
#define SANITIZER_SYMBOLIZER_LIBBACKTRACE_H

#include "sanitizer_platform.h"
#include "sanitizer_common.h"
#include "sanitizer_symbolizer.h"

#ifndef SANITIZER_LIBBACKTRACE
# define SANITIZER_LIBBACKTRACE 0
#endif

#ifndef SANITIZER_CP_DEMANGLE
# define SANITIZER_CP_DEMANGLE 0
#endif

namespace __sanitizer {

class LibbacktraceSymbolizer {
 public:
  static LibbacktraceSymbolizer *get(LowLevelAllocator *alloc);

  uptr SymbolizeCode(uptr addr, AddressInfo *frames, uptr max_frames,
                     const char *module_name, uptr module_offset);

  bool SymbolizeData(DataInfo *info);

  // May return NULL if demangling failed.
  static char *Demangle(const char *name, bool always_alloc = false);

 private:
  explicit LibbacktraceSymbolizer(void *state) : state_(state) {}

  void *state_;  // Leaked.
};

}  // namespace __sanitizer
#endif  // SANITIZER_SYMBOLIZER_LIBBACKTRACE_H
