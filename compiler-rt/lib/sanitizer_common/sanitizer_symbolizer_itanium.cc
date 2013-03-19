//===-- sanitizer_symbolizer_itanium.cc -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between the sanitizer run-time libraries.
// Itanium C++ ABI-specific implementation of symbolizer parts.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"
#if SANITIZER_MAC || SANITIZER_LINUX

#include "sanitizer_symbolizer.h"

#include <stdlib.h>

// C++ demangling function, as required by Itanium C++ ABI. This is weak,
// because we do not require a C++ ABI library to be linked to a program
// using sanitizers; if it's not present, we'll just use the mangled name.
namespace __cxxabiv1 {
  extern "C" char *__cxa_demangle(const char *mangled, char *buffer,
                                  size_t *length, int *status)
    SANITIZER_WEAK_ATTRIBUTE;
}

const char *__sanitizer::Demangle(const char *MangledName) {
  // FIXME: __cxa_demangle aggressively insists on allocating memory.
  // There's not much we can do about that, short of providing our
  // own demangler (libc++abi's implementation could be adapted so that
  // it does not allocate). For now, we just call it anyway, and we leak
  // the returned value.
  if (__cxxabiv1::__cxa_demangle)
    if (const char *Demangled =
          __cxxabiv1::__cxa_demangle(MangledName, 0, 0, 0))
      return Demangled;

  return MangledName;
}

#endif  // __APPLE__ || __linux__
