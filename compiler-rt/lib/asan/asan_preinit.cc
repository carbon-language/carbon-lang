//===-- asan_preinit.cc ---------------------------------------------------===//
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
// Call __asan_init at the very early stage of process startup.
// On Linux we use .preinit_array section (unless PIC macro is defined).
//===----------------------------------------------------------------------===//
#include "asan_internal.h"

#if ASAN_USE_PREINIT_ARRAY && !defined(PIC)
  // On Linux, we force __asan_init to be called before anyone else
  // by placing it into .preinit_array section.
  // FIXME: do we have anything like this on Mac?
  __attribute__((section(".preinit_array"), used))
  void (*__asan_preinit)(void) =__asan_init;
#elif SANITIZER_WINDOWS && defined(_DLL)
  // On Windows, when using dynamic CRT (/MD), we can put a pointer
  // to __asan_init into the global list of C initializers.
  // See crt0dat.c in the CRT sources for the details.
  #pragma section(".CRT$XIB", long, read)  // NOLINT
  __declspec(allocate(".CRT$XIB")) void (*__asan_preinit)() = __asan_init;
#endif
