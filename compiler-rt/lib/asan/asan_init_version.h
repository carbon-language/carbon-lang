//===-- asan_init_version.h -------------------------------------*- C++ -*-===//
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
// This header defines a versioned __asan_init function to be called at the
// startup of the instrumented program.
//===----------------------------------------------------------------------===//
#ifndef ASAN_INIT_VERSION_H
#define ASAN_INIT_VERSION_H

#include "sanitizer_common/sanitizer_internal_defs.h"

extern "C" {
  // This function should be called at the very beginning of the process,
  // before any instrumented code is executed and before any call to malloc.
  // Every time the ASan ABI changes we also change the version number in this
  // name. Objects build with incompatible ASan ABI version
  // will not link with run-time.
  // Changes between ABI versions:
  // v1=>v2: added 'module_name' to __asan_global
  // v2=>v3: stack frame description (created by the compiler)
  //         contains the function PC as the 3-rd field (see
  //         DescribeAddressIfStack).
  // v3=>v4: added '__asan_global_source_location' to __asan_global.
  SANITIZER_INTERFACE_ATTRIBUTE void __asan_init_v4();
  #define __asan_init __asan_init_v4
  #define __asan_init_name "__asan_init_v4"
}

#endif  // ASAN_INIT_VERSION_H
