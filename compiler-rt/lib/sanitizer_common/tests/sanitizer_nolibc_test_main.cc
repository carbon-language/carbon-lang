//===-- sanitizer_nolibc_test_main.cc -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer/AddressSanitizer runtime.
// Tests for libc independence of sanitizer_common.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_libc.h"

// TODO(pcc): Remove these once the allocator dependency is gone.
extern "C" void *__libc_malloc() { return 0; }
extern "C" void __libc_free(void *p) {}

extern "C" void _start() {
  internal__exit(0);
}
