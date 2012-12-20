//===-- tsan_interface_java.cc --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//

#include "tsan_interface_java.h"
#include "tsan_rtl.h"

using namespace __tsan;  // NOLINT

void __tsan_java_init(jptr heap_begin, jptr heap_size) {
}

int  __tsan_java_fini() {
  return 0;
}

void __tsan_java_alloc(jptr ptr, jptr size) {
}

void __tsan_java_free(jptr ptr, jptr size) {
}

void __tsan_java_move(jptr src, jptr dst, jptr size) {
}

void __tsan_java_mutex_lock(jptr addr) {
}

void __tsan_java_mutex_unlock(jptr addr) {
}

void __tsan_java_mutex_read_lock(jptr addr) {
}

void __tsan_java_mutex_read_unlock(jptr addr) {
}

