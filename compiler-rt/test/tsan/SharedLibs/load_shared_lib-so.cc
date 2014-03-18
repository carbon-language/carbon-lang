//===----------- load_shared_lib-so.cc --------------------------*- C++ -*-===//
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

#include <stddef.h>
#include <unistd.h>

int GLOB_SHARED = 0;

extern "C"
void *write_from_so(void *unused) {
  if (unused)
    sleep(1);
  GLOB_SHARED++;
  return NULL;
}
