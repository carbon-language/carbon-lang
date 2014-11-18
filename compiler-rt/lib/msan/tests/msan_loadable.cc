//===-- msan_loadable.cc --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of MemorySanitizer.
//
// MemorySanitizer unit tests.
//===----------------------------------------------------------------------===//

#include "msan/msan_interface_internal.h"
#include <stdlib.h>

static void *dso_global;

// No name mangling.
extern "C" {

void **get_dso_global() {
  return &dso_global;
}

}
