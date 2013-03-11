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

__attribute__((constructor))
void loadable_module_init(void) {
  if (!__msan_has_dynamic_component())
    return;
  // The real test is that this compare should not make an uninit.
  if (dso_global == NULL)
    dso_global = malloc(4);
}

__attribute__((destructor))
void loadable_module_fini(void) {
  if (!__msan_has_dynamic_component())
    return;
  free(dso_global);
  // *Don't* overwrite it with NULL!  That would unpoison it, but our test
  // relies on reloading at the same address and keeping the poison.
}

void **get_dso_global() {
  return &dso_global;
}

}
