//===-- sanitizer_symbolizer.cc -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a stub for LLVM-based symbolizer.
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries. See sanitizer.h for details.
//===----------------------------------------------------------------------===//

// WARNING: Avoid using library functions - see comments in symbolizer.h.
#include "sanitizer_symbolizer.h"
// FIXME: replace library malloc/free with internal_malloc/internal_free
// that would be provided by ASan/TSan run-time libraries.
#include <stdlib.h>

namespace __sanitizer {

void AddressInfo::Clear() {
  free(module);
  free(function);
  free(file);
}

void AddressInfoList::Clear() {
  AddressInfoList *cur = this;
  while (cur) {
    cur->info.Clear();
    AddressInfoList *nxt = cur->next;
    free(cur);
    cur = nxt;
  }
}

AddressInfoList* SymbolizeCode(uptr address) {
  AddressInfoList *list = (AddressInfoList*)malloc(sizeof(AddressInfoList));
  list->next = 0;
  list->info.address = address;
  list->info.module = 0;
  list->info.module_offset = 0;
  list->info.function = 0;
  list->info.file = 0;
  list->info.line = 0;
  list->info.column = 0;
  return list;
}

}  // namespace __sanitizer
