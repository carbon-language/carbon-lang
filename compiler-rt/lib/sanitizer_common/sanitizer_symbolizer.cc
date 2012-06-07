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

#include "sanitizer_common.h"
#include "sanitizer_symbolizer.h"

namespace __sanitizer {

void AddressInfo::Clear() {
  InternalFree(module);
  InternalFree(function);
  InternalFree(file);
}

void AddressInfoList::Clear() {
  AddressInfoList *cur = this;
  while (cur) {
    cur->info.Clear();
    AddressInfoList *nxt = cur->next;
    InternalFree(cur);
    cur = nxt;
  }
}

AddressInfoList* SymbolizeCode(uptr address) {
  AddressInfoList *list = (AddressInfoList*)InternalAlloc(
      sizeof(AddressInfoList));
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
