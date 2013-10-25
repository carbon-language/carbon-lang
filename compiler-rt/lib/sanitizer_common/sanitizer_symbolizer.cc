//===-- sanitizer_symbolizer.cc -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_symbolizer.h"

namespace __sanitizer {

atomic_uintptr_t Symbolizer::symbolizer_;
LowLevelAllocator Symbolizer::symbolizer_allocator_;

Symbolizer *Symbolizer::GetOrNull() {
  return reinterpret_cast<Symbolizer *>(
      atomic_load(&symbolizer_, memory_order_acquire));
}

Symbolizer *Symbolizer::Get() {
  Symbolizer *sym = GetOrNull();
  CHECK(sym);
  return sym;
}

Symbolizer *Symbolizer::Disable() {
  CHECK_EQ(0, atomic_load(&symbolizer_, memory_order_acquire));
  Symbolizer *dummy_sym = new(symbolizer_allocator_) Symbolizer;
  atomic_store(&symbolizer_, reinterpret_cast<uptr>(&dummy_sym),
               memory_order_release);
  return dummy_sym;
}

}  // namespace __sanitizer
