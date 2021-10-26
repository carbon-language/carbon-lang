// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/heap.h"

#include "executable_semantics/common/error.h"
#include "executable_semantics/interpreter/raw_heap.h"
#include "llvm/ADT/StringExtras.h"

namespace Carbon {

auto Heap::Read(const Address& a, SourceLocation source_loc)
    -> Nonnull<const Value*> {
  CheckAlive(a.allocation_, source_loc);
  return RawRead(a.allocation_)->GetField(arena_, a.field_path_, source_loc);
}

void Heap::Write(const Address& a, Nonnull<const Value*> v,
                 SourceLocation source_loc) {
  CheckAlive(a.allocation_, source_loc);
  Nonnull<const Value*> new_allocation_value =
      RawRead(a.allocation_)->SetField(arena_, a.field_path_, v, source_loc);
  RawWrite(a.allocation_, new_allocation_value);
}

void Heap::CheckAlive(AllocationId allocation, SourceLocation source_loc) {
  if (!IsAlive(allocation)) {
    FATAL_RUNTIME_ERROR(source_loc)
        << "undefined behavior: access to dead value " << *RawRead(allocation);
  }
}

void Heap::Print(llvm::raw_ostream& out) const {
  llvm::ListSeparator sep;
  for (AllocationId allocation : AllAllocations()) {
    out << sep;
    PrintAllocation(allocation, out);
  }
}

void Heap::PrintAllocation(AllocationId allocation,
                           llvm::raw_ostream& out) const {
  if (!IsAlive(allocation)) {
    out << "!!";
  }
  out << *RawRead(allocation);
}

}  // namespace Carbon
