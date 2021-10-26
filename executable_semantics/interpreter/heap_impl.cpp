// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/heap_impl.h"

#include "executable_semantics/common/error.h"
#include "llvm/ADT/StringExtras.h"

namespace Carbon {

auto HeapImpl::AllocateValue(Nonnull<const Value*> v) -> AllocationId {
  // Putting the following two side effects together in this function
  // ensures that we don't do anything else in between, which is really bad!
  // Consider whether to include a copy of the input v in this function
  // or to leave it up to the caller.
  AllocationId a(values_.size());
  values_.push_back(v);
  alive_.push_back(true);
  return a;
}

auto HeapImpl::Read(const Address& a, SourceLocation source_loc)
    -> Nonnull<const Value*> {
  this->CheckAlive(a.allocation_, source_loc);
  return values_[a.allocation_.index_]->GetField(arena_, a.field_path_,
                                                 source_loc);
}

void HeapImpl::Write(const Address& a, Nonnull<const Value*> v,
                     SourceLocation source_loc) {
  this->CheckAlive(a.allocation_, source_loc);
  values_[a.allocation_.index_] = values_[a.allocation_.index_]->SetField(
      arena_, a.field_path_, v, source_loc);
}

void HeapImpl::CheckAlive(AllocationId allocation, SourceLocation source_loc) {
  if (!alive_[allocation.index_]) {
    FATAL_RUNTIME_ERROR(source_loc)
        << "undefined behavior: access to dead value "
        << *values_[allocation.index_];
  }
}

void HeapImpl::Deallocate(AllocationId allocation) {
  if (alive_[allocation.index_]) {
    alive_[allocation.index_] = false;
  } else {
    FATAL_RUNTIME_ERROR_NO_LINE() << "deallocating an already dead value";
  }
}

void HeapImpl::Print(llvm::raw_ostream& out) const {
  llvm::ListSeparator sep;
  for (size_t i = 0; i < values_.size(); ++i) {
    out << sep;
    PrintAllocation(AllocationId(i), out);
  }
}

void HeapImpl::PrintAllocation(AllocationId allocation,
                               llvm::raw_ostream& out) const {
  if (!alive_[allocation.index_]) {
    out << "!!";
  }
  out << *values_[allocation.index_];
}

}  // namespace Carbon
