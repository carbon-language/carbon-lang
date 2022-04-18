// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/heap.h"

#include "executable_semantics/common/error_builders.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"

namespace Carbon {

auto Heap::AllocateValue(Nonnull<const Value*> v) -> AllocationId {
  // Putting the following two side effects together in this function
  // ensures that we don't do anything else in between, which is really bad!
  // Consider whether to include a copy of the input v in this function
  // or to leave it up to the caller.
  AllocationId a(values_.size());
  values_.push_back(v);
  alive_.push_back(true);
  return a;
}

auto Heap::Read(const Address& a, SourceLocation source_loc) const
    -> ErrorOr<Nonnull<const Value*>> {
  RETURN_IF_ERROR(this->CheckAlive(a.allocation_, source_loc));
  return values_[a.allocation_.index_]->GetField(arena_, a.field_path_,
                                                 source_loc);
}

auto Heap::Write(const Address& a, Nonnull<const Value*> v,
                 SourceLocation source_loc) -> ErrorOr<Success> {
  RETURN_IF_ERROR(this->CheckAlive(a.allocation_, source_loc));
  ASSIGN_OR_RETURN(values_[a.allocation_.index_],
                   values_[a.allocation_.index_]->SetField(
                       arena_, a.field_path_, v, source_loc));
  return Success();
}

auto Heap::CheckAlive(AllocationId allocation, SourceLocation source_loc) const
    -> ErrorOr<Success> {
  if (!alive_[allocation.index_]) {
    return RuntimeError(source_loc)
           << "undefined behavior: access to dead value "
           << *values_[allocation.index_];
  }
  return Success();
}

void Heap::Deallocate(AllocationId allocation) {
  if (alive_[allocation.index_]) {
    alive_[allocation.index_] = false;
  } else {
    FATAL() << "deallocating an already dead value: "
            << *values_[allocation.index_];
  }
}

void Heap::Print(llvm::raw_ostream& out) const {
  llvm::ListSeparator sep;
  for (size_t i = 0; i < values_.size(); ++i) {
    out << sep;
    if (!alive_[i]) {
      out << "!!";
    }
    out << *values_[i];
  }
}

}  // namespace Carbon
