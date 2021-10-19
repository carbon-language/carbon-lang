// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/heap.h"

#include "executable_semantics/common/error.h"
#include "llvm/ADT/StringExtras.h"

namespace Carbon {

auto Heap::AllocateValue(Nonnull<const Value*> v) -> Address {
  // Putting the following two side effects together in this function
  // ensures that we don't do anything else in between, which is really bad!
  // Consider whether to include a copy of the input v in this function
  // or to leave it up to the caller.
  Address a(values_.size());
  values_.push_back(v);
  alive_.push_back(true);
  return a;
}

auto Heap::Read(const Address& a, SourceLocation source_loc)
    -> Nonnull<const Value*> {
  this->CheckAlive(a, source_loc);
  return values_[a.index_]->GetField(arena_, a.field_path_, source_loc);
}

void Heap::Write(const Address& a, Nonnull<const Value*> v,
                 SourceLocation source_loc) {
  this->CheckAlive(a, source_loc);
  values_[a.index_] =
      values_[a.index_]->SetField(arena_, a.field_path_, v, source_loc);
}

void Heap::CheckAlive(const Address& address, SourceLocation source_loc) {
  if (!alive_[address.index_]) {
    FATAL_RUNTIME_ERROR(source_loc)
        << "undefined behavior: access to dead value "
        << *values_[address.index_];
  }
}

void Heap::Deallocate(const Address& address) {
  CHECK(address.field_path_.IsEmpty());
  if (alive_[address.index_]) {
    alive_[address.index_] = false;
  } else {
    FATAL_RUNTIME_ERROR_NO_LINE() << "deallocating an already dead value";
  }
}

void Heap::Print(llvm::raw_ostream& out) const {
  llvm::ListSeparator sep;
  for (size_t i = 0; i < values_.size(); ++i) {
    out << sep;
    PrintAddress(Address(i), out);
  }
}

void Heap::PrintAddress(const Address& a, llvm::raw_ostream& out) const {
  if (!alive_[a.index_]) {
    out << "!!";
  }
  out << *values_[a.index_];
}

}  // namespace Carbon
