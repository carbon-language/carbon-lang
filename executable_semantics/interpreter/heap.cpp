// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/heap.h"

#include "executable_semantics/common/error.h"
#include "llvm/ADT/StringExtras.h"

namespace Carbon {

auto Heap::AllocateValue(Ptr<const Value> v) -> Address {
  // Putting the following two side effects together in this function
  // ensures that we don't do anything else in between, which is really bad!
  // Consider whether to include a copy of the input v in this function
  // or to leave it up to the caller.
  Address a(values.size());
  values.push_back(v);
  alive.push_back(true);
  return a;
}

auto Heap::Read(const Address& a, SourceLocation loc) -> Ptr<const Value> {
  this->CheckAlive(a, loc);
  return values[a.index]->GetField(arena.Get(), a.field_path, loc);
}

void Heap::Write(const Address& a, Ptr<const Value> v, SourceLocation loc) {
  this->CheckAlive(a, loc);
  values[a.index] =
      values[a.index]->SetField(arena.Get(), a.field_path, v, loc);
}

void Heap::CheckAlive(const Address& address, SourceLocation loc) {
  if (!alive[address.index]) {
    FATAL_RUNTIME_ERROR(loc) << "undefined behavior: access to dead value "
                             << *values[address.index];
  }
}

void Heap::Deallocate(const Address& address) {
  CHECK(address.field_path.IsEmpty());
  if (alive[address.index]) {
    alive[address.index] = false;
  } else {
    FATAL_RUNTIME_ERROR_NO_LINE() << "deallocating an already dead value";
  }
}

void Heap::Print(llvm::raw_ostream& out) const {
  llvm::ListSeparator sep;
  for (size_t i = 0; i < values.size(); ++i) {
    out << sep;
    PrintAddress(Address(i), out);
  }
}

void Heap::PrintAddress(const Address& a, llvm::raw_ostream& out) const {
  if (!alive[a.index]) {
    out << "!!";
  }
  out << *values[a.index];
}

}  // namespace Carbon
