// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/heap.h"

namespace Carbon {

auto Heap::AllocateValue(const Value* v) -> Address {
  // Putting the following two side effects together in this function
  // ensures that we don't do anything else in between, which is really bad!
  // Consider whether to include a copy of the input v in this function
  // or to leave it up to the caller.
  CHECK(v != nullptr);
  Address a(values_.size());
  values_.push_back(v);
  alive_.push_back(true);
  return a;
}

auto Heap::Read(const Address& a, int line_num) -> const Value* {
  this->CheckAlive(a, line_num);
  return values_[a.index]->GetField(a.field_path, line_num);
}

void Heap::Write(const Address& a, const Value* v, int line_num) {
  CHECK(v != nullptr);
  this->CheckAlive(a, line_num);
  values_[a.index] = values_[a.index]->SetField(a.field_path, v, line_num);
}

void Heap::CheckAlive(const Address& address, int line_num) {
  if (!alive_[address.index]) {
    llvm::errs() << line_num << ": undefined behavior: access to dead value "
                 << *values_[address.index] << "\n";
    exit(-1);
  }
}

void Heap::Deallocate(const Address& address) {
  CHECK(address.field_path.IsEmpty());
  if (alive_[address.index]) {
    alive_[address.index] = false;
  } else {
    llvm::errs() << "runtime error, deallocating an already dead value\n";
    exit(-1);
  }
}

void Heap::Print(llvm::raw_ostream& out) const {
  for (size_t i = 0; i < values_.size(); ++i) {
    PrintAddress(Address(i), out);
    out << ", ";
  }
}

void Heap::PrintAddress(const Address& a, llvm::raw_ostream& out) const {
  if (!alive_[a.index]) {
    out << "!!";
  }
  out << *values_[a.index];
}

}  // namespace Carbon
