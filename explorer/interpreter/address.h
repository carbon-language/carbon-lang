// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_ADDRESS_H_
#define CARBON_EXPLORER_INTERPRETER_ADDRESS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "common/ostream.h"
#include "explorer/interpreter/element_path.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

// An AllocationId identifies an _allocation_ produced by a Heap. An allocation
// is analogous to the C++ notion of a complete object: the `Value` in an
// allocation is not a sub-part of any other `Value`.
class AllocationId {
 public:
  AllocationId(const AllocationId&) = default;
  auto operator=(const AllocationId&) -> AllocationId& = default;

  // Prints a human-readable representation of *this to `out`.
  //
  // Currently that representation consists of an integer index.
  void Print(llvm::raw_ostream& out) const {
    out << "Allocation(" << index_ << ")";
  }

 private:
  // The representation of AllocationId describes how to locate an object within
  // a Heap, so its implementation details are tied to the implementation
  // details of Heap.
  friend class Heap;

  explicit AllocationId(size_t index) : index_(index) {}

  size_t index_;
};

// An Address represents a memory address in the Carbon virtual machine.
// Addresses are used to access values stored in a Heap. Unlike an
// AllocationId, an Address can refer to a sub-Value of some larger Value.
class Address {
 public:
  // Constructs an `Address` that refers to the value stored in `allocation`.
  explicit Address(AllocationId allocation) : allocation_(allocation) {}

  Address(const Address&) = default;
  Address(Address&&) = default;
  auto operator=(const Address&) -> Address& = default;
  auto operator=(Address&&) -> Address& = default;

  // Prints a human-readable representation of `a` to `out`.
  //
  // Currently, that representation consists of an AllocationId followed by an
  // optional ElementPath specifying a particular field within that allocation.
  void Print(llvm::raw_ostream& out) const {
    out << allocation_ << element_path_;
  }

  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

  // If *this represents the address of an object with a field named
  // `field_name`, this method returns the address of that field.
  auto ElementAddress(Nonnull<const Element*> element) const -> Address {
    Address result = *this;
    result.element_path_.Append(element);
    return result;
  }

 private:
  // The representation of Address describes how to locate an object within
  // the Heap, so its implementation details are tied to the implementation
  // details of the Heap.
  friend class Heap;

  AllocationId allocation_;
  ElementPath element_path_;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_ADDRESS_H_
