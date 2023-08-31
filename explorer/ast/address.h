// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_AST_ADDRESS_H_
#define CARBON_EXPLORER_AST_ADDRESS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "common/check.h"
#include "common/ostream.h"
#include "explorer/ast/element_path.h"

namespace Carbon {

// An AllocationId identifies an _allocation_ produced by a Heap. An allocation
// is analogous to the C++ notion of a complete object: the `Value` in an
// allocation is not a sub-part of any other `Value`.
class AllocationId : public Printable<AllocationId> {
 public:
  AllocationId(const AllocationId&) = default;
  auto operator=(const AllocationId&) -> AllocationId& = default;

  inline friend auto operator==(AllocationId lhs, AllocationId rhs) -> bool {
    return lhs.index_ == rhs.index_;
  }
  inline friend auto hash_value(AllocationId id) {
    return llvm::hash_combine(id.index_);
  }

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
class Address : public Printable<Address> {
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

  // If *this represents the address of an object with a field named
  // `field_name`, this method returns the address of that field.
  auto ElementAddress(Nonnull<const Element*> element) const -> Address {
    Address result = *this;
    result.element_path_.Append(element);
    return result;
  }

  // Drop all trailing BaseElements from the element path, returning the
  // downcasted address.
  auto DowncastedAddress() const -> Address {
    Address address = *this;
    address.element_path_.RemoveTrailingBaseElements();
    return address;
  }

  inline friend auto operator==(const Address& lhs, const Address& rhs)
      -> bool {
    return lhs.allocation_ == rhs.allocation_ &&
           lhs.element_path_ == rhs.element_path_;
  }
  inline friend auto hash_value(const Address& a) -> llvm::hash_code {
    return llvm::hash_combine(a.allocation_, a.element_path_);
  }

 private:
  // The representation of Address describes how to locate an object within
  // the Heap, so its implementation details are tied to the implementation
  // details of the Heap.
  friend class Heap;
  friend class RuntimeScope;

  AllocationId allocation_;
  ElementPath element_path_;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_AST_ADDRESS_H_
