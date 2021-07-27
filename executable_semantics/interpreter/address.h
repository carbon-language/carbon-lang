// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_ADDRESS_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_ADDRESS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "common/ostream.h"
#include "executable_semantics/interpreter/field_path.h"

namespace Carbon {

// An Address represents a memory address in the Carbon virtual machine.
// Addresses are used to access values stored in a Heap, and are obtained
// from a Heap (or by deriving them from other Addresses).
class Address {
 public:
  Address(const Address&) = default;
  Address(Address&&) = default;
  auto operator=(const Address&) -> Address& = default;
  auto operator=(Address&&) -> Address& = default;

  // Returns true if the two addresses refer to the same memory location.
  friend auto operator==(const Address& lhs, const Address& rhs) -> bool {
    return lhs.index == rhs.index;
  }

  friend auto operator!=(const Address& lhs, const Address& rhs) -> bool {
    return not(lhs == rhs);
  }

  // Prints a human-readable representation of `a` to `out`.
  //
  // Currently, that representation consists of an integer index identifying
  // the whole memory allocation, and an optional FieldPath specifying a
  // particular field within that allocation.
  void Print(llvm::raw_ostream& out) const {
    out << "Address(" << index << ")" << field_path;
  }

  // If *this represents the address of an object with a field named
  // `field_name`, this method returns the address of that field.
  auto SubobjectAddress(std::string field_name) const -> Address {
    Address result = *this;
    result.field_path.Append(std::move(field_name));
    return result;
  }

 private:
  // The representation of Address describes how to locate an object within
  // the Heap, so its implementation details are tied to the implementation
  // details of the Heap.
  friend class Heap;

  explicit Address(uint64_t index) : index(index) {}

  uint64_t index;
  FieldPath field_path;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_ADDRESS_H_
