// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_MEMORY_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_MEMORY_H_

#include <ostream>
#include <vector>

#include "executable_semantics/interpreter/address.h"
#include "executable_semantics/interpreter/value.h"

namespace Carbon {

// A Heap represents the abstract machine's dynamically allocated memory.
class Heap {
 public:
  // Constructs an empty Heap.
  Heap() = default;

  Heap(const Heap&) = delete;
  Heap& operator=(const Heap&) = delete;

  // Returns the value at the given address in the heap after
  // checking that it is alive.
  auto Read(const Address& a, int line_num) -> const Value*;

  // Writes the given value at the address in the heap after
  // checking that the address is alive.
  auto Write(const Address& a, const Value* v, int line_num) -> void;

  // Put the given value on the heap and mark it as alive.
  auto AllocateValue(const Value* v) -> Address;

  // Marks the object at this address, and all of its sub-objects, as dead.
  auto Deallocate(const Address& address) -> void;

  // Print the value at the given address to the stream `out`.
  auto PrintAddress(const Address& a, std::ostream& out) -> void;

  // Print all the values on the heap to the stream `out`.
  auto PrintHeap(std::ostream& out) -> void;

 private:
  // Signal an error if the address is no longer alive.
  void CheckAlive(const Address& address, int line_num);

  std::vector<const Value*> values_;
  std::vector<bool> alive_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_MEMORY_H_
