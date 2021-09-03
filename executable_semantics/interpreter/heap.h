// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_MEMORY_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_MEMORY_H_

#include <vector>

#include "common/ostream.h"
#include "executable_semantics/interpreter/address.h"
#include "executable_semantics/interpreter/value.h"
#include "llvm/Support/Compiler.h"

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
  auto Read(const Address& a, SourceLocation loc) -> Ptr<const Value>;

  // Writes the given value at the address in the heap after
  // checking that the address is alive.
  void Write(const Address& a, Ptr<const Value> v, SourceLocation loc);

  // Put the given value on the heap and mark it as alive.
  auto AllocateValue(Ptr<const Value> v) -> Address;

  // Marks the object at this address, and all of its sub-objects, as dead.
  void Deallocate(const Address& address);

  // Print the value at the given address to the stream `out`.
  void PrintAddress(const Address& a, llvm::raw_ostream& out) const;

  // Print all the values on the heap to the stream `out`.
  void Print(llvm::raw_ostream& out) const;

  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

 private:
  // Signal an error if the address is no longer alive.
  void CheckAlive(const Address& address, SourceLocation loc);

  std::vector<Ptr<const Value>> values_;
  std::vector<bool> alive_;
  Arena arena;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_MEMORY_H_
