// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_HEAP_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_HEAP_H_

#include <vector>

#include "common/ostream.h"
#include "executable_semantics/ast/source_location.h"
#include "executable_semantics/common/nonnull.h"
#include "executable_semantics/interpreter/address.h"

namespace Carbon {

class Value;

// A Heap represents the abstract machine's dynamically allocated memory.
// Heap is an abstract class for layering reasons rather than polymorphism;
// HeapImpl is intended to be its only implementation.
class Heap {
 public:
  Heap(const Heap&) = delete;
  auto operator=(const Heap&) -> Heap& = delete;

  // Returns the value at the given address in the heap after
  // checking that it is alive.
  virtual auto Read(const Address& a, SourceLocation source_loc)
      -> Nonnull<const Value*> = 0;

  // Writes the given value at the address in the heap after
  // checking that the address is alive.
  virtual void Write(const Address& a, Nonnull<const Value*> v,
                     SourceLocation source_loc) = 0;

  // Put the given value on the heap and mark it as alive.
  virtual auto AllocateValue(Nonnull<const Value*> v) -> AllocationId = 0;

  // Marks this allocation, and all of its sub-objects, as dead.
  virtual void Deallocate(AllocationId allocation) = 0;

  // Print the value at the given allocation to the stream `out`.
  virtual void PrintAllocation(AllocationId allocation,
                               llvm::raw_ostream& out) const = 0;

  // Print all the values on the heap to the stream `out`.
  virtual void Print(llvm::raw_ostream& out) const = 0;

  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

 protected:
  Heap() = default;
  virtual ~Heap() = default;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_HEAP_H_
