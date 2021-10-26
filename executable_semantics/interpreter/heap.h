// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_HEAP_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_HEAP_H_

#include <vector>

#include "common/ostream.h"
#include "executable_semantics/interpreter/address.h"
#include "executable_semantics/interpreter/raw_heap.h"
#include "executable_semantics/interpreter/value.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

// A Heap represents the abstract machine's dynamically allocated memory. It
// extends `RawHeap` with the ability to read and write individual addresses,
// not just whole allocations. It also provides some convenience functionality
// such as printing that can't be in `RawHeap` for layering reasons.
class Heap : public RawHeap {
 public:
  // Constructs an empty Heap.
  explicit Heap(Nonnull<Arena*> arena) : RawHeap(), arena_(arena){};

  Heap(const Heap&) = delete;
  auto operator=(const Heap&) -> Heap& = delete;

  // Returns the value at the given address in the heap after
  // checking that it is alive.
  auto Read(const Address& a, SourceLocation source_loc)
      -> Nonnull<const Value*>;

  // Writes the given value at the address in the heap after
  // checking that the address is alive.
  void Write(const Address& a, Nonnull<const Value*> v,
             SourceLocation source_loc);

  // Print the value at the given allocation to the stream `out`.
  void PrintAllocation(AllocationId allocation, llvm::raw_ostream& out) const;

  // Print all the allocations on the heap to the stream `out`.
  void Print(llvm::raw_ostream& out) const;

  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

 private:
  // Signal an error if the allocation is no longer alive.
  void CheckAlive(AllocationId allocation, SourceLocation source_loc);

  Nonnull<Arena*> arena_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_HEAP_H_
