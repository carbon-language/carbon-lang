// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_HEAP_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_HEAP_H_

#include <vector>

#include "common/ostream.h"
#include "executable_semantics/common/nonnull.h"
#include "executable_semantics/common/source_location.h"
#include "executable_semantics/interpreter/address.h"
#include "executable_semantics/interpreter/heap_allocation_interface.h"
#include "executable_semantics/interpreter/value.h"

namespace Carbon {

// A Heap represents the abstract machine's dynamically allocated memory.
class Heap : public HeapAllocationInterface {
 public:
  // Constructs an empty Heap.
  explicit Heap(Nonnull<Arena*> arena) : arena_(arena){};

  Heap(const Heap&) = delete;
  auto operator=(const Heap&) -> Heap& = delete;

  // Returns the value at the given address in the heap after
  // checking that it is alive.
  auto Read(const Address& a, SourceLocation source_loc) const
      -> ErrorOr<Nonnull<const Value*>>;

  // Writes the given value at the address in the heap after
  // checking that the address is alive.
  auto Write(const Address& a, Nonnull<const Value*> v,
             SourceLocation source_loc) -> ErrorOr<Success>;

  // Put the given value on the heap and mark it as alive.
  auto AllocateValue(Nonnull<const Value*> v) -> AllocationId override;

  // Marks this allocation, and all of its sub-objects, as dead.
  void Deallocate(AllocationId allocation) override;

  // Print all the values on the heap to the stream `out`.
  void Print(llvm::raw_ostream& out) const;

  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

  auto arena() const -> Arena& override { return *arena_; }

 private:
  // Signal an error if the allocation is no longer alive.
  auto CheckAlive(AllocationId allocation, SourceLocation source_loc) const
      -> ErrorOr<Success>;

  Nonnull<Arena*> arena_;
  std::vector<Nonnull<const Value*>> values_;
  std::vector<bool> alive_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_HEAP_H_
