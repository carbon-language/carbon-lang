// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_HEAP_ALLOCATION_INTERFACE_H_
#define CARBON_EXPLORER_INTERPRETER_HEAP_ALLOCATION_INTERFACE_H_

#include "common/error.h"
#include "explorer/ast/address.h"
#include "explorer/common/arena.h"
#include "explorer/common/nonnull.h"
#include "explorer/common/source_location.h"

namespace Carbon {

class Value;

// The allocation interface for Heap, factored out as an interface in order to
// resolve a layering issue. No other class should derive from this.
class HeapAllocationInterface {
 public:
  HeapAllocationInterface(const HeapAllocationInterface&) = delete;
  auto operator=(const HeapAllocationInterface&)
      -> HeapAllocationInterface& = delete;

  // Returns the value at the given address in the heap after
  // checking that it is alive.
  virtual auto Read(const Address& a, SourceLocation source_loc) const
      -> ErrorOr<Nonnull<const Value*>> = 0;

  // Writes the given value at the address in the heap after
  // checking that the address is alive.
  virtual auto Write(const Address& a, Nonnull<const Value*> v,
                     SourceLocation source_loc) -> ErrorOr<Success> = 0;

  // Put the given value on the heap and mark it as alive.
  virtual auto AllocateValue(Nonnull<const Value*> v) -> AllocationId = 0;

  // Marks this allocation, and all of its sub-objects, as dead.
  virtual auto Deallocate(AllocationId allocation, SourceLocation source_loc)
      -> ErrorOr<Success> = 0;

  // Returns the arena used to allocate the values in this heap.
  virtual auto arena() const -> Arena& = 0;

 protected:
  HeapAllocationInterface() = default;
  virtual ~HeapAllocationInterface() = default;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_HEAP_ALLOCATION_INTERFACE_H_
