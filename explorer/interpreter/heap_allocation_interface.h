// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_HEAP_ALLOCATION_INTERFACE_H_
#define CARBON_EXPLORER_INTERPRETER_HEAP_ALLOCATION_INTERFACE_H_

#include "explorer/common/arena.h"
#include "explorer/common/nonnull.h"
#include "explorer/interpreter/address.h"

namespace Carbon {

class Value;

// The allocation interface for Heap, factored out as an interface in order to
// resolve a layering issue. No other class should derive from this.
class HeapAllocationInterface {
 public:
  HeapAllocationInterface(const HeapAllocationInterface&) = delete;
  auto operator=(const HeapAllocationInterface&)
      -> HeapAllocationInterface& = delete;

  // Put the given value on the heap and mark it as alive.
  virtual auto AllocateValue(Nonnull<const Value*> v) -> AllocationId = 0;

  // Marks this allocation, and all of its sub-objects, as dead.
  virtual void Deallocate(AllocationId allocation) = 0;

  // Returns the arena used to allocate the values in this heap.
  virtual auto arena() const -> Arena& = 0;

  // Returns the ID of the first allocation that holds `v`, if one exists.
  // TODO: Find a way to remove this.
  virtual auto GetAllocationId(Nonnull<const Value*> v) const
      -> std::optional<AllocationId> = 0;

 protected:
  HeapAllocationInterface() = default;
  virtual ~HeapAllocationInterface() = default;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_HEAP_ALLOCATION_INTERFACE_H_
