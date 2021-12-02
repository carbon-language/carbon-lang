// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_HEAP_ALLOCATION_INTERFACE_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_HEAP_ALLOCATION_INTERFACE_H_

#include "executable_semantics/common/nonnull.h"
#include "executable_semantics/interpreter/address.h"

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

 protected:
  HeapAllocationInterface() = default;
  virtual ~HeapAllocationInterface() = default;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_HEAP_ALLOCATION_INTERFACE_H_
