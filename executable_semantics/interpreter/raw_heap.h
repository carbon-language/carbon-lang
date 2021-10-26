// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_RAW_HEAP_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_RAW_HEAP_H_

#include <vector>

#include "executable_semantics/common/nonnull.h"
#include "executable_semantics/interpreter/address.h"

namespace Carbon {

class Value;

// A RawHeap represents the abstract machine's dynamically allocated memory as
// a collection of _allocations_, each of which has a `Value`, and may be alive
// or dead. An allocation is analogous to the C++ notion of a complete object:
// the the `Value` in an allocation is not a sub-part of any other `Value`.
class RawHeap {
 public:
  RawHeap(const RawHeap&) = delete;
  auto operator=(const RawHeap&) = delete;

  // Put the given value on the heap and mark it as alive.
  auto AllocateValue(Nonnull<const Value*> v) -> AllocationId;

  // Marks the object at this address, and all of its sub-objects, as dead.
  void Deallocate(AllocationId allocation);

 protected:
  RawHeap() = default;
  ~RawHeap() = default;

  // Returns true if the given allocation is alive.
  auto IsAlive(AllocationId allocation) const -> bool {
    return alive_[allocation.index_];
  }

  // Returns the value stored in the given allocation. In order to support
  // logging and tracing use cases, the allocation is permitted to be dead,
  // in which case the last value stored is returned.
  auto RawRead(AllocationId allocation) const -> Nonnull<const Value*>;

  // Writes `value` into the given allocation, which must be alive.
  void RawWrite(AllocationId allocation, Nonnull<const Value*> value);

  // Returns all allocations in this heap (both alive and dead).
  auto AllAllocations() const -> std::vector<AllocationId>;

 private:
  std::vector<Nonnull<const Value*>> values_;
  std::vector<bool> alive_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_RAW_HEAP_H_
