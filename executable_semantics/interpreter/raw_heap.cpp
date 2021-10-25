// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/raw_heap.h"

#include "executable_semantics/common/error.h"

namespace Carbon {

auto RawHeap::AllocateValue(Nonnull<const Value*> v) -> AllocationId {
  // Putting the following two side effects together in this function
  // ensures that we don't do anything else in between, which is really bad!
  // Consider whether to include a copy of the input v in this function
  // or to leave it up to the caller.
  AllocationId a(values_.size());
  values_.push_back(v);
  alive_.push_back(true);
  return a;
}

auto RawHeap::RawRead(AllocationId a) const -> Nonnull<const Value*> {
  return values_[a.index_];
}

void RawHeap::RawWrite(AllocationId a, Nonnull<const Value*> v) {
  CHECK(alive_[a.index_]);
  values_[a.index_] = v;
}

void RawHeap::Deallocate(AllocationId address) {
  if (alive_[address.index_]) {
    alive_[address.index_] = false;
  } else {
    FATAL_RUNTIME_ERROR_NO_LINE() << "deallocating an already dead value";
  }
}

auto RawHeap::AllAllocations() const -> std::vector<AllocationId> {
  std::vector<AllocationId> result;
  for (size_t i = 0; i < values_.size(); ++i) {
    result.push_back(AllocationId(i));
  }
  return result;
}

}  // namespace Carbon
