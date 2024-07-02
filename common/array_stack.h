// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_ARRAY_STACK_H_
#define CARBON_COMMON_ARRAY_STACK_H_

#include "common/check.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace Carbon {

// Provides arrays in a stack form. Only the top of the stack can have elements
// added.
//
// By using a single vector for elements, the intent is that as arrays are
// pushed and popped, the same storage will be reused. This should yield
// efficiencies for heap allocations.
template <typename ValueT>
class ArrayStack {
 public:
  // Pushes a new array onto the stack.
  auto PushArray() -> void { array_offsets_.push_back(elements_.size()); }

  // Pops the top array from the stack.
  auto PopArray() -> void {
    auto region = array_offsets_.pop_back_val();
    elements_.truncate(region);
  }

  // Returns the top array from the stack.
  auto PeekArray() -> llvm::ArrayRef<ValueT> {
    CARBON_CHECK(!array_offsets_.empty());
    return llvm::ArrayRef(elements_).slice(array_offsets_.back());
  }

  // Returns the full set of values on the stack, regardless of whether any
  // arrays are pushed.
  auto PeekAllValues() -> llvm::ArrayRef<ValueT> { return elements_; }

  // Adds a value to the top array on the stack.
  auto PushValue(ValueT value) -> void {
    CARBON_CHECK(!array_offsets_.empty())
        << "Must call PushArray before PushValue.";
    elements_.push_back(value);
  }

  // Returns the current number of values in all arrays.
  auto elements_size() -> size_t { return elements_.size(); }

 private:
  // For each pushed array, the start index in elements_.
  llvm::SmallVector<int32_t> array_offsets_;

  // The full set of elements in all arrays.
  llvm::SmallVector<ValueT> elements_;
};

}  // namespace Carbon

#endif  // CARBON_COMMON_ARRAY_STACK_H_
