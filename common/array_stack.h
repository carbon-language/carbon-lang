// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_ARRAY_STACK_H_
#define CARBON_COMMON_ARRAY_STACK_H_

#include "common/check.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace Carbon {

// Provides a stack of arrays. Only the array at the top of the stack can have
// elements added.
//
// Example usage:
//   // Push to start.
//   PushArray();
//   // Add values.
//   PushValue(3);
//   // Look at values.
//   PeekArray();
//   // Pop when done.
//   PopArray();
//
// By using a single vector for elements, the intent is that as arrays are
// pushed and popped, the same storage will be reused. This should yield
// efficiencies for heap allocations. For example, in the toolchain we
// frequently have an array per scope, and only add to the current scope's
// array; this allows better reuse when entering and leaving scopes.
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
  auto PeekArray() const -> llvm::ArrayRef<ValueT> {
    CARBON_CHECK(!array_offsets_.empty());
    return llvm::ArrayRef(elements_).slice(array_offsets_.back());
  }

  // Returns the full set of values on the stack, regardless of whether any
  // arrays are pushed.
  auto PeekAllValues() const -> llvm::ArrayRef<ValueT> { return elements_; }

  // Adds a value to the top array on the stack.
  auto PushValue(ValueT value) -> void {
    CARBON_CHECK(!array_offsets_.empty())
        << "Must call PushArray before PushValue.";
    elements_.push_back(value);
  }

  // Returns the current number of values in all arrays.
  auto elements_size() const -> size_t { return elements_.size(); }

 private:
  // For each pushed array, the start index in elements_.
  llvm::SmallVector<int32_t> array_offsets_;

  // The full set of elements in all arrays.
  llvm::SmallVector<ValueT> elements_;
};

}  // namespace Carbon

#endif  // CARBON_COMMON_ARRAY_STACK_H_
