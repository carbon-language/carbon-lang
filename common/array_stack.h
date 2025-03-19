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
//   AppendToTop(3);
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
  auto PushArray() -> void { array_offsets_.push_back(values_.size()); }

  // Pops the top array from the stack.
  auto PopArray() -> void {
    auto region = array_offsets_.pop_back_val();
    values_.truncate(region);
  }

  // Returns the top array from the stack.
  auto PeekArray() const -> llvm::ArrayRef<ValueT> {
    CARBON_CHECK(!array_offsets_.empty());
    return llvm::ArrayRef(values_).slice(array_offsets_.back());
  }

  auto PeekArray() -> llvm::MutableArrayRef<ValueT> {
    CARBON_CHECK(!array_offsets_.empty());
    return llvm::MutableArrayRef(values_).slice(array_offsets_.back());
  }

  // Returns the array at a specific index.
  auto PeekArrayAt(int index) const -> llvm::ArrayRef<ValueT> {
    auto ref = llvm::ArrayRef(values_).slice(array_offsets_[index]);
    if (index + 1 < static_cast<int>(array_offsets_.size())) {
      ref = ref.take_front(array_offsets_[index + 1] - array_offsets_[index]);
    }
    return ref;
  }

  // Returns the full set of values on the stack, regardless of whether any
  // arrays are pushed.
  auto PeekAllValues() const -> llvm::ArrayRef<ValueT> { return values_; }

  // Appends a value to the top array on the stack.
  auto AppendToTop(ValueT value) -> void {
    CARBON_CHECK(!array_offsets_.empty(),
                 "Must call PushArray before AppendToTop.");
    values_.push_back(value);
  }

  // Prepends a value to the top array on the stack.
  auto PrependToTop(ValueT value) -> void {
    CARBON_CHECK(!array_offsets_.empty(),
                 "Must call PushArray before PrependToTop.");
    values_.insert(values_.begin() + array_offsets_.back(), value);
  }

  // Adds multiple values to the top array on the stack.
  auto AppendToTop(llvm::ArrayRef<ValueT> values) -> void {
    CARBON_CHECK(!array_offsets_.empty(),
                 "Must call PushArray before PushValues.");
    llvm::append_range(values_, values);
  }

  // Returns the current number of values in all arrays.
  auto all_values_size() const -> size_t { return values_.size(); }

  // Returns true if the stack has no arrays pushed.
  auto empty() const -> bool { return array_offsets_.empty(); }

 private:
  // For each pushed array, the start index in `values_`.
  llvm::SmallVector<int32_t> array_offsets_;

  // The full set of elements in all arrays.
  llvm::SmallVector<ValueT> values_;
};

}  // namespace Carbon

#endif  // CARBON_COMMON_ARRAY_STACK_H_
