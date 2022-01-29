// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_STACK_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_STACK_H_

#include <cstddef>
#include <iterator>
#include <vector>

#include "common/check.h"

namespace Carbon {

// A stack data structure.
template <class T>
struct Stack {
  // NOLINTNEXTLINE(readability-identifier-naming)
  using const_iterator = typename std::vector<T>::const_reverse_iterator;

  // Creates an empty instance.
  Stack() = default;

  // Creates an instance containing just `x`.
  explicit Stack(T x) : Stack() { Push(std::move(x)); }

  // Pushes `x` onto the top of the stack.
  void Push(T x) { elements_.push_back(std::move(x)); }

  // Removes and returns the top element of the stack.
  //
  // - Requires: !this->IsEmpty()
  auto Pop() -> T {
    CHECK(!IsEmpty()) << "Can't pop from empty stack.";
    auto r = std::move(elements_.back());
    elements_.pop_back();
    return r;
  }

  // Removes the top `n` elements of the stack.
  //
  // - Requires: n >= 0 && n <= Count()
  void Pop(int n) {
    CHECK(n >= 0) << "Negative pop count disallowed.";
    CHECK(static_cast<size_t>(n) <= elements_.size())
        << "Can only pop as many elements as stack has.";
    elements_.erase(elements_.end() - n, elements_.end());
  }

  // Returns the top element of the stack.
  //
  // - Requires: !this->IsEmpty()
  auto Top() const -> const T& {
    CHECK(!IsEmpty()) << "Empty stack has no Top().";
    return elements_.back();
  }

  // Returns `true` iff `Count() > 0`.
  auto IsEmpty() const -> bool { return elements_.empty(); }

  // Returns the number of elements in `*this`.
  auto Count() const -> int { return elements_.size(); }

  // Iterates over the Stack from top to bottom.
  auto begin() const -> const_iterator { return elements_.crbegin(); }
  auto end() const -> const_iterator { return elements_.crend(); }

 private:
  std::vector<T> elements_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_CONS_LIST_H_
