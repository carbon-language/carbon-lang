// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_STACK_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_STACK_H_

#include <cstddef>
#include <iterator>
#include <vector>

#include "common/check.h"
#include "executable_semantics/interpreter/list_node.h"

namespace Carbon {

// A stack data structure.
template <class T>
struct Stack {
  using const_iterator = typename std::vector<T>::const_reverse_iterator;

  // Creates an empty instance.
  Stack() = default;

  // Creates an instance containing just `x`.
  // TODO: consider removing this. It's somewhat unconventional, and the
  // callsite readability is debatable.
  explicit Stack(T x) : Stack() { Push(std::move(x)); }

  // Pushes `x` onto the top of the stack.
  void Push(T x) { elements.push_back(std::move(x)); }

  // Removes and returns the top element of the stack.
  //
  // - Requires: not this->IsEmpty()
  auto Pop() -> T {
    CHECK(not IsEmpty()) << "Can't pop from empty stack.";
    auto r = std::move(elements.back());
    elements.pop_back();
    return r;
  }

  // Removes the top `n` elements of the stack.
  //
  // - Requires: n >= 0 and n <= Count()
  void Pop(int n) {
    CHECK(n >= 0) << "Negative pop count disallowed.";
    CHECK(static_cast<size_t>(n) <= elements.size())
        << "Can only pop as many elements as stack has.";
    elements.resize(elements.size() - n);
  }

  // Returns the top element of the stack.
  //
  // - Requires: not this->IsEmpty()
  auto Top() const -> T {
    CHECK(not IsEmpty()) << "Empty stack has no Top().";
    return elements.back();
  }

  // Returns `true` iff `Count() > 0`.
  auto IsEmpty() const -> bool { return elements.empty(); }

  // Returns the number of elements in `*this`.
  auto Count() const -> int { return elements.size(); }

  // Iterates over the Stack from top to bottom.
  const_iterator begin() const { return elements.crbegin(); }
  const_iterator end() const { return elements.crend(); }

 private:
  std::vector<T> elements;
};

// Explicitly enable CTAD to silence warnings.
// TODO: consider removing this (and perhaps the associated constructor).
template <typename T>
Stack(T x) -> Stack<T>;

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_CONS_LIST_H_
