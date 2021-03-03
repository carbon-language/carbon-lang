// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_STACK_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_STACK_H_

#include <cassert>
#include <cstddef>
#include <iterator>

#include "executable_semantics/interpreter/list_node.h"

namespace Carbon {

/// A persistent stack data structure.
///
/// - Note: this data structure leaks memory.
template <class T>
struct Stack {
  /// A forward iterator over elements of a `Stack`.
  struct Iterator {
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = const T*;
    using reference = const T&;
    using iterator_category = std::forward_iterator_tag;

    Iterator(ListNode<T>* x) : p(x) {}
    Iterator(const Iterator& mit) : p(mit.p) {}
    Iterator& operator++() {
      p = p->next;
      return *this;
    }
    Iterator operator++(int) {
      Iterator tmp(*this);
      operator++();
      return tmp;
    }
    bool operator==(const Iterator& rhs) const { return p == rhs.p; }
    bool operator!=(const Iterator& rhs) const { return p != rhs.p; }
    const T& operator*() { return p->curr; }
    const T* operator->() { return &p->curr; }

   private:
    ListNode<T>* p;
  };

  /// The position of the first/`Top()` element, or `end()` if
  /// `this->IsEmpty()`.
  auto begin() const -> Iterator { return Iterator(head); }

  /// The position one past that of the last element.
  auto end() const -> Iterator { return Iterator(nullptr); }

  /// Creates an empty instance.
  Stack() { head = nullptr; }

  /// Creates an instance containing just `x`.
  Stack(T x) : Stack() { Push(x); }

  /// Pushes `x` onto the top of the stack.
  void Push(T x) { head = new ListNode<T>(x, head); }

  /// Returns a copy of `*this`, with `x` pushed onto the top.
  auto Pushing(T x) const -> Stack {
    auto r = *this;
    r.Push(x);
    return r;
  }

  /// Removes and returns the top element of the stack.
  ///
  /// - Requires: !this->IsEmpty()
  auto Pop() -> T {
    assert(!IsEmpty() && "Can't pop from empty stack.");
    auto r = head->curr;
    head = head->next;
    return r;
  }

  /// Removes the top `n` elements of the stack.
  ///
  /// - Requires: n >= 0 && n <= Count()
  void Pop(int n) {
    assert(n >= 0 && "Negative pop count disallowed.");
    while (n--) {
      assert(head != nullptr && "Can only pop as many elements as stack has.");
      head = head->next;
    }
  }

  /// Returns a copy of `*this`, sans the top element.
  ///
  /// - Requires: !this->IsEmpty()
  auto Popped() const -> Stack {
    auto r = *this;
    r.Pop();
    return r;
  }

  /// Returns the top element of the stack.
  ///
  /// - Requires: !this->IsEmpty()
  auto Top() const -> T {
    assert(!IsEmpty() && "Empty stack has no Top().");
    return head->curr;
  }

  /// Returns `true` iff `Count() > 0`.
  auto IsEmpty() const -> bool { return head == nullptr; }

  /// Returns `true` iff `Count() > n`.
  ///
  /// - Complexity: O(`n`)
  auto CountExceeds(int n) const -> bool {
    if (n < 0)
      return true;

    for (auto p = head; p != nullptr; p = p->next) {
      if (n-- == 0)
        return true;
    }

    return false;
  }

  /// Returns the number of elements in `*this`.
  auto Count() const -> int { return std::distance(begin(), end()); }

 private:
  /// An linked list of cells containing the elements of self.
  ListNode<T>* head;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_CONS_LIST_H_
