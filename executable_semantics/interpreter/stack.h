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

// A persistent stack data structure.
//
// - Note: this data structure leaks memory.
template <class T>
struct Stack {
  // A forward iterator over elements of a `Stack`.
  struct Iterator {
    // NOLINTNEXTLINE(readability-identifier-naming)
    using value_type = T;
    // NOLINTNEXTLINE(readability-identifier-naming)
    using difference_type = std::ptrdiff_t;
    // NOLINTNEXTLINE(readability-identifier-naming)
    using pointer = const T*;
    // NOLINTNEXTLINE(readability-identifier-naming)
    using reference = const T&;
    // NOLINTNEXTLINE(readability-identifier-naming)
    using iterator_category = std::forward_iterator_tag;

    explicit Iterator(ListNode<T>* x) : p(x) {}
    Iterator(const Iterator& mit) : p(mit.p) {}
    auto operator++() -> Iterator& {
      p = p->next;
      return *this;
    }
    auto operator++(int) -> Iterator {
      Iterator tmp(*this);
      operator++();
      return tmp;
    }
    auto operator==(const Iterator& rhs) const -> bool { return p == rhs.p; }
    auto operator!=(const Iterator& rhs) const -> bool { return p != rhs.p; }
    auto operator*() -> const T& { return p->curr; }
    auto operator->() -> const T* { return &p->curr; }

   private:
    ListNode<T>* p;
  };

  // The position of the first/`Top()` element, or `end()` if
  // `this->IsEmpty()`.
  // NOLINTNEXTLINE(readability-identifier-naming)
  auto begin() const -> Iterator { return Iterator(head); }

  // The position one past that of the last element.
  // NOLINTNEXTLINE(readability-identifier-naming)
  auto end() const -> Iterator { return Iterator(nullptr); }

  // Creates an empty instance.
  Stack() { head = nullptr; }

  // Creates an instance containing just `x`.
  explicit Stack(T x) : Stack() { Push(x); }

  // Pushes `x` onto the top of the stack.
  void Push(T x) { head = new ListNode<T>(x, head); }

  // Returns a copy of `*this`, with `x` pushed onto the top.
  auto Pushing(T x) const -> Stack {
    auto r = *this;
    r.Push(x);
    return r;
  }

  // Removes and returns the top element of the stack.
  //
  // - Requires: !this->IsEmpty()
  auto Pop() -> T {
    assert(!IsEmpty() && "Can't pop from empty stack.");
    auto r = head->curr;
    head = head->next;
    return r;
  }

  // Removes the top `n` elements of the stack.
  //
  // - Requires: n >= 0 && n <= Count()
  void Pop(int n) {
    assert(n >= 0 && "Negative pop count disallowed.");
    while (n--) {
      assert(head != nullptr && "Can only pop as many elements as stack has.");
      head = head->next;
    }
  }

  // Returns a copy of `*this`, sans the top element.
  //
  // - Requires: !this->IsEmpty()
  auto Popped() const -> Stack {
    auto r = *this;
    r.Pop();
    return r;
  }

  // Returns the top element of the stack.
  //
  // - Requires: !this->IsEmpty()
  auto Top() const -> T {
    assert(!IsEmpty() && "Empty stack has no Top().");
    return head->curr;
  }

  // Returns `true` iff `Count() > 0`.
  [[nodiscard]] auto IsEmpty() const -> bool { return head == nullptr; }

  // Returns `true` iff `Count() > n`.
  //
  // - Complexity: O(`n`)
  [[nodiscard]] auto CountExceeds(int n) const -> bool {
    if (n < 0) {
      return true;
    }

    for (auto p = head; p != nullptr; p = p->next) {
      if (n-- == 0) {
        return true;
      }
    }

    return false;
  }

  // Returns the number of elements in `*this`.
  [[nodiscard]] auto Count() const -> int {
    return std::distance(begin(), end());
  }

 private:
  // An linked list of cells containing the elements of self.
  ListNode<T>* head;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_CONS_LIST_H_
