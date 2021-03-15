// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_LIST_NODE_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_LIST_NODE_H_

#include <cstddef>
#include <iterator>

namespace Carbon {

template <class T>
struct ListNode {
  ListNode(T e, ListNode* n) : curr(e), next(n) {}

  const T curr;
  ListNode* const next;

  // ListNode cells are part of a "persistent data structure" and are thus
  // immutable.
  auto operator=(const ListNode&) -> ListNode& = delete;
  auto operator=(ListNode&&) -> ListNode& = delete;
};

// A forward iterator over elements of a `ListNode` list.
template <class T>
struct ListNodeIterator {
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

  explicit ListNodeIterator(ListNode<T>* x) : p(x) {}
  ListNodeIterator(const ListNodeIterator& iter) : p(iter.p) {}
  auto operator++() -> ListNodeIterator& {
    p = p->next;
    return *this;
  }
  auto operator++(int) -> ListNodeIterator {
    ListNodeIterator tmp(*this);
    operator++();
    return tmp;
  }
  auto operator==(const ListNodeIterator& rhs) const -> bool {
    return p == rhs.p;
  }
  auto operator!=(const ListNodeIterator& rhs) const -> bool {
    return p != rhs.p;
  }
  auto operator*() -> const T& { return p->curr; }
  auto operator->() -> const T* { return &p->curr; }

 private:
  ListNode<T>* p;
};

template <class T>
auto Length(ListNode<T>* ls) -> unsigned int {
  if (ls) {
    return 1 + Length(ls->next);
  } else {
    return 0;
  }
}

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_LIST_NODE_H_
