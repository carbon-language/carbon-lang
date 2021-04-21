// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_LIST_NODE_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_LIST_NODE_H_

namespace Carbon {

template <class T>
struct ListNode {
  ListNode(T e, ListNode* n) : curr(e), next(n) {}

  const T curr;
  ListNode* const next;

  // ListNode cells are part of a "persistent data structure" and are thus
  // immutable.
  ListNode& operator=(const ListNode&) = delete;
  ListNode& operator=(ListNode&&) = delete;
};

// A forward iterator over elements of a `ListNode` list.
template <class T>
struct ListNodeIterator {
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = const T*;
  using reference = const T&;
  using iterator_category = std::forward_iterator_tag;

  ListNodeIterator(ListNode<T>* x) : p(x) {}
  ListNodeIterator(const ListNodeIterator& iter) : p(iter.p) {}
  ListNodeIterator& operator++() {
    p = p->next;
    return *this;
  }
  ListNodeIterator operator++(int) {
    ListNodeIterator tmp(*this);
    operator++();
    return tmp;
  }
  bool operator==(const ListNodeIterator& rhs) const { return p == rhs.p; }
  bool operator!=(const ListNodeIterator& rhs) const { return p != rhs.p; }
  const T& operator*() { return p->curr; }
  const T* operator->() { return &p->curr; }

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
