// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_CONS_LIST_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_CONS_LIST_H_

namespace Carbon {

template <class T>
struct Cons {
  Cons(T e, Cons* n) : curr(e), next(n) {}

  const T curr;
  Cons* const next;

  // Cons cells are part of a "persistent data structure" and are thus
  // immutable.
  Cons& operator=(const Cons&) = delete;
  Cons& operator=(Cons&&) = delete;
};

// A forward iterator over elements of a `Cons` list.
template <class T>
struct ConsIterator {
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = const T*;
  using reference = const T&;
  using iterator_category = std::forward_iterator_tag;

  ConsIterator(Cons<T>* x) : p(x) {}
  ConsIterator(const ConsIterator& mit) : p(mit.p) {}
  ConsIterator& operator++() {
    p = p->next;
    return *this;
  }
  ConsIterator operator++(int) {
    ConsIterator tmp(*this);
    operator++();
    return tmp;
  }
  bool operator==(const ConsIterator& rhs) const { return p == rhs.p; }
  bool operator!=(const ConsIterator& rhs) const { return p != rhs.p; }
  const T& operator*() { return p->curr; }
  const T* operator->() { return &p->curr; }

 private:
  Cons<T>* p;
};

template <class T>
auto Length(Cons<T>* ls) -> unsigned int {
  if (ls) {
    return 1 + Length(ls->next);
  } else {
    return 0;
  }
}

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_CONS_LIST_H_
