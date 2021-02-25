// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_CONS_LIST_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_CONS_LIST_H_

namespace Carbon {

template <class T>
struct Stack;

template <class T>
struct Cons {
  friend struct Stack<T>;

 private:
  Cons(T e, Cons* n) : curr(e), next(n) {}

  const T curr;
  Cons* const next;

  // Cons cells are part of a "persistent data structure" and are thus
  // immutable.
  Cons& operator=(const Cons&) = delete;
  Cons& operator=(Cons&&) = delete;
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
