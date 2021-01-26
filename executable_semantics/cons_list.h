// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_CONS_LIST_H
#define EXECUTABLE_SEMANTICS_CONS_LIST_H

template <class T>
struct Cons {
  T curr;
  Cons* next;
  Cons(T e, Cons* n) : curr(e), next(n) {}
};

template <class T>
auto MakeCons(const T& x, Cons<T>* ls) -> Cons<T>* {
  return new Cons<T>(x, ls);
}

template <class T>
auto Length(Cons<T>* ls) -> unsigned int {
  if (ls) {
    return 1 + Length(ls->next);
  } else {
    return 0;
  }
}

#endif  // EXECUTABLE_SEMANTICS_CONS_LIST_H
