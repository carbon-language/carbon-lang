// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_CONS_LIST_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_CONS_LIST_H_

namespace Carbon {

template <class T>
struct Cons {
  Cons(T e, Cons* n) : curr(e), next(n) {}

  T curr;
  Cons* next;
};

template <class T>
auto MakeCons(const T& x) -> Cons<T>* {
  return new Cons<T>(x, nullptr);
}

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

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_CONS_LIST_H_
