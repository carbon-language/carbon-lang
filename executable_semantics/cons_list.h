// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CONS_LIST_H
#define CONS_LIST_H

template<class T>
struct Cons {
  T curr;
  Cons* next;
  Cons(T e, Cons* n) : curr(e), next(n) { }
};

template<class T>
Cons<T>* cons(const T& x, Cons<T>* ls) {
  return new Cons<T>(x, ls);
}

template<class T>
unsigned int length(Cons<T>* ls) {
  if (ls) {
    return 1 + length(ls->next);
  } else {
    return 0;
  }
}


#endif
