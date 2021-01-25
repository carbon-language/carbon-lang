// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ASSOC_LIST_H
#define ASSOC_LIST_H

#include <iostream>
#include <string>
using std::cerr;
using std::endl;

template<class K, class V>
struct AList {
  K key;
  V value;
  AList* next;
  AList(K k, V v, AList* n) : key(k), value(v), next(n) { }
};

template<class K, class V>
auto Lookup(int lineno, AList<K,V>* alist, K key, void (*print_key)(K)) -> V {
  if (alist == NULL) {
    cerr << lineno << ": could not find `";
    print_key(key);
    cerr << "`" << endl;
    exit(-1);
  } else if (alist->key == key) {
    return alist->value;
  } else {
    return Lookup(lineno, alist->next, key, print_key);
  }
}



#endif
