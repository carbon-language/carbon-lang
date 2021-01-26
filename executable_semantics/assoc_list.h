// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_ASSOC_LIST_H
#define EXECUTABLE_SEMANTICS_ASSOC_LIST_H

#include <iostream>
#include <string>

template <class K, class V>
struct AList {
  AList(K k, V v, AList* n) : key(k), value(v), next(n) {}

  K key;
  V value;
  AList* next;
};

template <class K, class V>
auto Lookup(int lineno, AList<K, V>* alist, const K& key,
            void (*print_key)(const K&)) -> V {
  if (alist == NULL) {
    std::cerr << lineno << ": could not find `";
    print_key(key);
    std::cerr << "`" << std::endl;
    exit(-1);
  } else if (alist->key == key) {
    return alist->value;
  } else {
    return Lookup(lineno, alist->next, key, print_key);
  }
}

#endif  // EXECUTABLE_SEMANTICS_ASSOC_LIST_H
