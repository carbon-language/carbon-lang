// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_ASSOC_LIST_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_ASSOC_LIST_H_

#include <iostream>
#include <list>
#include <string>

namespace Carbon {

template <class K, class V>
struct AssocList {
  AssocList(K k, V v, AssocList* n) : key(k), value(v), next(n) {}

  K key;
  V value;
  AssocList* next;
};

template <class K, class V>
auto Lookup(int line_num, AssocList<K, V>* alist, const K& key,
            void (*print_key)(const K&)) -> V {
  if (alist == NULL) {
    std::cerr << line_num << ": could not find `" << key << "`" << std::endl;
    exit(-1);
  } else if (alist->key == key) {
    return alist->value;
  } else {
    return Lookup(line_num, alist->next, key, print_key);
  }
}

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_ASSOC_LIST_H_
