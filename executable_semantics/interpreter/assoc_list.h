// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_ASSOC_LIST_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_ASSOC_LIST_H_

#include <iostream>
#include <list>
#include <optional>
#include <string>

#include "executable_semantics/interpreter/cons_list.h"

namespace Carbon {

template <class K, class V>
class AssocList {
 public:
  AssocList() { head = nullptr; }

  auto Lookup(const K& key) -> std::optional<V> {
    if (head == nullptr) {
      return std::nullopt;
    } else if (head->curr.first == key) {
      return head->curr.second;
    } else {
      auto next = AssocList(head->next);
      return next.Lookup(key);
    }
  }

  auto Extend(const K& k, const V& v) -> void {
    head = new Cons<std::pair<K, V> >(std::make_pair(k, v), head);
  }

  auto Extending(const K& k, const V& v) -> AssocList<K, V> {
    return AssocList(new Cons<std::pair<K, V> >(std::make_pair(k, v), head));
  }

  AssocList(Cons<std::pair<K, V> >* h) : head(h) {}

  Cons<std::pair<K, V> >* head;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_ASSOC_LIST_H_
