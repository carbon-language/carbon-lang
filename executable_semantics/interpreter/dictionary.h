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
class Dictionary {
 public:
  Dictionary() { head = nullptr; }

  // Return the value associated with the given key.
  auto Lookup(const K& key) -> std::optional<V> {
    if (head == nullptr) {
      return std::nullopt;
    } else if (head->curr.first == key) {
      return head->curr.second;
    } else {
      auto next = Dictionary(head->next);
      return next.Lookup(key);
    }
  }

  // Associate the value v with key k in this.
  auto SetValueAt(const K& k, const V& v) -> void {
    head = new Cons<std::pair<K, V> >(std::make_pair(k, v), head);
  }

  // Return a copy of this that is updated to associate
  // the value v with key k.
  auto SettingValueAt(const K& k, const V& v) -> Dictionary<K, V> {
    return Dictionary(new Cons<std::pair<K, V> >(std::make_pair(k, v), head));
  }

  Dictionary(Cons<std::pair<K, V> >* h) : head(h) {}

  // TODO: make this private once we've added iterators.
  Cons<std::pair<K, V> >* head;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_ASSOC_LIST_H_
