// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_DICTIONARY_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_DICTIONARY_H_

#include <iostream>
#include <list>
#include <optional>
#include <string>

#include "executable_semantics/interpreter/cons_list.h"

namespace Carbon {

// A persistent dictionary with a simple implementation.
template <class K, class V>
class Dictionary {
 public:
  // Create an empty dictionary.
  Dictionary() { head = nullptr; }

  // Copying the dictionary is O(1) time.

  // Return the value associated with the given key.
  // Time complexity: O(n) in the number of times
  //    any value has been set for any key.
  auto Lookup(const K& key) -> std::optional<V> {
    for (auto kv : *this)
      if (kv.first == key)
        return kv.second;
    return std::nullopt;
  }

  // Associate the value v with key k in the dictionary.
  // Time complexity: O(1).
  auto SetValueAt(const K& k, const V& v) -> void {
    head = new Cons<std::pair<K, V> >(std::make_pair(k, v), head);
  }

  // Return a copy of the dictionary that is updated to associate
  // the value v with key k.
  // Time complexity: O(1).
  auto SettingValueAt(const K& k, const V& v) -> Dictionary<K, V> {
    return Dictionary(new Cons<std::pair<K, V> >(std::make_pair(k, v), head));
  }

  typedef ConsIterator<std::pair<K, V> > Iterator;

  // The position of the first element of the dictionary
  // or `end()` if the dictionary is empty.
  auto begin() const -> Iterator { return Iterator(head); }

  // The position one past that of the last element.
  auto end() const -> Iterator { return Iterator(nullptr); }

 private:
  Dictionary(Cons<std::pair<K, V> >* h) : head(h) {}

  Cons<std::pair<K, V> >* head;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_DICTIONARY_H_
