// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_DICTIONARY_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_DICTIONARY_H_

#include <iostream>
#include <list>
#include <optional>
#include <string>

#include "executable_semantics/common/arena.h"
#include "executable_semantics/interpreter/list_node.h"

namespace Carbon {

// A persistent dictionary with a simple implementation.
// Copying the dictionary is O(1) time.
template <class K, class V>
class Dictionary {
 public:
  // Create an empty dictionary.
  Dictionary() { head = nullptr; }

  // Return the value associated with the given key.
  // Time complexity: O(n) where n is the number of times
  //    any value has been set across all keys.
  auto Get(const K& key) -> std::optional<V> {
    for (auto kv : *this) {
      if (kv.first == key) {
        return kv.second;
      }
    }
    return std::nullopt;
  }

  // Associate the value v with key k in the dictionary.
  // Time complexity: O(1).
  auto Set(const K& k, const V& v) -> void {
    head = global_arena->New<ListNode<std::pair<K, V>>>(std::make_pair(k, v),
                                                        head);
  }

  typedef ListNodeIterator<std::pair<K, V>> Iterator;

  // The position of the first element of the dictionary
  // or `end()` if the dictionary is empty.
  auto begin() const -> Iterator { return Iterator(head); }

  // The position one past that of the last element.
  auto end() const -> Iterator { return Iterator(nullptr); }

 private:
  Dictionary(ListNode<std::pair<K, V>>* h) : head(h) {}

  ListNode<std::pair<K, V>>* head;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_DICTIONARY_H_
