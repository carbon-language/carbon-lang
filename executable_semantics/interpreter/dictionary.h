// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_DICTIONARY_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_DICTIONARY_H_

#include <iterator>
#include <optional>

#include "executable_semantics/common/arena.h"

namespace Carbon {

// A persistent dictionary with a simple implementation.
// Copying the dictionary is O(1) time.
template <class K, class V>
class Dictionary {
 public:
  struct Node {
    using ValueType = std::pair<K, V>;

    Node(ValueType e, std::optional<Ptr<Node>> n) : curr(e), next(n) {}

    const ValueType curr;
    const std::optional<Ptr<Node>> next;

    // Node cells are part of a "persistent data structure" and are thus
    // immutable.
    Node& operator=(const Node&) = delete;
    Node& operator=(Node&&) = delete;
  };

  // A forward iterator over elements of a `Node` list.
  struct Iterator {
    using value_type = typename Node::ValueType;
    using difference_type = std::ptrdiff_t;
    using pointer = const value_type*;
    using reference = const value_type&;
    using iterator_category = std::forward_iterator_tag;

    Iterator(std::optional<Ptr<Node>> x) : p(x) {}
    Iterator(const Iterator& iter) : p(iter.p) {}
    Iterator& operator++() {
      p = (*p)->next;
      return *this;
    }
    Iterator operator++(int) {
      Iterator tmp(*this);
      operator++();
      return tmp;
    }
    bool operator==(const Iterator& rhs) const { return p == rhs.p; }
    bool operator!=(const Iterator& rhs) const { return p != rhs.p; }
    const value_type& operator*() { return (*p)->curr; }
    const value_type* operator->() { return &(*p)->curr; }

   private:
    std::optional<Ptr<Node>> p;
  };

  // Create an empty dictionary. Does not take ownership of arena.
  explicit Dictionary(Arena* arena) : arena(arena) {}

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
    head = arena->New<Node>(std::make_pair(k, v), head);
  }

  bool IsEmpty() { return !head; }

  // The position of the first element of the dictionary
  // or `end()` if the dictionary is empty.
  auto begin() const -> Iterator { return Iterator(head); }

  // The position one past that of the last element.
  auto end() const -> Iterator { return Iterator(std::nullopt); }

 private:
  std::optional<Ptr<Node>> head;
  Ptr<Arena> arena;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_DICTIONARY_H_
