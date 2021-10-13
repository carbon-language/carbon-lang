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

    Node(ValueType e, std::optional<Nonnull<Node*>> n) : curr(e), next(n) {}

    const ValueType curr;
    const std::optional<Nonnull<Node*>> next;

    // Node cells are part of a "persistent data structure" and are thus
    // immutable.
    auto operator=(const Node&) -> Node& = delete;
    auto operator=(Node&&) -> Node& = delete;
  };

  // A forward iterator over elements of a `Node` list.
  struct Iterator {
    using ValueType = typename Node::ValueType;
    using DifferenceType = std::ptrdiff_t;
    using Pointer = const value_type*;
    using Reference = const value_type&;
    using IteratorCategory = std::forward_iterator_tag;

    explicit Iterator(std::optional<Nonnull<Node*>> x) : p(x) {}
    Iterator(const Iterator& iter) : p(iter.p) {}
    auto operator++() -> Iterator& {
      p = (*p)->next;
      return *this;
    }
    auto operator++(int) -> Iterator {
      Iterator tmp(*this);
      operator++();
      return tmp;
    }
    auto operator==(const Iterator& rhs) const -> bool { return p == rhs.p; }
    auto operator!=(const Iterator& rhs) const -> bool { return p != rhs.p; }
    auto operator*() -> const value_type& { return (*p)->curr; }
    auto operator->() -> const value_type* { return &(*p)->curr; }

   private:
    std::optional<Nonnull<Node*>> p;
  };

  // Create an empty dictionary.
  explicit Dictionary(Nonnull<Arena*> arena) : arena(arena) {}

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

  auto IsEmpty() -> bool { return !head; }

  // The position of the first element of the dictionary
  // or `end()` if the dictionary is empty.
  auto begin() const -> Iterator { return Iterator(head); }

  // The position one past that of the last element.
  auto end() const -> Iterator { return Iterator(std::nullopt); }

 private:
  std::optional<Nonnull<Node*>> head;
  Nonnull<Arena*> arena;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_DICTIONARY_H_
