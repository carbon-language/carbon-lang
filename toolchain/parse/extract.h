// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSE_EXTRACT_H_
#define CARBON_TOOLCHAIN_PARSE_EXTRACT_H_

#include <tuple>
#include <utility>

#include "common/struct_reflection.h"
#include "toolchain/parse/tree.h"

namespace Carbon::Parse {

// Extract an `NodeId` as a single child.
template <>
struct Tree::Extractable<NodeId> {
  static auto Extract(const Tree* /*tree*/, SiblingIterator& it,
                      SiblingIterator end) -> std::optional<NodeId> {
    if (it == end) {
      return std::nullopt;
    }
    return NodeId(*it++);
  }
};

// Aliases for `NodeId` to describe particular kinds of nodes.
//
// TODO: We should check that the right kind of node is present.
using AnyExpr = NodeId;
using AnyDecl = NodeId;
using AnyStatement = NodeId;
using AnyPattern = NodeId;
using AnyNameComponent = NodeId;

// TODO: define Or<T, U> and Not<Kind>.

// Extract a `TypeNodeId<T>` as a single required child.
template <typename T>
struct Tree::Extractable<TypedNodeId<T>> {
  static auto Extract(const Tree* tree, SiblingIterator& it,
                      SiblingIterator end) -> std::optional<TypedNodeId<T>> {
    if (it == end || tree->node_kind(*it) != T::Kind) {
      llvm::errs() << "FIXME: Extract TypedNodeId " << T::Kind << " error\n";
      return std::nullopt;
    }
    llvm::errs() << "FIXME: Extract TypedNodeId " << T::Kind << " success\n";
    return TypedNodeId<T>(*it++);
  }
};

// An optional child. If this child is present, it will not be of kind `T`.
template <typename T>
class OptionalNot {
 public:
  explicit OptionalNot(NodeId node_id) : node_id_(node_id) {}
  explicit OptionalNot(std::nullopt_t) : node_id_(NodeId::Invalid) {}

  // Returns whether this element was present.
  auto is_present() -> bool { return node_id_ != NodeId::Invalid; }

  // Gets the `Node`, if this element was present.
  auto GetNode() const -> std::optional<NodeId> {
    return is_present() ? node_id_ : std::nullopt;
  }

 private:
  NodeId node_id_;
};

// Extract an `OptionalNot<T>` as either zero or one child.
template <typename T>
struct Tree::Extractable<OptionalNot<T>> {
  static auto Extract(const Tree* tree, SiblingIterator& it,
                      SiblingIterator end) -> std::optional<OptionalNot<T>> {
    if (it == end || tree->node_kind(*it) == T::Kind) {
      return OptionalNot<T>(std::nullopt);
    }
    return OptionalNot<T>(*it++);
  }
};

// A list of `T`s, terminated by a `Bracket`. Each `T` should implement
// `ChildTraits`, and `Bracket` should be the struct for a parse node kind.
template <typename T, typename Bracket>
class BracketedList : public std::vector<T> {};

// Extract a `BracketedList` by extracting `T`s until we reach `Bracket`.
template <typename T, typename Bracket>
struct Tree::Extractable<BracketedList<T, Bracket>> {
  static auto Extract(const Tree* tree, SiblingIterator& it,
                      SiblingIterator end)
      -> std::optional<BracketedList<T, Bracket>> {
    BracketedList<T, Bracket> result;
    while (it != end && tree->node_kind(*it) != Bracket::Kind) {
      auto item = Extractable<T>::Extract(tree, it, end);
      if (!item.has_value()) {
        return std::nullopt;
      }
      result.push_back(*item);
    }
    if (it == end) {
      return std::nullopt;
    }
    std::reverse(result.begin(), result.end());
    return result;
  }
};

// Extract an `optional<T>` from a list of child nodes by attempting to extract
// a `T`, and extracting nothing if that fails.
template <typename T>
struct Tree::Extractable<std::optional<T>> {
  static auto Extract(const Tree* tree, SiblingIterator& it,
                      SiblingIterator end) -> std::optional<std::optional<T>> {
    auto old_it = it;
    std::optional<T> value = Extractable<T>::Extract(tree, it, end);
    if (value) {
      llvm::errs() << "FIXME: Extract std::optional found\n";
      return value;
    }
    llvm::errs() << "FIXME: Extract std::optional missing\n";
    it = old_it;
    return value;
  }
};

// Extract a `tuple<T...>` from a list of child nodes by extracting each `T` in
// reverse order.
template <typename... T>
struct Tree::Extractable<std::tuple<T...>> {
  template <std::size_t... Index>
  static auto ExtractImpl(const Tree* tree, SiblingIterator& it,
                          SiblingIterator end, std::index_sequence<Index...>)
      -> std::optional<std::tuple<T...>> {
    std::tuple<std::optional<T>...> fields;

    llvm::errs() << "FIXME: Extract tuple\n";
    // Use a fold over the `=` operator to parse fields from right to left.
    [[maybe_unused]] int unused;
    static_cast<void>(
        ((std::get<Index>(fields) = Extractable<T>::Extract(tree, it, end),
          unused) = ... = 0));

    if (!(std::get<Index>(fields).has_value() && ...)) {
      llvm::errs() << "FIXME: Extract tuple error\n";
      return std::nullopt;
    }

    llvm::errs() << "FIXME: Extract tuple success\n";
    return std::tuple<T...>{std::move(std::get<Index>(fields).value())...};
  }

  static auto Extract(const Tree* tree, SiblingIterator& it,
                      SiblingIterator end) -> std::optional<std::tuple<T...>> {
    return ExtractImpl(tree, it, end, std::make_index_sequence<sizeof...(T)>());
  }
};

// Extract the fields of a simple aggregate type.
template <typename T>
struct Tree::Extractable {
  static_assert(std::is_aggregate_v<T>, "Unsupported child type");

  static auto Extract(const Tree* tree, SiblingIterator& it,
                      SiblingIterator end) -> std::optional<T> {
    // Extract the corresponding tuple type.
    using TupleType = decltype(StructReflection::AsTuple(std::declval<T>()));
    llvm::errs() << "FIXME: Extract simple aggregate\n";
    auto tuple = Extractable<TupleType>::Extract(tree, it, end);
    if (!tuple.has_value()) {
      llvm::errs() << "FIXME: Extract simple aggregate error\n";
      return std::nullopt;
    }
    llvm::errs() << "FIXME: Extract simple aggregate success\n";

    // Convert the tuple to the struct type.
    return std::apply(
        [](auto&&... value) {
          return T{std::forward<decltype(value)>(value)...};
        },
        *tuple);
  }
};

}  // namespace Carbon::Parse

#endif  // CARBON_TOOLCHAIN_PARSE_EXTRACT_H_
