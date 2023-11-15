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

// A child that can be any parse node.
class AnyNode : public Node {
 public:
  explicit AnyNode(Node node) : Node(node) {}
};

// Extract an `AnyNode` as a single child.
template <>
struct Tree::Extractable<AnyNode> {
  static auto Extract(const Tree* /*tree*/, SiblingIterator& it) -> AnyNode {
    return AnyNode(*it++);
  }
};

// Aliases for `AnyNode` to describe particular kinds of nodes.
//
// TODO: We should check that the right kind of node is present.
using AnyExpr = AnyNode;
using AnyDecl = AnyNode;
using AnyStatement = AnyNode;
using AnyPattern = AnyNode;

// A child that is expected to be of the specified kind `T`.
template <typename T>
class Required : public Node {
 public:
  explicit Required(Node node) : Node(node) {}

  // Get the representation of this child node. Returns `nullopt` if the node is
  // invalid, or valid but of the wrong kind.
  //
  // TODO: Can we CHECK that the latter doesn't happen?
  auto Extract(const Tree* tree) const -> std::optional<T> {
    return ExtractAs<T>(tree);
  }
};

// Extract a `Required<T>` as a single child.
template <typename T>
struct Tree::Extractable<Required<T>> {
  static auto Extract(const Tree* /*tree*/, SiblingIterator& it)
      -> Required<T> {
    // TODO: Can we CHECK that this node is of the right kind?
    return Required<T>(*it++);
  }
};

// An optional child. If this child is present, it will be of kind `T`.
template <typename T>
class Optional {
 public:
  explicit Optional(Node node) : node_(node) {}
  explicit Optional(std::nullopt_t) : node_(Node::Invalid) {}

  // Returns whether this element was present.
  auto is_present() -> bool { return node_ != Node::Invalid; }

  // Gets the `Node`, if this element was present.
  auto GetNode() const -> std::optional<Node> {
    return is_present() ? node_ : std::nullopt;
  }

  // Gets the typed node, if it is present and valid. Note that this returns
  // `std::nullopt` if this element was present but malformed. Use `is_present`
  // to determine if the element was present at all.
  auto Extract(const Tree* tree) const -> std::optional<T> {
    return is_present() ? node_.ExtractAs<T>(tree) : std::nullopt;
  }

 private:
  Node node_;
};

// Extract an `Optional<T>` as either zero or one child.
template <typename T>
struct Tree::Extractable<Optional<T>> {
  static auto Extract(const Tree* tree, SiblingIterator& it) -> Optional<T> {
    return (*it).kind(tree) == T::Kind ? Optional<T>(*it++)
                                       : Optional<T>(std::nullopt);
  }
};

// An optional child that is present if the previous child is a `T`.
template <typename T>
class IfPreviousNodeIs {
 public:
  explicit IfPreviousNodeIs(Node node) : node_(node) {}
  explicit IfPreviousNodeIs(std::nullopt_t) : node_(Node::Invalid) {}

  // Returns whether this element was present.
  auto is_present() -> bool { return node_ != Node::Invalid; }

  // Gets the `Node`, if this element was present.
  auto GetNode() const -> std::optional<Node> {
    return is_present() ? node_ : std::nullopt;
  }

  // Returns whether this child is present and is a valid node of type `U`.
  template <typename U>
  auto IsValid(const Tree* tree) -> bool {
    return is_present() && node_.IsValid<U>(tree);
  }

  // Extracts the node, if it's present, valid, and of type `U`.
  template <typename U>
  auto ExtractAs(const Tree* tree) const -> std::optional<T> {
    return is_present() ? node_.ExtractAs<T>(tree) : std::nullopt;
  }

 private:
  Node node_;
};

// Extract an `IfPreviousNodeIs<T>` as either zero or one child.
template <typename T>
struct Tree::Extractable<IfPreviousNodeIs<T>> {
  static auto Extract(const Tree* tree, SiblingIterator& it)
      -> IfPreviousNodeIs<T> {
    return (*std::next(it)).kind(tree) == T::Kind
               ? IfPreviousNodeIs<T>(*it++)
               : IfPreviousNodeIs<T>(std::nullopt);
  }
};

// A list of `T`s, terminated by a `Bracket`. Each `T` should implement
// `ChildTraits`, and `Bracket` should be the struct for a parse node kind.
template <typename T, typename Bracket>
class BracketedList : public std::vector<T> {};

// Extract a `BracketedList` by extracting `T`s until we reach `Bracket`.
template <typename T, typename Bracket>
struct Tree::Extractable<BracketedList<T, Bracket>> {
  static auto Extract(const Tree* tree, SiblingIterator& it)
      -> BracketedList<T, Bracket> {
    BracketedList<T, Bracket> result;
    while ((*it).kind(tree) != Bracket::Kind) {
      result.push_back(Extractable<T>::Extract(tree, it));
    }
    std::reverse(result.begin(), result.end());
    return result;
  }
};

// TODO: Move to common/?
namespace Internal {
// A wrapper around a type `T` that might not be default-constructible, that
// defers constructing the wrapped `T` until some time later. The wrapped value
// must be constructed before the wrapper is destroyed.
template <typename T>
union ManualConstruction {
  T value;

  ManualConstruction() {}
  auto Construct(T t) { new (&value) T(std::move(t)); }
  ~ManualConstruction() { value.~T(); }
};
}  // namespace Internal

// Extract a `tuple<T...>` from a list of child nodes by extracting each `T` in
// reverse order.
template <typename... T>
struct Tree::Extractable<std::tuple<T...>> {
  template <std::size_t... Index>
  static auto ExtractImpl(const Tree* tree, SiblingIterator& it,
                          std::index_sequence<Index...>) -> std::tuple<T...> {
    std::tuple<Internal::ManualConstruction<T>...> fields;

    // Use a fold over the `=` operator to parse fields from right to left.
    [[maybe_unused]] int unused;
    static_cast<void>(
        ((std::get<Index>(fields).Construct(Extractable<T>::Extract(tree, it)),
          unused) = ... = 0));

    return {std::move(std::get<Index>(fields).value)...};
  }

  static auto Extract(const Tree* tree, SiblingIterator& it)
      -> std::tuple<T...> {
    return ExtractImpl(tree, it, std::make_index_sequence<sizeof...(T)>());
  }
};

// Extract the fields of a simple aggregate type.
template <typename T>
struct Tree::Extractable {
  static_assert(std::is_aggregate_v<T>, "Unsupported child type");

  static auto Extract(const Tree* tree, SiblingIterator& it) -> T {
    // Extract the corresponding tuple type.
    using TupleType = decltype(StructReflection::AsTuple(std::declval<T>()));
    auto tuple = Extractable<TupleType>::Extract(tree, it);

    // Convert the tuple to the struct type.
    return std::apply(
        [](auto&&... value) {
          return T{std::forward<decltype(value)>(value)...};
        },
        tuple);
  }
};

}  // namespace Carbon::Parse

#endif  // CARBON_TOOLCHAIN_PARSE_EXTRACT_H_
