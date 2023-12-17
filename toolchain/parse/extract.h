// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSE_EXTRACT_H_
#define CARBON_TOOLCHAIN_PARSE_EXTRACT_H_

#include <tuple>
#include <utility>

#include "common/struct_reflection.h"
#include "toolchain/parse/tree.h"
#include "toolchain/parse/typed_nodes.h"

namespace Carbon::Parse {

// A complete source file. Note that there is no corresponding parse node for
// the file. The file is instead the complete contents of the parse tree.
struct File {
  TypedNodeId<FileStart> start;
  BracketedList<AnyDecl, FileStart> decls;
  TypedNodeId<FileEnd> end;

  static auto Make(const Tree* tree) -> File {
    return tree->ExtractNodeFromChildren<File>(tree->roots());
  }
};

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

// Extract a `TypedNodeId<T>` as a single required child.
template <typename T>
struct Tree::Extractable<TypedNodeId<T>> {
  static auto Extract(const Tree* tree, SiblingIterator& it,
                      SiblingIterator end) -> std::optional<TypedNodeId<T>> {
    if (it == end || tree->node_kind(*it) != T::Kind) {
      llvm::errs() << "FIXME: Extract TypedNodeId " << tree->node_kind(*it)
                   << " != " << T::Kind << " error\n";
      return std::nullopt;
    }
    llvm::errs() << "FIXME: Extract TypedNodeId " << T::Kind << " success\n";
    return TypedNodeId<T>(*it++);
  }
};

// Extract an `NodeIdInCategory<Category>` as a single child.
template <NodeCategory Category>
struct Tree::Extractable<NodeIdInCategory<Category>> {
  static auto Extract(const Tree* tree, SiblingIterator& it,
                      SiblingIterator end)
      -> std::optional<NodeIdInCategory<Category>> {
    if (!Category) {
      llvm::errs() << "FIXME NodeIdInCategory <none>\n";
    }
#define CARBON_NODE_CATEGORY(Name)                        \
  if (!!(Category & NodeCategory::Name)) {                \
    llvm::errs() << "FIXME NodeIdInCategory " #Name "\n"; \
  }
    CARBON_NODE_CATEGORY(Decl)
    CARBON_NODE_CATEGORY(Expr)
    CARBON_NODE_CATEGORY(Modifier)
    CARBON_NODE_CATEGORY(NameComponent)
    CARBON_NODE_CATEGORY(Pattern)
    CARBON_NODE_CATEGORY(Statement)

    if (it == end || !(tree->node_kind(*it).category() & Category)) {
      llvm::errs() << "FIXME: Extract NodeIdInCategory " << tree->node_kind(*it)
                   << " error\n";
      return std::nullopt;
    }
    llvm::errs() << "FIXME: Extract NodeIdInCategory " << tree->node_kind(*it)
                 << " success\n";
    return NodeIdInCategory<Category>(*it++);
  }
};

// Extract a `NodeIdOneOf<T, U>` as a single required child.
template <typename T, typename U>
struct Tree::Extractable<NodeIdOneOf<T, U>> {
  static auto Extract(const Tree* tree, SiblingIterator& it,
                      SiblingIterator end) -> std::optional<NodeIdOneOf<T, U>> {
    auto kind = tree->node_kind(*it);
    llvm::errs() << "FIXME: Extract NodeIdOneOf " << kind << " ? " << T::Kind
                 << " or " << U::Kind;
    if (it == end || (kind != T::Kind && kind != U::Kind)) {
      llvm::errs() << " error\n";
      return std::nullopt;
    }
    llvm::errs() << " success\n";
    return NodeIdOneOf<T, U>(*it++);
  }
};

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

// Extract a `llvm::SmallVector<T>` by extracting `T`s until we can't.
template <typename T>
struct Tree::Extractable<llvm::SmallVector<T>> {
  static auto Extract(const Tree* tree, SiblingIterator& it,
                      SiblingIterator end)
      -> std::optional<llvm::SmallVector<T>> {
    llvm::SmallVector<T> result;
    while (it != end) {
      auto old_it = it;
      auto item = Extractable<T>::Extract(tree, it, end);
      if (!item.has_value()) {
        it = old_it;
        break;
      }
      result.push_back(*item);
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
