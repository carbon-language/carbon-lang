// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSE_EXTRACT_H_
#define CARBON_TOOLCHAIN_PARSE_EXTRACT_H_

#include <initializer_list>
#include <optional>
#include <tuple>
#include <utility>

#include "common/error.h"
#include "common/find.h"
#include "common/struct_reflection.h"
#include "llvm/Support/TypeName.h"
#include "toolchain/parse/tree.h"
#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/parse/typed_nodes.h"

namespace Carbon::Parse {

// Implementation of the process of extracting a typed node structure from the
// parse tree. The extraction process uses the class `Extractable<T>`, defined
// below, to extract individual fields of type `T`.
class NodeExtractor {
 public:
  struct CheckpointState {
    TreeAndSubtrees::SiblingIterator it;
  };

  NodeExtractor(const TreeAndSubtrees* tree, const Lex::TokenizedBuffer* tokens,
                ErrorBuilder* trace, NodeId node_id,
                llvm::iterator_range<TreeAndSubtrees::SiblingIterator> children)
      : tree_(tree),
        tokens_(tokens),
        trace_(trace),
        node_id_(node_id),
        it_(children.begin()),
        end_(children.end()) {}

  auto at_end() const -> bool { return it_ == end_; }
  auto kind() const -> NodeKind { return tree_->tree().node_kind(*it_); }
  auto has_token() const -> bool { return node_id_.has_value(); }
  auto token() const -> Lex::TokenIndex {
    return tree_->tree().node_token(node_id_);
  }
  auto token_kind() const -> Lex::TokenKind {
    return tokens_->GetKind(token());
  }
  auto trace() const -> ErrorBuilder* { return trace_; }

  // Saves a checkpoint of our current position so we can return later if
  // extraction of a child node fails.
  auto Checkpoint() const -> CheckpointState { return {.it = it_}; }
  auto RestoreCheckpoint(CheckpointState checkpoint) -> void {
    it_ = checkpoint.it;
  }

  // Determines whether the current position matches the specified node kind. If
  // not, produces a suitable trace message.
  auto MatchesNodeIdForKind(NodeKind kind) const -> bool;

  // Determines whether the current position matches the specified node
  // category. If not, produces a suitable trace message.
  auto MatchesNodeIdInCategory(NodeCategory category) const -> bool;

  // Determines whether the current position matches any of the specified node
  // kinds. If not, produces a suitable trace message.
  auto MatchesNodeIdOneOf(std::initializer_list<NodeKind> kinds) const -> bool;

  // Determines whether the token corresponding to the enclosing node is of the
  // specified kind. If not, produces a suitable trace message.
  auto MatchesTokenKind(Lex::TokenKind expected_kind) const -> bool;

  // Extracts the next node from the tree.
  auto ExtractNode() -> NodeId { return *it_++; }

  // Extracts a tuple-like type `T` by extracting its components and then
  // assembling a `T` value.
  template <typename T, typename... U, size_t... Index>
  auto ExtractTupleLikeType(std::index_sequence<Index...> /*indices*/,
                            std::tuple<U...>* /*type*/) -> std::optional<T>;

  // Split out trace logic. The noinline saves a few seconds on compilation.
  // TODO: Switch format to `llvm::StringLiteral` if
  // `llvm::StringLiteral::c_str` is added.
  template <typename... ArgT>
  [[clang::noinline]] auto MaybeTrace(const char* format, ArgT... args) const
      -> void {
    if (trace_) {
      *trace_ << llvm::formatv(format, args...);
    }
  }

  auto tree() -> const Tree& { return tree_->tree(); }

 private:
  const TreeAndSubtrees* tree_;
  const Lex::TokenizedBuffer* tokens_;
  ErrorBuilder* trace_;
  NodeId node_id_;
  TreeAndSubtrees::SiblingIterator it_;
  TreeAndSubtrees::SiblingIterator end_;
};

// A trait type that should be specialized by types that can be extracted
// from a parse tree. A specialization should provide the following API:
//
// ```cpp
// template<>
// struct Extractable<T> {
//   // Extract a value of this type from the sequence of nodes starting at
//   // `it`, and increment `it` past this type. Returns `std::nullopt` if
//   // the tree is malformed. If `trace != nullptr`, writes what actions
//   // were taken to `*trace`.
//   static auto Extract(NodeExtractor* extractor) -> std::optional<T>;
// };
// ```
//
// Note that `TreeAndSubtrees::SiblingIterator`s iterate in reverse order
// through the children of a node.
template <typename T>
struct Extractable;

// Extract a `NodeId` as a single child.
template <>
struct Extractable<NodeId> {
  static auto Extract(NodeExtractor& extractor) -> std::optional<NodeId> {
    if (extractor.at_end()) {
      extractor.MaybeTrace("NodeId error: no more children\n");
      return std::nullopt;
    }
    extractor.MaybeTrace("NodeId: {0} consumed\n", extractor.kind());
    return extractor.ExtractNode();
  }
};

// Extract a `FooId`, which is the same as `NodeIdForKind<NodeKind::Foo>`,
// as a single required child.
template <const NodeKind& Kind>
struct Extractable<NodeIdForKind<Kind>> {
  static auto Extract(NodeExtractor& extractor)
      -> std::optional<NodeIdForKind<Kind>> {
    if (extractor.MatchesNodeIdForKind(Kind)) {
      return extractor.tree().As<NodeIdForKind<Kind>>(extractor.ExtractNode());
    } else {
      return std::nullopt;
    }
  }
};

// Extract a `NodeIdInCategory<Category>` as a single child.
template <NodeCategory::RawEnumType Category>
struct Extractable<NodeIdInCategory<Category>> {
  static auto Extract(NodeExtractor& extractor)
      -> std::optional<NodeIdInCategory<Category>> {
    if (extractor.MatchesNodeIdInCategory(Category)) {
      return extractor.tree().As<NodeIdInCategory<Category>>(
          extractor.ExtractNode());
    } else {
      return std::nullopt;
    }
  }
};

// Extract a `NodeIdOneOf<T...>` as a single required child.
template <typename... T>
struct Extractable<NodeIdOneOf<T...>> {
  static auto Extract(NodeExtractor& extractor)
      -> std::optional<NodeIdOneOf<T...>> {
    if (extractor.MatchesNodeIdOneOf({T::Kind...})) {
      return extractor.tree().As<NodeIdOneOf<T...>>(extractor.ExtractNode());
    } else {
      return std::nullopt;
    }
  }
};

// Extract a `NodeIdNot<T>` as a single required child.
// Note: this is only instantiated once, so no need to create a helper function.
template <typename T>
struct Extractable<NodeIdNot<T>> {
  static auto Extract(NodeExtractor& extractor) -> std::optional<NodeIdNot<T>> {
    // This converts NodeKind::Definition to NodeKind.
    constexpr NodeKind Kind = T::Kind;
    if (extractor.at_end()) {
      extractor.MaybeTrace("NodeIdNot {0} error: no more children\n", Kind);
      return std::nullopt;
    } else if (extractor.kind() == Kind) {
      extractor.MaybeTrace("NodeIdNot error: unexpected {0}\n", Kind);
      return std::nullopt;
    }
    extractor.MaybeTrace("NodeIdNot {0}: {1} consumed\n", Kind,
                         extractor.kind());
    return NodeIdNot<T>(extractor.ExtractNode());
  }
};

// Extract an `llvm::SmallVector<T>` by extracting `T`s until we can't.
template <typename T>
struct Extractable<llvm::SmallVector<T>> {
  static auto Extract(NodeExtractor& extractor)
      -> std::optional<llvm::SmallVector<T>> {
    extractor.MaybeTrace("Vector: begin\n");
    llvm::SmallVector<T> result;
    while (!extractor.at_end()) {
      auto checkpoint = extractor.Checkpoint();
      auto item = Extractable<T>::Extract(extractor);
      if (!item.has_value()) {
        extractor.RestoreCheckpoint(checkpoint);
        break;
      }
      result.push_back(*item);
    }
    std::reverse(result.begin(), result.end());
    extractor.MaybeTrace("Vector: end\n");
    return result;
  }
};

// Extract an `optional<T>` from a list of child nodes by attempting to extract
// a `T`, and extracting nothing if that fails.
template <typename T>
struct Extractable<std::optional<T>> {
  static auto Extract(NodeExtractor& extractor)
      -> std::optional<std::optional<T>> {
    extractor.MaybeTrace("Optional {0}: begin\n", llvm::getTypeName<T>());
    auto checkpoint = extractor.Checkpoint();
    std::optional<T> value = Extractable<T>::Extract(extractor);
    if (value) {
      extractor.MaybeTrace("Optional {0}: found\n", llvm::getTypeName<T>());
    } else {
      extractor.MaybeTrace("Optional {0}: missing\n", llvm::getTypeName<T>());
      extractor.RestoreCheckpoint(checkpoint);
    }
    return value;
  }
};

// Extract the token corresponding to a node.
template <const Lex::TokenKind& Kind>
struct Extractable<Lex::TokenIndexForKind<Kind>> {
  static auto Extract(NodeExtractor& extractor)
      -> std::optional<Lex::TokenIndexForKind<Kind>> {
    if (extractor.MatchesTokenKind(Kind)) {
      return static_cast<Lex::TokenIndexForKind<Kind>>(extractor.token());
    } else {
      return std::nullopt;
    }
  }
};

// Extract the token corresponding to a node.
template <>
struct Extractable<Lex::TokenIndex> {
  static auto Extract(NodeExtractor& extractor)
      -> std::optional<Lex::TokenIndex> {
    if (!extractor.has_token()) {
      extractor.MaybeTrace("Token expected but processing root node\n");
      return std::nullopt;
    }
    return extractor.token();
  }
};

template <typename T, typename... U, size_t... Index>
auto NodeExtractor::ExtractTupleLikeType(
    std::index_sequence<Index...> /*indices*/, std::tuple<U...>* /*type*/)
    -> std::optional<T> {
  std::tuple<std::optional<U>...> fields;
  MaybeTrace("Aggregate {0}: begin\n", llvm::getTypeName<T>());
  // Use a fold over the `=` operator to parse fields from right to left.
  [[maybe_unused]] int unused;
  bool ok = true;
  static_cast<void>(
      ((ok && (ok = (std::get<Index>(fields) = Extractable<U>::Extract(*this))
                        .has_value()),
        unused) = ... = 0));
  if (!ok) {
    MaybeTrace("Aggregate {0}: error\n", llvm::getTypeName<T>());
    return std::nullopt;
  }

  MaybeTrace("Aggregate {0}: success\n", llvm::getTypeName<T>());
  return T{std::move(std::get<Index>(fields).value())...};
}

// Extract the fields of a simple aggregate type.
template <typename T>
struct Extractable {
  static_assert(std::is_aggregate_v<T>, "Unsupported child type");
  static auto ExtractImpl(NodeExtractor& extractor) -> std::optional<T> {
    // Compute the corresponding tuple type.
    using TupleType = decltype(StructReflection::AsTuple(std::declval<T>()));
    return extractor.ExtractTupleLikeType<T>(
        std::make_index_sequence<std::tuple_size_v<TupleType>>(),
        static_cast<TupleType*>(nullptr));
  }

  static auto Extract(NodeExtractor& extractor) -> std::optional<T> {
    static_assert(!HasKindMember<T>, "Missing Id suffix");
    return ExtractImpl(extractor);
  }
};

template <typename T>
auto TreeAndSubtrees::TryExtractNodeFromChildren(
    NodeId node_id,
    llvm::iterator_range<TreeAndSubtrees::SiblingIterator> children,
    ErrorBuilder* trace) const -> std::optional<T> {
  NodeExtractor extractor(this, tokens_, trace, node_id, children);
  auto result = Extractable<T>::ExtractImpl(extractor);
  if (!extractor.at_end()) {
    if (trace) {
      *trace << "Error: " << tree_->node_kind(extractor.ExtractNode())
             << " node left unconsumed.";
    }
    return std::nullopt;
  }
  return result;
}

}  // namespace Carbon::Parse

#endif  // CARBON_TOOLCHAIN_PARSE_EXTRACT_H_
