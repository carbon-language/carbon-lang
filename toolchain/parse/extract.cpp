// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <tuple>
#include <typeinfo>
#include <utility>

#include "common/error.h"
#include "common/struct_reflection.h"
#include "toolchain/parse/tree.h"
#include "toolchain/parse/typed_nodes.h"

namespace Carbon::Parse {

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
//   static auto Extract(Tree* tree, Tree::SiblingIterator& it,
//                       Tree::SiblingIterator end,
//                       ErrorBuilder* trace) -> std::optional<T>;
// };
// ```
//
// Note that `Tree::SiblingIterator`s iterate in reverse order through the
// children of a node.
//
// This class is only in this file.
template <typename T>
struct Extractable;

namespace {
// Common state maintained during the node extraction process. This includes
// both the current position within the parse tree, as well as context
// information about the tree and node being extracted.
class NodeExtractor {
 public:
  struct CheckpointState {
    Tree::SiblingIterator it;
  };

  NodeExtractor(const Tree* tree, Lex::TokenizedBuffer* tokens,
               ErrorBuilder* trace, NodeId node_id,
               llvm::iterator_range<Tree::SiblingIterator> children)
      : tree_(tree),
        tokens_(tokens),
        trace_(trace),
        node_id_(node_id),
        it_(children.begin()),
        end_(children.end()) {}

  auto at_end() const -> bool { return it_ == end_; }
  auto kind() const -> NodeKind { return tree_->node_kind(*it_); }
  auto has_token() const -> bool { return node_id_.is_valid(); }
  auto token() const -> Lex::TokenIndex { return tree_->node_token(node_id_); }
  auto token_kind() const -> Lex::TokenKind { return tokens_->GetKind(token()); }
  auto trace() const -> ErrorBuilder* { return trace_; }

  // Saves a checkpoint of our current position so we can return later if
  // extraction of a child node fails.
  auto Checkpoint() const -> CheckpointState { return {.it = it_}; }
  auto RestoreCheckpoint(CheckpointState checkpoint) { it_ = checkpoint.it; }

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
  template <typename T, typename... U, std::size_t... Index>
  auto ExtractTupleLikeType(std::index_sequence<Index...> /*indices*/,
                            std::tuple<U...>* /*type*/) -> std::optional<T>;

 private:
  const Tree* tree_;
  Lex::TokenizedBuffer* tokens_;
  ErrorBuilder* trace_;
  NodeId node_id_;
  Tree::SiblingIterator it_;
  Tree::SiblingIterator end_;
};
}  // namespace

// Extract a `NodeId` as a single child.
template <>
struct Extractable<NodeId> {
  static auto Extract(NodeExtractor& state) -> std::optional<NodeId> {
    if (state.at_end()) {
      if (auto* trace = state.trace()) {
        *trace << "NodeId error: no more children\n";
      }
      return std::nullopt;
    }
    if (auto* trace = state.trace()) {
      *trace << "NodeId: " << state.kind() << " consumed\n";
    }
    return state.ExtractNode();
  }
};

auto NodeExtractor::MatchesNodeIdForKind(NodeKind expected_kind) const -> bool {
  if (at_end() || kind() != expected_kind) {
    if (trace_) {
      if (at_end()) {
        *trace_ << "NodeIdForKind error: no more children, expected "
               << expected_kind << "\n";
      } else {
        *trace_ << "NodeIdForKind error: wrong kind " << kind() << ", expected "
               << expected_kind << "\n";
      }
    }
    return false;
  }
  if (trace_) {
    *trace_ << "NodeIdForKind: " << expected_kind << " consumed\n";
  }
  return true;
}

// Extract a `FooId`, which is the same as `NodeIdForKind<NodeKind::Foo>`,
// as a single required child.
template <const NodeKind& Kind>
struct Extractable<NodeIdForKind<Kind>> {
  static auto Extract(NodeExtractor& state)
      -> std::optional<NodeIdForKind<Kind>> {
    if (state.MatchesNodeIdForKind(Kind)) {
      return NodeIdForKind<Kind>(state.ExtractNode());
    } else {
      return std::nullopt;
    }
  }
};

auto NodeExtractor::MatchesNodeIdInCategory(NodeCategory category) const
    -> bool {
  if (at_end() || !kind().category().HasAnyOf(category)) {
    if (trace_) {
      *trace_ << "NodeIdInCategory " << category << " error: ";
      if (at_end()) {
        *trace_ << "no more children\n";
      } else {
        *trace_ << "kind " << kind() << " doesn't match\n";
      }
    }
    return false;
  }
  if (trace_) {
    *trace_ << "NodeIdInCategory " << category << ": kind " << kind()
           << " consumed\n";
  }
  return true;
}

// Extract a `NodeIdInCategory<Category>` as a single child.
template <NodeCategory::RawEnumType Category>
struct Extractable<NodeIdInCategory<Category>> {
  static auto Extract(NodeExtractor& state)
      -> std::optional<NodeIdInCategory<Category>> {
    if (state.MatchesNodeIdInCategory(Category)) {
      return NodeIdInCategory<Category>(state.ExtractNode());
    } else {
      return std::nullopt;
    }
  }
};

auto NodeExtractor::MatchesNodeIdOneOf(
    std::initializer_list<NodeKind> kinds) const -> bool {
  auto trace_kinds = [&] {
    llvm::ListSeparator sep(" or ");
    for (auto kind : kinds) {
      *trace_ << sep << kind;
    }
  };
  auto node_kind = kind();
  if (at_end() ||
      std::find(kinds.begin(), kinds.end(), node_kind) == kinds.end()) {
    if (trace_) {
      if (at_end()) {
        *trace_ << "NodeIdOneOf error: no more children, expected ";
        trace_kinds();
        *trace_ << "\n";
      } else {
        *trace_ << "NodeIdOneOf error: wrong kind " << node_kind
               << ", expected ";
        trace_kinds();
        *trace_ << "\n";
      }
    }
    return false;
  }
  if (trace_) {
    *trace_ << "NodeIdOneOf ";
    trace_kinds();
    *trace_ << ": " << node_kind << " consumed\n";
  }
  return true;
}

// Extract a `NodeIdOneOf<T...>` as a single required child.
template <typename... T>
struct Extractable<NodeIdOneOf<T...>> {
  static auto Extract(NodeExtractor& state) -> std::optional<NodeIdOneOf<T...>> {
    if (state.MatchesNodeIdOneOf({T::Kind...})) {
      return NodeIdOneOf<T...>(state.ExtractNode());
    } else {
      return std::nullopt;
    }
  }
};

// Extract a `NodeIdNot<T>` as a single required child.
// Note: this is only instantiated once, so no need to create a helper function.
template <typename T>
struct Extractable<NodeIdNot<T>> {
  static auto Extract(NodeExtractor& state) -> std::optional<NodeIdNot<T>> {
    if (state.at_end() || state.kind() == T::Kind) {
      if (auto* trace = state.trace()) {
        if (state.at_end()) {
          *trace << "NodeIdNot " << T::Kind
                       << " error: no more children\n";
        } else {
          *trace << "NodeIdNot error: unexpected " << T::Kind << "\n";
        }
      }
      return std::nullopt;
    }
    if (auto* trace = state.trace()) {
      *trace << "NodeIdNot " << T::Kind << ": " << state.kind()
                   << " consumed\n";
    }
    return NodeIdNot<T>(state.ExtractNode());
  }
};

// Extract an `llvm::SmallVector<T>` by extracting `T`s until we can't.
template <typename T>
struct Extractable<llvm::SmallVector<T>> {
  static auto Extract(NodeExtractor& state)
      -> std::optional<llvm::SmallVector<T>> {
    if (auto* trace = state.trace()) {
      *trace << "Vector: begin\n";
    }
    llvm::SmallVector<T> result;
    while (!state.at_end()) {
      auto checkpoint = state.Checkpoint();
      auto item = Extractable<T>::Extract(state);
      if (!item.has_value()) {
        state.RestoreCheckpoint(checkpoint);
        break;
      }
      result.push_back(*item);
    }
    std::reverse(result.begin(), result.end());
    if (auto* trace = state.trace()) {
      *trace << "Vector: end\n";
    }
    return result;
  }
};

// Extract an `optional<T>` from a list of child nodes by attempting to extract
// a `T`, and extracting nothing if that fails.
template <typename T>
struct Extractable<std::optional<T>> {
  static auto Extract(NodeExtractor& state) -> std::optional<std::optional<T>> {
    if (auto* trace = state.trace()) {
      *trace << "Optional " << typeid(T).name() << ": begin\n";
    }
    auto checkpoint = state.Checkpoint();
    std::optional<T> value = Extractable<T>::Extract(state);
    if (value) {
      if (auto* trace = state.trace()) {
        *trace << "Optional " << typeid(T).name() << ": found\n";
      }
      return value;
    }
    if (auto* trace = state.trace()) {
      *trace << "Optional " << typeid(T).name() << ": missing\n";
    }
    state.RestoreCheckpoint(checkpoint);
    return value;
    }
};

auto NodeExtractor::MatchesTokenKind(Lex::TokenKind expected_kind) const
    -> bool {
  if (!node_id_.is_valid()) {
    if (trace_) {
      *trace_ << "Token " << expected_kind
              << " expected but processing root node\n";
    }
    return false;
  }
  if (token_kind() != expected_kind) {
    if (trace_) {
      *trace_ << "Token " << expected_kind << " expected for "
              << tree_->node_kind(node_id_) << ", found " << token_kind()
              << "\n";
    }
    return false;
  }
  return true;
}

// Extract the token corresponding to a node.
template <const Lex::TokenKind& Kind, bool RequireIfInvalid>
struct Extractable<Token<Kind, RequireIfInvalid>> {
  static auto Extract(NodeExtractor& state)
      -> std::optional<Token<Kind, RequireIfInvalid>> {
    if (state.MatchesTokenKind(Kind)) {
      return Token<Kind, RequireIfInvalid>{.index = state.token()};
    } else {
      return std::nullopt;
    }
  }
};

// Extract the token corresponding to a node.
template <>
struct Extractable<AnyToken> {
  static auto Extract(NodeExtractor& state) -> std::optional<AnyToken> {
    if (!state.has_token()) {
      if (auto* trace = state.trace()) {
        *trace << "Token expected but processing root node\n";
      }
      return std::nullopt;
    }
    return AnyToken{.index = state.token()};
  }
};

template <typename T, typename... U, std::size_t... Index>
auto NodeExtractor::ExtractTupleLikeType(
    std::index_sequence<Index...> /*indices*/, std::tuple<U...>* /*type*/)
    -> std::optional<T> {
  std::tuple<std::optional<U>...> fields;
  if (trace_) {
    *trace_ << "Aggregate " << typeid(T).name() << ": begin\n";
  }
  // Use a fold over the `=` operator to parse fields from right to left.
  [[maybe_unused]] int unused;
  bool ok = true;
  static_cast<void>(
      ((ok && (ok = (std::get<Index>(fields) = Extractable<U>::Extract(*this))
                        .has_value()),
        unused) = ... = 0));
  if (!ok) {
    if (trace_) {
      *trace_ << "Aggregate " << typeid(T).name() << ": error\n";
    }
    return std::nullopt;
  }

  if (trace_) {
    *trace_ << "Aggregate " << typeid(T).name() << ": success\n";
  }
  return T{std::move(std::get<Index>(fields).value())...};
}

// Extract the fields of a simple aggregate type.
template <typename T>
struct Extractable {
  static_assert(std::is_aggregate_v<T>, "Unsupported child type");
  static auto ExtractImpl(NodeExtractor& state) -> std::optional<T> {
    // Compute the corresponding tuple type.
    using TupleType = decltype(StructReflection::AsTuple(std::declval<T>()));
    return state.ExtractTupleLikeType<T>(
        std::make_index_sequence<std::tuple_size_v<TupleType>>(),
        static_cast<TupleType*>(nullptr));
  }

  static auto Extract(NodeExtractor& state) -> std::optional<T> {
    static_assert(!HasKindMember<T>, "Missing Id suffix");
    return ExtractImpl(state);
  }
};

template <typename T>
auto Tree::TryExtractNodeFromChildren(
    NodeId node_id, llvm::iterator_range<Tree::SiblingIterator> children,
    ErrorBuilder* trace) const -> std::optional<T> {
  NodeExtractor extractor(this, tokens_, trace, node_id, children);
  auto result = Extractable<T>::ExtractImpl(extractor);
  if (!extractor.at_end()) {
    if (trace) {
      *trace << "Error: " << node_kind(extractor.ExtractNode())
             << " node left unconsumed.";
    }
    return std::nullopt;
  }
  return result;
}

// Manually instantiate Tree::TryExtractNodeFromChildren
#define CARBON_PARSE_NODE_KIND(KindName)                                    \
  template auto Tree::TryExtractNodeFromChildren<KindName>(                 \
      NodeId node_id, llvm::iterator_range<Tree::SiblingIterator> children, \
      ErrorBuilder * trace) const -> std::optional<KindName>;

// Also instantiate for `File`, even though it isn't a parse node.
CARBON_PARSE_NODE_KIND(File)
#include "toolchain/parse/node_kind.def"

auto Tree::ExtractFile() const -> File {
  return ExtractNodeFromChildren<File>(NodeId::Invalid, roots());
}

}  // namespace Carbon::Parse
