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
struct ExtractState {
  struct CheckpointState {
    Tree::SiblingIterator it;
  };

  const Tree* tree;
  Lex::TokenizedBuffer* tokens;
  Tree::SiblingIterator it;
  Tree::SiblingIterator end;
  ErrorBuilder* trace;
  NodeId node_id;

  auto at_end() const -> bool { return it == end; }
  auto kind() const -> NodeKind { return tree->node_kind(*it); }
  auto token() const -> Lex::TokenIndex { return tree->node_token(node_id); }
  auto token_kind() const -> Lex::TokenKind { return tokens->GetKind(token()); }

  // Saves a checkpoint of our current position so we can return later if
  // extraction of a child node fails.
  auto Checkpoint() const -> CheckpointState { return {.it = it}; }
  auto RestoreCheckpoint(CheckpointState checkpoint) { it = checkpoint.it; }

  // Determines whether the current position matches the specified node kind. If
  // not, produces a suitable trace message.
  auto MatchesNodeIdForKind(NodeKind kind) const -> bool;

  // Determines whether the current position matches the specified node
  // category. If not, produces a suitable trace message.
  auto MatchesNodeIdInCategory(NodeCategory category) const -> bool;

  // Determines whether the current position matches any of the specified node
  // kinds. If not, produces a suitable trace message.
  auto MatchesNodeIdOneOf(std::initializer_list<NodeKind> kinds) const -> bool;

  // Extracts the next node from the tree.
  auto ExtractNode() -> NodeId { return *it++; }

  // Extracts a tuple-like type `T` by extracting its components and then
  // assembling a `T` value.
  template <typename T, typename... U, std::size_t... Index>
  auto ExtractTupleLikeType(std::index_sequence<Index...> /*indices*/,
                            std::tuple<U...>* /*type*/) -> std::optional<T>;
};
}  // namespace

// Extract a `NodeId` as a single child.
template <>
struct Extractable<NodeId> {
  static auto Extract(ExtractState& state) -> std::optional<NodeId> {
    if (state.at_end()) {
      if (state.trace) {
        *state.trace << "NodeId error: no more children\n";
      }
      return std::nullopt;
    }
    if (state.trace) {
      *state.trace << "NodeId: " << state.kind() << " consumed\n";
    }
    return state.ExtractNode();
  }
};

auto ExtractState::MatchesNodeIdForKind(NodeKind expected_kind) const -> bool {
  if (at_end() || kind() != expected_kind) {
    if (trace) {
      if (at_end()) {
        *trace << "NodeIdForKind error: no more children, expected "
               << expected_kind << "\n";
      } else {
        *trace << "NodeIdForKind error: wrong kind " << kind() << ", expected "
               << expected_kind << "\n";
      }
    }
    return false;
  }
  if (trace) {
    *trace << "NodeIdForKind: " << expected_kind << " consumed\n";
  }
  return true;
}

// Extract a `FooId`, which is the same as `NodeIdForKind<NodeKind::Foo>`,
// as a single required child.
template <const NodeKind& Kind>
struct Extractable<NodeIdForKind<Kind>> {
  static auto Extract(ExtractState& state)
      -> std::optional<NodeIdForKind<Kind>> {
    if (state.MatchesNodeIdForKind(Kind)) {
      return NodeIdForKind<Kind>(state.ExtractNode());
    } else {
      return std::nullopt;
    }
  }
};

auto ExtractState::MatchesNodeIdInCategory(NodeCategory category) const
    -> bool {
  if (at_end() || !kind().category().HasAnyOf(category)) {
    if (trace) {
      *trace << "NodeIdInCategory " << category << " error: ";
      if (at_end()) {
        *trace << "no more children\n";
      } else {
        *trace << "kind " << kind() << " doesn't match\n";
      }
    }
    return false;
  }
  if (trace) {
    *trace << "NodeIdInCategory " << category << ": kind " << kind()
           << " consumed\n";
  }
  return true;
}

// Extract a `NodeIdInCategory<Category>` as a single child.
template <NodeCategory::RawEnumType Category>
struct Extractable<NodeIdInCategory<Category>> {
  static auto Extract(ExtractState& state)
      -> std::optional<NodeIdInCategory<Category>> {
    if (state.MatchesNodeIdInCategory(Category)) {
      return NodeIdInCategory<Category>(state.ExtractNode());
    } else {
      return std::nullopt;
    }
  }
};

auto ExtractState::MatchesNodeIdOneOf(
    std::initializer_list<NodeKind> kinds) const -> bool {
  auto trace_kinds = [&] {
    llvm::ListSeparator sep(" or ");
    for (auto kind : kinds) {
      *trace << sep << kind;
    }
  };
  auto node_kind = kind();
  if (at_end() ||
      std::find(kinds.begin(), kinds.end(), node_kind) == kinds.end()) {
    if (trace) {
      if (at_end()) {
        *trace << "NodeIdOneOf error: no more children, expected ";
        trace_kinds();
        *trace << "\n";
      } else {
        *trace << "NodeIdOneOf error: wrong kind " << node_kind
               << ", expected ";
        trace_kinds();
        *trace << "\n";
      }
    }
    return false;
  }
  if (trace) {
    *trace << "NodeIdOneOf ";
    trace_kinds();
    *trace << ": " << node_kind << " consumed\n";
  }
  return true;
}

// Extract a `NodeIdOneOf<T...>` as a single required child.
template <typename... T>
struct Extractable<NodeIdOneOf<T...>> {
  static auto Extract(ExtractState& state) -> std::optional<NodeIdOneOf<T...>> {
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
  static auto Extract(ExtractState& state) -> std::optional<NodeIdNot<T>> {
    if (state.at_end() || state.kind() == T::Kind) {
      if (state.trace) {
        if (state.at_end()) {
          *state.trace << "NodeIdNot " << T::Kind
                       << " error: no more children\n";
        } else {
          *state.trace << "NodeIdNot error: unexpected " << T::Kind << "\n";
        }
      }
      return std::nullopt;
    }
    if (state.trace) {
      *state.trace << "NodeIdNot " << T::Kind << ": " << state.kind()
                   << " consumed\n";
    }
    return NodeIdNot<T>(state.ExtractNode());
  }
};

// Extract an `llvm::SmallVector<T>` by extracting `T`s until we can't.
template <typename T>
struct Extractable<llvm::SmallVector<T>> {
  static auto Extract(ExtractState& state)
      -> std::optional<llvm::SmallVector<T>> {
    if (state.trace) {
      *state.trace << "Vector: begin\n";
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
    if (state.trace) {
      *state.trace << "Vector: end\n";
    }
    return result;
  }
};

// Extract an `optional<T>` from a list of child nodes by attempting to extract
// a `T`, and extracting nothing if that fails.
template <typename T>
struct Extractable<std::optional<T>> {
  static auto Extract(ExtractState& state) -> std::optional<std::optional<T>> {
    if (state.trace) {
      *state.trace << "Optional " << typeid(T).name() << ": begin\n";
    }
    auto checkpoint = state.Checkpoint();
    std::optional<T> value = Extractable<T>::Extract(state);
    if (value) {
      if (state.trace) {
        *state.trace << "Optional " << typeid(T).name() << ": found\n";
      }
      return value;
    }
    if (state.trace) {
      *state.trace << "Optional " << typeid(T).name() << ": missing\n";
    }
    state.RestoreCheckpoint(checkpoint);
    return value;
  }
};

// Extract the token corresponding to a node.
template <const Lex::TokenKind& Kind, bool RequireIfInvalid>
struct Extractable<Token<Kind, RequireIfInvalid>> {
  static auto Extract(ExtractState& state)
      -> std::optional<Token<Kind, RequireIfInvalid>> {
    if (!state.node_id.is_valid()) {
      if (state.trace) {
        *state.trace << "Token " << Kind
                     << " expected but processing root node\n";
      }
      return std::nullopt;
    }
    if ((RequireIfInvalid || !state.tree->node_has_error(state.node_id)) &&
        state.token_kind() != Kind) {
      if (state.trace) {
        *state.trace << "Token " << Kind << " expected for "
                     << state.tree->node_kind(state.node_id) << ", found "
                     << state.token_kind() << "\n";
      }
      return std::nullopt;
    }
    return Token<Kind, RequireIfInvalid>{.index = state.token()};
  }
};

// Extract the token corresponding to a node.
template <>
struct Extractable<AnyToken> {
  static auto Extract(ExtractState& state) -> std::optional<AnyToken> {
    if (!state.node_id.is_valid()) {
      if (state.trace) {
        *state.trace << "Token expected but processing root node\n";
      }
      return std::nullopt;
    }
    return AnyToken{.index = state.token()};
  }
};

template <typename T, typename... U, std::size_t... Index>
auto ExtractState::ExtractTupleLikeType(
    std::index_sequence<Index...> /*indices*/, std::tuple<U...>* /*type*/)
    -> std::optional<T> {
  std::tuple<std::optional<U>...> fields;
  if (trace) {
    *trace << "Aggregate " << typeid(T).name() << ": begin\n";
  }
  // Use a fold over the `=` operator to parse fields from right to left.
  [[maybe_unused]] int unused;
  bool ok = true;
  static_cast<void>(
      ((ok && (ok = (std::get<Index>(fields) = Extractable<U>::Extract(*this))
                        .has_value()),
        unused) = ... = 0));
  if (!ok) {
    if (trace) {
      *trace << "Aggregate " << typeid(T).name() << ": error\n";
    }
    return std::nullopt;
  }

  if (trace) {
    *trace << "Aggregate " << typeid(T).name() << ": success\n";
  }
  return T{std::move(std::get<Index>(fields).value())...};
}

// Extract the fields of a simple aggregate type.
template <typename T>
struct Extractable {
  static_assert(std::is_aggregate_v<T>, "Unsupported child type");
  static auto ExtractImpl(ExtractState& state) -> std::optional<T> {
    // Compute the corresponding tuple type.
    using TupleType = decltype(StructReflection::AsTuple(std::declval<T>()));
    return state.ExtractTupleLikeType<T>(
        std::make_index_sequence<std::tuple_size_v<TupleType>>(),
        static_cast<TupleType*>(nullptr));
  }

  static auto Extract(ExtractState& state) -> std::optional<T> {
    static_assert(!HasKindMember<T>, "Missing Id suffix");
    return ExtractImpl(state);
  }
};

template <typename T>
auto Tree::TryExtractNodeFromChildren(
    NodeId node_id, llvm::iterator_range<Tree::SiblingIterator> children,
    ErrorBuilder* trace) const -> std::optional<T> {
  ExtractState state = {.tree = this,
                        .tokens = tokens_,
                        .it = children.begin(),
                        .end = children.end(),
                        .trace = trace,
                        .node_id = node_id};
  auto result = Extractable<T>::ExtractImpl(state);
  if (state.it != children.end()) {
    if (trace) {
      *trace << "Error: " << node_kind(*state.it) << " node left unconsumed.";
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
