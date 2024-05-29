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

struct Tree::ExtractState {
  const Tree* tree;
  Tree::SiblingIterator it;
  Tree::SiblingIterator end;
  ErrorBuilder* trace;
  NodeId node_id;

  auto at_end() const -> bool { return it == end; }
  auto kind() const -> NodeKind { return tree->node_kind(*it); }
  auto token() const -> Lex::TokenIndex { return tree->node_token(node_id); }
  auto token_kind() const -> Lex::TokenKind {
    return tree->tokens_->GetKind(token());
  }
};

using ExtractState = Tree::ExtractState;

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
    return *state.it++;
  }
};

static auto NodeIdForKindAccept(const NodeKind& kind,
                                ExtractState& state) -> bool {
  if (state.at_end() || state.kind() != kind) {
    if (state.trace) {
      if (state.at_end()) {
        *state.trace << "NodeIdForKind error: no more children, expected "
                     << kind << "\n";
      } else {
        *state.trace << "NodeIdForKind error: wrong kind " << state.kind()
                     << ", expected " << kind << "\n";
      }
    }
    return false;
  }
  if (state.trace) {
    *state.trace << "NodeIdForKind: " << kind << " consumed\n";
  }
  return true;
}

// Extract a `FooId`, which is the same as `NodeIdForKind<NodeKind::Foo>`,
// as a single required child.
template <const NodeKind& Kind>
struct Extractable<NodeIdForKind<Kind>> {
  static auto Extract(ExtractState& state)
      -> std::optional<NodeIdForKind<Kind>> {
    if (NodeIdForKindAccept(Kind, state)) {
      return NodeIdForKind<Kind>(*state.it++);
    } else {
      return std::nullopt;
    }
  }
};

static auto NodeIdInCategoryAccept(NodeCategory category, ExtractState& state) -> bool {
  if (state.at_end() || !(state.kind().category() & category)) {
    if (state.trace) {
      *state.trace << "NodeIdInCategory " << category << " error: ";
      if (state.at_end()) {
        *state.trace << "no more children\n";
      } else {
        *state.trace << "kind " << state.kind() << " doesn't match\n";
      }
    }
    return false;
  }
  if (state.trace) {
    *state.trace << "NodeIdInCategory " << category << ": kind " << state.kind()
                 << " consumed\n";
  }
  return true;
}

// Extract a `NodeIdInCategory<Category>` as a single child.
template <NodeCategory Category>
struct Extractable<NodeIdInCategory<Category>> {
  static auto Extract(ExtractState& state)
      -> std::optional<NodeIdInCategory<Category>> {
    if (NodeIdInCategoryAccept(Category, state)) {
      return NodeIdInCategory<Category>(*state.it++);
    } else {
      return std::nullopt;
    }
  }
};

static auto NodeIdOneOfAccept(std::initializer_list<NodeKind> kinds,
                              ExtractState& state) -> bool {
  auto trace_kinds = [&] {
    llvm::ListSeparator sep(" or ");
    for (auto kind : kinds) {
      *state.trace << sep << kind;
    }
  };
  auto kind = state.kind();
  if (state.at_end() || std::find(kinds.begin(), kinds.end(), kind) == kinds.end()) {
    if (state.trace) {
      if (state.at_end()) {
        *state.trace << "NodeIdOneOf error: no more children, expected ";
        trace_kinds();
        *state.trace << "\n";
      } else {
        *state.trace << "NodeIdOneOf error: wrong kind " << state.kind()
                     << ", expected ";
        trace_kinds();
        *state.trace << "\n";
      }
    }
    return false;
  }
  if (state.trace) {
    *state.trace << "NodeIdOneOf ";
    trace_kinds();
    *state.trace << ": " << kind << " consumed\n";
  }
  return true;
}

// Extract a `NodeIdOneOf<T...>` as a single required child.
template <typename... T>
struct Extractable<NodeIdOneOf<T...>> {
  static auto Extract(ExtractState& state) -> std::optional<NodeIdOneOf<T...>> {
    if (NodeIdOneOfAccept({T::Kind...}, state)) {
      return NodeIdOneOf<T...>(*state.it++);
    } else {
      return std::nullopt;
    }
  }
};

// Extract a `NodeIdNot<T>` as a single required child.
// Note: this is only instantiated once, so no need to create a helper function.
template <typename T>
struct Extractable<NodeIdNot<T>> {
  static auto Extract(ExtractState& state)
      -> std::optional<NodeIdNot<T>> {
    if (state.at_end() || state.kind() == T::Kind) {
      if (state.trace) {
        if (state.at_end()) {
          *state.trace << "NodeIdNot " << T::Kind << " error: no more children\n";
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
    return NodeIdNot<T>(*state.it++);
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
      auto old_state = state;
      auto item = Extractable<T>::Extract(state);
      if (!item.has_value()) {
        state = old_state;
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
  static auto Extract(ExtractState& state)
      -> std::optional<std::optional<T>> {
    if (state.trace) {
      *state.trace << "Optional " << typeid(T).name() << ": begin\n";
    }
    auto old_state = state;
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
    state = old_state;
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
        *state.trace << "No token for root node\n";
      }
      return std::nullopt;
    }
    if ((RequireIfInvalid || !state.tree->node_has_error(state.node_id)) &&
        state.token_kind() != Kind) {
      if (state.trace) {
        *state.trace << "Expected token kind " << Kind << ", found "
                     << state.token_kind() << "\n";
        return std::nullopt;
      }
    }
    return Token<Kind, RequireIfInvalid>(state.token());
  }
};

// Extract the token corresponding to a node.
template <>
struct Extractable<AnyToken> {
  static auto Extract(ExtractState& state) -> std::optional<AnyToken> {
    if (!state.node_id.is_valid()) {
      if (state.trace) {
        *state.trace << "No token for root node\n";
      }
      return std::nullopt;
    }
    return AnyToken(state.token());
  }
};

template <typename T, typename... U, std::size_t... Index>
static auto ExtractTupleLikeType(
    ExtractState& state, std::index_sequence<Index...> /*indices*/,
    std::tuple<U...>* /*type*/) -> std::optional<T> {
  std::tuple<std::optional<U>...> fields;
  if (state.trace) {
    *state.trace << "Aggregate " << typeid(T).name() << ": begin\n";
  }
  // Use a fold over the `=` operator to parse fields from right to left.
  [[maybe_unused]] int unused;
  bool ok = true;
  static_cast<void>(
      ((ok && (ok = (std::get<Index>(fields) = Extractable<U>::Extract(state))
                        .has_value()),
        unused) = ... = 0));
  if (!ok) {
    if (state.trace) {
      *state.trace << "Aggregate " << typeid(T).name() << ": error\n";
    }
    return std::nullopt;
  }

  if (state.trace) {
    *state.trace << "Aggregate " << typeid(T).name() << ": success\n";
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
    return ExtractTupleLikeType<T>(
        state, std::make_index_sequence<std::tuple_size_v<TupleType>>(),
        static_cast<TupleType*>(nullptr));
  }

  static auto Extract(ExtractState& state) -> std::optional<T> {
    static_assert(!HasKindMember<T>, "Missing Id suffix");
    return ExtractImpl(state);
  }
};

template <typename T>
auto Tree::TryExtractNodeFromChildren(
  NodeId node_id,
    llvm::iterator_range<Tree::SiblingIterator> children,
    ErrorBuilder* trace) const -> std::optional<T> {
  ExtractState state = {.tree = this,
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
