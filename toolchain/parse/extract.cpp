// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <tuple>
#include <utility>

#include "common/error.h"
#include "common/struct_reflection.h"
#include "toolchain/parse/extract_file.h"
#include "toolchain/parse/tree.h"
#include "toolchain/parse/typed_nodes.h"

namespace Carbon::Parse {

// A trait type that should be specialized by types that can be extracted
// from a parse tree. A specialization should provide the following API:
//
// ```
// interface Extractable {
//   // Extract a value of this type from the sequence of nodes starting at
//   // `it`, and increment `it` past this type. Returns `std::nullopt` if
//   // the tree is malformed. If `trace != nullptr`, writes what actions
//   // were taken to `*trace`.
//   static auto Extract(Tree* tree, Tree::SiblingIterator& it,
//                       Tree::SiblingIterator end,
//                       ErrorBuilder* trace) -> std::optional<Self>;
// }
// ```
//
// Note that `Tree::SiblingIterator`s iterate in reverse order through the
// children of a node.
//
// This class is only in this file.
template <typename T>
struct Extractable;

// Extract an `NodeId` as a single child.
template <>
struct Extractable<NodeId> {
  static auto Extract(const Tree* tree, Tree::SiblingIterator& it,
                      Tree::SiblingIterator end, ErrorBuilder* trace)
      -> std::optional<NodeId> {
    if (it == end) {
      if (trace) {
        *trace << "NodeId error: no more children\n";
      }
      return std::nullopt;
    }
    if (trace) {
      *trace << "NodeId: " << tree->node_kind(*it) << " consumed\n";
    }
    return NodeId(*it++);
  }
};

// Extract a `FooId`, which is the same as `NodeIdForKind<NodeKind::Foo>`,
// as a single required child.
template <const NodeKind& Kind>
struct Extractable<NodeIdForKind<Kind>> {
  static auto Extract(const Tree* tree, Tree::SiblingIterator& it,
                      Tree::SiblingIterator end, ErrorBuilder* trace)
      -> std::optional<NodeIdForKind<Kind>> {
    if (it == end || tree->node_kind(*it) != Kind) {
      if (trace) {
        if (it == end) {
          *trace << "NodeIdForKind error: no more children, expected " << Kind
                 << "\n";
        } else {
          *trace << "NodeIdForKind error: wrong kind " << tree->node_kind(*it)
                 << ", expected " << Kind << "\n";
        }
      }
      return std::nullopt;
    }
    if (trace) {
      *trace << "NodeIdForKind: " << Kind << " consumed\n";
    }
    return NodeIdForKind<Kind>(*it++);
  }
};

// Extract an `NodeIdInCategory<Category>` as a single child.
template <NodeCategory Category>
struct Extractable<NodeIdInCategory<Category>> {
  static auto Extract(const Tree* tree, Tree::SiblingIterator& it,
                      Tree::SiblingIterator end, ErrorBuilder* trace)
      -> std::optional<NodeIdInCategory<Category>> {
    if (trace) {
      *trace << "NodeIdInCategory";
      if (!Category) {
        *trace << " <none>";
      }
#define CARBON_NODE_CATEGORY(Name)         \
  if (!!(Category & NodeCategory::Name)) { \
    *trace << " " #Name;                   \
  }
      CARBON_NODE_CATEGORY(Decl);
      CARBON_NODE_CATEGORY(Expr);
      CARBON_NODE_CATEGORY(Modifier);
      CARBON_NODE_CATEGORY(NameComponent);
      CARBON_NODE_CATEGORY(Pattern);
      CARBON_NODE_CATEGORY(Statement);
#undef CARBON_NODE_CATEGORY
    }

    if (it == end || !(tree->node_kind(*it).category() & Category)) {
      if (trace) {
        if (it == end) {
          *trace << " error: no more children\n";
        } else {
          *trace << " error: kind " << tree->node_kind(*it)
                 << " doesn't match\n";
        }
      }
      return std::nullopt;
    }
    if (trace) {
      *trace << ": kind " << tree->node_kind(*it) << " consumed\n";
    }
    return NodeIdInCategory<Category>(*it++);
  }
};

// Extract a `NodeIdOneOf<T, U>` as a single required child.
template <typename T, typename U>
struct Extractable<NodeIdOneOf<T, U>> {
  static auto Extract(const Tree* tree, Tree::SiblingIterator& it,
                      Tree::SiblingIterator end, ErrorBuilder* trace)
      -> std::optional<NodeIdOneOf<T, U>> {
    auto kind = tree->node_kind(*it);
    if (it == end || (kind != T::Kind && kind != U::Kind)) {
      if (trace) {
        if (it == end) {
          *trace << "NodeIdOneOf error: no more children, expected " << T::Kind
                 << " or " << U::Kind << "\n";
        } else {
          *trace << "NodeIdOneOf error: wrong kind " << tree->node_kind(*it)
                 << ", expected " << T::Kind << " or " << U::Kind << "\n";
        }
      }
      return std::nullopt;
    }
    if (trace) {
      *trace << "NodeIdOneOf " << T::Kind << " or " << U::Kind << ": "
             << tree->node_kind(*it) << " consumed";
    }
    return NodeIdOneOf<T, U>(*it++);
  }
};

// Extract a `NodeIdNot<T>` as a single required child.
template <typename T>
struct Extractable<NodeIdNot<T>> {
  static auto Extract(const Tree* tree, Tree::SiblingIterator& it,
                      Tree::SiblingIterator end, ErrorBuilder* trace)
      -> std::optional<NodeIdNot<T>> {
    if (it == end || tree->node_kind(*it) == T::Kind) {
      if (trace) {
        if (it == end) {
          *trace << "NodeIdNot " << T::Kind << " error: no more children\n";
        } else {
          *trace << "NodeIdNot error: unexpected " << T::Kind << "\n";
        }
      }
      return std::nullopt;
    }
    if (trace) {
      *trace << "NodeIdNot " << T::Kind << ": " << tree->node_kind(*it)
             << " consumed\n";
    }
    return NodeIdNot<T>(*it++);
  }
};

// Extract a `llvm::SmallVector<T>` by extracting `T`s until we can't.
template <typename T>
struct Extractable<llvm::SmallVector<T>> {
  static auto Extract(const Tree* tree, Tree::SiblingIterator& it,
                      Tree::SiblingIterator end, ErrorBuilder* trace)
      -> std::optional<llvm::SmallVector<T>> {
    if (trace) {
      *trace << "Vector: begin\n";
    }
    llvm::SmallVector<T> result;
    while (it != end) {
      auto old_it = it;
      auto item = Extractable<T>::Extract(tree, it, end, trace);
      if (!item.has_value()) {
        it = old_it;
        break;
      }
      result.push_back(*item);
    }
    std::reverse(result.begin(), result.end());
    if (trace) {
      *trace << "Vector: end\n";
    }
    return result;
  }
};

// Extract an `optional<T>` from a list of child nodes by attempting to extract
// a `T`, and extracting nothing if that fails.
template <typename T>
struct Extractable<std::optional<T>> {
  static auto Extract(const Tree* tree, Tree::SiblingIterator& it,
                      Tree::SiblingIterator end, ErrorBuilder* trace)
      -> std::optional<std::optional<T>> {
    if (trace) {
      *trace << "Optional: begin\n";
    }
    auto old_it = it;
    std::optional<T> value = Extractable<T>::Extract(tree, it, end, trace);
    if (value) {
      if (trace) {
        *trace << "Optional: found\n";
      }
      return value;
    }
    if (trace) {
      *trace << "Optional: missing\n";
    }
    it = old_it;
    return value;
  }
};

// Extract a `tuple<T...>` from a list of child nodes by extracting each `T` in
// reverse order.
template <typename... T>
struct Extractable<std::tuple<T...>> {
  template <std::size_t... Index>
  static auto ExtractImpl(const Tree* tree, Tree::SiblingIterator& it,
                          Tree::SiblingIterator end, ErrorBuilder* trace,
                          std::index_sequence<Index...>)
      -> std::optional<std::tuple<T...>> {
    std::tuple<std::optional<T>...> fields;
    if (trace) {
      *trace << "Tuple: begin\n";
    }

    // Use a fold over the `=` operator to parse fields from right to left.
    [[maybe_unused]] int unused;
    bool ok = true;
    static_cast<void>(((ok &&
                        ok = (std::get<Index>(fields) =
                              Extractable<T>::Extract(tree, it, end, trace)).has_value(),
                        unused) = ... = 0));

    if (!ok) {
      if (trace) {
        *trace << "Tuple: error\n";
      }
      return std::nullopt;
    }

    if (trace) {
      *trace << "Tuple: success\n";
    }
    return std::tuple<T...>{std::move(std::get<Index>(fields).value())...};
  }

  static auto Extract(const Tree* tree, Tree::SiblingIterator& it,
                      Tree::SiblingIterator end, ErrorBuilder* trace)
      -> std::optional<std::tuple<T...>> {
    return ExtractImpl(tree, it, end, trace,
                       std::make_index_sequence<sizeof...(T)>());
  }
};

// Extract the fields of a simple aggregate type.
template <typename T>
struct Extractable {
  static_assert(std::is_aggregate_v<T>, "Unsupported child type");
  static auto ExtractImpl(const Tree* tree, Tree::SiblingIterator& it,
                          Tree::SiblingIterator end, ErrorBuilder* trace)
      -> std::optional<T> {
    if (trace) {
      *trace << "Aggregate: begin\n";
    }
    // Extract the corresponding tuple type.
    using TupleType = decltype(StructReflection::AsTuple(std::declval<T>()));
    auto tuple = Extractable<TupleType>::Extract(tree, it, end, trace);
    if (!tuple.has_value()) {
      if (trace) {
        *trace << "Aggregate: error\n";
      }
      return std::nullopt;
    }

    if (trace) {
      *trace << "Aggregate: success\n";
    }
    // Convert the tuple to the struct type.
    return std::apply(
        [](auto&&... value) {
          return T{std::forward<decltype(value)>(value)...};
        },
        *tuple);
  }

  static auto Extract(const Tree* tree, Tree::SiblingIterator& it,
                      Tree::SiblingIterator end, ErrorBuilder* trace)
      -> std::optional<T> {
    static_assert(!HasKindMember<T>, "Missing Id suffix");
    return ExtractImpl(tree, it, end, trace);
  }
};

template <typename T>
auto Tree::TryExtractNodeFromChildren(
    llvm::iterator_range<Tree::SiblingIterator> children,
    ErrorBuilder* trace) const -> std::optional<T> {
  auto it = children.begin();
  auto result = Extractable<T>::ExtractImpl(this, it, children.end(), trace);
  if (it != children.end()) {
    if (trace) {
      *trace << "Error: " << node_kind(*it) << " node left unconsumed.";
    }
    return std::nullopt;
  }
  return result;
}

// Manually instantiate Tree::TryExtractNodeFromChildren
#define CARBON_PARSE_NODE_KIND(KindName)                    \
  template auto Tree::TryExtractNodeFromChildren<KindName>( \
      llvm::iterator_range<Tree::SiblingIterator> children, \
      ErrorBuilder * trace) const -> std::optional<KindName>;

// Also instantiate for `File`, even though it isn't a parse node.
CARBON_PARSE_NODE_KIND(File)
#include "toolchain/parse/node_kind.def"

}  // namespace Carbon::Parse
