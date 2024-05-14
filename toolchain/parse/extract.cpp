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

// Extract a `NodeId` as a single child.
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
    return *it++;
  }
};

static auto NodeIdForKindAccept(const NodeKind& kind, const Tree* tree,
                                const Tree::SiblingIterator& it,
                                Tree::SiblingIterator end, ErrorBuilder* trace)
    -> bool {
  if (it == end || tree->node_kind(*it) != kind) {
    if (trace) {
      if (it == end) {
        *trace << "NodeIdForKind error: no more children, expected " << kind
               << "\n";
      } else {
        *trace << "NodeIdForKind error: wrong kind " << tree->node_kind(*it)
               << ", expected " << kind << "\n";
      }
    }
    return false;
  }
  if (trace) {
    *trace << "NodeIdForKind: " << kind << " consumed\n";
  }
  return true;
}

// Extract a `FooId`, which is the same as `NodeIdForKind<NodeKind::Foo>`,
// as a single required child.
template <const NodeKind& Kind>
struct Extractable<NodeIdForKind<Kind>> {
  static auto Extract(const Tree* tree, Tree::SiblingIterator& it,
                      Tree::SiblingIterator end, ErrorBuilder* trace)
      -> std::optional<NodeIdForKind<Kind>> {
    if (NodeIdForKindAccept(Kind, tree, it, end, trace)) {
      return NodeIdForKind<Kind>(*it++);
    } else {
      return std::nullopt;
    }
  }
};

static auto NodeIdInCategoryAccept(NodeCategory category, const Tree* tree,
                                   const Tree::SiblingIterator& it,
                                   Tree::SiblingIterator end,
                                   ErrorBuilder* trace) -> bool {
  if (it == end || !(tree->node_kind(*it).category() & category)) {
    if (trace) {
      *trace << "NodeIdInCategory " << category << " error: ";
      if (it == end) {
        *trace << "no more children\n";
      } else {
        *trace << "kind " << tree->node_kind(*it) << " doesn't match\n";
      }
    }
    return false;
  }
  if (trace) {
    *trace << "NodeIdInCategory " << category << ": kind "
           << tree->node_kind(*it) << " consumed\n";
  }
  return true;
}

// Extract a `NodeIdInCategory<Category>` as a single child.
template <NodeCategory Category>
struct Extractable<NodeIdInCategory<Category>> {
  static auto Extract(const Tree* tree, Tree::SiblingIterator& it,
                      Tree::SiblingIterator end, ErrorBuilder* trace)
      -> std::optional<NodeIdInCategory<Category>> {
    if (NodeIdInCategoryAccept(Category, tree, it, end, trace)) {
      return NodeIdInCategory<Category>(*it++);
    } else {
      return std::nullopt;
    }
  }
};

static auto NodeIdOneOfAccept(std::initializer_list<NodeKind> kinds,
                              const Tree* tree, const Tree::SiblingIterator& it,
                              Tree::SiblingIterator end, ErrorBuilder* trace)
    -> bool {
  auto trace_kinds = [&] {
    llvm::ListSeparator sep(" or ");
    for (auto kind : kinds) {
      *trace << sep << kind;
    }
  };
  auto kind = tree->node_kind(*it);
  if (it == end || std::find(kinds.begin(), kinds.end(), kind) == kinds.end()) {
    if (trace) {
      if (it == end) {
        *trace << "NodeIdOneOf error: no more children, expected ";
        trace_kinds();
        *trace << "\n";
      } else {
        *trace << "NodeIdOneOf error: wrong kind " << tree->node_kind(*it)
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
    *trace << ": " << tree->node_kind(*it) << " consumed\n";
  }
  return true;
}

// Extract a `NodeIdOneOf<T...>` as a single required child.
template <typename... T>
struct Extractable<NodeIdOneOf<T...>> {
  static auto Extract(const Tree* tree, Tree::SiblingIterator& it,
                      Tree::SiblingIterator end, ErrorBuilder* trace)
      -> std::optional<NodeIdOneOf<T...>> {
    if (NodeIdOneOfAccept({T::Kind...}, tree, it, end, trace)) {
      return NodeIdOneOf<T...>(*it++);
    } else {
      return std::nullopt;
    }
  }
};

// Extract a `NodeIdNot<T>` as a single required child.
// Note: this is only instantiated once, so no need to create a helper function.
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

// Extract an `llvm::SmallVector<T>` by extracting `T`s until we can't.
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
      *trace << "Optional " << typeid(T).name() << ": begin\n";
    }
    auto old_it = it;
    std::optional<T> value = Extractable<T>::Extract(tree, it, end, trace);
    if (value) {
      if (trace) {
        *trace << "Optional " << typeid(T).name() << ": found\n";
      }
      return value;
    }
    if (trace) {
      *trace << "Optional " << typeid(T).name() << ": missing\n";
    }
    it = old_it;
    return value;
  }
};

template <typename T, typename... U, std::size_t... Index>
static auto ExtractTupleLikeType(const Tree* tree, Tree::SiblingIterator& it,
                                 Tree::SiblingIterator end, ErrorBuilder* trace,
                                 std::index_sequence<Index...> /*indices*/,
                                 std::tuple<U...>* /*type*/)
    -> std::optional<T> {
  std::tuple<std::optional<U>...> fields;
  if (trace) {
    *trace << "Aggregate " << typeid(T).name() << ": begin\n";
  }
  // Use a fold over the `=` operator to parse fields from right to left.
  [[maybe_unused]] int unused;
  bool ok = true;
  static_cast<void>(
      ((ok && (ok = (std::get<Index>(fields) =
                         Extractable<U>::Extract(tree, it, end, trace))
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
  static auto ExtractImpl(const Tree* tree, Tree::SiblingIterator& it,
                          Tree::SiblingIterator end, ErrorBuilder* trace)
      -> std::optional<T> {
    // Compute the corresponding tuple type.
    using TupleType = decltype(StructReflection::AsTuple(std::declval<T>()));
    return ExtractTupleLikeType<T>(
        tree, it, end, trace,
        std::make_index_sequence<std::tuple_size_v<TupleType>>(),
        static_cast<TupleType*>(nullptr));
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

auto Tree::ExtractFile() const -> File {
  return ExtractNodeFromChildren<File>(roots());
}

}  // namespace Carbon::Parse
