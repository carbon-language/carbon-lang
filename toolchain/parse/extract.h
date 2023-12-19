// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSE_EXTRACT_H_
#define CARBON_TOOLCHAIN_PARSE_EXTRACT_H_

#include <tuple>
#include <utility>

#include "common/error.h"
#include "common/struct_reflection.h"
#include "toolchain/parse/tree.h"
#include "toolchain/parse/typed_nodes.h"

namespace Carbon::Parse {

// A complete source file. Note that there is no corresponding parse node for
// the file. The file is instead the complete contents of the parse tree.
struct File {
  FileStartId start;
  llvm::SmallVector<AnyDecl> decls;
  FileEndId end;

  static auto Make(const Tree* tree) -> File {
    return tree->ExtractNodeFromChildren<File>(tree->roots());
  }
};

// Extract an `NodeId` as a single child.
template <>
struct Tree::Extractable<NodeId> {
  static auto Extract(const Tree* tree, SiblingIterator& it,
                      SiblingIterator end, ErrorBuilder* trace)
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

// Extract a `FooId`, which is the same as `KindId<NodeKind::Foo>`,
// as a single required child.
template <const NodeKind& Kind>
struct Tree::Extractable<KindId<Kind>> {
  static auto Extract(const Tree* tree, SiblingIterator& it,
                      SiblingIterator end, ErrorBuilder* trace)
      -> std::optional<KindId<Kind>> {
    if (it == end || tree->node_kind(*it) != Kind) {
      if (trace) {
        if (it == end) {
          *trace << "KindId error: no more children, expected " << Kind << "\n";
        } else {
          *trace << "KindId error: wrong kind " << tree->node_kind(*it)
                 << ", expected " << Kind << "\n";
        }
      }
      return std::nullopt;
    }
    if (trace) {
      *trace << "KindId: " << Kind << " consumed\n";
    }
    return KindId<Kind>(*it++);
  }
};

// Extract an `NodeIdInCategory<Category>` as a single child.
template <NodeCategory Category>
struct Tree::Extractable<NodeIdInCategory<Category>> {
  static auto Extract(const Tree* tree, SiblingIterator& it,
                      SiblingIterator end, ErrorBuilder* trace)
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
struct Tree::Extractable<NodeIdOneOf<T, U>> {
  static auto Extract(const Tree* tree, SiblingIterator& it,
                      SiblingIterator end, ErrorBuilder* trace)
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
struct Tree::Extractable<NodeIdNot<T>> {
  static auto Extract(const Tree* tree, SiblingIterator& it,
                      SiblingIterator end, ErrorBuilder* trace)
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
struct Tree::Extractable<llvm::SmallVector<T>> {
  static auto Extract(const Tree* tree, SiblingIterator& it,
                      SiblingIterator end, ErrorBuilder* trace)
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
struct Tree::Extractable<std::optional<T>> {
  static auto Extract(const Tree* tree, SiblingIterator& it,
                      SiblingIterator end, ErrorBuilder* trace)
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
struct Tree::Extractable<std::tuple<T...>> {
  template <std::size_t... Index>
  static auto ExtractImpl(const Tree* tree, SiblingIterator& it,
                          SiblingIterator end, ErrorBuilder* trace,
                          std::index_sequence<Index...>)
      -> std::optional<std::tuple<T...>> {
    std::tuple<std::optional<T>...> fields;
    if (trace) {
      *trace << "Tuple: begin\n";
    }

    // Use a fold over the `=` operator to parse fields from right to left.
    [[maybe_unused]] int unused;
    static_cast<void>(((std::get<Index>(fields) =
                            Extractable<T>::Extract(tree, it, end, trace),
                        unused) = ... = 0));

    if (!(std::get<Index>(fields).has_value() && ...)) {
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

  static auto Extract(const Tree* tree, SiblingIterator& it,
                      SiblingIterator end, ErrorBuilder* trace)
      -> std::optional<std::tuple<T...>> {
    return ExtractImpl(tree, it, end, trace,
                       std::make_index_sequence<sizeof...(T)>());
  }
};

// Extract the fields of a simple aggregate type.
template <typename T>
struct Tree::Extractable {
  static_assert(std::is_aggregate_v<T>, "Unsupported child type");
  static auto ExtractImpl(const Tree* tree, SiblingIterator& it,
                          SiblingIterator end, ErrorBuilder* trace)
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

  static auto Extract(const Tree* tree, SiblingIterator& it,
                      SiblingIterator end, ErrorBuilder* trace)
      -> std::optional<T> {
    static_assert(!HasKindMember<T>, "Missing Id suffix");
    return ExtractImpl(tree, it, end, trace);
  }
};

}  // namespace Carbon::Parse

#endif  // CARBON_TOOLCHAIN_PARSE_EXTRACT_H_
