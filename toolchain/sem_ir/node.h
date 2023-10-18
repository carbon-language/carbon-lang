// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_NODE_H_
#define CARBON_TOOLCHAIN_SEM_IR_NODE_H_

#include <cstdint>

#include "common/check.h"
#include "common/ostream.h"
#include "common/struct_reflection.h"
#include "toolchain/base/index_base.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/builtin_kind.h"
#include "toolchain/sem_ir/node_kind.h"
#include "toolchain/sem_ir/typed_nodes.h"

namespace Carbon::SemIR {

// Data about the arguments of a typed node, to aid in type erasure. The `KindT`
// parameter is used to check that `TypedNode` is a typed node.
template <typename TypedNode, const NodeKind& KindT = TypedNode::Kind>
struct TypedNodeArgsInfo {
  // A corresponding std::tuple<...> type.
  using Tuple = decltype(StructReflection::AsTuple(std::declval<TypedNode>()));

  static constexpr int Arg0Field =
      HasParseNode<TypedNode> + HasTypeId<TypedNode>;

  static constexpr int NumArgs = std::tuple_size_v<Tuple> - Arg0Field;
  static_assert(NumArgs <= 2,
                "Unsupported: typed node has more than two data fields");

  template <int N>
  using ArgType = std::tuple_element_t<N + Arg0Field, Tuple>;
};

// A type-erased representation of a SemIR node, that may be constructed from
// the specific kinds of node defined in `typed_nodes.h`. This provides access
// to common fields present on most or all kinds of nodes:
//
// - `parse_node` for error placement.
// - `kind` for run-time logic when the input Kind is unknown.
// - `type_id` for quick type checking.
//
// In addition, any kind-specific data is stored and can be accessed by casting
// to the specific kind of node:
//
// - Use `node.kind()` or `Is<TypedNode>` to determine what kind of node it is.
// - Cast to a specific type using `node.As<TypedNode>()`
//   - Using the wrong kind in `node.As<TypedNode>()` is a programming error,
//     and will CHECK-fail in debug modes (opt may too, but it's not an API
//     guarantee).
// - Use `node.TryAs<TypedNode>()` to safely access type-specific node data
//   where the node's kind is not known.
class Node : public Printable<Node> {
 public:
  template <typename TypedNode, typename Info = TypedNodeArgsInfo<TypedNode>>
  /*implicit*/
  Node(TypedNode typed_node)
      : parse_node_(Parse::Node::Invalid),
        kind_(TypedNode::Kind),
        type_id_(TypeId::Invalid),
        arg0_(NodeId::InvalidIndex),
        arg1_(NodeId::InvalidIndex) {
    if constexpr (HasParseNode<TypedNode>) {
      parse_node_ = typed_node.parse_node;
    }
    if constexpr (HasTypeId<TypedNode>) {
      type_id_ = typed_node.type_id;
    }
    if constexpr (Info::NumArgs > 0) {
      auto tuple = StructReflection::AsTuple(typed_node);
      arg0_ = ToRaw(std::get<Info::Arg0Field>(tuple));
      if constexpr (Info::NumArgs > 1) {
        arg1_ = ToRaw(std::get<Info::Arg0Field + 1>(tuple));
      }
    }
  }

  // Returns whether this node has the specified type.
  template <typename TypedNode>
  auto Is() const -> bool {
    return kind() == TypedNode::Kind;
  }

  // Casts this node to the given typed node, which must match the node's kind,
  // and returns the typed node.
  template <typename TypedNode, typename Info = TypedNodeArgsInfo<TypedNode>>
  auto As() const -> TypedNode {
    CARBON_CHECK(Is<TypedNode>()) << "Casting node of kind " << kind()
                                  << " to wrong kind " << TypedNode::Kind;
    auto build_with_type_id_and_args = [&](auto... type_id_and_args) {
      if constexpr (HasParseNode<TypedNode>) {
        return TypedNode{parse_node(), type_id_and_args...};
      } else {
        return TypedNode{type_id_and_args...};
      }
    };

    auto build_with_args = [&](auto... args) {
      if constexpr (HasTypeId<TypedNode>) {
        return build_with_type_id_and_args(type_id(), args...);
      } else {
        return build_with_type_id_and_args(args...);
      }
    };

    if constexpr (Info::NumArgs == 0) {
      return build_with_args();
    } else if constexpr (Info::NumArgs == 1) {
      return build_with_args(
          FromRaw<typename Info::template ArgType<0>>(arg0_));
    } else if constexpr (Info::NumArgs == 2) {
      return build_with_args(
          FromRaw<typename Info::template ArgType<0>>(arg0_),
          FromRaw<typename Info::template ArgType<1>>(arg1_));
    }
  }

  // If this node is the given kind, returns a typed node, otherwise returns
  // nullopt.
  template <typename TypedNode>
  auto TryAs() const -> std::optional<TypedNode> {
    if (Is<TypedNode>()) {
      return As<TypedNode>();
    } else {
      return std::nullopt;
    }
  }

  auto parse_node() const -> Parse::Node { return parse_node_; }
  auto kind() const -> NodeKind { return kind_; }

  // Gets the type of the value produced by evaluating this node.
  auto type_id() const -> TypeId { return type_id_; }

  auto Print(llvm::raw_ostream& out) const -> void;

 private:
  // Convert a field to its raw representation, used as `arg0_` / `arg1_`.
  static constexpr auto ToRaw(IndexBase base) -> int32_t { return base.index; }
  static constexpr auto ToRaw(BuiltinKind kind) -> int32_t {
    return kind.AsInt();
  }

  // Convert a field from its raw representation.
  template <typename T>
  static constexpr auto FromRaw(int32_t raw) -> T {
    return T(raw);
  }
  template <>
  constexpr auto FromRaw<BuiltinKind>(int32_t raw) -> BuiltinKind {
    return BuiltinKind::FromInt(raw);
  }

  Parse::Node parse_node_;
  NodeKind kind_;
  TypeId type_id_;

  // Use `As` to access arg0 and arg1.
  int32_t arg0_;
  int32_t arg1_;
};

// TODO: This is currently 20 bytes because we sometimes have 2 arguments for a
// pair of Nodes. However, NodeKind is 1 byte; if args
// were 3.5 bytes, we could potentially shrink Node by 4 bytes. This
// may be worth investigating further.
static_assert(sizeof(Node) == 20, "Unexpected Node size");

// Typed nodes can be printed by converting them to nodes.
template <typename TypedNode, typename = TypedNodeArgsInfo<TypedNode>>
inline llvm::raw_ostream& operator<<(llvm::raw_ostream& out, TypedNode node) {
  Node(node).Print(out);
  return out;
}

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_NODE_H_
