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
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

// Data about the arguments of a typed node, to aid in type erasure. The `KindT`
// parameter is used to check that `TypedInst` is a typed node.
template <typename TypedInst,
          const NodeKind::Definition& KindT = TypedInst::Kind>
struct TypedInstArgsInfo {
  // A corresponding std::tuple<...> type.
  using Tuple = decltype(StructReflection::AsTuple(std::declval<TypedInst>()));

  static constexpr int FirstArgField =
      HasParseNode<TypedInst> + HasTypeId<TypedInst>;

  static constexpr int NumArgs = std::tuple_size_v<Tuple> - FirstArgField;
  static_assert(NumArgs <= 2,
                "Unsupported: typed node has more than two data fields");

  template <int N>
  using ArgType = std::tuple_element_t<FirstArgField + N, Tuple>;

  template <int N>
  static auto Get(TypedInst node) -> ArgType<N> {
    return std::get<FirstArgField + N>(StructReflection::AsTuple(node));
  }
};

// A type-erased representation of a SemIR node, that may be constructed from
// the specific kinds of node defined in `typed_insts.h`. This provides access
// to common fields present on most or all kinds of nodes:
//
// - `parse_node` for error placement.
// - `kind` for run-time logic when the input Kind is unknown.
// - `type_id` for quick type checking.
//
// In addition, kind-specific data can be accessed by casting to the specific
// kind of node:
//
// - Use `node.kind()` or `Is<TypedInst>` to determine what kind of node it is.
// - Cast to a specific type using `node.As<TypedInst>()`
//   - Using the wrong kind in `node.As<TypedInst>()` is a programming error,
//     and will CHECK-fail in debug modes (opt may too, but it's not an API
//     guarantee).
// - Use `node.TryAs<TypedInst>()` to safely access type-specific node data
//   where the node's kind is not known.
class Node : public Printable<Node> {
 public:
  template <typename TypedInst, typename Info = TypedInstArgsInfo<TypedInst>>
  // NOLINTNEXTLINE(google-explicit-constructor)
  Node(TypedInst typed_inst)
      : parse_node_(Parse::Node::Invalid),
        kind_(TypedInst::Kind),
        type_id_(TypeId::Invalid),
        arg0_(InstId::InvalidIndex),
        arg1_(InstId::InvalidIndex) {
    if constexpr (HasParseNode<TypedInst>) {
      parse_node_ = typed_inst.parse_node;
    }
    if constexpr (HasTypeId<TypedInst>) {
      type_id_ = typed_inst.type_id;
    }
    if constexpr (Info::NumArgs > 0) {
      arg0_ = ToRaw(Info::template Get<0>(typed_inst));
    }
    if constexpr (Info::NumArgs > 1) {
      arg1_ = ToRaw(Info::template Get<1>(typed_inst));
    }
  }

  // Returns whether this node has the specified type.
  template <typename TypedInst>
  auto Is() const -> bool {
    return kind() == TypedInst::Kind;
  }

  // Casts this node to the given typed node, which must match the node's kind,
  // and returns the typed node.
  template <typename TypedInst, typename Info = TypedInstArgsInfo<TypedInst>>
  auto As() const -> TypedInst {
    CARBON_CHECK(Is<TypedInst>()) << "Casting node of kind " << kind()
                                  << " to wrong kind " << TypedInst::Kind;
    auto build_with_type_id_and_args = [&](auto... type_id_and_args) {
      if constexpr (HasParseNode<TypedInst>) {
        return TypedInst{parse_node(), type_id_and_args...};
      } else {
        return TypedInst{type_id_and_args...};
      }
    };

    auto build_with_args = [&](auto... args) {
      if constexpr (HasTypeId<TypedInst>) {
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
  template <typename TypedInst>
  auto TryAs() const -> std::optional<TypedInst> {
    if (Is<TypedInst>()) {
      return As<TypedInst>();
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
  friend class NodeTestHelper;

  // Raw constructor, used for testing.
  explicit Node(NodeKind kind, Parse::Node parse_node, TypeId type_id,
                int32_t arg0, int32_t arg1)
      : parse_node_(parse_node),
        kind_(kind),
        type_id_(type_id),
        arg0_(arg0),
        arg1_(arg1) {}

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
template <typename TypedInst, typename = TypedInstArgsInfo<TypedInst>>
inline llvm::raw_ostream& operator<<(llvm::raw_ostream& out, TypedInst node) {
  Node(node).Print(out);
  return out;
}

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_NODE_H_
