// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSE_NODE_IDS_H_
#define CARBON_TOOLCHAIN_PARSE_NODE_IDS_H_

#include "toolchain/base/index_base.h"
#include "toolchain/parse/node_kind.h"

namespace Carbon::Parse {

// Represents an invalid node id of any type
struct InvalidNodeId {};

// A lightweight handle representing a node in the tree.
//
// Objects of this type are small and cheap to copy and store. They don't
// contain any of the information about the node, and serve as a handle that
// can be used with the underlying tree to query for detailed information.
struct NodeId : public IdBase {
  // An explicitly invalid instance.
  static constexpr InvalidNodeId Invalid;

  using IdBase::IdBase;
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr NodeId(InvalidNodeId /*invalid*/) : IdBase(NodeId::InvalidIndex) {}
};

// For looking up the type associated with a given id type.
template <typename T>
struct NodeForId;

// `<KindName>Id` is a typed version of `NodeId` that references a node of kind
// `<KindName>`:
template <const NodeKind& K>
struct NodeIdForKind : public NodeId {
  // NOLINTNEXTLINE(readability-identifier-naming)
  static const NodeKind& Kind;
  constexpr explicit NodeIdForKind(NodeId node_id) : NodeId(node_id) {}
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr NodeIdForKind(InvalidNodeId /*invalid*/)
      : NodeId(NodeId::InvalidIndex) {}
};
template <const NodeKind& K>
const NodeKind& NodeIdForKind<K>::Kind = K;

#define CARBON_PARSE_NODE_KIND(KindName) \
  using KindName##Id = NodeIdForKind<NodeKind::KindName>;
#include "toolchain/parse/node_kind.def"

// NodeId that matches any NodeKind whose `category()` overlaps with `Category`.
template <NodeCategory Category>
struct NodeIdInCategory : public NodeId {
  // Support conversion from `NodeIdForKind<Kind>` if Kind's category
  // overlaps with `Category`.
  template <const NodeKind& Kind>
  // NOLINTNEXTLINE(google-explicit-constructor)
  NodeIdInCategory(NodeIdForKind<Kind> node_id) : NodeId(node_id) {
    CARBON_CHECK(!!(Kind.category() & Category));
  }

  constexpr explicit NodeIdInCategory(NodeId node_id) : NodeId(node_id) {}
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr NodeIdInCategory(InvalidNodeId /*invalid*/)
      : NodeId(NodeId::InvalidIndex) {}
};

// Aliases for `NodeIdInCategory` to describe particular categories of nodes.
using AnyDeclId = NodeIdInCategory<NodeCategory::Decl>;
using AnyExprId = NodeIdInCategory<NodeCategory::Expr>;
using AnyImplAsId = NodeIdInCategory<NodeCategory::ImplAs>;
using AnyMemberNameOrMemberExprId =
    NodeIdInCategory<NodeCategory::MemberName | NodeCategory::MemberExpr>;
using AnyModifierId = NodeIdInCategory<NodeCategory::Modifier>;
using AnyNameComponentId = NodeIdInCategory<NodeCategory::NameComponent>;
using AnyPatternId = NodeIdInCategory<NodeCategory::Pattern>;
using AnyStatementId = NodeIdInCategory<NodeCategory::Statement>;

// NodeId with kind that matches one of the `T::Kind`s.
template <typename... T>
struct NodeIdOneOf : public NodeId {
  static_assert(sizeof...(T) >= 2, "Expected at least two types.");
  constexpr explicit NodeIdOneOf(NodeId node_id) : NodeId(node_id) {}
  template <const NodeKind& Kind>
  // NOLINTNEXTLINE(google-explicit-constructor)
  NodeIdOneOf(NodeIdForKind<Kind> node_id) : NodeId(node_id) {
    static_assert(((T::Kind == Kind) || ...));
  }
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr NodeIdOneOf(InvalidNodeId /*invalid*/)
      : NodeId(NodeId::InvalidIndex) {}
};

using AnyClassDeclId = NodeIdOneOf<ClassDeclId, ClassDefinitionStartId>;
using AnyFunctionDeclId = NodeIdOneOf<FunctionDeclId, FunctionDefinitionStartId,
                                      BuiltinFunctionDefinitionStartId>;
using AnyImplDeclId = NodeIdOneOf<ImplDeclId, ImplDefinitionStartId>;
using AnyInterfaceDeclId =
    NodeIdOneOf<InterfaceDeclId, InterfaceDefinitionStartId>;
using AnyNamespaceId = NodeIdOneOf<NamespaceId, ImportDirectiveId>;
using AnyPointerDeferenceExprId =
    NodeIdOneOf<PrefixOperatorStarId, PointerMemberAccessExprId>;

// NodeId with kind that is anything but T::Kind.
template <typename T>
struct NodeIdNot : public NodeId {
  constexpr explicit NodeIdNot(NodeId node_id) : NodeId(node_id) {}
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr NodeIdNot(InvalidNodeId /*invalid*/)
      : NodeId(NodeId::InvalidIndex) {}
};

// Note that the support for extracting these types using the `Tree::Extract*`
// functions is defined in `extract.cpp`.

}  // namespace Carbon::Parse

#endif  // CARBON_TOOLCHAIN_PARSE_NODE_IDS_H_
