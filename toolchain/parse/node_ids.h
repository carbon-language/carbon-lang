// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSE_NODE_IDS_H_
#define CARBON_TOOLCHAIN_PARSE_NODE_IDS_H_

#include "toolchain/base/index_base.h"
#include "toolchain/parse/node_kind.h"

namespace Carbon::Parse {

// A lightweight handle representing a node in the tree.
//
// Objects of this type are small and cheap to copy and store. They don't
// contain any of the information about the node, and serve as a handle that
// can be used with the underlying tree to query for detailed information.
struct NodeId : public IdBase {
  // An explicitly invalid instance.
  static const NodeId Invalid;

  using IdBase::IdBase;
};

constexpr NodeId NodeId::Invalid = NodeId(NodeId::InvalidIndex);

// For looking up the type associated with a given id type.
template <typename T>
struct NodeForId;

// `<KindName>Id` is a typed version of `NodeId` that references a node of kind
// `<KindName>`:
template <const NodeKind&>
struct NodeIdForKind : public NodeId {
  static const NodeIdForKind Invalid;

  explicit NodeIdForKind(NodeId node_id) : NodeId(node_id) {}
};
template <const NodeKind& Kind>
constexpr NodeIdForKind<Kind> NodeIdForKind<Kind>::Invalid =
    NodeIdForKind(NodeId::Invalid.index);

#define CARBON_PARSE_NODE_KIND(KindName) \
  using KindName##Id = NodeIdForKind<NodeKind::KindName>;
#include "toolchain/parse/node_kind.def"

// NodeId that matches any NodeKind whose `category()` overlaps with `Category`.
template <NodeCategory Category>
struct NodeIdInCategory : public NodeId {
  // An explicitly invalid instance.
  static const NodeIdInCategory<Category> Invalid;

  // TODO: Support conversion from `NodeIdForKind<Kind>` if `Kind::category()`
  // overlaps with `Category`.

  explicit NodeIdInCategory(NodeId node_id) : NodeId(node_id) {}
};

template <NodeCategory Category>
constexpr NodeIdInCategory<Category> NodeIdInCategory<Category>::Invalid =
    NodeIdInCategory<Category>(NodeId::InvalidIndex);

// Aliases for `NodeIdInCategory` to describe particular categories of nodes.
using AnyDeclId = NodeIdInCategory<NodeCategory::Decl>;
using AnyExprId = NodeIdInCategory<NodeCategory::Expr>;
using AnyMemberNameId = NodeIdInCategory<NodeCategory::MemberName>;
using AnyModifierId = NodeIdInCategory<NodeCategory::Modifier>;
using AnyNameComponentId = NodeIdInCategory<NodeCategory::NameComponent>;
using AnyPatternId = NodeIdInCategory<NodeCategory::Pattern>;
using AnyStatementId = NodeIdInCategory<NodeCategory::Statement>;

// NodeId with kind that matches either T::Kind or U::Kind.
template <typename T, typename U>
struct NodeIdOneOf : public NodeId {
  // An explicitly invalid instance.
  static const NodeIdOneOf<T, U> Invalid;

  // TODO: Support conversion from `NodeIdForKind<Kind>` if `Kind` is
  // `T::Kind` or `U::Kind`.

  explicit NodeIdOneOf(NodeId node_id) : NodeId(node_id) {}
};

template <typename T, typename U>
constexpr NodeIdOneOf<T, U> NodeIdOneOf<T, U>::Invalid =
    NodeIdOneOf<T, U>(NodeId::InvalidIndex);

// NodeId with kind that is anything but T::Kind.
template <typename T>
struct NodeIdNot : public NodeId {
  // An explicitly invalid instance.
  static const NodeIdNot<T> Invalid;

  explicit NodeIdNot(NodeId node_id) : NodeId(node_id) {}
};

template <typename T>
constexpr NodeIdNot<T> NodeIdNot<T>::Invalid =
    NodeIdNot<T>(NodeId::InvalidIndex);

// Note that the support for extracting these types using the `Tree::Extract*`
// functions is defined in `extract.cpp`.

}  // namespace Carbon::Parse

#endif  // CARBON_TOOLCHAIN_PARSE_NODE_IDS_H_
