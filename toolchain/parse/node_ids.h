// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSE_NODE_IDS_H_
#define CARBON_TOOLCHAIN_PARSE_NODE_IDS_H_

#include "toolchain/base/index_base.h"
#include "toolchain/lex/token_index.h"
#include "toolchain/parse/node_kind.h"

namespace Carbon::Parse {

// Represents an invalid node id of any type
struct NoneNodeId {};

// A lightweight handle representing a node in the tree.
//
// Objects of this type are small and cheap to copy and store. They don't
// contain any of the information about the node, and serve as a handle that
// can be used with the underlying tree to query for detailed information.
struct NodeId : public IdBase<NodeId> {
  static constexpr llvm::StringLiteral Label = "node";

  // A node ID with no value.
  static constexpr NoneNodeId None;

  using IdBase::IdBase;
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr NodeId(NoneNodeId /*none*/) : IdBase(NoneIndex) {}
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

  // Provide a factory function for construction from `NodeId`. This doesn't
  // validate the type, so it's unsafe.
  static constexpr auto UnsafeMake(NodeId node_id) -> NodeIdForKind {
    return NodeIdForKind(node_id);
  }

  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr NodeIdForKind(NoneNodeId /*none*/) : NodeId(NoneIndex) {}

 private:
  // Private to prevent accidental explicit construction from an untyped
  // NodeId.
  explicit constexpr NodeIdForKind(NodeId node_id) : NodeId(node_id) {}
};
template <const NodeKind& K>
const NodeKind& NodeIdForKind<K>::Kind = K;

#define CARBON_PARSE_NODE_KIND(KindName) \
  using KindName##Id = NodeIdForKind<NodeKind::KindName>;
#include "toolchain/parse/node_kind.def"

// NodeId that matches any NodeKind whose `category()` overlaps with `Category`.
template <NodeCategory::RawEnumType Category>
struct NodeIdInCategory : public NodeId {
  // Provide a factory function for construction from `NodeId`. This doesn't
  // validate the type, so it's unsafe.
  static constexpr auto UnsafeMake(NodeId node_id) -> NodeIdInCategory {
    return NodeIdInCategory(node_id);
  }

  // Support conversion from `NodeIdForKind<Kind>` if Kind's category
  // overlaps with `Category`.
  template <const NodeKind& Kind>
  // NOLINTNEXTLINE(google-explicit-constructor)
  NodeIdInCategory(NodeIdForKind<Kind> node_id) : NodeId(node_id) {
    CARBON_CHECK(Kind.category().HasAnyOf(Category));
  }

  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr NodeIdInCategory(NoneNodeId /*none*/) : NodeId(NoneIndex) {}

 private:
  // Private to prevent accidental explicit construction from an untyped
  // NodeId.
  explicit constexpr NodeIdInCategory(NodeId node_id) : NodeId(node_id) {}
};

// Aliases for `NodeIdInCategory` to describe particular categories of nodes.
using AnyDeclId = NodeIdInCategory<NodeCategory::Decl>;
using AnyExprId = NodeIdInCategory<NodeCategory::Expr>;
using AnyImplAsId = NodeIdInCategory<NodeCategory::ImplAs>;
using AnyMemberAccessId =
    NodeIdInCategory<NodeCategory::MemberName | NodeCategory::MemberExpr |
                     NodeCategory::IntConst>;
using AnyModifierId = NodeIdInCategory<NodeCategory::Modifier>;
using AnyPatternId = NodeIdInCategory<NodeCategory::Pattern>;
using AnyStatementId =
    NodeIdInCategory<NodeCategory::Statement | NodeCategory::Decl>;
using AnyRequirementId = NodeIdInCategory<NodeCategory::Requirement>;
using AnyNonExprNameId = NodeIdInCategory<NodeCategory::NonExprName>;
using AnyPackageNameId = NodeIdInCategory<NodeCategory::PackageName>;

// NodeId with kind that matches one of the `T::Kind`s.
template <typename... T>
  requires(sizeof...(T) >= 2)
struct NodeIdOneOf : public NodeId {
  // Provide a factory function for construction from `NodeId`. This doesn't
  // validate the type, so it's unsafe.
  static constexpr auto UnsafeMake(NodeId node_id) -> NodeIdOneOf {
    return NodeIdOneOf(node_id);
  }

  template <const NodeKind& Kind>
    requires((T::Kind == Kind) || ...)
  // NOLINTNEXTLINE(google-explicit-constructor)
  NodeIdOneOf(NodeIdForKind<Kind> node_id) : NodeId(node_id) {}

  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr NodeIdOneOf(NoneNodeId /*none*/) : NodeId(NoneIndex) {}

 private:
  // Private to prevent accidental explicit construction from an untyped
  // NodeId.
  explicit constexpr NodeIdOneOf(NodeId node_id) : NodeId(node_id) {}
};

using AnyClassDeclId =
    NodeIdOneOf<ClassDeclId, ClassDefinitionStartId,
                // TODO: This may be wrong? But we have choice types produce a
                // class, so they are a form of class decls. This avoids
                // duplicating all of SemIR::ClassDecl.
                ChoiceDefinitionStartId>;
using AnyFunctionDeclId = NodeIdOneOf<FunctionDeclId, FunctionDefinitionStartId,
                                      BuiltinFunctionDefinitionStartId>;
using AnyImplDeclId = NodeIdOneOf<ImplDeclId, ImplDefinitionStartId>;
using AnyInterfaceDeclId =
    NodeIdOneOf<InterfaceDeclId, InterfaceDefinitionStartId>;
using AnyNamespaceId =
    NodeIdOneOf<NamespaceId, ImportDeclId, LibraryDeclId, PackageDeclId>;
using AnyPackagingDeclId =
    NodeIdOneOf<ImportDeclId, LibraryDeclId, PackageDeclId>;
using AnyPointerDeferenceExprId =
    NodeIdOneOf<PrefixOperatorStarId, PointerMemberAccessExprId>;
using AnyRuntimeBindingPatternName =
    NodeIdOneOf<IdentifierNameNotBeforeParamsId, SelfValueNameId,
                UnderscoreNameId>;

// NodeId with kind that is anything but T::Kind.
template <typename T>
struct NodeIdNot : public NodeId {
  constexpr explicit NodeIdNot(NodeId node_id) : NodeId(node_id) {}
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr NodeIdNot(NoneNodeId /*none*/) : NodeId(NoneIndex) {}
};

// Note that the support for extracting these types using the `Tree::Extract*`
// functions is defined in `extract.cpp`.

}  // namespace Carbon::Parse

#endif  // CARBON_TOOLCHAIN_PARSE_NODE_IDS_H_
