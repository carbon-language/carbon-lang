// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_H_

#include <cstdint>

#include "common/check.h"
#include "common/ostream.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/semantics_builtin_kind.h"
#include "toolchain/semantics/semantics_node_kind.h"

namespace Carbon {

// The ID of a node.
struct SemanticsNodeId : public IndexBase {
  // An explicitly invalid node ID.
  static const SemanticsNodeId Invalid;

// Builtin node IDs.
#define CARBON_SEMANTICS_BUILTIN_KIND_NAME(Name) \
  static const SemanticsNodeId Builtin##Name;
#include "toolchain/semantics/semantics_builtin_kind.def"

  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "node";
    if (!is_valid()) {
      IndexBase::Print(out);
    } else if (index < SemanticsBuiltinKind::ValidCount) {
      out << SemanticsBuiltinKind::FromInt(index);
    } else {
      // Use the `+` as a small reminder that this is a delta, rather than an
      // absolute index.
      out << "+" << index - SemanticsBuiltinKind::ValidCount;
    }
  }
};

constexpr SemanticsNodeId SemanticsNodeId::Invalid =
    SemanticsNodeId(SemanticsNodeId::InvalidIndex);

// Uses the cross-reference node ID for a builtin. This relies on SemanticsIR
// guarantees for builtin cross-reference placement.
#define CARBON_SEMANTICS_BUILTIN_KIND_NAME(Name)             \
  constexpr SemanticsNodeId SemanticsNodeId::Builtin##Name = \
      SemanticsNodeId(SemanticsBuiltinKind::Name.AsInt());
#include "toolchain/semantics/semantics_builtin_kind.def"

// The ID of a call.
struct SemanticsCallId : public IndexBase {
  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "call";
    IndexBase::Print(out);
  }
};

// The ID of a callable, such as a function.
struct SemanticsCallableId : public IndexBase {
  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "callable";
    IndexBase::Print(out);
  }
};

// The ID of a cross-referenced IR.
struct SemanticsCrossReferenceIRId : public IndexBase {
  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "ir";
    IndexBase::Print(out);
  }
};

// The ID of an integer literal.
struct SemanticsIntegerLiteralId : public IndexBase {
  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "int";
    IndexBase::Print(out);
  }
};

// The ID of a node block.
struct SemanticsNodeBlockId : public IndexBase {
  // All SemanticsIR instances must provide the 0th node block as empty.
  static const SemanticsNodeBlockId Empty;

  // An explicitly invalid ID.
  static const SemanticsNodeBlockId Invalid;

  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "block";
    IndexBase::Print(out);
  }
};

constexpr SemanticsNodeBlockId SemanticsNodeBlockId::Empty =
    SemanticsNodeBlockId(0);
constexpr SemanticsNodeBlockId SemanticsNodeBlockId::Invalid =
    SemanticsNodeBlockId(SemanticsNodeBlockId::InvalidIndex);

// The ID of a real literal.
struct SemanticsRealLiteralId : public IndexBase {
  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "real";
    IndexBase::Print(out);
  }
};

// The ID of a string.
struct SemanticsStringId : public IndexBase {
  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "str";
    IndexBase::Print(out);
  }
};

// The standard structure for SemanticsNode. For each Kind in SemanticsNodeKind:
//
// - Create a `SemanticsNode` using `SemanticsNode::Kind::Make()`
// - Access cross-Kind members using `node.type_id()` and similar.
// - Access Kind-specific members using `node.GetAsKind()`, which depending on
//   the number of members will return one of NoArgs, a single value, or a
//   `std::pair` of values.
//   - `node.kind()` can be used to ensure the correct Kind is accessed. Using
//     the wrong `node.GetAsKind()` is a programming error, and should
//     CHECK-fail in debug modes (opt may too, but it's not an API guarantee).
//
// Internally, each Kind uses the `Factory*` types to provide a boilerplate
// `Make` and `Get` methods.
class SemanticsNode {
 public:
  struct NoArgs {};

  // Factory base classes are private, then used for public classes. This class
  // has two public and two private sections to prevent accidents.
 private:
  // Factory templates need to use the raw enum instead of the class wrapper.
  using KindTemplateEnum = Internal::SemanticsNodeKindRawEnum;

  // The FactoryBase and FactoryNBase structs provide Make and Get to support 0,
  // 1, or 2 arguments for a SemanticsNode. These are protected so that child
  // factories can opt in to what pieces they want to use.
  template <KindTemplateEnum Kind, typename... ArgTypes>
  struct FactoryBase {
   protected:
    static auto Make(ParseTree::Node parse_node, SemanticsNodeId type_id,
                     ArgTypes... arg_ids) -> SemanticsNode {
      return SemanticsNode(parse_node, SemanticsNodeKind::Create(Kind), type_id,
                           arg_ids.index...);
    }
  };
  template <KindTemplateEnum Kind>
  struct Factory0Base : public FactoryBase<Kind> {
   protected:
    static auto Get(SemanticsNode node) -> NoArgs {
      CARBON_CHECK(node.kind() == Kind);
      return NoArgs();
    }
  };
  template <KindTemplateEnum Kind, typename Arg0Type>
  struct Factory1Base : public FactoryBase<Kind, Arg0Type> {
   protected:
    static auto Get(SemanticsNode node) -> Arg0Type {
      CARBON_CHECK(node.kind() == Kind);
      return Arg0Type(node.arg0_);
    }
  };
  template <KindTemplateEnum Kind, typename Arg0Type, typename Arg1Type>
  struct Factory2Base : public FactoryBase<Kind, Arg0Type, Arg1Type> {
   protected:
    static auto Get(SemanticsNode node) -> std::pair<Arg0Type, Arg1Type> {
      CARBON_CHECK(node.kind() == Kind);
      return {Arg0Type(node.arg0_), Arg1Type(node.arg1_)};
    }
  };

  // The Factory structs provide public Make and Get for nodes.
  template <KindTemplateEnum Kind>
  struct Factory0 : public Factory0Base<Kind> {
    using Factory0Base<Kind>::Make;
    using Factory0Base<Kind>::Get;
  };
  template <KindTemplateEnum Kind, typename Arg0Type>
  struct Factory1 : public Factory1Base<Kind, Arg0Type> {
    using Factory1Base<Kind, Arg0Type>::Make;
    using Factory1Base<Kind, Arg0Type>::Get;
  };
  template <KindTemplateEnum Kind, typename Arg0Type, typename Arg1Type>
  struct Factory2 : public Factory2Base<Kind, Arg0Type, Arg1Type> {
    using Factory2Base<Kind, Arg0Type, Arg1Type>::Make;
    using Factory2Base<Kind, Arg0Type, Arg1Type>::Get;
  };

  // The FactoryNPreTyped structs provide a Make that assumes a non-changging
  // type.
  template <KindTemplateEnum Kind, int32_t TypeIndex>
  struct Factory0PreTyped : public Factory0Base<Kind> {
    static auto Make(ParseTree::Node parse_node) -> SemanticsNode {
      SemanticsNodeId type_id(TypeIndex);
      return Factory0Base<Kind>::Make(parse_node, type_id);
    }
    using Factory0Base<Kind>::Get;
  };
  template <KindTemplateEnum Kind, int32_t TypeIndex, typename Arg0Type>
  struct Factory1PreTyped : public Factory1Base<Kind, Arg0Type> {
    static auto Make(ParseTree::Node parse_node, Arg0Type arg0_id)
        -> SemanticsNode {
      SemanticsNodeId type_id(TypeIndex);
      return Factory1Base<Kind, Arg0Type>::Make(parse_node, type_id, arg0_id);
    }
    using Factory1Base<Kind, Arg0Type>::Get;
  };
  template <KindTemplateEnum Kind, int32_t TypeIndex, typename Arg0Type,
            typename Arg1Type>
  struct Factory2PreTyped : public Factory2Base<Kind, Arg0Type, Arg1Type> {
    static auto Make(ParseTree::Node parse_node, Arg0Type arg0_id,
                     Arg1Type arg1_id) -> SemanticsNode {
      SemanticsNodeId type_id(TypeIndex);
      return Factory2Base<Kind, Arg0Type, Arg1Type>::Make(parse_node, type_id,
                                                          arg0_id, arg1_id);
    }
    using Factory2Base<Kind, Arg0Type, Arg1Type>::Get;
  };

 public:
  // Invalid is in the SemanticsNodeKind enum, but should never be used.
  struct Invalid {
    static auto Get(SemanticsNode /*node*/) -> SemanticsNode::NoArgs {
      CARBON_FATAL() << "Invalid access";
    }
  };

  struct Assign : public SemanticsNode::Factory2<SemanticsNodeKind::Assign,
                                                 SemanticsNodeId /*lhs_id*/,
                                                 SemanticsNodeId /*rhs_id*/> {};

  struct BinaryOperatorAdd
      : public SemanticsNode::Factory2<SemanticsNodeKind::BinaryOperatorAdd,
                                       SemanticsNodeId /*lhs_id*/,
                                       SemanticsNodeId /*rhs_id*/> {};

  struct BindName
      : public SemanticsNode::Factory2<SemanticsNodeKind::BindName,
                                       SemanticsStringId /*name_id*/,
                                       SemanticsNodeId /*node_id*/> {};

  struct Builtin {
    static auto Make(SemanticsBuiltinKind builtin_kind, SemanticsNodeId type_id)
        -> SemanticsNode {
      // Builtins won't have a ParseTree node associated, so we provide the
      // default invalid one.
      // This can't use the standard Make function because of the `AsInt()` cast
      // instead of `.index`.
      return SemanticsNode(ParseTree::Node::Invalid, SemanticsNodeKind::Builtin,
                           type_id, builtin_kind.AsInt());
    }
    static auto Get(SemanticsNode node) -> SemanticsBuiltinKind {
      return SemanticsBuiltinKind::FromInt(node.arg0_);
    }
  };

  struct Call
      : public Factory2<SemanticsNodeKind::Call, SemanticsCallId /*call_id*/,
                        SemanticsCallableId /*callable_id*/> {};

  struct CodeBlock
      : public Factory1PreTyped<SemanticsNodeKind::CodeBlock,
                                SemanticsNodeId::InvalidIndex,
                                SemanticsNodeBlockId /*node_block_id*/> {};

  struct CrossReference
      : public Factory2Base<SemanticsNodeKind::CrossReference,
                            SemanticsCrossReferenceIRId /*ir_id*/,
                            SemanticsNodeId /*node_id*/> {
    static auto Make(SemanticsNodeId type_id, SemanticsCrossReferenceIRId ir_id,
                     SemanticsNodeId node_id) -> SemanticsNode {
      // A node's parse tree node must refer to a node in the current parse
      // tree. This cannot use the cross-referenced node's parse tree node
      // because it will be in a different parse tree.
      return Factory2Base::Make(ParseTree::Node::Invalid, type_id, ir_id,
                                node_id);
    }
    using Factory2Base::Get;
  };

  struct FunctionDeclaration
      : public Factory2PreTyped<SemanticsNodeKind::FunctionDeclaration,
                                SemanticsNodeId::InvalidIndex,
                                SemanticsStringId /*name_id*/,
                                SemanticsCallableId /*signature_id*/> {};

  struct FunctionDefinition
      : public Factory2PreTyped<SemanticsNodeKind::FunctionDefinition,
                                SemanticsNodeId::InvalidIndex,
                                SemanticsNodeId /*decl_id*/,
                                SemanticsNodeBlockId /*node_block_id*/> {};

  struct IntegerLiteral
      : public Factory1PreTyped<SemanticsNodeKind::IntegerLiteral,
                                SemanticsBuiltinKind::IntegerType.AsInt(),
                                SemanticsIntegerLiteralId /*integer_id*/> {};

  struct RealLiteral
      : public Factory1PreTyped<SemanticsNodeKind::RealLiteral,
                                SemanticsBuiltinKind::FloatingPointType.AsInt(),
                                SemanticsRealLiteralId /*real_id*/> {};

  struct Return : public Factory0PreTyped<SemanticsNodeKind::Return,
                                          SemanticsNodeId::InvalidIndex> {};

  struct ReturnExpression : public Factory1<SemanticsNodeKind::ReturnExpression,
                                            SemanticsNodeId /*expr_id*/> {};

  struct StringLiteral
      : public Factory1PreTyped<SemanticsNodeKind::StringLiteral,
                                SemanticsBuiltinKind::StringType.AsInt(),
                                SemanticsStringId /*string_id*/> {};

  struct VarStorage : public Factory0<SemanticsNodeKind::VarStorage> {};

  SemanticsNode()
      : SemanticsNode(ParseTree::Node::Invalid, SemanticsNodeKind::Invalid,
                      SemanticsNodeId::Invalid) {}

  // Provide `node.GetAsKind()` as an instance method for all kinds, essentially
  // an alias for`SemanticsNode::Kind::Get(node)`.
#define CARBON_SEMANTICS_NODE_KIND(Name) \
  auto GetAs##Name() const->auto{ return Name::Get(*this); }
#include "toolchain/semantics/semantics_node_kind.def"

  auto parse_node() const -> ParseTree::Node { return parse_node_; }
  auto kind() const -> SemanticsNodeKind { return kind_; }
  auto type_id() const -> SemanticsNodeId { return type_id_; }

  auto Print(llvm::raw_ostream& out) const -> void;

 private:
  // Builtins have peculiar construction, so they are a friend rather than using
  // a factory base class.
  friend struct SemanticsNodeForBuiltin;

  explicit SemanticsNode(ParseTree::Node parse_node, SemanticsNodeKind kind,
                         SemanticsNodeId type_id,
                         int32_t arg0 = SemanticsNodeId::InvalidIndex,
                         int32_t arg1 = SemanticsNodeId::InvalidIndex)
      : parse_node_(parse_node),
        kind_(kind),
        type_id_(type_id),
        arg0_(arg0),
        arg1_(arg1) {}

  ParseTree::Node parse_node_;
  SemanticsNodeKind kind_;
  SemanticsNodeId type_id_;

  // Use SemanticsKindNode::Get to access arg0 and arg1.
  int32_t arg0_;
  int32_t arg1_;
};

// TODO: This is currently 20 bytes because we sometimes have 2 arguments for a
// pair of SemanticsNodes. However, SemanticsNodeKind is 1 byte; if args
// were 3.5 bytes, we could potentially shrink SemanticsNode by 4 bytes. This
// may be worth investigating further.
static_assert(sizeof(SemanticsNode) == 20, "Unexpected SemanticsNode size");

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_NODE_H_
