// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_NODE_STACK_H_
#define CARBON_TOOLCHAIN_CHECK_NODE_STACK_H_

#include <type_traits>

#include "common/vlog.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

// Wraps the stack of parse nodes for Context. Each parse node can have an
// associated id of some kind (instruction, instruction block, function, class,
// ...).
//
// All pushes and pops will be vlogged.
//
// Pop APIs will run basic verification:
//
// - If receiving a pop_parse_kind, verify that the parse_node being popped is
//   of pop_parse_kind.
// - Validates presence of inst_id based on whether it's a solo
//   parse_node.
//
// These should be assumed API constraints unless otherwise mentioned on a
// method. The main exception is PopAndIgnore, which doesn't do verification.
class NodeStack {
 public:
  explicit NodeStack(const Parse::Tree& parse_tree,
                     llvm::raw_ostream* vlog_stream)
      : parse_tree_(&parse_tree), vlog_stream_(vlog_stream) {}

  // Pushes a solo parse tree node onto the stack. Used when there is no
  // IR generated by the node.
  auto Push(Parse::NodeId parse_node) -> void {
    auto kind = parse_tree_->node_kind(parse_node);
    CARBON_CHECK(ParseNodeKindToIdKind(kind) == IdKind::SoloParseNode)
        << "Parse kind expects an Id: " << kind;
    CARBON_VLOG() << "Node Push " << stack_.size() << ": " << kind
                  << " -> <none>\n";
    CARBON_CHECK(stack_.size() < (1 << 20))
        << "Excessive stack size: likely infinite loop";
    stack_.push_back(Entry(parse_node, SemIR::InstId::Invalid));
  }

  // Pushes a parse tree node onto the stack with an ID.
  template <typename IdT>
  auto Push(Parse::NodeId parse_node, IdT id) -> void {
    auto kind = parse_tree_->node_kind(parse_node);
    CARBON_CHECK(ParseNodeKindToIdKind(kind) == IdTypeToIdKind<IdT>())
        << "Parse kind expected a different IdT: " << kind << " -> " << id
        << "\n";
    CARBON_CHECK(id.is_valid()) << "Push called with invalid id: "
                                << parse_tree_->node_kind(parse_node);
    CARBON_VLOG() << "Node Push " << stack_.size() << ": " << kind << " -> "
                  << id << "\n";
    CARBON_CHECK(stack_.size() < (1 << 20))
        << "Excessive stack size: likely infinite loop";
    stack_.push_back(Entry(parse_node, id));
  }

  // Returns whether there is a node of the specified kind on top of the stack.
  auto PeekIs(Parse::NodeKind kind) const -> bool {
    return !stack_.empty() && PeekParseNodeKind() == kind;
  }

  // Returns whether there is a node of the specified kind on top of the stack.
  // Templated for consistency with other functions taking a parse node kind.
  template <const Parse::NodeKind& RequiredParseKind>
  auto PeekIs() const -> bool {
    return PeekIs(RequiredParseKind);
  }

  // Returns whether there is a name on top of the stack.
  auto PeekIsName() const -> bool {
    return !stack_.empty() &&
           ParseNodeKindToIdKind(PeekParseNodeKind()) == IdKind::NameId;
  }

  // Pops the top of the stack without any verification.
  auto PopAndIgnore() -> void {
    Entry back = stack_.pop_back_val();
    CARBON_VLOG() << "Node Pop " << stack_.size() << ": "
                  << parse_tree_->node_kind(back.parse_node)
                  << " -> <ignored>\n";
  }

  // Pops the top of the stack and returns the parse_node.
  template <const Parse::NodeKind& RequiredParseKind>
  auto PopForSoloParseNode() -> Parse::NodeIdForKind<RequiredParseKind> {
    Entry back = PopEntry<SemIR::InstId>();
    RequireIdKind(RequiredParseKind, IdKind::SoloParseNode);
    RequireParseKind<RequiredParseKind>(back.parse_node);
    return Parse::NodeIdForKind<RequiredParseKind>(back.parse_node);
  }

  // Pops the top of the stack if it is the given kind, and returns the
  // parse_node. Otherwise, returns std::nullopt.
  template <const Parse::NodeKind& RequiredParseKind>
  auto PopForSoloParseNodeIf()
      -> std::optional<Parse::NodeIdForKind<RequiredParseKind>> {
    if (PeekIs<RequiredParseKind>()) {
      return PopForSoloParseNode<RequiredParseKind>();
    }
    return std::nullopt;
  }

  // Pops the top of the stack.
  template <const Parse::NodeKind& RequiredParseKind>
  auto PopAndDiscardSoloParseNode() -> void {
    PopForSoloParseNode<RequiredParseKind>();
  }

  // Pops the top of the stack if it is the given kind. Returns `true` if a node
  // was popped.
  template <const Parse::NodeKind& RequiredParseKind>
  auto PopAndDiscardSoloParseNodeIf() -> bool {
    if (!PeekIs<RequiredParseKind>()) {
      return false;
    }
    PopForSoloParseNode<RequiredParseKind>();
    return true;
  }

  // Pops an expression from the top of the stack and returns the parse_node and
  // the ID.
  auto PopExprWithParseNode() -> std::pair<Parse::NodeId, SemIR::InstId> {
    return PopWithParseNode<SemIR::InstId>();
  }

  // Pops a pattern from the top of the stack and returns the parse_node and
  // the ID.
  auto PopPatternWithParseNode() -> std::pair<Parse::NodeId, SemIR::InstId> {
    return PopWithParseNode<SemIR::InstId>();
  }

  // Pops a name from the top of the stack and returns the parse_node and
  // the ID.
  auto PopNameWithParseNode() -> std::pair<Parse::NodeId, SemIR::NameId> {
    return PopWithParseNode<SemIR::NameId>();
  }

  // Pops the top of the stack and returns the parse_node and the ID.
  template <const Parse::NodeKind& RequiredParseKind>
  auto PopWithParseNode() -> auto {
    constexpr IdKind RequiredIdKind = ParseNodeKindToIdKind(RequiredParseKind);
    auto node_id_cast = [&](auto back) {
      using NodeIdT = Parse::NodeIdForKind<RequiredParseKind>;
      return std::pair<NodeIdT, decltype(back.second)>(back);
    };

    if constexpr (RequiredIdKind == IdKind::InstId) {
      auto back = PopWithParseNode<SemIR::InstId>();
      RequireParseKind<RequiredParseKind>(back.first);
      return node_id_cast(back);
    }
    if constexpr (RequiredIdKind == IdKind::InstBlockId) {
      auto back = PopWithParseNode<SemIR::InstBlockId>();
      RequireParseKind<RequiredParseKind>(back.first);
      return node_id_cast(back);
    }
    if constexpr (RequiredIdKind == IdKind::FunctionId) {
      auto back = PopWithParseNode<SemIR::FunctionId>();
      RequireParseKind<RequiredParseKind>(back.first);
      return node_id_cast(back);
    }
    if constexpr (RequiredIdKind == IdKind::ClassId) {
      auto back = PopWithParseNode<SemIR::ClassId>();
      RequireParseKind<RequiredParseKind>(back.first);
      return node_id_cast(back);
    }
    if constexpr (RequiredIdKind == IdKind::InterfaceId) {
      auto back = PopWithParseNode<SemIR::InterfaceId>();
      RequireParseKind<RequiredParseKind>(back.first);
      return node_id_cast(back);
    }
    if constexpr (RequiredIdKind == IdKind::NameId) {
      auto back = PopWithParseNode<SemIR::NameId>();
      RequireParseKind<RequiredParseKind>(back.first);
      return node_id_cast(back);
    }
    if constexpr (RequiredIdKind == IdKind::TypeId) {
      auto back = PopWithParseNode<SemIR::TypeId>();
      RequireParseKind<RequiredParseKind>(back.first);
      return node_id_cast(back);
    }
    CARBON_FATAL() << "Unpoppable IdKind for parse kind: " << RequiredParseKind
                   << "; see value in ParseNodeKindToIdKind";
  }

  // Pops the top of the stack and returns the parse_node and the ID if it is
  // of the specified kind.
  template <const Parse::NodeKind& RequiredParseKind>
  auto PopWithParseNodeIf()
      -> std::optional<decltype(PopWithParseNode<RequiredParseKind>())> {
    if (!PeekIs<RequiredParseKind>()) {
      return std::nullopt;
    }
    return PopWithParseNode<RequiredParseKind>();
  }

  // Pops an expression from the top of the stack and returns the ID.
  // Expressions map multiple Parse::NodeKinds to SemIR::InstId always.
  auto PopExpr() -> SemIR::InstId { return PopExprWithParseNode().second; }

  // Pops a pattern from the top of the stack and returns the ID.
  // Patterns map multiple Parse::NodeKinds to SemIR::InstId always.
  auto PopPattern() -> SemIR::InstId {
    return PopPatternWithParseNode().second;
  }

  // Pops a name from the top of the stack and returns the ID.
  auto PopName() -> SemIR::NameId { return PopNameWithParseNode().second; }

  // TODO: Can we add a `Pop<...>` that takes a parse node category? See
  // https://github.com/carbon-language/carbon-lang/pull/3534/files#r1432067519

  // Pops the top of the stack and returns the ID.
  template <const Parse::NodeKind& RequiredParseKind>
  auto Pop() -> auto {
    return PopWithParseNode<RequiredParseKind>().second;
  }

  // Pops the top of the stack if it has the given kind, and returns the ID.
  // Otherwise returns std::nullopt.
  template <const Parse::NodeKind& RequiredParseKind>
  auto PopIf() -> std::optional<decltype(Pop<RequiredParseKind>())> {
    if (PeekIs<RequiredParseKind>()) {
      return Pop<RequiredParseKind>();
    }
    return std::nullopt;
  }

  // Peeks at the parse node of the top of the node stack.
  auto PeekParseNode() const -> Parse::NodeId {
    return stack_.back().parse_node;
  }

  // Peeks at the kind of the parse node of the top of the node stack.
  auto PeekParseNodeKind() const -> Parse::NodeKind {
    return parse_tree_->node_kind(PeekParseNode());
  }

  // Peeks at the ID associated with the top of the name stack.
  template <const Parse::NodeKind& RequiredParseKind>
  auto Peek() const -> auto {
    Entry back = stack_.back();
    RequireParseKind<RequiredParseKind>(back.parse_node);
    constexpr IdKind RequiredIdKind = ParseNodeKindToIdKind(RequiredParseKind);
    if constexpr (RequiredIdKind == IdKind::InstId) {
      return back.id<SemIR::InstId>();
    }
    if constexpr (RequiredIdKind == IdKind::InstBlockId) {
      return back.id<SemIR::InstBlockId>();
    }
    if constexpr (RequiredIdKind == IdKind::FunctionId) {
      return back.id<SemIR::FunctionId>();
    }
    if constexpr (RequiredIdKind == IdKind::ClassId) {
      return back.id<SemIR::ClassId>();
    }
    if constexpr (RequiredIdKind == IdKind::InterfaceId) {
      return back.id<SemIR::InterfaceId>();
    }
    if constexpr (RequiredIdKind == IdKind::NameId) {
      return back.id<SemIR::NameId>();
    }
    if constexpr (RequiredIdKind == IdKind::TypeId) {
      return back.id<SemIR::TypeId>();
    }
    CARBON_FATAL() << "Unpeekable IdKind for parse kind: " << RequiredParseKind
                   << "; see value in ParseNodeKindToIdKind";
  }

  // Prints the stack for a stack dump.
  auto PrintForStackDump(llvm::raw_ostream& output) const -> void;

  auto empty() const -> bool { return stack_.empty(); }
  auto size() const -> size_t { return stack_.size(); }

 private:
  // Possible associated ID types.
  enum class IdKind : int8_t {
    InstId,
    InstBlockId,
    FunctionId,
    ClassId,
    InterfaceId,
    NameId,
    TypeId,
    // No associated ID type.
    SoloParseNode,
    // Not expected in the node stack.
    Unused,
  };

  // An entry in stack_.
  struct Entry {
    explicit Entry(Parse::NodeId parse_node, SemIR::InstId inst_id)
        : parse_node(parse_node), inst_id(inst_id) {}
    explicit Entry(Parse::NodeId parse_node, SemIR::InstBlockId inst_block_id)
        : parse_node(parse_node), inst_block_id(inst_block_id) {}
    explicit Entry(Parse::NodeId parse_node, SemIR::FunctionId function_id)
        : parse_node(parse_node), function_id(function_id) {}
    explicit Entry(Parse::NodeId parse_node, SemIR::ClassId class_id)
        : parse_node(parse_node), class_id(class_id) {}
    explicit Entry(Parse::NodeId parse_node, SemIR::InterfaceId interface_id)
        : parse_node(parse_node), interface_id(interface_id) {}
    explicit Entry(Parse::NodeId parse_node, SemIR::NameId name_id)
        : parse_node(parse_node), name_id(name_id) {}
    explicit Entry(Parse::NodeId parse_node, SemIR::TypeId type_id)
        : parse_node(parse_node), type_id(type_id) {}

    // Returns the appropriate ID basaed on type.
    template <typename T>
    auto id() -> T& {
      if constexpr (std::is_same<T, SemIR::InstId>()) {
        return inst_id;
      }
      if constexpr (std::is_same<T, SemIR::InstBlockId>()) {
        return inst_block_id;
      }
      if constexpr (std::is_same<T, SemIR::FunctionId>()) {
        return function_id;
      }
      if constexpr (std::is_same<T, SemIR::ClassId>()) {
        return class_id;
      }
      if constexpr (std::is_same<T, SemIR::InterfaceId>()) {
        return interface_id;
      }
      if constexpr (std::is_same<T, SemIR::NameId>()) {
        return name_id;
      }
      if constexpr (std::is_same<T, SemIR::TypeId>()) {
        return type_id;
      }
    }

    // The parse node associated with the stack entry.
    Parse::NodeId parse_node;

    // The entries will evaluate as invalid if and only if they're a solo
    // parse_node. Invalid is used instead of optional to save space.
    //
    // A discriminator isn't needed because the caller can determine which field
    // is used based on the Parse::NodeKind.
    union {
      SemIR::InstId inst_id;
      SemIR::InstBlockId inst_block_id;
      SemIR::FunctionId function_id;
      SemIR::ClassId class_id;
      SemIR::InterfaceId interface_id;
      SemIR::NameId name_id;
      SemIR::TypeId type_id;
    };
  };
  static_assert(sizeof(Entry) == 8, "Unexpected Entry size");

  // Translate a parse node kind to the enum ID kind it should always provide.
  static constexpr auto ParseNodeKindToIdKind(Parse::NodeKind kind) -> IdKind {
    switch (kind) {
      case Parse::NodeKind::Addr:
      case Parse::NodeKind::ArrayExpr:
      case Parse::NodeKind::BindingPattern:
      case Parse::NodeKind::CallExpr:
      case Parse::NodeKind::CallExprStart:
      case Parse::NodeKind::GenericBindingPattern:
      case Parse::NodeKind::IdentifierNameExpr:
      case Parse::NodeKind::IfExprThen:
      case Parse::NodeKind::IfExprElse:
      case Parse::NodeKind::IndexExpr:
      case Parse::NodeKind::MemberAccessExpr:
      case Parse::NodeKind::PackageExpr:
      case Parse::NodeKind::ParenExpr:
      case Parse::NodeKind::ReturnType:
      case Parse::NodeKind::SelfTypeNameExpr:
      case Parse::NodeKind::SelfValueNameExpr:
      case Parse::NodeKind::ShortCircuitOperandAnd:
      case Parse::NodeKind::ShortCircuitOperandOr:
      case Parse::NodeKind::ShortCircuitOperatorAnd:
      case Parse::NodeKind::ShortCircuitOperatorOr:
      case Parse::NodeKind::StructFieldValue:
      case Parse::NodeKind::StructLiteral:
      case Parse::NodeKind::StructFieldType:
      case Parse::NodeKind::StructTypeLiteral:
      case Parse::NodeKind::TupleLiteral:
      case Parse::NodeKind::VariableInitializer:
        return IdKind::InstId;
      case Parse::NodeKind::IfCondition:
      case Parse::NodeKind::IfExprIf:
      case Parse::NodeKind::ImplicitParamList:
      case Parse::NodeKind::TuplePattern:
      case Parse::NodeKind::WhileCondition:
      case Parse::NodeKind::WhileConditionStart:
        return IdKind::InstBlockId;
      case Parse::NodeKind::FunctionDefinitionStart:
        return IdKind::FunctionId;
      case Parse::NodeKind::ClassDefinitionStart:
        return IdKind::ClassId;
      case Parse::NodeKind::InterfaceDefinitionStart:
        return IdKind::InterfaceId;
      case Parse::NodeKind::BaseName:
      case Parse::NodeKind::IdentifierName:
      case Parse::NodeKind::SelfValueName:
        return IdKind::NameId;
      case Parse::NodeKind::ArrayExprSemi:
      case Parse::NodeKind::ClassIntroducer:
      case Parse::NodeKind::CodeBlockStart:
      case Parse::NodeKind::ExprOpenParen:
      case Parse::NodeKind::FunctionIntroducer:
      case Parse::NodeKind::IfStatementElse:
      case Parse::NodeKind::ImplicitParamListStart:
      case Parse::NodeKind::InterfaceIntroducer:
      case Parse::NodeKind::LetIntroducer:
      case Parse::NodeKind::QualifiedName:
      case Parse::NodeKind::ReturnedModifier:
      case Parse::NodeKind::ReturnStatementStart:
      case Parse::NodeKind::ReturnVarModifier:
      case Parse::NodeKind::StructLiteralOrStructTypeLiteralStart:
      case Parse::NodeKind::TuplePatternStart:
      case Parse::NodeKind::VariableIntroducer:
        return IdKind::SoloParseNode;
// Use x-macros to handle boilerplate cases.
#define CARBON_PARSE_NODE_KIND(...)
#define CARBON_PARSE_NODE_KIND_INFIX_OPERATOR(Name, ...) \
  case Parse::NodeKind::InfixOperator##Name:             \
    return IdKind::InstId;
#define CARBON_PARSE_NODE_KIND_POSTFIX_OPERATOR(Name, ...) \
  case Parse::NodeKind::PostfixOperator##Name:             \
    return IdKind::InstId;
#define CARBON_PARSE_NODE_KIND_PREFIX_OPERATOR(Name, ...) \
  case Parse::NodeKind::PrefixOperator##Name:             \
    return IdKind::InstId;
#define CARBON_PARSE_NODE_KIND_TOKEN_LITERAL(Name, ...) \
  case Parse::NodeKind::Name:                           \
    return IdKind::InstId;
#define CARBON_PARSE_NODE_KIND_TOKEN_MODIFIER(Name, ...) \
  case Parse::NodeKind::Name##Modifier:                  \
    return IdKind::Unused;
#include "toolchain/parse/node_kind.def"
      default:
        return IdKind::Unused;
    }
  }

  // Translates an ID type to the enum ID kind for comparison with
  // ParseNodeKindToIdKind.
  template <typename IdT>
  static constexpr auto IdTypeToIdKind() -> IdKind {
    if constexpr (std::is_same_v<IdT, SemIR::InstId>) {
      return IdKind::InstId;
    }
    if constexpr (std::is_same_v<IdT, SemIR::InstBlockId>) {
      return IdKind::InstBlockId;
    }
    if constexpr (std::is_same_v<IdT, SemIR::FunctionId>) {
      return IdKind::FunctionId;
    }
    if constexpr (std::is_same_v<IdT, SemIR::ClassId>) {
      return IdKind::ClassId;
    }
    if constexpr (std::is_same_v<IdT, SemIR::InterfaceId>) {
      return IdKind::InterfaceId;
    }
    if constexpr (std::is_same_v<IdT, SemIR::NameId>) {
      return IdKind::NameId;
    }
    if constexpr (std::is_same_v<IdT, SemIR::TypeId>) {
      return IdKind::TypeId;
    }
  }

  // Pops an entry.
  template <typename IdT>
  auto PopEntry() -> Entry {
    Entry back = stack_.pop_back_val();
    CARBON_VLOG() << "Node Pop " << stack_.size() << ": "
                  << parse_tree_->node_kind(back.parse_node) << " -> "
                  << back.id<IdT>() << "\n";
    return back;
  }

  // Pops the top of the stack and returns the parse_node and the ID.
  template <typename IdT>
  auto PopWithParseNode() -> std::pair<Parse::NodeId, IdT> {
    Entry back = PopEntry<IdT>();
    RequireIdKind(parse_tree_->node_kind(back.parse_node),
                  IdTypeToIdKind<IdT>());
    return {back.parse_node, back.id<IdT>()};
  }

  // Require a Parse::NodeKind be mapped to a particular IdKind.
  auto RequireIdKind(Parse::NodeKind parse_kind, IdKind id_kind) const -> void {
    CARBON_CHECK(ParseNodeKindToIdKind(parse_kind) == id_kind)
        << "Unexpected IdKind mapping for " << parse_kind;
  }

  // Require an entry to have the given Parse::NodeKind.
  template <const Parse::NodeKind& RequiredParseKind>
  auto RequireParseKind(Parse::NodeId parse_node) const -> void {
    auto actual_kind = parse_tree_->node_kind(parse_node);
    CARBON_CHECK(RequiredParseKind == actual_kind)
        << "Expected " << RequiredParseKind << ", found " << actual_kind;
  }

  // The file's parse tree.
  const Parse::Tree* parse_tree_;

  // Whether to print verbose output.
  llvm::raw_ostream* vlog_stream_;

  // The actual stack.
  // PushEntry and PopEntry control modification in order to centralize
  // vlogging.
  llvm::SmallVector<Entry> stack_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_NODE_STACK_H_
