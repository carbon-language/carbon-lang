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

// Wraps the stack of nodes for Context.
//
// All pushes and pops will be vlogged.
//
// Pop APIs will run basic verification:
//
// - If receiving a pop_parse_kind, verify that the parse_lamp being popped is
//   of pop_parse_kind.
// - Validates presence of inst_id based on whether it's a solo
//   parse_lamp.
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
  auto Push(Parse::Lamp parse_lamp) -> void {
    CARBON_CHECK(ParseNodeKindToIdKind(parse_tree_->node_kind(parse_lamp)) ==
                 IdKind::SoloParseNode)
        << "Parse kind expects an Id: " << parse_tree_->node_kind(parse_lamp);
    CARBON_VLOG() << "Node Push " << stack_.size() << ": "
                  << parse_tree_->node_kind(parse_lamp) << " -> <none>\n";
    CARBON_CHECK(stack_.size() < (1 << 20))
        << "Excessive stack size: likely infinite loop";
    stack_.push_back(Entry(parse_lamp, SemIR::InstId::Invalid));
  }

  // Pushes a parse tree node onto the stack with an ID.
  template <typename IdT>
  auto Push(Parse::Lamp parse_lamp, IdT id) -> void {
    CARBON_CHECK(ParseNodeKindToIdKind(parse_tree_->node_kind(parse_lamp)) ==
                 IdTypeToIdKind<IdT>())
        << "Parse kind expected a different IdT: "
        << parse_tree_->node_kind(parse_lamp) << " -> " << id << "\n";
    CARBON_CHECK(id.is_valid()) << "Push called with invalid id: "
                                << parse_tree_->node_kind(parse_lamp);
    CARBON_VLOG() << "Node Push " << stack_.size() << ": "
                  << parse_tree_->node_kind(parse_lamp) << " -> " << id << "\n";
    CARBON_CHECK(stack_.size() < (1 << 20))
        << "Excessive stack size: likely infinite loop";
    stack_.push_back(Entry(parse_lamp, id));
  }

  // Returns whether the node on the top of the stack is the specified kind.
  template <Parse::LampKind::RawEnumType RequiredParseKind>
  auto PeekIs() const -> bool {
    return parse_tree_->node_kind(PeekParseNode()) == RequiredParseKind;
  }

  // Pops the top of the stack without any verification.
  auto PopAndIgnore() -> void { PopEntry<SemIR::InstId>(); }

  // Pops the top of the stack and returns the parse_lamp.
  template <Parse::LampKind::RawEnumType RequiredParseKind>
  auto PopForSoloParseNode() -> Parse::Lamp {
    Entry back = PopEntry<SemIR::InstId>();
    RequireIdKind(Parse::LampKind::Create(RequiredParseKind),
                  IdKind::SoloParseNode);
    RequireParseKind<RequiredParseKind>(back.parse_lamp);
    return back.parse_lamp;
  }

  // Pops the top of the stack if it is the given kind, and returns the
  // parse_lamp. Otherwise, returns std::nullopt.
  template <Parse::LampKind::RawEnumType RequiredParseKind>
  auto PopForSoloParseNodeIf() -> std::optional<Parse::Lamp> {
    if (PeekIs<RequiredParseKind>()) {
      return PopForSoloParseNode<RequiredParseKind>();
    }
    return std::nullopt;
  }

  // Pops the top of the stack.
  template <Parse::LampKind::RawEnumType RequiredParseKind>
  auto PopAndDiscardSoloParseNode() -> void {
    PopForSoloParseNode<RequiredParseKind>();
  }

  // Pops an expression from the top of the stack and returns the parse_lamp and
  // the ID.
  auto PopExpressionWithParseNode() -> std::pair<Parse::Lamp, SemIR::InstId> {
    return PopWithParseNode<SemIR::InstId>();
  }

  // Pops the top of the stack and returns the parse_lamp and the ID.
  template <Parse::LampKind::RawEnumType RequiredParseKind>
  auto PopWithParseNode() -> auto {
    constexpr IdKind RequiredIdKind =
        ParseNodeKindToIdKind(Parse::LampKind::Create(RequiredParseKind));
    if constexpr (RequiredIdKind == IdKind::InstId) {
      auto back = PopWithParseNode<SemIR::InstId>();
      RequireParseKind<RequiredParseKind>(back.first);
      return back;
    }
    if constexpr (RequiredIdKind == IdKind::InstBlockId) {
      auto back = PopWithParseNode<SemIR::InstBlockId>();
      RequireParseKind<RequiredParseKind>(back.first);
      return back;
    }
    if constexpr (RequiredIdKind == IdKind::FunctionId) {
      auto back = PopWithParseNode<SemIR::FunctionId>();
      RequireParseKind<RequiredParseKind>(back.first);
      return back;
    }
    if constexpr (RequiredIdKind == IdKind::ClassId) {
      auto back = PopWithParseNode<SemIR::ClassId>();
      RequireParseKind<RequiredParseKind>(back.first);
      return back;
    }
    if constexpr (RequiredIdKind == IdKind::StringId) {
      auto back = PopWithParseNode<StringId>();
      RequireParseKind<RequiredParseKind>(back.first);
      return back;
    }
    if constexpr (RequiredIdKind == IdKind::TypeId) {
      auto back = PopWithParseNode<SemIR::TypeId>();
      RequireParseKind<RequiredParseKind>(back.first);
      return back;
    }
    CARBON_FATAL() << "Unpoppable IdKind for parse kind: "
                   << Parse::LampKind::Create(RequiredParseKind)
                   << "; see value in ParseNodeKindToIdKind";
  }

  // Pops an expression from the top of the stack and returns the ID.
  // Expressions map multiple Parse::LampKinds to SemIR::InstId always.
  auto PopExpression() -> SemIR::InstId {
    return PopExpressionWithParseNode().second;
  }

  // Pops the top of the stack and returns the ID.
  template <Parse::LampKind::RawEnumType RequiredParseKind>
  auto Pop() -> auto {
    return PopWithParseNode<RequiredParseKind>().second;
  }

  // Pops the top of the stack if it has the given kind, and returns the ID.
  // Otherwise returns std::nullopt.
  template <Parse::LampKind::RawEnumType RequiredParseKind>
  auto PopIf() -> std::optional<decltype(Pop<RequiredParseKind>())> {
    if (PeekIs<RequiredParseKind>()) {
      return Pop<RequiredParseKind>();
    }
    return std::nullopt;
  }

  // Peeks at the parse_lamp of the given depth in the stack, or by default the
  // top node.
  auto PeekParseNode() const -> Parse::Lamp { return stack_.back().parse_lamp; }

  // Peeks at the ID of node at the given depth in the stack, or by default the
  // top node.
  template <Parse::LampKind::RawEnumType RequiredParseKind>
  auto Peek() const -> auto {
    Entry back = stack_.back();
    RequireParseKind<RequiredParseKind>(back.parse_lamp);
    constexpr IdKind RequiredIdKind =
        ParseNodeKindToIdKind(Parse::LampKind::Create(RequiredParseKind));
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
    if constexpr (RequiredIdKind == IdKind::StringId) {
      return back.id<StringId>();
    }
    if constexpr (RequiredIdKind == IdKind::TypeId) {
      return back.id<SemIR::TypeId>();
    }
    CARBON_FATAL() << "Unpeekable IdKind for parse kind: "
                   << Parse::LampKind::Create(RequiredParseKind)
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
    StringId,
    TypeId,
    // No associated ID type.
    SoloParseNode,
    // Not expected in the node stack.
    Unused,
  };

  // An entry in stack_.
  struct Entry {
    explicit Entry(Parse::Lamp parse_lamp, SemIR::InstId inst_id)
        : parse_lamp(parse_lamp), inst_id(inst_id) {}
    explicit Entry(Parse::Lamp parse_lamp, SemIR::InstBlockId inst_block_id)
        : parse_lamp(parse_lamp), inst_block_id(inst_block_id) {}
    explicit Entry(Parse::Lamp parse_lamp, SemIR::FunctionId function_id)
        : parse_lamp(parse_lamp), function_id(function_id) {}
    explicit Entry(Parse::Lamp parse_lamp, SemIR::ClassId class_id)
        : parse_lamp(parse_lamp), class_id(class_id) {}
    explicit Entry(Parse::Lamp parse_lamp, StringId name_id)
        : parse_lamp(parse_lamp), name_id(name_id) {}
    explicit Entry(Parse::Lamp parse_lamp, SemIR::TypeId type_id)
        : parse_lamp(parse_lamp), type_id(type_id) {}

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
      if constexpr (std::is_same<T, StringId>()) {
        return name_id;
      }
      if constexpr (std::is_same<T, SemIR::TypeId>()) {
        return type_id;
      }
    }

    // The node associated with the stack entry.
    Parse::Lamp parse_lamp;

    // The entries will evaluate as invalid if and only if they're a solo
    // parse_lamp. Invalid is used instead of optional to save space.
    //
    // A discriminator isn't needed because the caller can determine which field
    // is used based on the Parse::LampKind.
    union {
      SemIR::InstId inst_id;
      SemIR::InstBlockId inst_block_id;
      SemIR::FunctionId function_id;
      SemIR::ClassId class_id;
      StringId name_id;
      SemIR::TypeId type_id;
    };
  };
  static_assert(sizeof(Entry) == 8, "Unexpected Entry size");

  // Translate a parse node kind to the enum ID kind it should always provide.
  static constexpr auto ParseNodeKindToIdKind(Parse::LampKind kind) -> IdKind {
    switch (kind) {
      case Parse::LampKind::ArrayExpression:
      case Parse::LampKind::CallExpression:
      case Parse::LampKind::CallExpressionStart:
      case Parse::LampKind::IfExpressionThen:
      case Parse::LampKind::IfExpressionElse:
      case Parse::LampKind::IndexExpression:
      case Parse::LampKind::InfixOperator:
      case Parse::LampKind::Literal:
      case Parse::LampKind::MemberAccessExpression:
      case Parse::LampKind::NameExpression:
      case Parse::LampKind::ParenExpression:
      case Parse::LampKind::PatternBinding:
      case Parse::LampKind::PostfixOperator:
      case Parse::LampKind::PrefixOperator:
      case Parse::LampKind::ReturnType:
      case Parse::LampKind::SelfTypeNameExpression:
      case Parse::LampKind::SelfValueNameExpression:
      case Parse::LampKind::ShortCircuitOperand:
      case Parse::LampKind::StructFieldValue:
      case Parse::LampKind::StructLiteral:
      case Parse::LampKind::StructFieldType:
      case Parse::LampKind::StructTypeLiteral:
      case Parse::LampKind::TupleLiteral:
        return IdKind::InstId;
      case Parse::LampKind::IfCondition:
      case Parse::LampKind::IfExpressionIf:
      case Parse::LampKind::ImplicitParameterList:
      case Parse::LampKind::ParameterList:
      case Parse::LampKind::WhileCondition:
      case Parse::LampKind::WhileConditionStart:
        return IdKind::InstBlockId;
      case Parse::LampKind::FunctionDefinitionStart:
        return IdKind::FunctionId;
      case Parse::LampKind::ClassDefinitionStart:
        return IdKind::ClassId;
      case Parse::LampKind::Name:
        return IdKind::StringId;
      case Parse::LampKind::ArrayExpressionSemi:
      case Parse::LampKind::ClassIntroducer:
      case Parse::LampKind::CodeBlockStart:
      case Parse::LampKind::FunctionIntroducer:
      case Parse::LampKind::IfStatementElse:
      case Parse::LampKind::ImplicitParameterListStart:
      case Parse::LampKind::LetIntroducer:
      case Parse::LampKind::ParameterListStart:
      case Parse::LampKind::ParenExpressionOrTupleLiteralStart:
      case Parse::LampKind::QualifiedDeclaration:
      case Parse::LampKind::ReturnStatementStart:
      case Parse::LampKind::SelfValueName:
      case Parse::LampKind::StructLiteralOrStructTypeLiteralStart:
      case Parse::LampKind::VariableInitializer:
      case Parse::LampKind::VariableIntroducer:
        return IdKind::SoloParseNode;
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
    if constexpr (std::is_same_v<IdT, StringId>) {
      return IdKind::StringId;
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
                  << parse_tree_->node_kind(back.parse_lamp) << " -> "
                  << back.id<IdT>() << "\n";
    return back;
  }

  // Pops the top of the stack and returns the parse_lamp and the ID.
  template <typename IdT>
  auto PopWithParseNode() -> std::pair<Parse::Lamp, IdT> {
    Entry back = PopEntry<IdT>();
    RequireIdKind(parse_tree_->node_kind(back.parse_lamp),
                  IdTypeToIdKind<IdT>());
    return {back.parse_lamp, back.id<IdT>()};
  }

  // Require a Parse::LampKind be mapped to a particular IdKind.
  auto RequireIdKind(Parse::LampKind parse_kind, IdKind id_kind) const -> void {
    CARBON_CHECK(ParseNodeKindToIdKind(parse_kind) == id_kind)
        << "Unexpected IdKind mapping for " << parse_kind;
  }

  // Require an entry to have the given Parse::LampKind.
  template <Parse::LampKind::RawEnumType RequiredParseKind>
  auto RequireParseKind(Parse::Lamp parse_lamp) const -> void {
    auto actual_kind = parse_tree_->node_kind(parse_lamp);
    CARBON_CHECK(RequiredParseKind == actual_kind)
        << "Expected " << Parse::LampKind::Create(RequiredParseKind)
        << ", found " << actual_kind;
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
