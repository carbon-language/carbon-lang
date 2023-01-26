// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_parse_tree_handler.h"

#include <functional>
#include <utility>

#include "common/vlog.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "toolchain/lexer/token_kind.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"
#include "toolchain/semantics/semantics_builtin_kind.h"
#include "toolchain/semantics/semantics_ir.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon {

class PrettyStackTraceFunction : public llvm::PrettyStackTraceEntry {
 public:
  explicit PrettyStackTraceFunction(std::function<void(llvm::raw_ostream&)> fn)
      : fn_(std::move(fn)) {}
  ~PrettyStackTraceFunction() override = default;

  auto print(llvm::raw_ostream& output) const -> void override { fn_(output); }

 private:
  const std::function<void(llvm::raw_ostream&)> fn_;
};

auto SemanticsParseTreeHandler::Build() -> void {
  PrettyStackTraceFunction pretty_node_stack([&](llvm::raw_ostream& output) {
    node_stack_.PrintForStackDump(output);
  });
  PrettyStackTraceFunction pretty_node_block_stack(
      [&](llvm::raw_ostream& output) {
        node_block_stack_.PrintForStackDump(output);
      });

  // Add a block for the ParseTree.
  node_block_stack_.Push();
  PushScope();

  for (auto parse_node : parse_tree_->postorder()) {
    switch (auto parse_kind = parse_tree_->node_kind(parse_node)) {
#define CARBON_PARSE_NODE_KIND(Name) \
  case ParseNodeKind::Name: {        \
    Handle##Name(parse_node);        \
    break;                           \
  }
#include "toolchain/parser/parse_node_kind.def"
    }
  }

  node_block_stack_.Pop();

  PopScope();
  CARBON_CHECK(name_lookup_.empty()) << name_lookup_.size();
  CARBON_CHECK(scope_stack_.empty()) << scope_stack_.size();
}

auto SemanticsParseTreeHandler::AddNode(SemanticsNode node) -> SemanticsNodeId {
  auto block = node_block_stack_.PeekForAdd();
  CARBON_VLOG() << "AddNode " << block << ": " << node << "\n";
  return semantics_->AddNode(block, node);
}

auto SemanticsParseTreeHandler::AddNodeAndPush(ParseTree::Node parse_node,
                                               SemanticsNode node) -> void {
  auto node_id = AddNode(node);
  node_stack_.Push(parse_node, node_id);
}

auto SemanticsParseTreeHandler::BindName(ParseTree::Node name_node,
                                         SemanticsNodeId type_id,
                                         SemanticsNodeId target_id)
    -> SemanticsStringId {
  CARBON_CHECK(parse_tree_->node_kind(name_node) == ParseNodeKind::DeclaredName)
      << parse_tree_->node_kind(name_node);
  auto name_str = parse_tree_->GetNodeText(name_node);
  auto name_id = semantics_->AddString(name_str);

  AddNode(SemanticsNode::MakeBindName(name_node, type_id, name_id, target_id));
  auto [it, inserted] = current_scope().names.insert(name_id);
  if (inserted) {
    name_lookup_[name_id].push_back(target_id);
  } else {
    CARBON_DIAGNOSTIC(NameRedefined, Error, "Redefining {0} in the same scope.",
                      llvm::StringRef);
    CARBON_DIAGNOSTIC(PreviousDefinition, Note, "Previous definition is here.");
    auto prev_def_id = name_lookup_[name_id].back();
    auto prev_def = semantics_->GetNode(prev_def_id);

    emitter_->Build(name_node, NameRedefined, name_str)
        .Note(prev_def.parse_node(), PreviousDefinition)
        .Emit();
  }
  return name_id;
}

auto SemanticsParseTreeHandler::PushScope() -> void {
  scope_stack_.push_back({});
}

auto SemanticsParseTreeHandler::PopScope() -> void {
  auto scope = scope_stack_.pop_back_val();
  for (const auto& str_id : scope.names) {
    auto it = name_lookup_.find(str_id);
    if (it->second.size() == 1) {
      // Erase names that no longer resolve.
      name_lookup_.erase(it);
    } else {
      it->second.pop_back();
    }
  }
}

auto SemanticsParseTreeHandler::TryTypeConversion(ParseTree::Node parse_node,
                                                  SemanticsNodeId lhs_id,
                                                  SemanticsNodeId rhs_id,
                                                  bool /*can_convert_lhs*/)
    -> SemanticsNodeId {
  auto lhs_type = semantics_->GetType(lhs_id);
  auto rhs_type = semantics_->GetType(rhs_id);
  // TODO: This should attempt a type conversion, but there's not enough
  // implemented to do that right now.
  if (lhs_type != rhs_type) {
    auto invalid_type = SemanticsNodeId::MakeBuiltinReference(
        SemanticsBuiltinKind::InvalidType);
    if (lhs_type != invalid_type && rhs_type != invalid_type) {
      // TODO: This is a poor diagnostic, and should be expanded.
      CARBON_DIAGNOSTIC(TypeMismatch, Error,
                        "Type mismatch: lhs is {0}, rhs is {1}",
                        SemanticsNodeId, SemanticsNodeId);
      emitter_->Emit(parse_node, TypeMismatch, lhs_type, rhs_type);
    }
    return invalid_type;
  }
  return lhs_type;
}

auto SemanticsParseTreeHandler::HandleAddress(ParseTree::Node /*parse_node*/)
    -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleBreakStatement(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleBreakStatementStart(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleCallExpression(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleCallExpressionComma(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleCallExpressionStart(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleCodeBlock(ParseTree::Node /*parse_node*/)
    -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleCodeBlockStart(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleContinueStatement(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleContinueStatementStart(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleDeclaredName(ParseTree::Node parse_node)
    -> void {
  // The parent is responsible for binding the name.
  node_stack_.Push(parse_node);
}

auto SemanticsParseTreeHandler::HandleDeducedParameterList(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleDeducedParameterListStart(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleDesignatedName(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleDesignatorExpression(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleEmptyDeclaration(
    ParseTree::Node parse_node) -> void {
  // Empty declarations have no actions associated, but we still balance the
  // tree.
  node_stack_.Push(parse_node);
}

auto SemanticsParseTreeHandler::HandleExpressionStatement(
    ParseTree::Node parse_node) -> void {
  // Pop the expression without investigating its contents.
  // TODO: This will probably eventually need to do some "do not discard"
  // analysis.
  node_stack_.PopAndDiscardId();
  node_stack_.Push(parse_node);
}

auto SemanticsParseTreeHandler::HandleFileEnd(ParseTree::Node /*parse_node*/)
    -> void {
  // Do nothing, no need to balance this node.
}

auto SemanticsParseTreeHandler::HandleForHeader(ParseTree::Node /*parse_node*/)
    -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleForHeaderStart(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleForIn(ParseTree::Node /*parse_node*/)
    -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleForStatement(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleFunctionDeclaration(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleFunctionDefinition(
    ParseTree::Node parse_node) -> void {
  // Merges code block children up under the FunctionDefinitionStart.
  while (parse_tree_->node_kind(node_stack_.PeekParseNode()) !=
         ParseNodeKind::FunctionDefinitionStart) {
    node_stack_.PopAndIgnore();
  }
  auto decl_id =
      node_stack_.PopForNodeId(ParseNodeKind::FunctionDefinitionStart);

  PopScope();
  auto block_id = node_block_stack_.Pop();
  AddNode(SemanticsNode::MakeFunctionDefinition(parse_node, decl_id, block_id));
  node_stack_.Push(parse_node);
}

auto SemanticsParseTreeHandler::HandleFunctionDefinitionStart(
    ParseTree::Node parse_node) -> void {
  node_stack_.PopForSoloParseNode(ParseNodeKind::ParameterList);
  auto [param_ir_id, param_refs_id] = finished_params_stack_.pop_back_val();
  auto name_node = node_stack_.PopForSoloParseNode(ParseNodeKind::DeclaredName);
  auto fn_node =
      node_stack_.PopForSoloParseNode(ParseNodeKind::FunctionIntroducer);

  SemanticsCallable callable;
  callable.param_ir_id = param_ir_id;
  callable.param_refs_id = param_refs_id;
  auto callable_id = semantics_->AddCallable(callable);
  auto decl_id =
      AddNode(SemanticsNode::MakeFunctionDeclaration(fn_node, callable_id));
  // TODO: Propagate the type of the function.
  BindName(name_node, SemanticsNodeId::MakeInvalid(), decl_id);

  node_block_stack_.Push();
  PushScope();
  node_stack_.Push(parse_node, decl_id);
}

auto SemanticsParseTreeHandler::HandleFunctionIntroducer(
    ParseTree::Node parse_node) -> void {
  // No action, just a bracketing node.
  node_stack_.Push(parse_node);
}

auto SemanticsParseTreeHandler::HandleIfCondition(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleIfConditionStart(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleIfStatement(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleIfStatementElse(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleInfixOperator(ParseTree::Node parse_node)
    -> void {
  auto rhs_id = node_stack_.PopForNodeId();
  auto lhs_id = node_stack_.PopForNodeId();
  SemanticsNodeId result_type =
      TryTypeConversion(parse_node, lhs_id, rhs_id, /*can_convert_lhs=*/true);

  // Figure out the operator for the token.
  auto token = parse_tree_->node_token(parse_node);
  switch (auto token_kind = tokens_->GetKind(token)) {
    case TokenKind::Plus:
      AddNodeAndPush(parse_node, SemanticsNode::MakeBinaryOperatorAdd(
                                     parse_node, result_type, lhs_id, rhs_id));
      break;
    default:
      CARBON_FATAL() << "Unrecognized token kind: " << token_kind;
  }
}

auto SemanticsParseTreeHandler::HandleInterfaceBody(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleInterfaceBodyStart(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleInterfaceDefinition(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleLiteral(ParseTree::Node parse_node)
    -> void {
  auto token = parse_tree_->node_token(parse_node);
  switch (auto token_kind = tokens_->GetKind(token)) {
    case TokenKind::IntegerLiteral: {
      auto id =
          semantics_->AddIntegerLiteral(tokens_->GetIntegerLiteral(token));
      AddNodeAndPush(parse_node,
                     SemanticsNode::MakeIntegerLiteral(parse_node, id));
      break;
    }
    case TokenKind::RealLiteral: {
      // TODO: Add storage of the Real literal.
      AddNodeAndPush(parse_node, SemanticsNode::MakeRealLiteral(parse_node));
      break;
    }
    case TokenKind::IntegerTypeLiteral: {
      auto text = tokens_->GetTokenText(token);
      CARBON_CHECK(text == "i32") << "Currently only i32 is allowed";
      node_stack_.Push(parse_node, SemanticsNodeId::MakeBuiltinReference(
                                       SemanticsBuiltinKind::IntegerType));
      break;
    }
    default:
      CARBON_FATAL() << "Unhandled kind: " << token_kind;
  }
}

auto SemanticsParseTreeHandler::HandleNameReference(ParseTree::Node parse_node)
    -> void {
  auto name_str = parse_tree_->GetNodeText(parse_node);

  auto name_not_found = [&] {
    CARBON_DIAGNOSTIC(NameNotFound, Error, "Name {0} not found",
                      llvm::StringRef);
    emitter_->Emit(parse_node, NameNotFound, name_str);
    node_stack_.Push(parse_node, SemanticsNodeId::MakeBuiltinReference(
                                     SemanticsBuiltinKind::InvalidType));
  };

  auto name_id = semantics_->GetString(name_str);
  if (!name_id) {
    name_not_found();
    return;
  }

  auto it = name_lookup_.find(*name_id);
  if (it == name_lookup_.end()) {
    name_not_found();
    return;
  }
  CARBON_CHECK(!it->second.empty()) << "Should have been erased: " << name_str;

  // TODO: Check for ambiguous lookups.
  node_stack_.Push(parse_node, it->second.back());
}

auto SemanticsParseTreeHandler::HandlePackageApi(ParseTree::Node /*parse_node*/)
    -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandlePackageDirective(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandlePackageImpl(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandlePackageIntroducer(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandlePackageLibrary(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::SaveParam() -> bool {
  // Copy the last node added to the IR block into the params block.
  auto ir_id = node_block_stack_.Peek();
  if (!ir_id.is_valid()) {
    return false;
  }
  auto& ir = semantics_->GetNodeBlock(ir_id);
  CARBON_CHECK(!ir.empty())
      << "Should only have a valid ID if a node was added";
  auto& param = ir.back();
  auto& params = semantics_->GetNodeBlock(params_stack_.PeekForAdd());
  if (!params.empty() && param == params.back()) {
    // The param was already added after a comma.
    return false;
  }
  params.push_back(ir.back());
  return true;
}

auto SemanticsParseTreeHandler::HandleParameterList(ParseTree::Node parse_node)
    -> void {
  // If there's a node in the IR block that has yet to be added to the params
  // block, add it now.
  SaveParam();

  while (true) {
    switch (auto parse_kind =
                parse_tree_->node_kind(node_stack_.PeekParseNode())) {
      case ParseNodeKind::ParameterListStart:
        node_stack_.PopAndDiscardSoloParseNode(
            ParseNodeKind::ParameterListStart);
        finished_params_stack_.push_back(
            {node_block_stack_.Pop(), params_stack_.Pop()});
        node_stack_.Push(parse_node);
        return;

      case ParseNodeKind::ParameterListComma:
        node_stack_.PopAndDiscardSoloParseNode(
            ParseNodeKind::ParameterListComma);
        break;

      case ParseNodeKind::PatternBinding:
        node_stack_.PopAndDiscardId(ParseNodeKind::PatternBinding);
        break;

      default:
        // This should only occur for invalid parse trees.
        CARBON_FATAL() << "TODO: " << parse_kind;
    }
  }

  llvm_unreachable("loop always exits");
}

auto SemanticsParseTreeHandler::HandleParameterListComma(
    ParseTree::Node parse_node) -> void {
  node_stack_.Push(parse_node);

  // Copy the last node added to the IR block into the params block.
  bool had_param_before_comma = SaveParam();
  CARBON_CHECK(had_param_before_comma)
      << "TODO: Should have a param before comma, will need error recovery";
}

auto SemanticsParseTreeHandler::HandleParameterListStart(
    ParseTree::Node parse_node) -> void {
  node_stack_.Push(parse_node);

  params_stack_.Push();
  node_block_stack_.Push();
}

auto SemanticsParseTreeHandler::HandleParenExpression(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleParenExpressionOrTupleLiteralStart(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandlePatternBinding(ParseTree::Node parse_node)
    -> void {
  auto type = node_stack_.PopForNodeId();

  // Get the name.
  auto name_node = node_stack_.PopForSoloParseNode();

  // Allocate storage, linked to the name for error locations.
  auto storage_id = AddNode(SemanticsNode::MakeVarStorage(name_node, type));

  // Bind the name to storage.
  auto name_id = BindName(name_node, type, storage_id);

  // If this node's result is used, it'll be for either the name or the storage
  // address. The storage address can be found through the name, so we push the
  // name.
  node_stack_.Push(parse_node, name_id);
}

auto SemanticsParseTreeHandler::HandlePostfixOperator(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandlePrefixOperator(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleReturnStatement(
    ParseTree::Node parse_node) -> void {
  if (parse_tree_->node_kind(node_stack_.PeekParseNode()) ==
      ParseNodeKind::ReturnStatementStart) {
    node_stack_.PopAndDiscardSoloParseNode(ParseNodeKind::ReturnStatementStart);
    AddNodeAndPush(parse_node, SemanticsNode::MakeReturn(parse_node));
  } else {
    auto arg = node_stack_.PopForNodeId();
    auto arg_type = semantics_->GetType(arg);
    node_stack_.PopAndDiscardSoloParseNode(ParseNodeKind::ReturnStatementStart);
    AddNodeAndPush(parse_node, SemanticsNode::MakeReturnExpression(
                                   parse_node, arg_type, arg));
  }
}

auto SemanticsParseTreeHandler::HandleReturnStatementStart(
    ParseTree::Node parse_node) -> void {
  // No action, just a bracketing node.
  node_stack_.Push(parse_node);
}

auto SemanticsParseTreeHandler::HandleReturnType(ParseTree::Node /*parse_node*/)
    -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleSelfDeducedParameter(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleSelfType(ParseTree::Node /*parse_node*/)
    -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleStructComma(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleStructFieldDesignator(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleStructFieldType(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleStructFieldUnknown(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleStructFieldValue(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleStructLiteral(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleStructLiteralOrStructTypeLiteralStart(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleStructTypeLiteral(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleTupleLiteral(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleTupleLiteralComma(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleVariableDeclaration(
    ParseTree::Node parse_node) -> void {
  auto last_child = node_stack_.PopForParseNodeAndNodeId();

  if (parse_tree_->node_kind(last_child.first) !=
      ParseNodeKind::PatternBinding) {
    auto storage_id =
        node_stack_.PopForNodeId(ParseNodeKind::VariableInitializer);

    auto binding = node_stack_.PopForParseNodeAndNameId();
    CARBON_CHECK(parse_tree_->node_kind(binding.first) ==
                 ParseNodeKind::PatternBinding);
    CARBON_CHECK(binding.second.is_valid());

    // Restore the name now that the initializer is complete.
    AddNameToLookup(binding.second, storage_id);

    auto storage_type =
        TryTypeConversion(parse_node, storage_id, last_child.second,
                          /*can_convert_lhs=*/false);
    AddNode(SemanticsNode::MakeAssign(parse_node, storage_type, storage_id,
                                      last_child.second));
  }

  node_stack_.PopAndDiscardSoloParseNode(ParseNodeKind::VariableIntroducer);
  node_stack_.Push(parse_node);
}

auto SemanticsParseTreeHandler::HandleVariableIntroducer(
    ParseTree::Node parse_node) -> void {
  // No action, just a bracketing node.
  node_stack_.Push(parse_node);
}

auto SemanticsParseTreeHandler::HandleVariableInitializer(
    ParseTree::Node parse_node) -> void {
  // Temporarily remove name lookup entries added by the `var`. These will be
  // restored by `VariableDeclaration`.

  // Save the storage ID.
  auto it = name_lookup_.find(node_stack_.PeekForNameId());
  CARBON_CHECK(it != name_lookup_.end());
  CARBON_CHECK(!it->second.empty());
  auto storage_id = it->second.back();

  // Pop the name from lookup.
  if (it->second.size() == 1) {
    // Erase names that no longer resolve.
    name_lookup_.erase(it);
  } else {
    it->second.pop_back();
  }

  node_stack_.Push(parse_node, storage_id);
}

auto SemanticsParseTreeHandler::HandleWhileCondition(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleWhileConditionStart(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

auto SemanticsParseTreeHandler::HandleWhileStatement(
    ParseTree::Node /*parse_node*/) -> void {
  CARBON_FATAL() << "TODO";
}

}  // namespace Carbon
