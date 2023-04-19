// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_parse_tree_handler.h"

#include <functional>
#include <utility>

#include "common/vlog.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "toolchain/diagnostics/diagnostic_kind.h"
#include "toolchain/lexer/token_kind.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"
#include "toolchain/semantics/semantics_ir.h"
#include "toolchain/semantics/semantics_node.h"
#include "toolchain/semantics/semantics_node_block_stack.h"

namespace Carbon {

CARBON_DIAGNOSTIC(SemanticsTodo, Error, "Semantics TODO: {0}", std::string);

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

  // Loops over all nodes in the tree. On some errors, this may return early,
  // for example if an unrecoverable state is encountered.
  for (auto parse_node : parse_tree_->postorder()) {
    switch (auto parse_kind = parse_tree_->node_kind(parse_node)) {
#define CARBON_PARSE_NODE_KIND(Name) \
  case ParseNodeKind::Name: {        \
    if (!Handle##Name(parse_node)) { \
      return;                        \
    }                                \
    break;                           \
  }
#include "toolchain/parser/parse_node_kind.def"
    }
  }

  // Pop information for the file-level scope.
  semantics_->top_node_block_id_ = node_block_stack_.Pop();
  PopScope();

  // Information in all the various context objects should be cleaned up as
  // various pieces of context go out of scope. At this point, nothing should
  // remain.
  // node_stack_ will still contain top-level entities.
  CARBON_CHECK(name_lookup_.empty()) << name_lookup_.size();
  CARBON_CHECK(scope_stack_.empty()) << scope_stack_.size();
  CARBON_CHECK(node_block_stack_.empty()) << node_block_stack_.size();
  CARBON_CHECK(params_or_args_stack_.empty()) << params_or_args_stack_.size();
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

auto SemanticsParseTreeHandler::AddNameToLookup(ParseTree::Node name_node,
                                                SemanticsStringId name_id,
                                                SemanticsNodeId target_id)
    -> void {
  auto [it, inserted] = current_scope().names.insert(name_id);
  if (inserted) {
    name_lookup_[name_id].push_back(target_id);
  } else {
    CARBON_DIAGNOSTIC(NameRedefined, Error, "Redefining {0} in the same scope.",
                      llvm::StringRef);
    CARBON_DIAGNOSTIC(PreviousDefinition, Note, "Previous definition is here.");
    auto prev_def_id = name_lookup_[name_id].back();
    auto prev_def = semantics_->GetNode(prev_def_id);

    emitter_->Build(name_node, NameRedefined, semantics_->GetString(name_id))
        .Note(prev_def.parse_node(), PreviousDefinition)
        .Emit();
  }
}

auto SemanticsParseTreeHandler::BindName(ParseTree::Node name_node,
                                         SemanticsNodeId type_id,
                                         SemanticsNodeId target_id)
    -> SemanticsStringId {
  CARBON_CHECK(parse_tree_->node_kind(name_node) == ParseNodeKind::DeclaredName)
      << parse_tree_->node_kind(name_node);
  auto name_str = parse_tree_->GetNodeText(name_node);
  auto name_id = semantics_->AddString(name_str);

  AddNode(
      SemanticsNode::BindName::Make(name_node, type_id, name_id, target_id));
  AddNameToLookup(name_node, name_id, target_id);
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

auto SemanticsParseTreeHandler::ImplicitAsForArgs(
    SemanticsNodeBlockId /*arg_ir_id*/, SemanticsNodeBlockId arg_refs_id,
    ParseTree::Node param_parse_node, SemanticsNodeBlockId param_refs_id,
    DiagnosticEmitter<ParseTree::Node>::DiagnosticBuilder* diagnostic) -> bool {
  // If both arguments and parameters are empty, return quickly. Otherwise,
  // we'll fetch both so that errors are consistent.
  if (arg_refs_id == SemanticsNodeBlockId::Empty &&
      param_refs_id == SemanticsNodeBlockId::Empty) {
    return true;
  }

  auto arg_refs = semantics_->GetNodeBlock(arg_refs_id);
  auto param_refs = semantics_->GetNodeBlock(param_refs_id);

  // If sizes mismatch, fail early.
  if (arg_refs.size() != param_refs.size()) {
    CARBON_CHECK(diagnostic != nullptr) << "Should have validated first";
    CARBON_DIAGNOSTIC(CallArgCountMismatch, Note,
                      "Callable cannot be used: Received {0} argument(s), but "
                      "require {1} argument(s).",
                      int, int);
    diagnostic->Note(param_parse_node, CallArgCountMismatch, arg_refs.size(),
                     param_refs.size());
    return false;
  }

  // Check type conversions per-element.
  // TODO: arg_ir_id is passed so that implicit conversions can be inserted.
  // It's currently not supported, but will be needed.
  for (size_t i = 0; i < arg_refs.size(); ++i) {
    auto value_id = arg_refs[i];
    auto as_type_id = semantics_->GetNode(param_refs[i]).type_id();
    if (ImplicitAsImpl(value_id, as_type_id,
                       diagnostic == nullptr ? &value_id : nullptr) ==
        ImplicitAsKind::Incompatible) {
      CARBON_CHECK(diagnostic != nullptr) << "Should have validated first";
      CARBON_DIAGNOSTIC(CallArgTypeMismatch, Note,
                        "Callable cannot be used: Cannot implicityly convert "
                        "argument {0} from `{1}` to `{2}`.",
                        size_t, std::string, std::string);
      diagnostic->Note(
          param_parse_node, CallArgTypeMismatch, i,
          semantics_->StringifyNode(semantics_->GetNode(value_id).type_id()),
          semantics_->StringifyNode(as_type_id));
      return false;
    }
  }

  return true;
}

auto SemanticsParseTreeHandler::ImplicitAsRequired(ParseTree::Node parse_node,
                                                   SemanticsNodeId value_id,
                                                   SemanticsNodeId as_type_id)
    -> SemanticsNodeId {
  SemanticsNodeId output_value_id = value_id;
  if (ImplicitAsImpl(value_id, as_type_id, &output_value_id) ==
      ImplicitAsKind::Incompatible) {
    // Only error when the system is trying to use the result.
    CARBON_DIAGNOSTIC(ImplicitAsConversionFailure, Error,
                      "Cannot implicitly convert from `{0}` to `{1}`.",
                      std::string, std::string);
    emitter_
        ->Build(
            parse_node, ImplicitAsConversionFailure,
            semantics_->StringifyNode(semantics_->GetNode(value_id).type_id()),
            semantics_->StringifyNode(as_type_id))
        .Emit();
  }
  return output_value_id;
}

auto SemanticsParseTreeHandler::ImplicitAsImpl(SemanticsNodeId value_id,
                                               SemanticsNodeId as_type_id,
                                               SemanticsNodeId* output_value_id)
    -> ImplicitAsKind {
  // Start by making sure both sides are valid. If any part is invalid, the
  // result is invalid and we shouldn't error.
  if (value_id == SemanticsNodeId::BuiltinInvalidType) {
    // If the value is invalid, we can't do much, but do "succeed".
    return ImplicitAsKind::Identical;
  }
  auto value_type_id = semantics_->GetNode(value_id).type_id();
  if (value_type_id == SemanticsNodeId::BuiltinInvalidType) {
    return ImplicitAsKind::Identical;
  }
  if (as_type_id == SemanticsNodeId::BuiltinInvalidType) {
    // Although the target type is invalid, this still changes the value.
    if (output_value_id != nullptr) {
      *output_value_id = SemanticsNodeId::BuiltinInvalidType;
    }
    return ImplicitAsKind::Compatible;
  }

  if (value_type_id == as_type_id) {
    // Type doesn't need to change.
    return ImplicitAsKind::Identical;
  }

  // When converting to a Type, there are some automatic conversions that can be
  // done.
  if (as_type_id == SemanticsNodeId::BuiltinTypeType) {
    if (value_id == SemanticsNodeId::BuiltinEmptyTuple) {
      if (output_value_id != nullptr) {
        *output_value_id = SemanticsNodeId::BuiltinEmptyTupleType;
      }
      return ImplicitAsKind::Compatible;
    }
    if (value_id == SemanticsNodeId::BuiltinEmptyStruct) {
      if (output_value_id != nullptr) {
        *output_value_id = SemanticsNodeId::BuiltinEmptyStructType;
      }
      return ImplicitAsKind::Compatible;
    }
  }

  auto value_type = semantics_->GetNode(value_type_id);
  auto as_type = semantics_->GetNode(as_type_id);
  if (CanImplicitAsStruct(value_type, as_type)) {
    // Under the current implementation, struct types are only allowed to
    // ImplicitAs when they're equivalent. What's really missing is type
    // consolidation such that this would fall under the above `value_type_id ==
    // as_type_id` case. In the future, this will need to handle actual
    // conversions.
    return ImplicitAsKind::Identical;
  }

  if (output_value_id != nullptr) {
    *output_value_id = SemanticsNodeId::BuiltinInvalidType;
  }
  return ImplicitAsKind::Incompatible;
}

auto SemanticsParseTreeHandler::CanImplicitAsStruct(SemanticsNode value_type,
                                                    SemanticsNode as_type)
    -> bool {
  if (value_type.kind() != SemanticsNodeKind::StructType ||
      as_type.kind() != SemanticsNodeKind::StructType) {
    return false;
  }
  auto value_type_refs =
      semantics_->GetNodeBlock(value_type.GetAsStructType().second);
  auto as_type_refs =
      semantics_->GetNodeBlock(as_type.GetAsStructType().second);
  if (value_type_refs.size() != as_type_refs.size()) {
    return false;
  }

  for (int i = 0; i < static_cast<int>(value_type_refs.size()); ++i) {
    auto value_type_field = semantics_->GetNode(value_type_refs[i]);
    auto as_type_field = semantics_->GetNode(as_type_refs[i]);
    if (value_type_field.type_id() != as_type_field.type_id() ||
        value_type_field.GetAsStructTypeField() !=
            as_type_field.GetAsStructTypeField()) {
      return false;
    }
  }
  return true;
}

auto SemanticsParseTreeHandler::ParamOrArgStart() -> void {
  params_or_args_stack_.Push();
  node_block_stack_.Push();
}

auto SemanticsParseTreeHandler::ParamOrArgComma(bool for_args) -> void {
  ParamOrArgSave(for_args);
}

auto SemanticsParseTreeHandler::ParamOrArgEnd(bool for_args,
                                              ParseNodeKind start_kind)
    -> std::pair<SemanticsNodeBlockId, SemanticsNodeBlockId> {
  if (parse_tree_->node_kind(node_stack_.PeekParseNode()) != start_kind) {
    ParamOrArgSave(for_args);
  }
  return {node_block_stack_.Pop(), params_or_args_stack_.Pop()};
}

auto SemanticsParseTreeHandler::ParamOrArgSave(bool for_args) -> void {
  SemanticsNodeId param_or_arg_id = SemanticsNodeId::Invalid;
  if (for_args) {
    // For an argument, we add a stub reference to the expression on the top of
    // the stack. There may not be anything on the IR prior to this.
    auto [entry_parse_node, entry_node_id] =
        node_stack_.PopForParseNodeAndNodeId();
    param_or_arg_id = AddNode(SemanticsNode::StubReference::Make(
        entry_parse_node, semantics_->GetNode(entry_node_id).type_id(),
        entry_node_id));
  } else {
    // For a parameter, there should always be something in the IR.
    node_stack_.PopAndIgnore();
    auto ir_id = node_block_stack_.Peek();
    CARBON_CHECK(ir_id.is_valid());
    auto& ir = semantics_->GetNodeBlock(ir_id);
    CARBON_CHECK(!ir.empty()) << "Should have had a param";
    param_or_arg_id = ir.back();
  }

  // Save the param or arg ID.
  auto& params_or_args =
      semantics_->GetNodeBlock(params_or_args_stack_.PeekForAdd());
  params_or_args.push_back(param_or_arg_id);
}

auto SemanticsParseTreeHandler::HandleAddress(ParseTree::Node parse_node)
    -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleAddress");
  return false;
}

auto SemanticsParseTreeHandler::HandleBreakStatement(ParseTree::Node parse_node)
    -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleBreakStatement");
  return false;
}

auto SemanticsParseTreeHandler::HandleBreakStatementStart(
    ParseTree::Node parse_node) -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleBreakStatementStart");
  return false;
}

auto SemanticsParseTreeHandler::HandleCallExpression(ParseTree::Node parse_node)
    -> bool {
  auto [ir_id, refs_id] =
      ParamOrArgEnd(/*for_args=*/true, ParseNodeKind::CallExpressionStart);

  // TODO: Convert to call expression.
  auto [call_expr_parse_node, name_id] =
      node_stack_.PopForParseNodeAndNodeId(ParseNodeKind::CallExpressionStart);
  auto name_node = semantics_->GetNode(name_id);
  if (name_node.kind() != SemanticsNodeKind::FunctionDeclaration) {
    // TODO: Work on error.
    emitter_->Emit(parse_node, SemanticsTodo, "Not a callable name");
    node_stack_.Push(parse_node, name_id);
    return true;
  }

  auto [_, callable_id] = name_node.GetAsFunctionDeclaration();
  auto callable = semantics_->GetCallable(callable_id);

  CARBON_DIAGNOSTIC(NoMatchingCall, Error, "No matching callable was found.");
  auto diagnostic = emitter_->Build(call_expr_parse_node, NoMatchingCall);
  if (!ImplicitAsForArgs(ir_id, refs_id, name_node.parse_node(),
                         callable.param_refs_id, &diagnostic)) {
    diagnostic.Emit();
    node_stack_.Push(parse_node, SemanticsNodeId::BuiltinInvalidType);
    return true;
  }

  CARBON_CHECK(ImplicitAsForArgs(ir_id, refs_id, name_node.parse_node(),
                                 callable.param_refs_id,
                                 /*diagnostic=*/nullptr));

  auto call_id = semantics_->AddCall({ir_id, refs_id});
  // TODO: Propagate return types from callable.
  auto call_node_id = AddNode(SemanticsNode::Call::Make(
      call_expr_parse_node, callable.return_type_id, call_id, callable_id));

  node_stack_.Push(parse_node, call_node_id);
  return true;
}

auto SemanticsParseTreeHandler::HandleCallExpressionComma(
    ParseTree::Node /*parse_node*/) -> bool {
  ParamOrArgComma(/*for_args=*/true);
  return true;
}

auto SemanticsParseTreeHandler::HandleCallExpressionStart(
    ParseTree::Node parse_node) -> bool {
  auto name_id = node_stack_.PopForNodeId(ParseNodeKind::NameReference);
  node_stack_.Push(parse_node, name_id);
  ParamOrArgStart();
  return true;
}

auto SemanticsParseTreeHandler::HandleClassDeclaration(
    ParseTree::Node parse_node) -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleClassDeclaration");
  return false;
}

auto SemanticsParseTreeHandler::HandleClassDefinition(
    ParseTree::Node parse_node) -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleClassDefinition");
  return false;
}

auto SemanticsParseTreeHandler::HandleClassDefinitionStart(
    ParseTree::Node parse_node) -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleClassDefinitionStart");
  return false;
}

auto SemanticsParseTreeHandler::HandleClassIntroducer(
    ParseTree::Node parse_node) -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleClassIntroducer");
  return false;
}

auto SemanticsParseTreeHandler::HandleCodeBlock(ParseTree::Node parse_node)
    -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleCodeBlock");
  return false;
}

auto SemanticsParseTreeHandler::HandleCodeBlockStart(ParseTree::Node parse_node)
    -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleCodeBlockStart");
  return false;
}

auto SemanticsParseTreeHandler::HandleContinueStatement(
    ParseTree::Node parse_node) -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleContinueStatement");
  return false;
}

auto SemanticsParseTreeHandler::HandleContinueStatementStart(
    ParseTree::Node parse_node) -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleContinueStatementStart");
  return false;
}

auto SemanticsParseTreeHandler::HandleDeclaredName(ParseTree::Node parse_node)
    -> bool {
  // The parent is responsible for binding the name.
  node_stack_.Push(parse_node);
  return true;
}

auto SemanticsParseTreeHandler::HandleDeducedParameterList(
    ParseTree::Node parse_node) -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleDeducedParameterList");
  return false;
}

auto SemanticsParseTreeHandler::HandleDeducedParameterListStart(
    ParseTree::Node parse_node) -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleDeducedParameterListStart");
  return false;
}

auto SemanticsParseTreeHandler::HandleDesignatedName(ParseTree::Node parse_node)
    -> bool {
  auto name_str = parse_tree_->GetNodeText(parse_node);
  auto name_id = semantics_->AddString(name_str);
  // The parent is responsible for binding the name.
  node_stack_.Push(parse_node, name_id);
  return true;
}

auto SemanticsParseTreeHandler::HandleDesignatorExpression(
    ParseTree::Node parse_node) -> bool {
  auto [_, name_id] =
      node_stack_.PopForParseNodeAndNameId(ParseNodeKind::DesignatedName);

  auto base_id = node_stack_.PopForNodeId();
  auto base = semantics_->GetNode(base_id);
  auto base_type = semantics_->GetNode(base.type_id());

  switch (base_type.kind()) {
    case SemanticsNodeKind::StructType: {
      auto refs = semantics_->GetNodeBlock(base_type.GetAsStructType().second);
      // TODO: Do we need to optimize this with a lookup table for O(1)?
      for (int i = 0; i < static_cast<int>(refs.size()); ++i) {
        auto ref = semantics_->GetNode(refs[i]);
        if (name_id == ref.GetAsStructTypeField()) {
          AddNodeAndPush(parse_node, SemanticsNode::StructMemberAccess::Make(
                                         parse_node, ref.type_id(), base_id,
                                         SemanticsMemberIndex(i)));
          return true;
        }
      }
      CARBON_DIAGNOSTIC(DesignatorExpressionNameNotFound, Error,
                        "Type `{0}` does not have a member `{1}`.", std::string,
                        llvm::StringRef);
      emitter_->Emit(parse_node, DesignatorExpressionNameNotFound,
                     semantics_->StringifyNode(base.type_id()),
                     semantics_->GetString(name_id));
      break;
    }
    default: {
      CARBON_DIAGNOSTIC(DesignatorExpressionUnsupported, Error,
                        "Type `{0}` does not support designator expressions.",
                        std::string);
      emitter_->Emit(parse_node, DesignatorExpressionUnsupported,
                     semantics_->StringifyNode(base.type_id()));
      break;
    }
  }

  // Should only be reached on error.
  node_stack_.Push(parse_node, SemanticsNodeId::BuiltinInvalidType);
  return true;
}

auto SemanticsParseTreeHandler::HandleEmptyDeclaration(
    ParseTree::Node parse_node) -> bool {
  // Empty declarations have no actions associated, but we still balance the
  // tree.
  node_stack_.Push(parse_node);
  return true;
}

auto SemanticsParseTreeHandler::HandleExpressionStatement(
    ParseTree::Node parse_node) -> bool {
  // Pop the expression without investigating its contents.
  // TODO: This will probably eventually need to do some "do not discard"
  // analysis.
  node_stack_.PopAndDiscardId();
  node_stack_.Push(parse_node);
  return true;
}

auto SemanticsParseTreeHandler::HandleFileEnd(ParseTree::Node /*parse_node*/)
    -> bool {
  // Do nothing, no need to balance this node.
  return true;
}

auto SemanticsParseTreeHandler::HandleForHeader(ParseTree::Node parse_node)
    -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleForHeader");
  return false;
}

auto SemanticsParseTreeHandler::HandleForHeaderStart(ParseTree::Node parse_node)
    -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleForHeaderStart");
  return false;
}

auto SemanticsParseTreeHandler::HandleForIn(ParseTree::Node parse_node)
    -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleForIn");
  return false;
}

auto SemanticsParseTreeHandler::HandleForStatement(ParseTree::Node parse_node)
    -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleForStatement");
  return false;
}

auto SemanticsParseTreeHandler::HandleFunctionDeclaration(
    ParseTree::Node parse_node) -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleFunctionDeclaration");
  return false;
}

auto SemanticsParseTreeHandler::HandleFunctionDefinition(
    ParseTree::Node parse_node) -> bool {
  // Merges code block children up under the FunctionDefinitionStart.
  while (parse_tree_->node_kind(node_stack_.PeekParseNode()) !=
         ParseNodeKind::FunctionDefinitionStart) {
    node_stack_.PopAndIgnore();
  }
  auto decl_id =
      node_stack_.PopForNodeId(ParseNodeKind::FunctionDefinitionStart);

  return_scope_stack_.pop_back();
  PopScope();
  auto block_id = node_block_stack_.Pop();
  AddNode(
      SemanticsNode::FunctionDefinition::Make(parse_node, decl_id, block_id));
  node_stack_.Push(parse_node);

  return true;
}

auto SemanticsParseTreeHandler::HandleFunctionDefinitionStart(
    ParseTree::Node parse_node) -> bool {
  SemanticsNodeId return_type_id = SemanticsNodeId::Invalid;
  if (parse_tree_->node_kind(node_stack_.PeekParseNode()) ==
      ParseNodeKind::ReturnType) {
    return_type_id = node_stack_.PopForNodeId(ParseNodeKind::ReturnType);
  }
  node_stack_.PopForSoloParseNode(ParseNodeKind::ParameterList);
  auto [param_ir_id, param_refs_id] = finished_params_stack_.pop_back_val();
  auto name_node = node_stack_.PopForSoloParseNode(ParseNodeKind::DeclaredName);
  auto fn_node =
      node_stack_.PopForSoloParseNode(ParseNodeKind::FunctionIntroducer);

  auto name_str = parse_tree_->GetNodeText(name_node);
  auto name_id = semantics_->AddString(name_str);

  auto callable_id =
      semantics_->AddCallable({.param_ir_id = param_ir_id,
                               .param_refs_id = param_refs_id,
                               .return_type_id = return_type_id});
  auto decl_id = AddNode(
      SemanticsNode::FunctionDeclaration::Make(fn_node, name_id, callable_id));
  AddNameToLookup(name_node, name_id, decl_id);

  node_block_stack_.Push();
  PushScope();
  return_scope_stack_.push_back(decl_id);
  node_stack_.Push(parse_node, decl_id);

  return true;
}

auto SemanticsParseTreeHandler::HandleFunctionIntroducer(
    ParseTree::Node parse_node) -> bool {
  // No action, just a bracketing node.
  node_stack_.Push(parse_node);
  return true;
}

auto SemanticsParseTreeHandler::HandleGenericPatternBinding(
    ParseTree::Node parse_node) -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "GenericPatternBinding");
  return false;
}

auto SemanticsParseTreeHandler::HandleIfCondition(ParseTree::Node parse_node)
    -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleIfCondition");
  return false;
}

auto SemanticsParseTreeHandler::HandleIfConditionStart(
    ParseTree::Node parse_node) -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleIfConditionStart");
  return false;
}

auto SemanticsParseTreeHandler::HandleIfStatement(ParseTree::Node parse_node)
    -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleIfStatement");
  return false;
}

auto SemanticsParseTreeHandler::HandleIfStatementElse(
    ParseTree::Node parse_node) -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleIfStatementElse");
  return false;
}

auto SemanticsParseTreeHandler::HandleInfixOperator(ParseTree::Node parse_node)
    -> bool {
  auto rhs_id = node_stack_.PopForNodeId();
  auto lhs_id = node_stack_.PopForNodeId();

  // TODO: This should search for a compatible interface. For now, it's a very
  // trivial check of validity on the operation.
  lhs_id = ImplicitAsRequired(parse_node, lhs_id,
                              semantics_->GetNode(rhs_id).type_id());

  // Figure out the operator for the token.
  auto token = parse_tree_->node_token(parse_node);
  switch (auto token_kind = tokens_->GetKind(token)) {
    case TokenKind::Plus:
      AddNodeAndPush(parse_node,
                     SemanticsNode::BinaryOperatorAdd::Make(
                         parse_node, semantics_->GetNode(lhs_id).type_id(),
                         lhs_id, rhs_id));
      break;
    default:
      emitter_->Emit(parse_node, SemanticsTodo,
                     llvm::formatv("Handle {0}", token_kind));
      return false;
  }

  return true;
}

auto SemanticsParseTreeHandler::HandleInterfaceDeclaration(
    ParseTree::Node parse_node) -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleInterfaceDeclaration");
  return false;
}

auto SemanticsParseTreeHandler::HandleInterfaceDefinition(
    ParseTree::Node parse_node) -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleInterfaceDefinition");
  return false;
}

auto SemanticsParseTreeHandler::HandleInterfaceDefinitionStart(
    ParseTree::Node parse_node) -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleInterfaceDefinitionStart");
  return false;
}

auto SemanticsParseTreeHandler::HandleInterfaceIntroducer(
    ParseTree::Node parse_node) -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleInterfaceIntroducer");
  return false;
}

auto SemanticsParseTreeHandler::HandleInvalidParse(ParseTree::Node parse_node)
    -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleInvalidParse");
  return false;
}

auto SemanticsParseTreeHandler::HandleLiteral(ParseTree::Node parse_node)
    -> bool {
  auto token = parse_tree_->node_token(parse_node);
  switch (auto token_kind = tokens_->GetKind(token)) {
    case TokenKind::IntegerLiteral: {
      auto id =
          semantics_->AddIntegerLiteral(tokens_->GetIntegerLiteral(token));
      AddNodeAndPush(parse_node,
                     SemanticsNode::IntegerLiteral::Make(parse_node, id));
      break;
    }
    case TokenKind::RealLiteral: {
      auto token_value = tokens_->GetRealLiteral(token);
      auto id =
          semantics_->AddRealLiteral({.mantissa = token_value.Mantissa(),
                                      .exponent = token_value.Exponent(),
                                      .is_decimal = token_value.IsDecimal()});
      AddNodeAndPush(parse_node,
                     SemanticsNode::RealLiteral::Make(parse_node, id));
      break;
    }
    case TokenKind::StringLiteral: {
      auto id = semantics_->AddString(tokens_->GetStringLiteral(token));
      AddNodeAndPush(parse_node,
                     SemanticsNode::StringLiteral::Make(parse_node, id));
      break;
    }
    case TokenKind::IntegerTypeLiteral: {
      auto text = tokens_->GetTokenText(token);
      if (text != "i32") {
        emitter_->Emit(parse_node, SemanticsTodo,
                       "Currently only i32 is allowed");
        return false;
      }
      node_stack_.Push(parse_node, SemanticsNodeId::BuiltinIntegerType);
      break;
    }
    case TokenKind::FloatingPointTypeLiteral: {
      auto text = tokens_->GetTokenText(token);
      if (text != "f64") {
        emitter_->Emit(parse_node, SemanticsTodo,
                       "Currently only f64 is allowed");
        return false;
      }
      node_stack_.Push(parse_node, SemanticsNodeId::BuiltinFloatingPointType);
      break;
    }
    case TokenKind::StringTypeLiteral: {
      node_stack_.Push(parse_node, SemanticsNodeId::BuiltinStringType);
      break;
    }
    default: {
      emitter_->Emit(parse_node, SemanticsTodo,
                     llvm::formatv("Handle {0}", token_kind));
      return false;
    }
  }

  return true;
}

auto SemanticsParseTreeHandler::HandleNameReference(ParseTree::Node parse_node)
    -> bool {
  auto name_str = parse_tree_->GetNodeText(parse_node);

  auto name_not_found = [&] {
    CARBON_DIAGNOSTIC(NameNotFound, Error, "Name {0} not found",
                      llvm::StringRef);
    emitter_->Emit(parse_node, NameNotFound, name_str);
    node_stack_.Push(parse_node, SemanticsNodeId::BuiltinInvalidType);
  };

  auto name_id = semantics_->GetStringID(name_str);
  if (!name_id) {
    name_not_found();
    return true;
  }

  auto it = name_lookup_.find(*name_id);
  if (it == name_lookup_.end()) {
    name_not_found();
    return true;
  }
  CARBON_CHECK(!it->second.empty()) << "Should have been erased: " << name_str;

  // TODO: Check for ambiguous lookups.
  node_stack_.Push(parse_node, it->second.back());

  return true;
}

auto SemanticsParseTreeHandler::HandleNamedConstraintDeclaration(
    ParseTree::Node parse_node) -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleNamedConstraintDeclaration");
  return false;
}

auto SemanticsParseTreeHandler::HandleNamedConstraintDefinition(
    ParseTree::Node parse_node) -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleNamedConstraintDefinition");
  return false;
}

auto SemanticsParseTreeHandler::HandleNamedConstraintDefinitionStart(
    ParseTree::Node parse_node) -> bool {
  emitter_->Emit(parse_node, SemanticsTodo,
                 "HandleNamedConstraintDefinitionStart");
  return false;
}

auto SemanticsParseTreeHandler::HandleNamedConstraintIntroducer(
    ParseTree::Node parse_node) -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleNamedConstraintIntroducer");
  return false;
}

auto SemanticsParseTreeHandler::HandlePackageApi(ParseTree::Node parse_node)
    -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandlePackageApi");
  return false;
}

auto SemanticsParseTreeHandler::HandlePackageDirective(
    ParseTree::Node parse_node) -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandlePackageDirective");
  return false;
}

auto SemanticsParseTreeHandler::HandlePackageImpl(ParseTree::Node parse_node)
    -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandlePackageImpl");
  return false;
}

auto SemanticsParseTreeHandler::HandlePackageIntroducer(
    ParseTree::Node parse_node) -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandlePackageIntroducer");
  return false;
}

auto SemanticsParseTreeHandler::HandlePackageLibrary(ParseTree::Node parse_node)
    -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandlePackageLibrary");
  return false;
}

auto SemanticsParseTreeHandler::HandleParameterList(ParseTree::Node parse_node)
    -> bool {
  auto [ir_id, refs_id] =
      ParamOrArgEnd(/*for_args=*/false, ParseNodeKind::ParameterListStart);

  PopScope();
  node_stack_.PopAndDiscardSoloParseNode(ParseNodeKind::ParameterListStart);
  finished_params_stack_.push_back({ir_id, refs_id});
  node_stack_.Push(parse_node);
  return true;
}

auto SemanticsParseTreeHandler::HandleParameterListComma(
    ParseTree::Node /*parse_node*/) -> bool {
  ParamOrArgComma(/*for_args=*/false);
  return true;
}

auto SemanticsParseTreeHandler::HandleParameterListStart(
    ParseTree::Node parse_node) -> bool {
  PushScope();
  node_stack_.Push(parse_node);
  ParamOrArgStart();
  return true;
}

auto SemanticsParseTreeHandler::HandleParenExpression(
    ParseTree::Node parse_node) -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleParenExpression");
  return false;
}

auto SemanticsParseTreeHandler::HandleParenExpressionOrTupleLiteralStart(
    ParseTree::Node parse_node) -> bool {
  emitter_->Emit(parse_node, SemanticsTodo,
                 "HandleParenExpressionOrTupleLiteralStart");
  return false;
}

auto SemanticsParseTreeHandler::HandlePatternBinding(ParseTree::Node parse_node)
    -> bool {
  auto [type_node, parsed_type_id] = node_stack_.PopForParseNodeAndNodeId();
  SemanticsNodeId cast_type_id = ImplicitAsRequired(
      type_node, parsed_type_id, SemanticsNodeId::BuiltinTypeType);

  // Get the name.
  auto name_node = node_stack_.PopForSoloParseNode();

  // Allocate storage, linked to the name for error locations.
  auto storage_id =
      AddNode(SemanticsNode::VarStorage::Make(name_node, cast_type_id));

  // Bind the name to storage.
  auto name_id = BindName(name_node, cast_type_id, storage_id);

  // If this node's result is used, it'll be for either the name or the
  // storage address. The storage address can be found through the name, so we
  // push the name.
  node_stack_.Push(parse_node, name_id);

  return true;
}

auto SemanticsParseTreeHandler::HandlePostfixOperator(
    ParseTree::Node parse_node) -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandlePostfixOperator");
  return false;
}

auto SemanticsParseTreeHandler::HandlePrefixOperator(ParseTree::Node parse_node)
    -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandlePrefixOperator");
  return false;
}

auto SemanticsParseTreeHandler::HandleReturnStatement(
    ParseTree::Node parse_node) -> bool {
  CARBON_CHECK(!return_scope_stack_.empty());
  const auto& fn_node = semantics_->GetNode(return_scope_stack_.back());
  const auto callable =
      semantics_->GetCallable(fn_node.GetAsFunctionDeclaration().second);

  if (parse_tree_->node_kind(node_stack_.PeekParseNode()) ==
      ParseNodeKind::ReturnStatementStart) {
    node_stack_.PopAndDiscardSoloParseNode(ParseNodeKind::ReturnStatementStart);

    if (callable.return_type_id.is_valid()) {
      // TODO: Add a note pointing at the return type's parse node.
      CARBON_DIAGNOSTIC(ReturnStatementMissingExpression, Error,
                        "Must return a {0}.", std::string);
      emitter_
          ->Build(parse_node, ReturnStatementMissingExpression,
                  semantics_->StringifyNode(callable.return_type_id))
          .Emit();
    }

    AddNodeAndPush(parse_node, SemanticsNode::Return::Make(parse_node));
  } else {
    auto arg = node_stack_.PopForNodeId();
    node_stack_.PopAndDiscardSoloParseNode(ParseNodeKind::ReturnStatementStart);

    if (!callable.return_type_id.is_valid()) {
      CARBON_DIAGNOSTIC(
          ReturnStatementDisallowExpression, Error,
          "No return expression should be provided in this context.");
      CARBON_DIAGNOSTIC(ReturnStatementImplicitNote, Note,
                        "There was no return type provided.");
      emitter_->Build(parse_node, ReturnStatementDisallowExpression)
          .Note(fn_node.parse_node(), ReturnStatementImplicitNote)
          .Emit();
    } else {
      arg = ImplicitAsRequired(parse_node, arg, callable.return_type_id);
    }

    AddNodeAndPush(parse_node,
                   SemanticsNode::ReturnExpression::Make(
                       parse_node, semantics_->GetNode(arg).type_id(), arg));
  }
  return true;
}

auto SemanticsParseTreeHandler::HandleReturnStatementStart(
    ParseTree::Node parse_node) -> bool {
  // No action, just a bracketing node.
  node_stack_.Push(parse_node);
  return true;
}

auto SemanticsParseTreeHandler::HandleReturnType(ParseTree::Node parse_node)
    -> bool {
  // Propagate the type expression.
  auto [type_parse_node, type_node_id] = node_stack_.PopForParseNodeAndNodeId();
  auto cast_node_id = ImplicitAsRequired(type_parse_node, type_node_id,
                                         SemanticsNodeId::BuiltinTypeType);
  node_stack_.Push(parse_node, cast_node_id);
  return true;
}

auto SemanticsParseTreeHandler::HandleSelfTypeIdentifier(
    ParseTree::Node parse_node) -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleSelfTypeIdentifier");
  return false;
}

auto SemanticsParseTreeHandler::HandleSelfValueIdentifier(
    ParseTree::Node parse_node) -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleSelfValueIdentifier");
  return false;
}

auto SemanticsParseTreeHandler::HandleStructComma(
    ParseTree::Node /*parse_node*/) -> bool {
  ParamOrArgComma(
      /*for_args=*/parse_tree_->node_kind(node_stack_.PeekParseNode()) !=
      ParseNodeKind::StructFieldType);
  return true;
}

auto SemanticsParseTreeHandler::HandleStructFieldDesignator(
    ParseTree::Node /*parse_node*/) -> bool {
  // This leaves the designated name on top because the `.` isn't interesting.
  CARBON_CHECK(parse_tree_->node_kind(node_stack_.PeekParseNode()) ==
               ParseNodeKind::DesignatedName);
  return true;
}

auto SemanticsParseTreeHandler::HandleStructFieldType(
    ParseTree::Node parse_node) -> bool {
  auto [type_node, type_id] = node_stack_.PopForParseNodeAndNodeId();
  SemanticsNodeId cast_type_id =
      ImplicitAsRequired(type_node, type_id, SemanticsNodeId::BuiltinTypeType);

  auto [name_node, name_id] =
      node_stack_.PopForParseNodeAndNameId(ParseNodeKind::DesignatedName);

  AddNode(
      SemanticsNode::StructTypeField::Make(name_node, cast_type_id, name_id));
  node_stack_.Push(parse_node);
  return true;
}

auto SemanticsParseTreeHandler::HandleStructFieldUnknown(
    ParseTree::Node parse_node) -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleStructFieldUnknown");
  return false;
}

auto SemanticsParseTreeHandler::HandleStructFieldValue(
    ParseTree::Node parse_node) -> bool {
  auto [value_parse_node, value_node_id] =
      node_stack_.PopForParseNodeAndNodeId();
  auto [_, name_id] =
      node_stack_.PopForParseNodeAndNameId(ParseNodeKind::DesignatedName);

  // Store the name for the type.
  auto type_block_id = args_type_info_stack_.PeekForAdd();
  semantics_->AddNode(
      type_block_id,
      SemanticsNode::StructTypeField::Make(
          parse_node, semantics_->GetNode(value_node_id).type_id(), name_id));

  // Push the value back on the stack as an argument.
  node_stack_.Push(parse_node, value_node_id);
  return true;
}

auto SemanticsParseTreeHandler::HandleStructLiteral(ParseTree::Node parse_node)
    -> bool {
  auto [ir_id, refs_id] = ParamOrArgEnd(
      /*for_args=*/true, ParseNodeKind::StructLiteralOrStructTypeLiteralStart);

  PopScope();
  node_stack_.PopAndDiscardSoloParseNode(
      ParseNodeKind::StructLiteralOrStructTypeLiteralStart);
  auto type_block_id = args_type_info_stack_.Pop();

  // Special-case `{}`.
  if (refs_id == SemanticsNodeBlockId::Empty) {
    node_stack_.Push(parse_node, SemanticsNodeId::BuiltinEmptyStruct);
    return true;
  }

  // Construct a type for the literal. Each field is one node, so ir_id and
  // refs_id match.
  auto refs = semantics_->GetNodeBlock(refs_id);
  auto type_id = AddNode(SemanticsNode::StructType::Make(
      parse_node, type_block_id, type_block_id));

  auto value_id = AddNode(
      SemanticsNode::StructValue::Make(parse_node, type_id, ir_id, refs_id));
  node_stack_.Push(parse_node, value_id);
  return true;
}

auto SemanticsParseTreeHandler::HandleStructLiteralOrStructTypeLiteralStart(
    ParseTree::Node parse_node) -> bool {
  PushScope();
  node_stack_.Push(parse_node);
  // At this point we aren't sure whether this will be a value or type literal,
  // so we push onto args irrespective. It just won't be used for a type
  // literal.
  args_type_info_stack_.Push();
  ParamOrArgStart();
  return true;
}

auto SemanticsParseTreeHandler::HandleStructTypeLiteral(
    ParseTree::Node parse_node) -> bool {
  auto [ir_id, refs_id] = ParamOrArgEnd(
      /*for_args=*/false, ParseNodeKind::StructLiteralOrStructTypeLiteralStart);

  PopScope();
  node_stack_.PopAndDiscardSoloParseNode(
      ParseNodeKind::StructLiteralOrStructTypeLiteralStart);
  // This is only used for value literals.
  args_type_info_stack_.Pop();

  CARBON_CHECK(refs_id != SemanticsNodeBlockId::Empty)
      << "{} is handled by StructLiteral.";

  auto type_id =
      AddNode(SemanticsNode::StructType::Make(parse_node, ir_id, refs_id));
  node_stack_.Push(parse_node, type_id);
  return true;
}

auto SemanticsParseTreeHandler::HandleTemplate(ParseTree::Node parse_node)
    -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleTemplate");
  return false;
}

auto SemanticsParseTreeHandler::HandleTupleLiteral(ParseTree::Node parse_node)
    -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleTupleLiteral");
  return false;
}

auto SemanticsParseTreeHandler::HandleTupleLiteralComma(
    ParseTree::Node parse_node) -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleTupleLiteralComma");
  return false;
}

auto SemanticsParseTreeHandler::HandleVariableDeclaration(
    ParseTree::Node parse_node) -> bool {
  auto [last_parse_node, last_node_id] = node_stack_.PopForParseNodeAndNodeId();

  if (parse_tree_->node_kind(last_parse_node) !=
      ParseNodeKind::PatternBinding) {
    auto storage_id =
        node_stack_.PopForNodeId(ParseNodeKind::VariableInitializer);

    auto binding =
        node_stack_.PopForParseNodeAndNameId(ParseNodeKind::PatternBinding);

    // Restore the name now that the initializer is complete.
    ReaddNameToLookup(binding.second, storage_id);

    auto cast_value_id = ImplicitAsRequired(
        parse_node, last_node_id, semantics_->GetNode(storage_id).type_id());
    AddNode(SemanticsNode::Assign::Make(
        parse_node, semantics_->GetNode(cast_value_id).type_id(), storage_id,
        cast_value_id));
  }

  node_stack_.PopAndDiscardSoloParseNode(ParseNodeKind::VariableIntroducer);
  node_stack_.Push(parse_node);

  return true;
}

auto SemanticsParseTreeHandler::HandleVariableIntroducer(
    ParseTree::Node parse_node) -> bool {
  // No action, just a bracketing node.
  node_stack_.Push(parse_node);
  return true;
}

auto SemanticsParseTreeHandler::HandleVariableInitializer(
    ParseTree::Node parse_node) -> bool {
  // Temporarily remove name lookup entries added by the `var`. These will be
  // restored by `VariableDeclaration`.

  // Save the storage ID.
  auto it = name_lookup_.find(
      node_stack_.PeekForNameId(ParseNodeKind::PatternBinding));
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

  return true;
}

auto SemanticsParseTreeHandler::HandleWhileCondition(ParseTree::Node parse_node)
    -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleWhileCondition");
  return false;
}

auto SemanticsParseTreeHandler::HandleWhileConditionStart(
    ParseTree::Node parse_node) -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleWhileConditionStart");
  return false;
}

auto SemanticsParseTreeHandler::HandleWhileStatement(ParseTree::Node parse_node)
    -> bool {
  emitter_->Emit(parse_node, SemanticsTodo, "HandleWhileStatement");
  return false;
}

}  // namespace Carbon
