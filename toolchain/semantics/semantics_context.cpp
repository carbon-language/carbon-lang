// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_context.h"

#include <utility>

#include "common/check.h"
#include "common/vlog.h"
#include "llvm/ADT/STLExtras.h"
#include "toolchain/diagnostics/diagnostic_kind.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"
#include "toolchain/semantics/semantics_declaration_name_stack.h"
#include "toolchain/semantics/semantics_ir.h"
#include "toolchain/semantics/semantics_node.h"
#include "toolchain/semantics/semantics_node_block_stack.h"
#include "toolchain/semantics/semantics_node_kind.h"

namespace Carbon::Check {

Context::Context(const Lex::TokenizedBuffer& tokens,
                 DiagnosticEmitter<Parse::Node>& emitter,
                 const Parse::Tree& parse_tree, SemIR::File& semantics_ir,
                 llvm::raw_ostream* vlog_stream)
    : tokens_(&tokens),
      emitter_(&emitter),
      parse_tree_(&parse_tree),
      semantics_ir_(&semantics_ir),
      vlog_stream_(vlog_stream),
      node_stack_(parse_tree, vlog_stream),
      node_block_stack_("node_block_stack_", semantics_ir, vlog_stream),
      params_or_args_stack_("params_or_args_stack_", semantics_ir, vlog_stream),
      args_type_info_stack_("args_type_info_stack_", semantics_ir, vlog_stream),
      declaration_name_stack_(this) {
  // Inserts the "Error" and "Type" types as "used types" so that
  // canonicalization can skip them. We don't emit either for lowering.
  canonical_types_.insert({SemIR::NodeId::BuiltinError, SemIR::TypeId::Error});
  canonical_types_.insert(
      {SemIR::NodeId::BuiltinTypeType, SemIR::TypeId::TypeType});
}

auto Context::TODO(Parse::Node parse_node, std::string label) -> bool {
  CARBON_DIAGNOSTIC(SemanticsTodo, Error, "Semantics TODO: `{0}`.",
                    std::string);
  emitter_->Emit(parse_node, SemanticsTodo, std::move(label));
  return false;
}

auto Context::VerifyOnFinish() -> void {
  // Information in all the various context objects should be cleaned up as
  // various pieces of context go out of scope. At this point, nothing should
  // remain.
  // node_stack_ will still contain top-level entities.
  CARBON_CHECK(name_lookup_.empty()) << name_lookup_.size();
  CARBON_CHECK(scope_stack_.empty()) << scope_stack_.size();
  CARBON_CHECK(node_block_stack_.empty()) << node_block_stack_.size();
  CARBON_CHECK(params_or_args_stack_.empty()) << params_or_args_stack_.size();
}

auto Context::AddNode(SemIR::Node node) -> SemIR::NodeId {
  return AddNodeToBlock(node_block_stack_.PeekForAdd(), node);
}

auto Context::AddNodeToBlock(SemIR::NodeBlockId block, SemIR::Node node)
    -> SemIR::NodeId {
  CARBON_VLOG() << "AddNode " << block << ": " << node << "\n";
  return semantics_ir_->AddNode(block, node);
}

auto Context::AddNodeAndPush(Parse::Node parse_node, SemIR::Node node) -> void {
  auto node_id = AddNode(node);
  node_stack_.Push(parse_node, node_id);
}

auto Context::DiagnoseDuplicateName(Parse::Node parse_node,
                                    SemIR::NodeId prev_def_id) -> void {
  CARBON_DIAGNOSTIC(NameDeclarationDuplicate, Error,
                    "Duplicate name being declared in the same scope.");
  CARBON_DIAGNOSTIC(NameDeclarationPrevious, Note,
                    "Name is previously declared here.");
  auto prev_def = semantics_ir_->GetNode(prev_def_id);
  emitter_->Build(parse_node, NameDeclarationDuplicate)
      .Note(prev_def.parse_node(), NameDeclarationPrevious)
      .Emit();
}

auto Context::DiagnoseNameNotFound(Parse::Node parse_node,
                                   SemIR::StringId name_id) -> void {
  CARBON_DIAGNOSTIC(NameNotFound, Error, "Name `{0}` not found.",
                    llvm::StringRef);
  emitter_->Emit(parse_node, NameNotFound, semantics_ir_->GetString(name_id));
}

auto Context::AddNameToLookup(Parse::Node name_node, SemIR::StringId name_id,
                              SemIR::NodeId target_id) -> void {
  if (current_scope().names.insert(name_id).second) {
    name_lookup_[name_id].push_back(target_id);
  } else {
    DiagnoseDuplicateName(name_node, name_lookup_[name_id].back());
  }
}

auto Context::LookupName(Parse::Node parse_node, SemIR::StringId name_id,
                         SemIR::NameScopeId scope_id, bool print_diagnostics)
    -> SemIR::NodeId {
  if (scope_id == SemIR::NameScopeId::Invalid) {
    auto it = name_lookup_.find(name_id);
    if (it == name_lookup_.end()) {
      if (print_diagnostics) {
        DiagnoseNameNotFound(parse_node, name_id);
      }
      return SemIR::NodeId::BuiltinError;
    }
    CARBON_CHECK(!it->second.empty())
        << "Should have been erased: " << semantics_ir_->GetString(name_id);

    // TODO: Check for ambiguous lookups.
    return it->second.back();
  } else {
    const auto& scope = semantics_ir_->GetNameScope(scope_id);
    auto it = scope.find(name_id);
    if (it == scope.end()) {
      if (print_diagnostics) {
        DiagnoseNameNotFound(parse_node, name_id);
      }
      return SemIR::NodeId::BuiltinError;
    }

    return it->second;
  }
}

auto Context::PushScope() -> void { scope_stack_.push_back({}); }

auto Context::PopScope() -> void {
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

template <typename BranchNode, typename... Args>
static auto AddDominatedBlockAndBranchImpl(Context& context,
                                           Parse::Node parse_node, Args... args)
    -> SemIR::NodeBlockId {
  if (!context.node_block_stack().is_current_block_reachable()) {
    return SemIR::NodeBlockId::Unreachable;
  }
  auto block_id = context.semantics_ir().AddNodeBlock();
  context.AddNode(BranchNode::Make(parse_node, block_id, args...));
  return block_id;
}

auto Context::AddDominatedBlockAndBranch(Parse::Node parse_node)
    -> SemIR::NodeBlockId {
  return AddDominatedBlockAndBranchImpl<SemIR::Node::Branch>(*this, parse_node);
}

auto Context::AddDominatedBlockAndBranchWithArg(Parse::Node parse_node,
                                                SemIR::NodeId arg_id)
    -> SemIR::NodeBlockId {
  return AddDominatedBlockAndBranchImpl<SemIR::Node::BranchWithArg>(
      *this, parse_node, arg_id);
}

auto Context::AddDominatedBlockAndBranchIf(Parse::Node parse_node,
                                           SemIR::NodeId cond_id)
    -> SemIR::NodeBlockId {
  return AddDominatedBlockAndBranchImpl<SemIR::Node::BranchIf>(
      *this, parse_node, cond_id);
}

auto Context::AddConvergenceBlockAndPush(
    Parse::Node parse_node, std::initializer_list<SemIR::NodeBlockId> blocks)
    -> void {
  CARBON_CHECK(blocks.size() >= 2) << "no convergence";

  SemIR::NodeBlockId new_block_id = SemIR::NodeBlockId::Unreachable;
  for (SemIR::NodeBlockId block_id : blocks) {
    if (block_id != SemIR::NodeBlockId::Unreachable) {
      if (new_block_id == SemIR::NodeBlockId::Unreachable) {
        new_block_id = semantics_ir().AddNodeBlock();
      }
      AddNodeToBlock(block_id,
                     SemIR::Node::Branch::Make(parse_node, new_block_id));
    }
  }
  node_block_stack().Push(new_block_id);
}

auto Context::AddConvergenceBlockWithArgAndPush(
    Parse::Node parse_node,
    std::initializer_list<std::pair<SemIR::NodeBlockId, SemIR::NodeId>>
        blocks_and_args) -> SemIR::NodeId {
  CARBON_CHECK(blocks_and_args.size() >= 2) << "no convergence";

  SemIR::NodeBlockId new_block_id = SemIR::NodeBlockId::Unreachable;
  for (auto [block_id, arg_id] : blocks_and_args) {
    if (block_id != SemIR::NodeBlockId::Unreachable) {
      if (new_block_id == SemIR::NodeBlockId::Unreachable) {
        new_block_id = semantics_ir().AddNodeBlock();
      }
      AddNodeToBlock(block_id, SemIR::Node::BranchWithArg::Make(
                                   parse_node, new_block_id, arg_id));
    }
  }
  node_block_stack().Push(new_block_id);

  // Acquire the result value.
  SemIR::TypeId result_type_id =
      semantics_ir().GetNode(blocks_and_args.begin()->second).type_id();
  return AddNode(
      SemIR::Node::BlockArg::Make(parse_node, result_type_id, new_block_id));
}

// Add the current code block to the enclosing function.
auto Context::AddCurrentCodeBlockToFunction() -> void {
  CARBON_CHECK(!node_block_stack().empty()) << "no current code block";
  CARBON_CHECK(!return_scope_stack().empty()) << "no current function";

  if (!node_block_stack().is_current_block_reachable()) {
    // Don't include unreachable blocks in the function.
    return;
  }

  auto function_id = semantics_ir()
                         .GetNode(return_scope_stack().back())
                         .GetAsFunctionDeclaration();
  semantics_ir()
      .GetFunction(function_id)
      .body_block_ids.push_back(node_block_stack().PeekForAdd());
}

auto Context::is_current_position_reachable() -> bool {
  switch (auto block_id = node_block_stack().Peek(); block_id.index) {
    case SemIR::NodeBlockId::Unreachable.index: {
      return false;
    }
    case SemIR::NodeBlockId::Invalid.index: {
      return true;
    }
    default: {
      // Our current position is at the end of a real block. That position is
      // reachable unless the previous instruction is a terminator instruction.
      const auto& block_contents = semantics_ir().GetNodeBlock(block_id);
      if (block_contents.empty()) {
        return true;
      }
      const auto& last_node = semantics_ir().GetNode(block_contents.back());
      return last_node.kind().terminator_kind() !=
             SemIR::TerminatorKind::Terminator;
    }
  }
}

auto Context::Initialize(Parse::Node parse_node, SemIR::NodeId target_id,
                         SemIR::NodeId value_id) -> SemIR::NodeId {
  // Implicitly convert the value to the type of the target.
  auto type_id = semantics_ir().GetNode(target_id).type_id();
  auto expr_id = ImplicitAsRequired(parse_node, value_id, type_id);
  SemIR::Node expr = semantics_ir().GetNode(expr_id);

  // Perform initialization now that we have an expression of the right type.
  switch (SemIR::GetExpressionCategory(semantics_ir(), expr_id)) {
    case SemIR::ExpressionCategory::NotExpression:
      CARBON_FATAL() << "Converting non-expression node " << expr
                     << " to initializing expression";

    case SemIR::ExpressionCategory::DurableReference:
    case SemIR::ExpressionCategory::EphemeralReference:
      // The design uses a custom "copy initialization" process here. We model
      // that as value binding followed by direct initialization.
      //
      // TODO: Determine whether this is observably different from the design,
      // and change either the toolchain or the design so they match.
      return AddNode(SemIR::Node::BindValue::Make(expr.parse_node(),
                                                  expr.type_id(), expr_id));

    case SemIR::ExpressionCategory::Value:
      // TODO: For class types, use an interface to determine how to perform
      // this operation.
      return expr_id;

    case SemIR::ExpressionCategory::Initializing:
      MarkInitializerFor(expr_id, target_id);
      return expr_id;
  }
}

auto Context::ConvertToValueExpression(SemIR::NodeId expr_id) -> SemIR::NodeId {
  switch (SemIR::GetExpressionCategory(semantics_ir(), expr_id)) {
    case SemIR::ExpressionCategory::NotExpression:
      CARBON_FATAL() << "Converting non-expression node "
                     << semantics_ir().GetNode(expr_id)
                     << " to value expression";

    case SemIR::ExpressionCategory::Initializing:
      // Commit to using a temporary for this initializing expression.
      // TODO: Don't create a temporary if the initializing representation is
      // already a value representation.
      expr_id = FinalizeTemporary(expr_id, /*discarded=*/false);
      [[fallthrough]];

    case SemIR::ExpressionCategory::DurableReference:
    case SemIR::ExpressionCategory::EphemeralReference: {
      // TODO: Support types with custom value representations.
      SemIR::Node expr = semantics_ir().GetNode(expr_id);
      return AddNode(SemIR::Node::BindValue::Make(expr.parse_node(),
                                                  expr.type_id(), expr_id));
    }

    case SemIR::ExpressionCategory::Value:
      return expr_id;
  }
}

auto Context::FinalizeTemporary(SemIR::NodeId init_id, bool discarded)
    -> SemIR::NodeId {
  // TODO: See if we can refactor this with MarkInitializerFor once recursion
  // through struct and tuple values is properly handled.
  auto orig_init_id = init_id;
  while (true) {
    SemIR::Node init = semantics_ir().GetNode(init_id);
    CARBON_CHECK(SemIR::GetExpressionCategory(semantics_ir(), init_id) ==
                 SemIR::ExpressionCategory::Initializing)
        << "Can only materialize initializing expressions, found " << init;
    switch (init.kind()) {
      default:
        CARBON_FATAL() << "Initialization from unexpected node " << init;

      case SemIR::NodeKind::StructValue:
      case SemIR::NodeKind::TupleValue:
        CARBON_FATAL() << init << " is not modeled as initializing yet";

      case SemIR::NodeKind::StubReference: {
        init_id = init.GetAsStubReference();
        continue;
      }

      case SemIR::NodeKind::Call: {
        auto [refs_id, callee_id] = init.GetAsCall();
        if (semantics_ir().GetFunction(callee_id).return_slot_id.is_valid()) {
          // The return slot should have a materialized temporary in it.
          auto temporary_id = semantics_ir().GetNodeBlock(refs_id).back();
          CARBON_CHECK(semantics_ir().GetNode(temporary_id).kind() ==
                       SemIR::NodeKind::TemporaryStorage)
              << "Return slot for function call does not contain a temporary; "
              << "initialized multiple times? Have "
              << semantics_ir().GetNode(temporary_id);
          return AddNode(SemIR::Node::Temporary::Make(
              init.parse_node(), init.type_id(), temporary_id, orig_init_id));
        }

        if (discarded) {
          // Don't invent a temporary that we're going to discard.
          return SemIR::NodeId::Invalid;
        }

        // The function has no return slot, but we want to produce a temporary
        // object. Materialize one now.
        // TODO: Consider using an invalid ID to mean that we immediately
        // materialize and initialize a temporary, rather than two separate
        // nodes.
        auto temporary_id = AddNode(SemIR::Node::TemporaryStorage::Make(
            init.parse_node(), init.type_id()));
        return AddNode(SemIR::Node::Temporary::Make(
            init.parse_node(), init.type_id(), temporary_id, init_id));
      }
    }
  }
}

auto Context::MarkInitializerFor(SemIR::NodeId init_id, SemIR::NodeId target_id)
    -> void {
  while (true) {
    SemIR::Node init = semantics_ir().GetNode(init_id);
    CARBON_CHECK(SemIR::GetExpressionCategory(semantics_ir(), init_id) ==
                 SemIR::ExpressionCategory::Initializing)
        << "initialization from non-initializing node " << init;
    switch (init.kind()) {
      default:
        CARBON_FATAL() << "Initialization from unexpected node " << init;

      case SemIR::NodeKind::StructValue:
      case SemIR::NodeKind::TupleValue:
        CARBON_FATAL() << init << " is not modeled as initializing yet";

      case SemIR::NodeKind::StubReference:
        init_id = init.GetAsStubReference();
        continue;

      case SemIR::NodeKind::Call: {
        // If the callee has a return slot, point it at our target.
        auto [refs_id, callee_id] = init.GetAsCall();
        if (semantics_ir().GetFunction(callee_id).return_slot_id.is_valid()) {
          // Replace the return slot with our given target, and remove the
          // tentatively-created temporary.
          auto temporary_id = std::exchange(
              semantics_ir().GetNodeBlock(refs_id).back(), target_id);
          auto temporary = semantics_ir().GetNode(temporary_id);
          CARBON_CHECK(temporary.kind() == SemIR::NodeKind::TemporaryStorage)
              << "Return slot for function call does not contain a temporary; "
              << "initialized multiple times? Have " << temporary;
          semantics_ir().ReplaceNode(
              temporary_id, SemIR::Node::NoOp::Make(temporary.parse_node()));
        }
        return;
      }
    }
  }
}

auto Context::HandleDiscardedExpression(SemIR::NodeId expr_id) -> void {
  // If we discard an initializing expression, materialize it first.
  if (SemIR::GetExpressionCategory(semantics_ir(), expr_id) ==
      SemIR::ExpressionCategory::Initializing) {
    FinalizeTemporary(expr_id, /*discarded=*/true);
  }

  // TODO: This will eventually need to do some "do not discard" analysis.
  (void)expr_id;
}

auto Context::ImplicitAsForArgs(
    SemIR::NodeBlockId arg_refs_id, Parse::Node param_parse_node,
    SemIR::NodeBlockId param_refs_id,
    DiagnosticEmitter<Parse::Node>::DiagnosticBuilder* diagnostic) -> bool {
  // If both arguments and parameters are empty, return quickly. Otherwise,
  // we'll fetch both so that errors are consistent.
  if (arg_refs_id == SemIR::NodeBlockId::Empty &&
      param_refs_id == SemIR::NodeBlockId::Empty) {
    return true;
  }

  auto& arg_refs = semantics_ir_->GetNodeBlock(arg_refs_id);
  const auto& param_refs = semantics_ir_->GetNodeBlock(param_refs_id);

  // If sizes mismatch, fail early.
  if (arg_refs.size() != param_refs.size()) {
    CARBON_CHECK(diagnostic != nullptr) << "Should have validated first";
    CARBON_DIAGNOSTIC(CallArgCountMismatch, Note,
                      "Function cannot be used: Received {0} argument(s), but "
                      "require {1} argument(s).",
                      int, int);
    diagnostic->Note(param_parse_node, CallArgCountMismatch, arg_refs.size(),
                     param_refs.size());
    return false;
  }

  // Check type conversions per-element.
  // TODO: arg_ir_id is passed so that implicit conversions can be inserted.
  // It's currently not supported, but will be needed.
  for (auto [i, value_id, param_ref] : llvm::enumerate(arg_refs, param_refs)) {
    auto as_type_id = semantics_ir_->GetNode(param_ref).type_id();
    if (ImplicitAsImpl(value_id, as_type_id,
                       diagnostic == nullptr ? &value_id : nullptr) ==
        ImplicitAsKind::Incompatible) {
      CARBON_CHECK(diagnostic != nullptr) << "Should have validated first";
      CARBON_DIAGNOSTIC(CallArgTypeMismatch, Note,
                        "Function cannot be used: Cannot implicitly convert "
                        "argument {0} from `{1}` to `{2}`.",
                        size_t, std::string, std::string);
      diagnostic->Note(param_parse_node, CallArgTypeMismatch, i,
                       semantics_ir_->StringifyType(
                           semantics_ir_->GetNode(value_id).type_id()),
                       semantics_ir_->StringifyType(as_type_id));
      return false;
    }

    // TODO: Convert to the proper expression category. For now, we assume
    // parameters are all `let` bindings.
    if (!diagnostic) {
      // TODO: Insert the conversion in the proper place in the node block.
      arg_refs[i] = ConvertToValueExpression(value_id);
    }
  }

  return true;
}

auto Context::ImplicitAsRequired(Parse::Node parse_node, SemIR::NodeId value_id,
                                 SemIR::TypeId as_type_id) -> SemIR::NodeId {
  SemIR::NodeId output_value_id = value_id;
  if (ImplicitAsImpl(value_id, as_type_id, &output_value_id) ==
      ImplicitAsKind::Incompatible) {
    // Only error when the system is trying to use the result.
    CARBON_DIAGNOSTIC(ImplicitAsConversionFailure, Error,
                      "Cannot implicitly convert from `{0}` to `{1}`.",
                      std::string, std::string);
    emitter_
        ->Build(parse_node, ImplicitAsConversionFailure,
                semantics_ir_->StringifyType(
                    semantics_ir_->GetNode(value_id).type_id()),
                semantics_ir_->StringifyType(as_type_id))
        .Emit();
  }
  return output_value_id;
}

auto Context::ImplicitAsImpl(SemIR::NodeId value_id, SemIR::TypeId as_type_id,
                             SemIR::NodeId* output_value_id) -> ImplicitAsKind {
  // Start by making sure both sides are valid. If any part is invalid, the
  // result is invalid and we shouldn't error.
  if (value_id == SemIR::NodeId::BuiltinError) {
    // If the value is invalid, we can't do much, but do "succeed".
    return ImplicitAsKind::Identical;
  }
  auto value = semantics_ir_->GetNode(value_id);
  auto value_type_id = value.type_id();
  if (value_type_id == SemIR::TypeId::Error) {
    // Although the source type is invalid, this still changes the value.
    if (output_value_id != nullptr) {
      *output_value_id = SemIR::NodeId::BuiltinError;
    }
    return ImplicitAsKind::Compatible;
  }

  if (as_type_id == SemIR::TypeId::Error) {
    // Although the target type is invalid, this still changes the value.
    if (output_value_id != nullptr) {
      *output_value_id = SemIR::NodeId::BuiltinError;
    }
    return ImplicitAsKind::Compatible;
  }

  if (value_type_id == as_type_id) {
    // Type doesn't need to change.
    return ImplicitAsKind::Identical;
  }

  auto as_type = semantics_ir_->GetTypeAllowBuiltinTypes(as_type_id);
  auto as_type_node = semantics_ir_->GetNode(as_type);
  if (as_type_node.kind() == SemIR::NodeKind::ArrayType) {
    auto [bound_node_id, element_type_id] = as_type_node.GetAsArrayType();
    // To resolve lambda issue.
    auto element_type = element_type_id;
    auto value_type_node = semantics_ir_->GetNode(
        semantics_ir_->GetTypeAllowBuiltinTypes(value_type_id));
    if (value_type_node.kind() == SemIR::NodeKind::TupleType) {
      auto tuple_type_block_id = value_type_node.GetAsTupleType();
      const auto& type_block = semantics_ir_->GetTypeBlock(tuple_type_block_id);
      if (type_block.size() ==
              semantics_ir_->GetArrayBoundValue(bound_node_id) &&
          std::all_of(type_block.begin(), type_block.end(),
                      [&](auto type) { return type == element_type; })) {
        if (output_value_id != nullptr) {
          // TODO: We should convert an initializing expression of tuple type
          // to an initializing expression of array type.
          value_id = ConvertToValueExpression(value_id);
          *output_value_id = AddNode(SemIR::Node::ArrayValue::Make(
              value.parse_node(), as_type_id, value_id));
        }
        return ImplicitAsKind::Compatible;
      }
    }
  }

  if (as_type_id == SemIR::TypeId::TypeType) {
    if (value.kind() == SemIR::NodeKind::TupleValue) {
      auto tuple_block_id = value.GetAsTupleValue();
      llvm::SmallVector<SemIR::TypeId> type_ids;
      // If it is empty tuple type, we don't fetch anything.
      if (tuple_block_id != SemIR::NodeBlockId::Empty) {
        const auto& tuple_block = semantics_ir_->GetNodeBlock(tuple_block_id);
        for (auto tuple_node_id : tuple_block) {
          // TODO: Eventually ExpressionAsType will insert implicit cast
          // instructions. When that happens, this will need to verify the full
          // tuple conversion will work before calling it.
          type_ids.push_back(
              ExpressionAsType(value.parse_node(), tuple_node_id));
        }
      }
      auto tuple_type_id =
          CanonicalizeTupleType(value.parse_node(), std::move(type_ids));
      if (output_value_id != nullptr) {
        *output_value_id =
            semantics_ir_->GetTypeAllowBuiltinTypes(tuple_type_id);
      }
      return ImplicitAsKind::Compatible;
    }
    // When converting `{}` to a type, the result is `{} as Type`.
    if (value.kind() == SemIR::NodeKind::StructValue &&
        value.GetAsStructValue() == SemIR::NodeBlockId::Empty) {
      if (output_value_id != nullptr) {
        *output_value_id = semantics_ir_->GetType(value_type_id);
      }
      return ImplicitAsKind::Compatible;
    }
  }

  // TODO: Handle ImplicitAs for compatible structs and tuples.

  if (output_value_id != nullptr) {
    *output_value_id = SemIR::NodeId::BuiltinError;
  }
  return ImplicitAsKind::Incompatible;
}

auto Context::ParamOrArgStart() -> void { params_or_args_stack_.Push(); }

auto Context::ParamOrArgComma(bool for_args) -> void {
  ParamOrArgSave(for_args);
}

auto Context::ParamOrArgEnd(bool for_args, Parse::NodeKind start_kind)
    -> SemIR::NodeBlockId {
  if (parse_tree_->node_kind(node_stack_.PeekParseNode()) != start_kind) {
    ParamOrArgSave(for_args);
  }
  return params_or_args_stack_.Pop();
}

auto Context::ParamOrArgSave(bool for_args) -> void {
  auto [entry_parse_node, entry_node_id] =
      node_stack_.PopExpressionWithParseNode();
  if (for_args) {
    // For an argument, we add a stub reference to the expression on the top of
    // the stack. There may not be anything on the IR prior to this.
    entry_node_id = AddNode(SemIR::Node::StubReference::Make(
        entry_parse_node, semantics_ir_->GetNode(entry_node_id).type_id(),
        entry_node_id));
  }

  // Save the param or arg ID.
  auto& params_or_args =
      semantics_ir_->GetNodeBlock(params_or_args_stack_.PeekForAdd());
  params_or_args.push_back(entry_node_id);
}

auto Context::CanonicalizeTypeImpl(
    SemIR::NodeKind kind,
    llvm::function_ref<void(llvm::FoldingSetNodeID& canonical_id)> profile_type,
    llvm::function_ref<SemIR::NodeId()> make_node) -> SemIR::TypeId {
  llvm::FoldingSetNodeID canonical_id;
  kind.Profile(canonical_id);
  profile_type(canonical_id);

  void* insert_pos;
  auto* node =
      canonical_type_nodes_.FindNodeOrInsertPos(canonical_id, insert_pos);
  if (node != nullptr) {
    return node->type_id();
  }

  auto node_id = make_node();
  auto type_id = semantics_ir_->AddType(node_id);
  CARBON_CHECK(canonical_types_.insert({node_id, type_id}).second);
  type_node_storage_.push_back(
      std::make_unique<TypeNode>(canonical_id, type_id));

  // In a debug build, check that our insertion position is still valid. It
  // could have been invalidated by a misbehaving `make_node`.
  CARBON_DCHECK([&] {
    void* check_insert_pos;
    auto* check_node = canonical_type_nodes_.FindNodeOrInsertPos(
        canonical_id, check_insert_pos);
    return !check_node && insert_pos == check_insert_pos;
  }()) << "Type was created recursively during canonicalization";

  canonical_type_nodes_.InsertNode(type_node_storage_.back().get(), insert_pos);
  return type_id;
}

// Compute a fingerprint for a tuple type, for use as a key in a folding set.
static auto ProfileTupleType(const llvm::SmallVector<SemIR::TypeId>& type_ids,
                             llvm::FoldingSetNodeID& canonical_id) -> void {
  for (const auto& type_id : type_ids) {
    canonical_id.AddInteger(type_id.index);
  }
}

// Compute a fingerprint for a type, for use as a key in a folding set.
static auto ProfileType(Context& semantics_context, SemIR::Node node,
                        llvm::FoldingSetNodeID& canonical_id) -> void {
  switch (node.kind()) {
    case SemIR::NodeKind::ArrayType: {
      auto [bound_id, element_type_id] = node.GetAsArrayType();
      canonical_id.AddInteger(
          semantics_context.semantics_ir().GetArrayBoundValue(bound_id));
      canonical_id.AddInteger(element_type_id.index);
      break;
    }
    case SemIR::NodeKind::Builtin:
      canonical_id.AddInteger(node.GetAsBuiltin().AsInt());
      break;
    case SemIR::NodeKind::CrossReference: {
      // TODO: Cross-references should be canonicalized by looking at their
      // target rather than treating them as new unique types.
      auto [xref_id, node_id] = node.GetAsCrossReference();
      canonical_id.AddInteger(xref_id.index);
      canonical_id.AddInteger(node_id.index);
      break;
    }
    case SemIR::NodeKind::ConstType:
      canonical_id.AddInteger(
          semantics_context.GetUnqualifiedType(node.GetAsConstType()).index);
      break;
    case SemIR::NodeKind::PointerType:
      canonical_id.AddInteger(node.GetAsPointerType().index);
      break;
    case SemIR::NodeKind::StructType: {
      auto refs =
          semantics_context.semantics_ir().GetNodeBlock(node.GetAsStructType());
      for (const auto& ref_id : refs) {
        auto ref = semantics_context.semantics_ir().GetNode(ref_id);
        auto [name_id, type_id] = ref.GetAsStructTypeField();
        canonical_id.AddInteger(name_id.index);
        canonical_id.AddInteger(type_id.index);
      }
      break;
    }
    case SemIR::NodeKind::StubReference: {
      // We rely on stub references not referring to each other to ensure we
      // only recurse once here.
      auto inner =
          semantics_context.semantics_ir().GetNode(node.GetAsStubReference());
      CARBON_CHECK(inner.kind() != SemIR::NodeKind::StubReference)
          << "A stub reference should never refer to another stub reference.";
      ProfileType(semantics_context, inner, canonical_id);
      break;
    }
    case SemIR::NodeKind::TupleType:
      ProfileTupleType(
          semantics_context.semantics_ir().GetTypeBlock(node.GetAsTupleType()),
          canonical_id);
      break;
    default:
      CARBON_FATAL() << "Unexpected type node " << node;
  }
}

auto Context::CanonicalizeTypeAndAddNodeIfNew(SemIR::Node node)
    -> SemIR::TypeId {
  auto profile_node = [&](llvm::FoldingSetNodeID& canonical_id) {
    ProfileType(*this, node, canonical_id);
  };
  auto make_node = [&] { return AddNode(node); };
  return CanonicalizeTypeImpl(node.kind(), profile_node, make_node);
}

auto Context::CanonicalizeType(SemIR::NodeId node_id) -> SemIR::TypeId {
  auto it = canonical_types_.find(node_id);
  if (it != canonical_types_.end()) {
    return it->second;
  }

  auto node = semantics_ir_->GetNode(node_id);
  auto profile_node = [&](llvm::FoldingSetNodeID& canonical_id) {
    ProfileType(*this, node, canonical_id);
  };
  auto make_node = [&] { return node_id; };
  return CanonicalizeTypeImpl(node.kind(), profile_node, make_node);
}

auto Context::CanonicalizeStructType(Parse::Node parse_node,
                                     SemIR::NodeBlockId refs_id)
    -> SemIR::TypeId {
  return CanonicalizeTypeAndAddNodeIfNew(SemIR::Node::StructType::Make(
      parse_node, SemIR::TypeId::TypeType, refs_id));
}

auto Context::CanonicalizeTupleType(Parse::Node parse_node,
                                    llvm::SmallVector<SemIR::TypeId>&& type_ids)
    -> SemIR::TypeId {
  // Defer allocating a SemIR::TypeBlockId until we know this is a new type.
  auto profile_tuple = [&](llvm::FoldingSetNodeID& canonical_id) {
    ProfileTupleType(type_ids, canonical_id);
  };
  auto make_tuple_node = [&] {
    auto type_block_id = semantics_ir_->AddTypeBlock();
    auto& type_block = semantics_ir_->GetTypeBlock(type_block_id);
    type_block = std::move(type_ids);
    return AddNode(SemIR::Node::TupleType::Make(
        parse_node, SemIR::TypeId::TypeType, type_block_id));
  };
  return CanonicalizeTypeImpl(SemIR::NodeKind::TupleType, profile_tuple,
                              make_tuple_node);
}

auto Context::GetPointerType(Parse::Node parse_node,
                             SemIR::TypeId pointee_type_id) -> SemIR::TypeId {
  return CanonicalizeTypeAndAddNodeIfNew(SemIR::Node::PointerType::Make(
      parse_node, SemIR::TypeId::TypeType, pointee_type_id));
}

auto Context::GetUnqualifiedType(SemIR::TypeId type_id) -> SemIR::TypeId {
  SemIR::Node type_node =
      semantics_ir_->GetNode(semantics_ir_->GetTypeAllowBuiltinTypes(type_id));
  if (type_node.kind() == SemIR::NodeKind::ConstType) {
    return type_node.GetAsConstType();
  }
  return type_id;
}

auto Context::PrintForStackDump(llvm::raw_ostream& output) const -> void {
  node_stack_.PrintForStackDump(output);
  node_block_stack_.PrintForStackDump(output);
  params_or_args_stack_.PrintForStackDump(output);
  args_type_info_stack_.PrintForStackDump(output);
}

}  // namespace Carbon::Check
