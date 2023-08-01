// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_context.h"

#include <utility>

#include "common/check.h"
#include "common/vlog.h"
#include "toolchain/diagnostics/diagnostic_kind.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"
#include "toolchain/semantics/semantics_declaration_name_stack.h"
#include "toolchain/semantics/semantics_ir.h"
#include "toolchain/semantics/semantics_node.h"
#include "toolchain/semantics/semantics_node_block_stack.h"
#include "toolchain/semantics/semantics_node_kind.h"

namespace Carbon {

SemanticsContext::SemanticsContext(const TokenizedBuffer& tokens,
                                   DiagnosticEmitter<ParseTree::Node>& emitter,
                                   const ParseTree& parse_tree,
                                   SemanticsIR& semantics_ir,
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
  canonical_types_.insert(
      {SemanticsNodeId::BuiltinError, SemanticsTypeId::Error});
  canonical_types_.insert(
      {SemanticsNodeId::BuiltinTypeType, SemanticsTypeId::TypeType});
}

auto SemanticsContext::TODO(ParseTree::Node parse_node, std::string label)
    -> bool {
  CARBON_DIAGNOSTIC(SemanticsTodo, Error, "Semantics TODO: {0}", std::string);
  emitter_->Emit(parse_node, SemanticsTodo, std::move(label));
  return false;
}

auto SemanticsContext::VerifyOnFinish() -> void {
  // Information in all the various context objects should be cleaned up as
  // various pieces of context go out of scope. At this point, nothing should
  // remain.
  // node_stack_ will still contain top-level entities.
  CARBON_CHECK(name_lookup_.empty()) << name_lookup_.size();
  CARBON_CHECK(scope_stack_.empty()) << scope_stack_.size();
  CARBON_CHECK(node_block_stack_.empty()) << node_block_stack_.size();
  CARBON_CHECK(params_or_args_stack_.empty()) << params_or_args_stack_.size();
}

auto SemanticsContext::AddNode(SemanticsNode node) -> SemanticsNodeId {
  return AddNodeToBlock(node_block_stack_.PeekForAdd(), node);
}

auto SemanticsContext::AddNodeToBlock(SemanticsNodeBlockId block,
                                      SemanticsNode node) -> SemanticsNodeId {
  CARBON_VLOG() << "AddNode " << block << ": " << node << "\n";
  return semantics_ir_->AddNode(block, node);
}

auto SemanticsContext::AddNodeAndPush(ParseTree::Node parse_node,
                                      SemanticsNode node) -> void {
  auto node_id = AddNode(node);
  node_stack_.Push(parse_node, node_id);
}

auto SemanticsContext::DiagnoseDuplicateName(ParseTree::Node parse_node,
                                             SemanticsNodeId prev_def_id)
    -> void {
  CARBON_DIAGNOSTIC(NameDeclarationDuplicate, Error,
                    "Duplicate name being declared in the same scope.");
  CARBON_DIAGNOSTIC(NameDeclarationPrevious, Note,
                    "Name is previously declared here.");
  auto prev_def = semantics_ir_->GetNode(prev_def_id);
  emitter_->Build(parse_node, NameDeclarationDuplicate)
      .Note(prev_def.parse_node(), NameDeclarationPrevious)
      .Emit();
}

auto SemanticsContext::DiagnoseNameNotFound(ParseTree::Node parse_node,
                                            SemanticsStringId name_id) -> void {
  CARBON_DIAGNOSTIC(NameNotFound, Error, "Name {0} not found", llvm::StringRef);
  emitter_->Emit(parse_node, NameNotFound, semantics_ir_->GetString(name_id));
}

auto SemanticsContext::AddNameToLookup(ParseTree::Node name_node,
                                       SemanticsStringId name_id,
                                       SemanticsNodeId target_id) -> void {
  if (current_scope().names.insert(name_id).second) {
    name_lookup_[name_id].push_back(target_id);
  } else {
    DiagnoseDuplicateName(name_node, name_lookup_[name_id].back());
  }
}

auto SemanticsContext::LookupName(ParseTree::Node parse_node,
                                  SemanticsStringId name_id,
                                  SemanticsNameScopeId scope_id,
                                  bool print_diagnostics) -> SemanticsNodeId {
  if (scope_id == SemanticsNameScopeId::Invalid) {
    auto it = name_lookup_.find(name_id);
    if (it == name_lookup_.end()) {
      if (print_diagnostics) {
        DiagnoseNameNotFound(parse_node, name_id);
      }
      return SemanticsNodeId::BuiltinError;
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
      return SemanticsNodeId::BuiltinError;
    }

    return it->second;
  }
}

auto SemanticsContext::PushScope() -> void { scope_stack_.push_back({}); }

auto SemanticsContext::PopScope() -> void {
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
static auto AddDominatedBlockAndBranchImpl(SemanticsContext& context,
                                           ParseTree::Node parse_node,
                                           Args... args)
    -> SemanticsNodeBlockId {
  if (!context.node_block_stack().is_current_block_reachable()) {
    return SemanticsNodeBlockId::Unreachable;
  }
  auto block_id = context.semantics_ir().AddNodeBlock();
  context.AddNode(BranchNode::Make(parse_node, block_id, args...));
  return block_id;
}

auto SemanticsContext::AddDominatedBlockAndBranch(ParseTree::Node parse_node)
    -> SemanticsNodeBlockId {
  return AddDominatedBlockAndBranchImpl<SemanticsNode::Branch>(*this,
                                                               parse_node);
}

auto SemanticsContext::AddDominatedBlockAndBranchWithArg(
    ParseTree::Node parse_node, SemanticsNodeId arg_id)
    -> SemanticsNodeBlockId {
  return AddDominatedBlockAndBranchImpl<SemanticsNode::BranchWithArg>(
      *this, parse_node, arg_id);
}

auto SemanticsContext::AddDominatedBlockAndBranchIf(ParseTree::Node parse_node,
                                                    SemanticsNodeId cond_id)
    -> SemanticsNodeBlockId {
  return AddDominatedBlockAndBranchImpl<SemanticsNode::BranchIf>(
      *this, parse_node, cond_id);
}

auto SemanticsContext::AddConvergenceBlockAndPush(
    ParseTree::Node parse_node,
    std::initializer_list<SemanticsNodeBlockId> blocks) -> void {
  CARBON_CHECK(blocks.size() >= 2) << "no convergence";

  SemanticsNodeBlockId new_block_id = SemanticsNodeBlockId::Unreachable;
  for (SemanticsNodeBlockId block_id : blocks) {
    if (block_id != SemanticsNodeBlockId::Unreachable) {
      if (new_block_id == SemanticsNodeBlockId::Unreachable) {
        new_block_id = semantics_ir().AddNodeBlock();
      }
      AddNodeToBlock(block_id,
                     SemanticsNode::Branch::Make(parse_node, new_block_id));
    }
  }
  node_block_stack().Push(new_block_id);
}

auto SemanticsContext::AddConvergenceBlockWithArgAndPush(
    ParseTree::Node parse_node,
    std::initializer_list<std::pair<SemanticsNodeBlockId, SemanticsNodeId>>
        blocks_and_args) -> SemanticsNodeId {
  CARBON_CHECK(blocks_and_args.size() >= 2) << "no convergence";

  SemanticsNodeBlockId new_block_id = SemanticsNodeBlockId::Unreachable;
  for (auto [block_id, arg_id] : blocks_and_args) {
    if (block_id != SemanticsNodeBlockId::Unreachable) {
      if (new_block_id == SemanticsNodeBlockId::Unreachable) {
        new_block_id = semantics_ir().AddNodeBlock();
      }
      AddNodeToBlock(block_id, SemanticsNode::BranchWithArg::Make(
                                   parse_node, new_block_id, arg_id));
    }
  }
  node_block_stack().Push(new_block_id);

  // Acquire the result value.
  SemanticsTypeId result_type_id =
      semantics_ir().GetNode(blocks_and_args.begin()->second).type_id();
  return AddNode(
      SemanticsNode::BlockArg::Make(parse_node, result_type_id, new_block_id));
}

// Add the current code block to the enclosing function.
auto SemanticsContext::AddCurrentCodeBlockToFunction() -> void {
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

auto SemanticsContext::is_current_position_reachable() -> bool {
  switch (auto block_id = node_block_stack().Peek(); block_id.index) {
    case SemanticsNodeBlockId::Unreachable.index: {
      return false;
    }
    case SemanticsNodeBlockId::Invalid.index: {
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
             SemanticsTerminatorKind::Terminator;
    }
  }
}

auto SemanticsContext::ImplicitAsForArgs(
    SemanticsNodeBlockId arg_refs_id, ParseTree::Node param_parse_node,
    SemanticsNodeBlockId param_refs_id,
    DiagnosticEmitter<ParseTree::Node>::DiagnosticBuilder* diagnostic) -> bool {
  // If both arguments and parameters are empty, return quickly. Otherwise,
  // we'll fetch both so that errors are consistent.
  if (arg_refs_id == SemanticsNodeBlockId::Empty &&
      param_refs_id == SemanticsNodeBlockId::Empty) {
    return true;
  }

  auto arg_refs = semantics_ir_->GetNodeBlock(arg_refs_id);
  auto param_refs = semantics_ir_->GetNodeBlock(param_refs_id);

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
  for (size_t i = 0; i < arg_refs.size(); ++i) {
    auto value_id = arg_refs[i];
    auto as_type_id = semantics_ir_->GetNode(param_refs[i]).type_id();
    if (ImplicitAsImpl(value_id, as_type_id,
                       diagnostic == nullptr ? &value_id : nullptr) ==
        ImplicitAsKind::Incompatible) {
      CARBON_CHECK(diagnostic != nullptr) << "Should have validated first";
      CARBON_DIAGNOSTIC(CallArgTypeMismatch, Note,
                        "Function cannot be used: Cannot implicityly convert "
                        "argument {0} from `{1}` to `{2}`.",
                        size_t, std::string, std::string);
      diagnostic->Note(param_parse_node, CallArgTypeMismatch, i,
                       semantics_ir_->StringifyType(
                           semantics_ir_->GetNode(value_id).type_id()),
                       semantics_ir_->StringifyType(as_type_id));
      return false;
    }
  }

  return true;
}

auto SemanticsContext::ImplicitAsRequired(ParseTree::Node parse_node,
                                          SemanticsNodeId value_id,
                                          SemanticsTypeId as_type_id)
    -> SemanticsNodeId {
  SemanticsNodeId output_value_id = value_id;
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

auto SemanticsContext::ImplicitAsBool(ParseTree::Node parse_node,
                                      SemanticsNodeId value_id)
    -> SemanticsNodeId {
  return ImplicitAsRequired(parse_node, value_id,
                            CanonicalizeType(SemanticsNodeId::BuiltinBoolType));
}

auto SemanticsContext::ImplicitAsImpl(SemanticsNodeId value_id,
                                      SemanticsTypeId as_type_id,
                                      SemanticsNodeId* output_value_id)
    -> ImplicitAsKind {
  // Start by making sure both sides are valid. If any part is invalid, the
  // result is invalid and we shouldn't error.
  if (value_id == SemanticsNodeId::BuiltinError) {
    // If the value is invalid, we can't do much, but do "succeed".
    return ImplicitAsKind::Identical;
  }
  auto value = semantics_ir_->GetNode(value_id);
  auto value_type_id = value.type_id();
  if (value_type_id == SemanticsTypeId::Error) {
    return ImplicitAsKind::Identical;
  }

  if (as_type_id == SemanticsTypeId::Error) {
    // Although the target type is invalid, this still changes the value.
    if (output_value_id != nullptr) {
      *output_value_id = SemanticsNodeId::BuiltinError;
    }
    return ImplicitAsKind::Compatible;
  }

  if (value_type_id == as_type_id) {
    // Type doesn't need to change.
    return ImplicitAsKind::Identical;
  }
  if (as_type_id == SemanticsTypeId::TypeType) {
    if (value.kind() == SemanticsNodeKind::TupleValue) {
      auto tuple_block_id = value.GetAsTupleValue();
      llvm::SmallVector<SemanticsTypeId> type_ids;
      // If it is empty tuple type, we don't fetch anything.
      if (tuple_block_id != SemanticsNodeBlockId::Empty) {
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
    if (value.kind() == SemanticsNodeKind::StructValue &&
        value.GetAsStructValue() == SemanticsNodeBlockId::Empty) {
      if (output_value_id != nullptr) {
        *output_value_id = semantics_ir_->GetType(value_type_id);
      }
      return ImplicitAsKind::Compatible;
    }
  }

  // TODO: Handle ImplicitAs for compatible structs and tuples.

  if (output_value_id != nullptr) {
    *output_value_id = SemanticsNodeId::BuiltinError;
  }
  return ImplicitAsKind::Incompatible;
}

auto SemanticsContext::ParamOrArgStart() -> void {
  params_or_args_stack_.Push();
}

auto SemanticsContext::ParamOrArgComma(bool for_args) -> void {
  ParamOrArgSave(for_args);
}

auto SemanticsContext::ParamOrArgEnd(bool for_args, ParseNodeKind start_kind)
    -> SemanticsNodeBlockId {
  if (parse_tree_->node_kind(node_stack_.PeekParseNode()) != start_kind) {
    ParamOrArgSave(for_args);
  }
  return params_or_args_stack_.Pop();
}

auto SemanticsContext::ParamOrArgSave(bool for_args) -> void {
  auto [entry_parse_node, entry_node_id] =
      node_stack_.PopExpressionWithParseNode();
  if (for_args) {
    // For an argument, we add a stub reference to the expression on the top of
    // the stack. There may not be anything on the IR prior to this.
    entry_node_id = AddNode(SemanticsNode::StubReference::Make(
        entry_parse_node, semantics_ir_->GetNode(entry_node_id).type_id(),
        entry_node_id));
  }

  // Save the param or arg ID.
  auto& params_or_args =
      semantics_ir_->GetNodeBlock(params_or_args_stack_.PeekForAdd());
  params_or_args.push_back(entry_node_id);
}

template <typename ProfileType, typename MakeNode>
auto SemanticsContext::CanonicalizeTypeImpl(SemanticsNodeKind kind,
                                            ProfileType profile_type,
                                            MakeNode make_node)
    -> SemanticsTypeId {
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

#ifndef NDEBUG
  // In a debug build, check that our insertion position is still valid. It
  // could have been invalidated by a misbehaving `make_node`.
  void* check_insert_pos;
  auto* check_node =
      canonical_type_nodes_.FindNodeOrInsertPos(canonical_id, check_insert_pos);
  CARBON_CHECK(!check_node)
      << "Type was created recursively during canonicalization";
  CARBON_CHECK(insert_pos == check_insert_pos)
      << "Insertion position changed during canonicalization";
#endif

  canonical_type_nodes_.InsertNode(type_node_storage_.back().get(), insert_pos);
  return type_id;
}

auto SemanticsContext::CanonicalizeType(SemanticsNodeId node_id)
    -> SemanticsTypeId {
  auto node = semantics_ir_->GetNode(node_id);
  if (node.kind() == SemanticsNodeKind::StubReference) {
    node_id = node.GetAsStubReference();
    CARBON_CHECK(semantics_ir_->GetNode(node_id).kind() !=
                 SemanticsNodeKind::StubReference)
        << "Stub reference should not point to another stub reference";
  }

  auto it = canonical_types_.find(node_id);
  if (it != canonical_types_.end()) {
    return it->second;
  }

  switch (node.kind()) {
    case SemanticsNodeKind::Builtin:
    case SemanticsNodeKind::CrossReference: {
      // TODO: Cross-references should be canonicalized by looking at their
      // target rather than treating them as new unique types.
      auto type_id = semantics_ir_->AddType(node_id);
      CARBON_CHECK(canonical_types_.insert({node_id, type_id}).second);
      return type_id;
    }
    case SemanticsNodeKind::ConstType: {
      return CanonicalizeTypeImpl(
          node.kind(), node_id, [&](llvm::FoldingSetNodeID& canonical_id) {
            canonical_id.AddInteger(
                GetUnqualifiedType(node.GetAsConstType()).index);
          });
    }
    case SemanticsNodeKind::PointerType: {
      return CanonicalizeTypeImpl(
          node.kind(), node_id, [&](llvm::FoldingSetNodeID& canonical_id) {
            canonical_id.AddInteger(node.GetAsPointerType().index);
          });
    }
    case SemanticsNodeKind::StructType:
    case SemanticsNodeKind::TupleType: {
      CARBON_FATAL() << "Type should have been canonizalized when created: "
                     << node;
    }
    default: {
      CARBON_FATAL() << "Unexpected non-canonical type node " << node;
    }
  }
}

auto SemanticsContext::CanonicalizeStructType(ParseTree::Node parse_node,
                                              SemanticsNodeBlockId refs_id)
    -> SemanticsTypeId {
  auto profile_struct = [&](llvm::FoldingSetNodeID& canonical_id) {
    auto refs = semantics_ir_->GetNodeBlock(refs_id);
    for (const auto& ref_id : refs) {
      auto ref = semantics_ir_->GetNode(ref_id);
      canonical_id.AddInteger(ref.GetAsStructTypeField().index);
      canonical_id.AddInteger(ref.type_id().index);
    }
  };
  auto make_struct_node = [&] {
    return AddNode(SemanticsNode::StructType::Make(
        parse_node, SemanticsTypeId::TypeType, refs_id));
  };
  return CanonicalizeTypeImpl(SemanticsNodeKind::StructType, profile_struct,
                              make_struct_node);
}

auto SemanticsContext::CanonicalizeTupleType(
    ParseTree::Node parse_node, llvm::SmallVector<SemanticsTypeId>&& type_ids)
    -> SemanticsTypeId {
  auto profile_tuple = [&](llvm::FoldingSetNodeID& canonical_id) {
    for (const auto& type_id : type_ids) {
      canonical_id.AddInteger(type_id.index);
    }
  };
  auto make_tuple_node = [&] {
    auto type_block_id = semantics_ir_->AddTypeBlock();
    auto& type_block = semantics_ir_->GetTypeBlock(type_block_id);
    type_block = std::move(type_ids);
    return AddNode(SemanticsNode::TupleType::Make(
        parse_node, SemanticsTypeId::TypeType, type_block_id));
  };
  return CanonicalizeTypeImpl(SemanticsNodeKind::TupleType, profile_tuple,
                              make_tuple_node);
}

auto SemanticsContext::GetUnqualifiedType(SemanticsTypeId type_id)
    -> SemanticsTypeId {
  SemanticsNode type_node =
      semantics_ir_->GetNode(semantics_ir_->GetType(type_id));
  if (type_node.kind() == SemanticsNodeKind::ConstType)
    return type_node.GetAsConstType();
  return type_id;
}

auto SemanticsContext::PrintForStackDump(llvm::raw_ostream& output) const
    -> void {
  node_stack_.PrintForStackDump(output);
  node_block_stack_.PrintForStackDump(output);
  params_or_args_stack_.PrintForStackDump(output);
  args_type_info_stack_.PrintForStackDump(output);
}

}  // namespace Carbon
