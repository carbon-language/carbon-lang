// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_context.h"

#include <utility>

#include "common/vlog.h"
#include "toolchain/diagnostics/diagnostic_kind.h"
#include "toolchain/lexer/token_kind.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"
#include "toolchain/semantics/semantics_ir.h"
#include "toolchain/semantics/semantics_node.h"
#include "toolchain/semantics/semantics_node_block_stack.h"

namespace Carbon {

SemanticsContext::SemanticsContext(const TokenizedBuffer& tokens,
                                   DiagnosticEmitter<ParseTree::Node>& emitter,
                                   const ParseTree& parse_tree,
                                   SemanticsIR& semantics,
                                   llvm::raw_ostream* vlog_stream)
    : tokens_(&tokens),
      emitter_(&emitter),
      parse_tree_(&parse_tree),
      semantics_(&semantics),
      vlog_stream_(vlog_stream),
      node_stack_(parse_tree, vlog_stream),
      node_block_stack_("node_block_stack_", semantics.node_blocks(),
                        vlog_stream),
      params_or_args_stack_("params_or_args_stack_", semantics.node_blocks(),
                            vlog_stream),
      args_type_info_stack_("args_type_info_stack_", semantics.node_blocks(),
                            vlog_stream) {
  // Inserts the "Invalid" and "Type" types as "used types" so that
  // canonicalization can skip them. We don't emit either for lowering.
  canonical_types_.insert(SemanticsNodeId::BuiltinInvalidType);
  canonical_types_.insert(SemanticsNodeId::BuiltinTypeType);
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
  CARBON_CHECK(!node.type_id().is_valid() ||
               node.type_id() == SemanticsNodeId::BuiltinInvalidType ||
               canonical_types_.contains(node.type_id()))
      << "Added node without canonicalizing its type: " << node;
  auto block = node_block_stack_.PeekForAdd();
  CARBON_VLOG() << "AddNode " << block << ": " << node << "\n";
  return semantics_->AddNode(block, node);
}

auto SemanticsContext::AddNodeAndPush(ParseTree::Node parse_node,
                                      SemanticsNode node) -> void {
  auto node_id = AddNode(node);
  node_stack_.Push(parse_node, node_id);
}

auto SemanticsContext::AddNameToLookup(ParseTree::Node name_node,
                                       SemanticsStringId name_id,
                                       SemanticsNodeId target_id) -> void {
  if (!AddNameToLookupImpl(name_id, target_id)) {
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

auto SemanticsContext::AddNameToLookupImpl(SemanticsStringId name_id,
                                           SemanticsNodeId target_id) -> bool {
  if (current_scope().names.insert(name_id).second) {
    name_lookup_[name_id].push_back(target_id);
    return true;
  } else {
    return false;
  }
}

auto SemanticsContext::BindName(ParseTree::Node name_node,
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

auto SemanticsContext::TempRemoveLatestNameFromLookup() -> SemanticsNodeId {
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
  return storage_id;
}

auto SemanticsContext::LookupName(ParseTree::Node parse_node,
                                  llvm::StringRef name) -> SemanticsNodeId {
  CARBON_DIAGNOSTIC(NameNotFound, Error, "Name {0} not found", llvm::StringRef);

  auto name_id = semantics_->GetStringID(name);
  if (!name_id) {
    emitter_->Emit(parse_node, NameNotFound, name);
    return SemanticsNodeId::BuiltinInvalidType;
  }

  auto it = name_lookup_.find(*name_id);
  if (it == name_lookup_.end()) {
    emitter_->Emit(parse_node, NameNotFound, name);
    return SemanticsNodeId::BuiltinInvalidType;
  }
  CARBON_CHECK(!it->second.empty()) << "Should have been erased: " << name;

  // TODO: Check for ambiguous lookups.
  return it->second.back();
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

auto SemanticsContext::ImplicitAsRequired(ParseTree::Node parse_node,
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

auto SemanticsContext::ImplicitAsImpl(SemanticsNodeId value_id,
                                      SemanticsNodeId as_type_id,
                                      SemanticsNodeId* output_value_id)
    -> ImplicitAsKind {
  // Start by making sure both sides are valid. If any part is invalid, the
  // result is invalid and we shouldn't error.
  if (value_id == SemanticsNodeId::BuiltinInvalidType) {
    // If the value is invalid, we can't do much, but do "succeed".
    return ImplicitAsKind::Identical;
  }
  auto value = semantics_->GetNode(value_id);
  auto value_type_id = value.type_id();
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

  if (as_type_id == SemanticsNodeId::BuiltinTypeType) {
    // When converting `()` to a type, the result is `() as Type`.
    // TODO: This might switch to be closer to the struct conversion below.
    if (value_id == SemanticsNodeId::BuiltinEmptyTuple) {
      if (output_value_id != nullptr) {
        *output_value_id = SemanticsNodeId::BuiltinEmptyTupleType;
      }
      return ImplicitAsKind::Compatible;
    }

    // When converting `{}` to a type, the result is `{} as Type`.
    if (value.kind() == SemanticsNodeKind::StructValue &&
        value.GetAsStructValue() == SemanticsNodeBlockId::Empty) {
      if (output_value_id != nullptr) {
        *output_value_id = value_type_id;
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

auto SemanticsContext::CanImplicitAsStruct(SemanticsNode value_type,
                                           SemanticsNode as_type) -> bool {
  if (value_type.kind() != SemanticsNodeKind::StructType ||
      as_type.kind() != SemanticsNodeKind::StructType) {
    return false;
  }
  auto value_type_refs = semantics_->GetNodeBlock(value_type.GetAsStructType());
  auto as_type_refs = semantics_->GetNodeBlock(as_type.GetAsStructType());
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

auto SemanticsContext::CanonicalizeType(SemanticsNodeId node_id)
    -> SemanticsNodeId {
  if (canonical_types_.insert(node_id).second) {
    semantics_->AddType(node_id);
  }
  return node_id;
}

auto SemanticsContext::PrintForStackDump(llvm::raw_ostream& output) const
    -> void {
  node_stack_.PrintForStackDump(output);
  node_block_stack_.PrintForStackDump(output);
  params_or_args_stack_.PrintForStackDump(output);
  args_type_info_stack_.PrintForStackDump(output);
}

}  // namespace Carbon
