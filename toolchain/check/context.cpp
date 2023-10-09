// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

#include <string>
#include <utility>

#include "common/check.h"
#include "common/vlog.h"
#include "llvm/ADT/Sequence.h"
#include "toolchain/check/declaration_name_stack.h"
#include "toolchain/check/node_block_stack.h"
#include "toolchain/diagnostics/diagnostic_kind.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/node.h"
#include "toolchain/sem_ir/node_kind.h"

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
  auto node_id = node_block_stack_.AddNode(node);
  CARBON_VLOG() << "AddNode: " << node << "\n";
  return node_id;
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

auto Context::FollowNameReferences(SemIR::NodeId node_id) -> SemIR::NodeId {
  while (true) {
    auto node = semantics_ir().GetNode(node_id);
    switch (node.kind()) {
      case SemIR::NodeKind::NameReference: {
        node_id = node.As<SemIR::NameReference>().value_id;
        break;
      }
      case SemIR::NodeKind::NameReferenceUntyped: {
        node_id = node.As<SemIR::NameReferenceUntyped>().value_id;
        break;
      }
      default:
        return node_id;
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
  auto block_id = context.semantics_ir().AddNodeBlockId();
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

auto Context::AddConvergenceBlockAndPush(Parse::Node parse_node, int num_blocks)
    -> void {
  CARBON_CHECK(num_blocks >= 2) << "no convergence";

  SemIR::NodeBlockId new_block_id = SemIR::NodeBlockId::Unreachable;
  for ([[maybe_unused]] auto _ : llvm::seq(num_blocks)) {
    if (node_block_stack().is_current_block_reachable()) {
      if (new_block_id == SemIR::NodeBlockId::Unreachable) {
        new_block_id = semantics_ir().AddNodeBlockId();
      }
      AddNode(SemIR::Node::Branch::Make(parse_node, new_block_id));
    }
    node_block_stack().Pop();
  }
  node_block_stack().Push(new_block_id);
}

auto Context::AddConvergenceBlockWithArgAndPush(
    Parse::Node parse_node, std::initializer_list<SemIR::NodeId> block_args)
    -> SemIR::NodeId {
  CARBON_CHECK(block_args.size() >= 2) << "no convergence";

  SemIR::NodeBlockId new_block_id = SemIR::NodeBlockId::Unreachable;
  for (auto arg_id : block_args) {
    if (node_block_stack().is_current_block_reachable()) {
      if (new_block_id == SemIR::NodeBlockId::Unreachable) {
        new_block_id = semantics_ir().AddNodeBlockId();
      }
      AddNode(
          SemIR::Node::BranchWithArg::Make(parse_node, new_block_id, arg_id));
    }
    node_block_stack().Pop();
  }
  node_block_stack().Push(new_block_id);

  // Acquire the result value.
  SemIR::TypeId result_type_id =
      semantics_ir().GetNode(*block_args.begin()).type_id();
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

  auto function_id =
      semantics_ir()
          .GetNodeAs<SemIR::FunctionDeclaration>(return_scope_stack().back())
          .function_id;
  semantics_ir()
      .GetFunction(function_id)
      .body_block_ids.push_back(node_block_stack().PeekOrAdd());
}

auto Context::is_current_position_reachable() -> bool {
  if (!node_block_stack().is_current_block_reachable()) {
    return false;
  }

  // Our current position is at the end of a reachable block. That position is
  // reachable unless the previous instruction is a terminator instruction.
  auto block_contents = node_block_stack().PeekCurrentBlockContents();
  if (block_contents.empty()) {
    return true;
  }
  const auto& last_node = semantics_ir().GetNode(block_contents.back());
  return last_node.kind().terminator_kind() !=
         SemIR::TerminatorKind::Terminator;
}

auto Context::ParamOrArgStart() -> void { params_or_args_stack_.Push(); }

auto Context::ParamOrArgComma() -> void {
  ParamOrArgSave(node_stack_.PopExpression());
}

auto Context::ParamOrArgEndNoPop(Parse::NodeKind start_kind) -> void {
  if (parse_tree_->node_kind(node_stack_.PeekParseNode()) != start_kind) {
    ParamOrArgSave(node_stack_.PopExpression());
  }
}

auto Context::ParamOrArgPop() -> SemIR::NodeBlockId {
  return params_or_args_stack_.Pop();
}

auto Context::ParamOrArgEnd(Parse::NodeKind start_kind) -> SemIR::NodeBlockId {
  ParamOrArgEndNoPop(start_kind);
  return ParamOrArgPop();
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
static auto ProfileTupleType(llvm::ArrayRef<SemIR::TypeId> type_ids,
                             llvm::FoldingSetNodeID& canonical_id) -> void {
  for (auto type_id : type_ids) {
    canonical_id.AddInteger(type_id.index);
  }
}

// Compute a fingerprint for a type, for use as a key in a folding set.
static auto ProfileType(Context& semantics_context, SemIR::Node node,
                        llvm::FoldingSetNodeID& canonical_id) -> void {
  switch (node.kind()) {
    case SemIR::NodeKind::ArrayType: {
      auto array_type = node.As<SemIR::ArrayType>();
      canonical_id.AddInteger(
          semantics_context.semantics_ir().GetArrayBoundValue(
              array_type.bound_id));
      canonical_id.AddInteger(array_type.element_type_id.index);
      break;
    }
    case SemIR::NodeKind::Builtin:
      canonical_id.AddInteger(node.As<SemIR::Builtin>().builtin_kind.AsInt());
      break;
    case SemIR::NodeKind::CrossReference: {
      // TODO: Cross-references should be canonicalized by looking at their
      // target rather than treating them as new unique types.
      auto xref = node.As<SemIR::CrossReference>();
      canonical_id.AddInteger(xref.ir_id.index);
      canonical_id.AddInteger(xref.node_id.index);
      break;
    }
    case SemIR::NodeKind::ConstType:
      canonical_id.AddInteger(
          semantics_context.GetUnqualifiedType(node.As<SemIR::ConstType>().inner_id)
              .index);
      break;
    case SemIR::NodeKind::PointerType:
      canonical_id.AddInteger(node.As<SemIR::PointerType>().pointee_id.index);
      break;
    case SemIR::NodeKind::StructType: {
      auto fields = semantics_context.semantics_ir().GetNodeBlock(
          node.As<SemIR::StructType>().fields_id);
      for (const auto& field_id : fields) {
        auto field =
            semantics_context.semantics_ir().GetNodeAs<SemIR::StructTypeField>(
                field_id);
        canonical_id.AddInteger(field.name_id.index);
        canonical_id.AddInteger(field.type_id.index);
      }
      break;
    }
    case SemIR::NodeKind::TupleType:
      ProfileTupleType(semantics_context.semantics_ir().GetTypeBlock(
                           node.As<SemIR::TupleType>().elements_id),
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
                                    llvm::ArrayRef<SemIR::TypeId> type_ids)
    -> SemIR::TypeId {
  // Defer allocating a SemIR::TypeBlockId until we know this is a new type.
  auto profile_tuple = [&](llvm::FoldingSetNodeID& canonical_id) {
    ProfileTupleType(type_ids, canonical_id);
  };
  auto make_tuple_node = [&] {
    return AddNode(
        SemIR::Node::TupleType::Make(parse_node, SemIR::TypeId::TypeType,
                                     semantics_ir_->AddTypeBlock(type_ids)));
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
  if (auto const_type = type_node.TryAs<SemIR::ConstType>()) {
    return const_type->inner_id;
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
