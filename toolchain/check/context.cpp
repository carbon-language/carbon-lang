// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

#include <string>
#include <utility>

#include "common/check.h"
#include "common/vlog.h"
#include "llvm/ADT/Sequence.h"
#include "toolchain/check/decl_name_stack.h"
#include "toolchain/check/eval.h"
#include "toolchain/check/inst_block_stack.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/inst_kind.h"
#include "toolchain/sem_ir/typed_insts.h"
#include "toolchain/sem_ir/value_stores.h"

namespace Carbon::Check {

Context::Context(const Lex::TokenizedBuffer& tokens, DiagnosticEmitter& emitter,
                 const Parse::Tree& parse_tree, SemIR::File& sem_ir,
                 llvm::raw_ostream* vlog_stream)
    : tokens_(&tokens),
      emitter_(&emitter),
      parse_tree_(&parse_tree),
      sem_ir_(&sem_ir),
      vlog_stream_(vlog_stream),
      node_stack_(parse_tree, vlog_stream),
      inst_block_stack_("inst_block_stack_", sem_ir, vlog_stream),
      params_or_args_stack_("params_or_args_stack_", sem_ir, vlog_stream),
      args_type_info_stack_("args_type_info_stack_", sem_ir, vlog_stream),
      decl_name_stack_(this),
      lexical_lookup_(sem_ir_->identifiers()) {
  // Inserts the "Error" and "Type" types as "used types" so that
  // canonicalization can skip them. We don't emit either for lowering.
  canonical_types_.insert({SemIR::InstId::BuiltinError, SemIR::TypeId::Error});
  canonical_types_.insert(
      {SemIR::InstId::BuiltinTypeType, SemIR::TypeId::TypeType});
}

auto Context::TODO(Parse::NodeId parse_node, std::string label) -> bool {
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
  CARBON_CHECK(scope_stack_.empty()) << scope_stack_.size();
  CARBON_CHECK(inst_block_stack_.empty()) << inst_block_stack_.size();
  CARBON_CHECK(params_or_args_stack_.empty()) << params_or_args_stack_.size();
}

auto Context::AddInstInNoBlock(SemIR::ParseNodeAndInst parse_node_and_inst)
    -> SemIR::InstId {
  auto inst_id = sem_ir().insts().AddInNoBlock(parse_node_and_inst);
  CARBON_VLOG() << "AddInst: " << parse_node_and_inst.inst << "\n";

  auto const_id = TryEvalInst(*this, inst_id, parse_node_and_inst.inst);
  if (const_id.is_constant()) {
    CARBON_VLOG() << "Constant: " << parse_node_and_inst.inst << " -> "
                  << const_id.inst_id() << "\n";
    constant_values().Set(inst_id, const_id);
  }

  return inst_id;
}

auto Context::AddInst(SemIR::ParseNodeAndInst parse_node_and_inst)
    -> SemIR::InstId {
  auto inst_id = AddInstInNoBlock(parse_node_and_inst);
  inst_block_stack_.AddInstId(inst_id);
  return inst_id;
}

auto Context::AddPlaceholderInstInNoBlock(
    SemIR::ParseNodeAndInst parse_node_and_inst) -> SemIR::InstId {
  auto inst_id = sem_ir().insts().AddInNoBlock(parse_node_and_inst);
  CARBON_VLOG() << "AddPlaceholderInst: " << parse_node_and_inst.inst << "\n";
  return inst_id;
}

auto Context::AddPlaceholderInst(SemIR::ParseNodeAndInst parse_node_and_inst)
    -> SemIR::InstId {
  auto inst_id = AddPlaceholderInstInNoBlock(parse_node_and_inst);
  inst_block_stack_.AddInstId(inst_id);
  return inst_id;
}

auto Context::AddConstant(SemIR::Inst inst, bool is_symbolic)
    -> SemIR::ConstantId {
  auto [inst_id, added] = constants().GetOrAdd(inst);
  auto const_id = is_symbolic ? SemIR::ConstantId::ForSymbolicConstant(inst_id)
                              : SemIR::ConstantId::ForTemplateConstant(inst_id);
  if (added) {
    // TODO: Should `ConstantStore` do this for us?
    constant_values().Set(inst_id, const_id);
  }

  CARBON_VLOG() << "AddConstantInst: " << inst << "\n";
  return const_id;
}

auto Context::AddInstAndPush(SemIR::ParseNodeAndInst parse_node_and_inst)
    -> void {
  auto inst_id = AddInst(parse_node_and_inst);
  node_stack_.Push(parse_node_and_inst.parse_node, inst_id);
}

auto Context::ReplaceInstBeforeConstantUse(
    SemIR::InstId inst_id, SemIR::ParseNodeAndInst parse_node_and_inst)
    -> void {
  sem_ir().insts().Set(inst_id, parse_node_and_inst);

  CARBON_VLOG() << "ReplaceInst: " << inst_id << " -> "
                << parse_node_and_inst.inst << "\n";

  // Redo evaluation. This is only safe to do if this instruction has not
  // already been used as a constant, which is the caller's responsibility to
  // ensure.
  auto const_id = TryEvalInst(*this, inst_id, parse_node_and_inst.inst);
  if (const_id.is_constant()) {
    CARBON_VLOG() << "Constant: " << parse_node_and_inst.inst << " -> "
                  << const_id.inst_id() << "\n";
  }
  constant_values().Set(inst_id, const_id);
}

auto Context::DiagnoseDuplicateName(Parse::NodeId parse_node,
                                    SemIR::InstId prev_def_id) -> void {
  CARBON_DIAGNOSTIC(NameDeclDuplicate, Error,
                    "Duplicate name being declared in the same scope.");
  CARBON_DIAGNOSTIC(NameDeclPrevious, Note,
                    "Name is previously declared here.");
  emitter_->Build(parse_node, NameDeclDuplicate)
      .Note(prev_def_id, NameDeclPrevious)
      .Emit();
}

auto Context::DiagnoseNameNotFound(Parse::NodeId parse_node,
                                   SemIR::NameId name_id) -> void {
  CARBON_DIAGNOSTIC(NameNotFound, Error, "Name `{0}` not found.", std::string);
  emitter_->Emit(parse_node, NameNotFound, names().GetFormatted(name_id).str());
}

auto Context::NoteIncompleteClass(SemIR::ClassId class_id,
                                  DiagnosticBuilder& builder) -> void {
  CARBON_DIAGNOSTIC(ClassForwardDeclaredHere, Note,
                    "Class was forward declared here.");
  CARBON_DIAGNOSTIC(ClassIncompleteWithinDefinition, Note,
                    "Class is incomplete within its definition.");
  const auto& class_info = classes().Get(class_id);
  CARBON_CHECK(!class_info.is_defined()) << "Class is not incomplete";
  if (class_info.definition_id.is_valid()) {
    builder.Note(class_info.definition_id, ClassIncompleteWithinDefinition);
  } else {
    builder.Note(class_info.decl_id, ClassForwardDeclaredHere);
  }
}

auto Context::AddPackageImports(Parse::NodeId import_node,
                                IdentifierId package_id,
                                llvm::ArrayRef<const SemIR::File*> sem_irs,
                                bool has_load_error) -> void {
  CARBON_CHECK(has_load_error || !sem_irs.empty())
      << "There should be either a load error or at least one IR.";

  auto name_id = SemIR::NameId::ForIdentifier(package_id);

  SemIR::CrossRefIRId first_id(cross_ref_irs().size());
  for (const auto* sem_ir : sem_irs) {
    cross_ref_irs().Add(sem_ir);
  }
  if (has_load_error) {
    cross_ref_irs().Add(nullptr);
  }
  SemIR::CrossRefIRId last_id(cross_ref_irs().size() - 1);

  auto type_id = GetBuiltinType(SemIR::BuiltinKind::NamespaceType);
  auto inst_id =
      AddInst({import_node, SemIR::Import{.type_id = type_id,
                                          .first_cross_ref_ir_id = first_id,
                                          .last_cross_ref_ir_id = last_id}});

  // Add the import to lookup. Should always succeed because imports will be
  // uniquely named.
  AddNameToLookup(import_node, name_id, inst_id);
  // Add a name for formatted output. This isn't used in name lookup in order
  // to reduce indirection, but it's separate from the Import because it
  // otherwise fits in an Inst.
  auto bind_name_id = bind_names().Add(
      {.name_id = name_id, .enclosing_scope_id = SemIR::NameScopeId::Package});
  AddInst({import_node, SemIR::BindName{.type_id = type_id,
                                        .bind_name_id = bind_name_id,
                                        .value_id = inst_id}});
}

auto Context::AddNameToLookup(Parse::NodeId name_node, SemIR::NameId name_id,
                              SemIR::InstId target_id) -> void {
  if (current_scope().names.insert(name_id).second) {
    // TODO: Reject if we previously performed a failed lookup for this name in
    // this scope or a scope nested within it.
    auto& lexical_results = lexical_lookup_.Get(name_id);
    CARBON_CHECK(lexical_results.empty() ||
                 lexical_results.back().scope_index < current_scope_index())
        << "Failed to clean up after scope nested within the current scope";
    lexical_results.push_back(
        {.inst_id = target_id, .scope_index = current_scope_index()});
  } else {
    DiagnoseDuplicateName(name_node,
                          lexical_lookup_.Get(name_id).back().inst_id);
  }
}

auto Context::ResolveIfLazyImportRef(SemIR::InstId inst_id) -> void {
  auto inst = insts().Get(inst_id);
  auto lazy_inst = inst.TryAs<SemIR::LazyImportRef>();
  if (!lazy_inst) {
    return;
  }
  const SemIR::File& import_ir = *cross_ref_irs().Get(lazy_inst->ir_id);
  auto import_inst = import_ir.insts().Get(lazy_inst->inst_id);
  switch (import_inst.kind()) {
    case SemIR::InstKind::FunctionDecl: {
      // TODO: Fill this in better.
      auto function_id =
          functions().Add({.name_id = SemIR::NameId::Invalid,
                           .enclosing_scope_id = SemIR::NameScopeId::Invalid,
                           .decl_id = inst_id,
                           .implicit_param_refs_id = SemIR::InstBlockId::Empty,
                           .param_refs_id = SemIR::InstBlockId::Empty,
                           .return_type_id = SemIR::TypeId::Invalid,
                           .return_slot_id = SemIR::InstId::Invalid});
      ReplaceInstBeforeConstantUse(
          inst_id,
          // TODO: For diagnostic purposes, we should provide some form of
          // location for the function.
          {Parse::NodeId::Invalid,
           SemIR::FunctionDecl{GetBuiltinType(SemIR::BuiltinKind::FunctionType),
                               function_id}});
      constant_values().Set(inst_id,
                            SemIR::ConstantId::ForTemplateConstant(inst_id));
      break;
    }

    default:
      // TODO: We need more type support. For now we inject an arbitrary
      // invalid node that's unrelated to the underlying value. The TODO
      // diagnostic is used since this section shouldn't typically be able to
      // error.
      TODO(Parse::NodeId::Invalid,
           (llvm::Twine("TODO: support ") + import_inst.kind().name()).str());
      ReplaceInstBeforeConstantUse(
          inst_id, {Parse::NodeId::Invalid,
                    SemIR::VarStorage{SemIR::TypeId::Error,
                                      SemIR::NameId::PackageNamespace}});
      break;
  }
}

auto Context::LookupNameInDecl(Parse::NodeId /*parse_node*/,
                               SemIR::NameId name_id,
                               SemIR::NameScopeId scope_id) -> SemIR::InstId {
  if (!scope_id.is_valid()) {
    // Look for a name in the current scope only. There are two cases where the
    // name would be in an outer scope:
    //
    //  - The name is the sole component of the declared name:
    //
    //    class A;
    //    fn F() {
    //      class A;
    //    }
    //
    //    In this case, the inner A is not the same class as the outer A, so
    //    lookup should not find the outer A.
    //
    //  - The name is a qualifier of some larger declared name:
    //
    //    class A { class B; }
    //    fn F() {
    //      class A.B {}
    //    }
    //
    //    In this case, we're not in the correct scope to define a member of
    //    class A, so we should reject, and we achieve this by not finding the
    //    name A from the outer scope.
    auto& lexical_results = lexical_lookup_.Get(name_id);
    if (!lexical_results.empty()) {
      auto result = lexical_results.back();
      if (result.scope_index == current_scope_index()) {
        ResolveIfLazyImportRef(result.inst_id);
        return result.inst_id;
      }
    }
    return SemIR::InstId::Invalid;
  } else {
    // We do not look into `extend`ed scopes here. A qualified name in a
    // declaration must specify the exact scope in which the name was originally
    // introduced:
    //
    //    base class A { fn F(); }
    //    class B { extend base: A; }
    //
    //    // Error, no `F` in `B`.
    //    fn B.F() {}
    return LookupNameInExactScope(name_id, name_scopes().Get(scope_id));
  }
}

auto Context::LookupUnqualifiedName(Parse::NodeId parse_node,
                                    SemIR::NameId name_id) -> SemIR::InstId {
  // TODO: Check for shadowed lookup results.

  // Find the results from enclosing lexical scopes. These will be combined with
  // results from non-lexical scopes such as namespaces and classes.
  llvm::ArrayRef<LexicalLookup::Result> lexical_results =
      lexical_lookup_.Get(name_id);

  // Walk the non-lexical scopes and perform lookups into each of them.
  for (auto [index, name_scope_id] : llvm::reverse(non_lexical_scope_stack_)) {
    // If the innermost lexical result is within this non-lexical scope, then
    // it shadows all further non-lexical results and we're done.
    if (!lexical_results.empty() &&
        lexical_results.back().scope_index > index) {
      auto inst_id = lexical_results.back().inst_id;
      ResolveIfLazyImportRef(inst_id);
      return inst_id;
    }

    if (auto non_lexical_result =
            LookupQualifiedName(parse_node, name_id, name_scope_id,
                                /*required=*/false);
        non_lexical_result.is_valid()) {
      return non_lexical_result;
    }
  }

  if (!lexical_results.empty()) {
    auto inst_id = lexical_results.back().inst_id;
    ResolveIfLazyImportRef(inst_id);
    return inst_id;
  }

  // We didn't find anything at all.
  if (!lexical_lookup_has_load_error_) {
    DiagnoseNameNotFound(parse_node, name_id);
  }
  return SemIR::InstId::BuiltinError;
}

auto Context::LookupNameInExactScope(SemIR::NameId name_id,
                                     const SemIR::NameScope& scope)
    -> SemIR::InstId {
  if (auto it = scope.names.find(name_id); it != scope.names.end()) {
    ResolveIfLazyImportRef(it->second);
    return it->second;
  }
  return SemIR::InstId::Invalid;
}

auto Context::LookupQualifiedName(Parse::NodeId parse_node,
                                  SemIR::NameId name_id,
                                  SemIR::NameScopeId scope_id, bool required)
    -> SemIR::InstId {
  llvm::SmallVector<SemIR::NameScopeId> scope_ids = {scope_id};
  auto result_id = SemIR::InstId::Invalid;
  bool has_error = false;

  // Walk this scope and, if nothing is found here, the scopes it extends.
  while (!scope_ids.empty()) {
    const auto& scope = name_scopes().Get(scope_ids.pop_back_val());
    has_error |= scope.has_error;

    auto scope_result_id = LookupNameInExactScope(name_id, scope);
    if (!scope_result_id.is_valid()) {
      // Nothing found in this scope: also look in its extended scopes.
      auto extended = llvm::reverse(scope.extended_scopes);
      scope_ids.append(extended.begin(), extended.end());
      continue;
    }

    // If this is our second lookup result, diagnose an ambiguity.
    if (result_id.is_valid()) {
      // TODO: This is currently not reachable because the only scope that can
      // extend is a class scope, and it can only extend a single base class.
      // Add test coverage once this is possible.
      CARBON_DIAGNOSTIC(
          NameAmbiguousDueToExtend, Error,
          "Ambiguous use of name `{0}` found in multiple extended scopes.",
          std::string);
      emitter_->Emit(parse_node, NameAmbiguousDueToExtend,
                     names().GetFormatted(name_id).str());
      // TODO: Add notes pointing to the scopes.
      return SemIR::InstId::BuiltinError;
    }

    result_id = scope_result_id;
  }

  if (required && !result_id.is_valid()) {
    if (!has_error) {
      DiagnoseNameNotFound(parse_node, name_id);
    }
    return SemIR::InstId::BuiltinError;
  }

  return result_id;
}

auto Context::PushScope(SemIR::InstId scope_inst_id,
                        SemIR::NameScopeId scope_id,
                        bool lexical_lookup_has_load_error) -> void {
  scope_stack_.push_back(
      {.index = next_scope_index_,
       .scope_inst_id = scope_inst_id,
       .scope_id = scope_id,
       .prev_lexical_lookup_has_load_error = lexical_lookup_has_load_error_});
  if (scope_id.is_valid()) {
    non_lexical_scope_stack_.push_back({next_scope_index_, scope_id});
  }

  lexical_lookup_has_load_error_ |= lexical_lookup_has_load_error;

  // TODO: Handle this case more gracefully.
  CARBON_CHECK(next_scope_index_.index != std::numeric_limits<int32_t>::max())
      << "Ran out of scopes";
  ++next_scope_index_.index;
}

auto Context::PopScope() -> void {
  auto scope = scope_stack_.pop_back_val();

  lexical_lookup_has_load_error_ = scope.prev_lexical_lookup_has_load_error;

  for (const auto& str_id : scope.names) {
    auto& lexical_results = lexical_lookup_.Get(str_id);
    CARBON_CHECK(lexical_results.back().scope_index == scope.index)
        << "Inconsistent scope index for name " << names().GetFormatted(str_id);
    lexical_results.pop_back();
  }

  if (scope.scope_id.is_valid()) {
    CARBON_CHECK(non_lexical_scope_stack_.back().first == scope.index);
    non_lexical_scope_stack_.pop_back();
  }

  if (scope.has_returned_var) {
    CARBON_CHECK(!return_scope_stack_.empty());
    CARBON_CHECK(return_scope_stack_.back().returned_var.is_valid());
    return_scope_stack_.back().returned_var = SemIR::InstId::Invalid;
  }
}

auto Context::PopToScope(ScopeIndex index) -> void {
  while (current_scope_index() > index) {
    PopScope();
  }
  CARBON_CHECK(current_scope_index() == index)
      << "Scope index " << index << " does not enclose the current scope "
      << current_scope_index();
}

auto Context::SetReturnedVarOrGetExisting(SemIR::InstId inst_id)
    -> SemIR::InstId {
  CARBON_CHECK(!return_scope_stack_.empty()) << "`returned var` in no function";
  auto& returned_var = return_scope_stack_.back().returned_var;
  if (returned_var.is_valid()) {
    return returned_var;
  }

  returned_var = inst_id;
  CARBON_CHECK(!current_scope().has_returned_var)
      << "Scope has returned var but none is set";
  if (inst_id.is_valid()) {
    current_scope().has_returned_var = true;
  }
  return SemIR::InstId::Invalid;
}

auto Context::FollowNameRefs(SemIR::InstId inst_id) -> SemIR::InstId {
  while (auto name_ref = insts().Get(inst_id).TryAs<SemIR::NameRef>()) {
    inst_id = name_ref->value_id;
  }
  return inst_id;
}

template <typename BranchNode, typename... Args>
static auto AddDominatedBlockAndBranchImpl(Context& context,
                                           Parse::NodeId parse_node,
                                           Args... args) -> SemIR::InstBlockId {
  if (!context.inst_block_stack().is_current_block_reachable()) {
    return SemIR::InstBlockId::Unreachable;
  }
  auto block_id = context.inst_blocks().AddDefaultValue();
  context.AddInst({parse_node, BranchNode{block_id, args...}});
  return block_id;
}

auto Context::AddDominatedBlockAndBranch(Parse::NodeId parse_node)
    -> SemIR::InstBlockId {
  return AddDominatedBlockAndBranchImpl<SemIR::Branch>(*this, parse_node);
}

auto Context::AddDominatedBlockAndBranchWithArg(Parse::NodeId parse_node,
                                                SemIR::InstId arg_id)
    -> SemIR::InstBlockId {
  return AddDominatedBlockAndBranchImpl<SemIR::BranchWithArg>(*this, parse_node,
                                                              arg_id);
}

auto Context::AddDominatedBlockAndBranchIf(Parse::NodeId parse_node,
                                           SemIR::InstId cond_id)
    -> SemIR::InstBlockId {
  return AddDominatedBlockAndBranchImpl<SemIR::BranchIf>(*this, parse_node,
                                                         cond_id);
}

auto Context::AddConvergenceBlockAndPush(Parse::NodeId parse_node,
                                         int num_blocks) -> void {
  CARBON_CHECK(num_blocks >= 2) << "no convergence";

  SemIR::InstBlockId new_block_id = SemIR::InstBlockId::Unreachable;
  for ([[maybe_unused]] auto _ : llvm::seq(num_blocks)) {
    if (inst_block_stack().is_current_block_reachable()) {
      if (new_block_id == SemIR::InstBlockId::Unreachable) {
        new_block_id = inst_blocks().AddDefaultValue();
      }
      AddInst({parse_node, SemIR::Branch{new_block_id}});
    }
    inst_block_stack().Pop();
  }
  inst_block_stack().Push(new_block_id);
}

auto Context::AddConvergenceBlockWithArgAndPush(
    Parse::NodeId parse_node, std::initializer_list<SemIR::InstId> block_args)
    -> SemIR::InstId {
  CARBON_CHECK(block_args.size() >= 2) << "no convergence";

  SemIR::InstBlockId new_block_id = SemIR::InstBlockId::Unreachable;
  for (auto arg_id : block_args) {
    if (inst_block_stack().is_current_block_reachable()) {
      if (new_block_id == SemIR::InstBlockId::Unreachable) {
        new_block_id = inst_blocks().AddDefaultValue();
      }
      AddInst({parse_node, SemIR::BranchWithArg{new_block_id, arg_id}});
    }
    inst_block_stack().Pop();
  }
  inst_block_stack().Push(new_block_id);

  // Acquire the result value.
  SemIR::TypeId result_type_id = insts().Get(*block_args.begin()).type_id();
  return AddInst({parse_node, SemIR::BlockArg{result_type_id, new_block_id}});
}

// Add the current code block to the enclosing function.
auto Context::AddCurrentCodeBlockToFunction(Parse::NodeId parse_node) -> void {
  CARBON_CHECK(!inst_block_stack().empty()) << "no current code block";

  if (return_scope_stack().empty()) {
    CARBON_CHECK(parse_node.is_valid())
        << "No current function, but parse_node not provided";
    TODO(parse_node,
         "Control flow expressions are currently only supported inside "
         "functions.");
    return;
  }

  if (!inst_block_stack().is_current_block_reachable()) {
    // Don't include unreachable blocks in the function.
    return;
  }

  auto function_id =
      insts()
          .GetAs<SemIR::FunctionDecl>(return_scope_stack().back().decl_id)
          .function_id;
  functions()
      .Get(function_id)
      .body_block_ids.push_back(inst_block_stack().PeekOrAdd());
}

auto Context::is_current_position_reachable() -> bool {
  if (!inst_block_stack().is_current_block_reachable()) {
    return false;
  }

  // Our current position is at the end of a reachable block. That position is
  // reachable unless the previous instruction is a terminator instruction.
  auto block_contents = inst_block_stack().PeekCurrentBlockContents();
  if (block_contents.empty()) {
    return true;
  }
  const auto& last_inst = insts().Get(block_contents.back());
  return last_inst.kind().terminator_kind() !=
         SemIR::TerminatorKind::Terminator;
}

auto Context::ParamOrArgStart() -> void { params_or_args_stack_.Push(); }

auto Context::ParamOrArgComma() -> void {
  ParamOrArgSave(node_stack_.PopExpr());
}

auto Context::ParamOrArgEndNoPop(Parse::NodeKind start_kind) -> void {
  if (!node_stack_.PeekIs(start_kind)) {
    ParamOrArgSave(node_stack_.PopExpr());
  }
}

auto Context::ParamOrArgPop() -> SemIR::InstBlockId {
  return params_or_args_stack_.Pop();
}

auto Context::ParamOrArgEnd(Parse::NodeKind start_kind) -> SemIR::InstBlockId {
  ParamOrArgEndNoPop(start_kind);
  return ParamOrArgPop();
}

namespace {
// Worklist-based type completion mechanism.
//
// When attempting to complete a type, we may find other types that also need to
// be completed: types nested within that type, and the value representation of
// the type. In order to complete a type without recursing arbitrarily deeply,
// we use a worklist of tasks:
//
// - An `AddNestedIncompleteTypes` step adds a task for all incomplete types
//   nested within a type to the work list.
// - A `BuildValueRepr` step computes the value representation for a
//   type, once all of its nested types are complete, and marks the type as
//   complete.
class TypeCompleter {
 public:
  TypeCompleter(
      Context& context,
      std::optional<llvm::function_ref<auto()->Context::DiagnosticBuilder>>
          diagnoser)
      : context_(context), diagnoser_(diagnoser) {}

  // Attempts to complete the given type. Returns true if it is now complete,
  // false if it could not be completed.
  auto Complete(SemIR::TypeId type_id) -> bool {
    Push(type_id);
    while (!work_list_.empty()) {
      if (!ProcessStep()) {
        return false;
      }
    }
    return true;
  }

 private:
  // Adds `type_id` to the work list, if it's not already complete.
  auto Push(SemIR::TypeId type_id) -> void {
    if (!context_.types().IsComplete(type_id)) {
      work_list_.push_back({type_id, Phase::AddNestedIncompleteTypes});
    }
  }

  // Runs the next step.
  auto ProcessStep() -> bool {
    auto [type_id, phase] = work_list_.back();

    // We might have enqueued the same type more than once. Just skip the
    // type if it's already complete.
    if (context_.types().IsComplete(type_id)) {
      work_list_.pop_back();
      return true;
    }

    auto inst_id = context_.types().GetInstId(type_id);
    auto inst = context_.insts().Get(inst_id);
    auto old_work_list_size = work_list_.size();

    switch (phase) {
      case Phase::AddNestedIncompleteTypes:
        if (!AddNestedIncompleteTypes(inst)) {
          return false;
        }
        CARBON_CHECK(work_list_.size() >= old_work_list_size)
            << "AddNestedIncompleteTypes should not remove work items";
        work_list_[old_work_list_size - 1].phase = Phase::BuildValueRepr;
        break;

      case Phase::BuildValueRepr: {
        auto value_rep = BuildValueRepr(type_id, inst);
        context_.sem_ir().CompleteType(type_id, value_rep);
        CARBON_CHECK(old_work_list_size == work_list_.size())
            << "BuildValueRepr should not change work items";
        work_list_.pop_back();

        // Also complete the value representation type, if necessary. This
        // should never fail: the value representation shouldn't require any
        // additional nested types to be complete.
        if (!context_.types().IsComplete(value_rep.type_id)) {
          work_list_.push_back({value_rep.type_id, Phase::BuildValueRepr});
        }
        // For a pointer representation, the pointee also needs to be complete.
        if (value_rep.kind == SemIR::ValueRepr::Pointer) {
          auto pointee_type_id =
              context_.sem_ir().GetPointeeType(value_rep.type_id);
          if (!context_.types().IsComplete(pointee_type_id)) {
            work_list_.push_back({pointee_type_id, Phase::BuildValueRepr});
          }
        }
        break;
      }
    }

    return true;
  }

  // Adds any types nested within `type_inst` that need to be complete for
  // `type_inst` to be complete to our work list.
  auto AddNestedIncompleteTypes(SemIR::Inst type_inst) -> bool {
    switch (type_inst.kind()) {
      case SemIR::ArrayType::Kind:
        Push(type_inst.As<SemIR::ArrayType>().element_type_id);
        break;

      case SemIR::StructType::Kind:
        for (auto field_id : context_.inst_blocks().Get(
                 type_inst.As<SemIR::StructType>().fields_id)) {
          Push(context_.insts()
                   .GetAs<SemIR::StructTypeField>(field_id)
                   .field_type_id);
        }
        break;

      case SemIR::TupleType::Kind:
        for (auto element_type_id : context_.type_blocks().Get(
                 type_inst.As<SemIR::TupleType>().elements_id)) {
          Push(element_type_id);
        }
        break;

      case SemIR::ClassType::Kind: {
        auto class_type = type_inst.As<SemIR::ClassType>();
        auto& class_info = context_.classes().Get(class_type.class_id);
        if (!class_info.is_defined()) {
          if (diagnoser_) {
            auto builder = (*diagnoser_)();
            context_.NoteIncompleteClass(class_type.class_id, builder);
            builder.Emit();
          }
          return false;
        }
        Push(class_info.object_repr_id);
        break;
      }

      case SemIR::ConstType::Kind:
        Push(type_inst.As<SemIR::ConstType>().inner_id);
        break;

      default:
        break;
    }

    return true;
  }

  // Makes an empty value representation, which is used for types that have no
  // state, such as empty structs and tuples.
  auto MakeEmptyValueRepr() const -> SemIR::ValueRepr {
    return {.kind = SemIR::ValueRepr::None,
            .type_id = context_.CanonicalizeTupleType({})};
  }

  // Makes a value representation that uses pass-by-copy, copying the given
  // type.
  auto MakeCopyValueRepr(SemIR::TypeId rep_id,
                         SemIR::ValueRepr::AggregateKind aggregate_kind =
                             SemIR::ValueRepr::NotAggregate) const
      -> SemIR::ValueRepr {
    return {.kind = SemIR::ValueRepr::Copy,
            .aggregate_kind = aggregate_kind,
            .type_id = rep_id};
  }

  // Makes a value representation that uses pass-by-address with the given
  // pointee type.
  auto MakePointerValueRepr(SemIR::TypeId pointee_id,
                            SemIR::ValueRepr::AggregateKind aggregate_kind =
                                SemIR::ValueRepr::NotAggregate) const
      -> SemIR::ValueRepr {
    // TODO: Should we add `const` qualification to `pointee_id`?
    return {.kind = SemIR::ValueRepr::Pointer,
            .aggregate_kind = aggregate_kind,
            .type_id = context_.GetPointerType(pointee_id)};
  }

  // Gets the value representation of a nested type, which should already be
  // complete.
  auto GetNestedValueRepr(SemIR::TypeId nested_type_id) const {
    CARBON_CHECK(context_.types().IsComplete(nested_type_id))
        << "Nested type should already be complete";
    auto value_rep = context_.types().GetValueRepr(nested_type_id);
    CARBON_CHECK(value_rep.kind != SemIR::ValueRepr::Unknown)
        << "Complete type should have a value representation";
    return value_rep;
  };

  auto BuildCrossRefValueRepr(SemIR::TypeId type_id, SemIR::CrossRef xref) const
      -> SemIR::ValueRepr {
    auto xref_inst =
        context_.cross_ref_irs().Get(xref.ir_id)->insts().Get(xref.inst_id);

    // The canonical description of a type should only have cross-references
    // for entities owned by another File, such as builtins, which are owned
    // by the prelude, and named entities like classes and interfaces, which
    // we don't support yet.
    CARBON_CHECK(xref_inst.kind() == SemIR::Builtin::Kind)
        << "TODO: Handle other kinds of inst cross-references";

    // clang warns on unhandled enum values; clang-tidy is incorrect here.
    // NOLINTNEXTLINE(bugprone-switch-missing-default-case)
    switch (xref_inst.As<SemIR::Builtin>().builtin_kind) {
      case SemIR::BuiltinKind::TypeType:
      case SemIR::BuiltinKind::Error:
      case SemIR::BuiltinKind::Invalid:
      case SemIR::BuiltinKind::BoolType:
      case SemIR::BuiltinKind::IntType:
      case SemIR::BuiltinKind::FloatType:
      case SemIR::BuiltinKind::NamespaceType:
      case SemIR::BuiltinKind::FunctionType:
      case SemIR::BuiltinKind::BoundMethodType:
        return MakeCopyValueRepr(type_id);

      case SemIR::BuiltinKind::StringType:
        // TODO: Decide on string value semantics. This should probably be a
        // custom value representation carrying a pointer and size or
        // similar.
        return MakePointerValueRepr(type_id);
    }
    llvm_unreachable("All builtin kinds were handled above");
  }

  auto BuildStructOrTupleValueRepr(std::size_t num_elements,
                                   SemIR::TypeId elementwise_rep,
                                   bool same_as_object_rep) const
      -> SemIR::ValueRepr {
    SemIR::ValueRepr::AggregateKind aggregate_kind =
        same_as_object_rep ? SemIR::ValueRepr::ValueAndObjectAggregate
                           : SemIR::ValueRepr::ValueAggregate;

    if (num_elements == 1) {
      // The value representation for a struct or tuple with a single element
      // is a struct or tuple containing the value representation of the
      // element.
      // TODO: Consider doing the same whenever `elementwise_rep` is
      // sufficiently small.
      return MakeCopyValueRepr(elementwise_rep, aggregate_kind);
    }
    // For a struct or tuple with multiple fields, we use a pointer
    // to the elementwise value representation.
    return MakePointerValueRepr(elementwise_rep, aggregate_kind);
  }

  auto BuildStructTypeValueRepr(SemIR::TypeId type_id,
                                SemIR::StructType struct_type) const
      -> SemIR::ValueRepr {
    // TODO: Share more code with tuples.
    auto fields = context_.inst_blocks().Get(struct_type.fields_id);
    if (fields.empty()) {
      return MakeEmptyValueRepr();
    }

    // Find the value representation for each field, and construct a struct
    // of value representations.
    llvm::SmallVector<SemIR::InstId> value_rep_fields;
    value_rep_fields.reserve(fields.size());
    bool same_as_object_rep = true;
    for (auto field_id : fields) {
      auto field = context_.insts().GetAs<SemIR::StructTypeField>(field_id);
      auto field_value_rep = GetNestedValueRepr(field.field_type_id);
      if (field_value_rep.type_id != field.field_type_id) {
        same_as_object_rep = false;
        field.field_type_id = field_value_rep.type_id;
        // TODO: Use `TryEvalInst` to form this value.
        field_id = context_
                       .AddConstant(field, context_.constant_values()
                                               .Get(context_.types().GetInstId(
                                                   field.field_type_id))
                                               .is_symbolic())
                       .inst_id();
      }
      value_rep_fields.push_back(field_id);
    }

    auto value_rep = same_as_object_rep
                         ? type_id
                         : context_.CanonicalizeStructType(
                               context_.inst_blocks().Add(value_rep_fields));
    return BuildStructOrTupleValueRepr(fields.size(), value_rep,
                                       same_as_object_rep);
  }

  auto BuildTupleTypeValueRepr(SemIR::TypeId type_id,
                               SemIR::TupleType tuple_type) const
      -> SemIR::ValueRepr {
    // TODO: Share more code with structs.
    auto elements = context_.type_blocks().Get(tuple_type.elements_id);
    if (elements.empty()) {
      return MakeEmptyValueRepr();
    }

    // Find the value representation for each element, and construct a tuple
    // of value representations.
    llvm::SmallVector<SemIR::TypeId> value_rep_elements;
    value_rep_elements.reserve(elements.size());
    bool same_as_object_rep = true;
    for (auto element_type_id : elements) {
      auto element_value_rep = GetNestedValueRepr(element_type_id);
      if (element_value_rep.type_id != element_type_id) {
        same_as_object_rep = false;
      }
      value_rep_elements.push_back(element_value_rep.type_id);
    }

    auto value_rep = same_as_object_rep
                         ? type_id
                         : context_.CanonicalizeTupleType(value_rep_elements);
    return BuildStructOrTupleValueRepr(elements.size(), value_rep,
                                       same_as_object_rep);
  }

  // Builds and returns the value representation for the given type. All nested
  // types, as found by AddNestedIncompleteTypes, are known to be complete.
  auto BuildValueRepr(SemIR::TypeId type_id, SemIR::Inst inst) const
      -> SemIR::ValueRepr {
    // TODO: This can emit new SemIR instructions. Consider emitting them into a
    // dedicated file-scope instruction block where possible, or somewhere else
    // that better reflects the definition of the type, rather than wherever the
    // type happens to first be required to be complete.

    // clang warns on unhandled enum values; clang-tidy is incorrect here.
    // NOLINTNEXTLINE(bugprone-switch-missing-default-case)
    switch (inst.kind()) {
      case SemIR::AddrOf::Kind:
      case SemIR::AddrPattern::Kind:
      case SemIR::ArrayIndex::Kind:
      case SemIR::ArrayInit::Kind:
      case SemIR::Assign::Kind:
      case SemIR::BaseDecl::Kind:
      case SemIR::BindName::Kind:
      case SemIR::BindValue::Kind:
      case SemIR::BlockArg::Kind:
      case SemIR::BoolLiteral::Kind:
      case SemIR::BoundMethod::Kind:
      case SemIR::Branch::Kind:
      case SemIR::BranchIf::Kind:
      case SemIR::BranchWithArg::Kind:
      case SemIR::Call::Kind:
      case SemIR::ClassDecl::Kind:
      case SemIR::ClassElementAccess::Kind:
      case SemIR::ClassInit::Kind:
      case SemIR::Converted::Kind:
      case SemIR::Deref::Kind:
      case SemIR::FieldDecl::Kind:
      case SemIR::FunctionDecl::Kind:
      case SemIR::Import::Kind:
      case SemIR::InitializeFrom::Kind:
      case SemIR::InterfaceDecl::Kind:
      case SemIR::IntLiteral::Kind:
      case SemIR::LazyImportRef::Kind:
      case SemIR::NameRef::Kind:
      case SemIR::Namespace::Kind:
      case SemIR::Param::Kind:
      case SemIR::RealLiteral::Kind:
      case SemIR::Return::Kind:
      case SemIR::ReturnExpr::Kind:
      case SemIR::SpliceBlock::Kind:
      case SemIR::StringLiteral::Kind:
      case SemIR::StructAccess::Kind:
      case SemIR::StructTypeField::Kind:
      case SemIR::StructLiteral::Kind:
      case SemIR::StructInit::Kind:
      case SemIR::StructValue::Kind:
      case SemIR::Temporary::Kind:
      case SemIR::TemporaryStorage::Kind:
      case SemIR::TupleAccess::Kind:
      case SemIR::TupleIndex::Kind:
      case SemIR::TupleLiteral::Kind:
      case SemIR::TupleInit::Kind:
      case SemIR::TupleValue::Kind:
      case SemIR::UnaryOperatorNot::Kind:
      case SemIR::ValueAsRef::Kind:
      case SemIR::ValueOfInitializer::Kind:
      case SemIR::VarStorage::Kind:
        CARBON_FATAL() << "Type refers to non-type inst " << inst;

      case SemIR::CrossRef::Kind:
        return BuildCrossRefValueRepr(type_id, inst.As<SemIR::CrossRef>());

      case SemIR::ArrayType::Kind: {
        // For arrays, it's convenient to always use a pointer representation,
        // even when the array has zero or one element, in order to support
        // indexing.
        return MakePointerValueRepr(type_id, SemIR::ValueRepr::ObjectAggregate);
      }

      case SemIR::StructType::Kind:
        return BuildStructTypeValueRepr(type_id, inst.As<SemIR::StructType>());

      case SemIR::TupleType::Kind:
        return BuildTupleTypeValueRepr(type_id, inst.As<SemIR::TupleType>());

      case SemIR::ClassType::Kind:
        // The value representation for a class is a pointer to the object
        // representation.
        // TODO: Support customized value representations for classes.
        // TODO: Pick a better value representation when possible.
        return MakePointerValueRepr(
            context_.classes()
                .Get(inst.As<SemIR::ClassType>().class_id)
                .object_repr_id,
            SemIR::ValueRepr::ObjectAggregate);

      case SemIR::Builtin::Kind:
        CARBON_FATAL() << "Builtins should be named as cross-references";

      case SemIR::BindSymbolicName::Kind:
      case SemIR::PointerType::Kind:
      case SemIR::UnboundElementType::Kind:
        return MakeCopyValueRepr(type_id);

      case SemIR::ConstType::Kind:
        // The value representation of `const T` is the same as that of `T`.
        // Objects are not modifiable through their value representations.
        return GetNestedValueRepr(inst.As<SemIR::ConstType>().inner_id);
    }
  }

  enum class Phase : int8_t {
    // The next step is to add nested types to the list of types to complete.
    AddNestedIncompleteTypes,
    // The next step is to build the value representation for the type.
    BuildValueRepr,
  };

  struct WorkItem {
    SemIR::TypeId type_id;
    Phase phase;
  };

  Context& context_;
  llvm::SmallVector<WorkItem> work_list_;
  std::optional<llvm::function_ref<auto()->Context::DiagnosticBuilder>>
      diagnoser_;
};
}  // namespace

auto Context::TryToCompleteType(
    SemIR::TypeId type_id,
    std::optional<llvm::function_ref<auto()->DiagnosticBuilder>> diagnoser)
    -> bool {
  return TypeCompleter(*this, diagnoser).Complete(type_id);
}

auto Context::CanonicalizeTypeImpl(
    SemIR::InstKind kind,
    llvm::function_ref<bool(llvm::FoldingSetNodeID& canonical_id)> profile_type,
    llvm::function_ref<SemIR::InstId()> make_inst) -> SemIR::TypeId {
  llvm::FoldingSetNodeID canonical_id;
  kind.Profile(canonical_id);
  if (!profile_type(canonical_id)) {
    return SemIR::TypeId::Error;
  }

  void* insert_pos;
  auto* node =
      canonical_type_nodes_.FindNodeOrInsertPos(canonical_id, insert_pos);
  if (node != nullptr) {
    return node->type_id();
  }

  auto inst_id = make_inst();
  auto type_id = types().Add({.inst_id = inst_id});
  CARBON_CHECK(canonical_types_.insert({inst_id, type_id}).second);
  type_node_storage_.push_back(
      std::make_unique<TypeNode>(canonical_id, type_id));

  // In a debug build, check that our insertion position is still valid. It
  // could have been invalidated by a misbehaving `make_inst`.
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

// Compute a fingerprint for a type, for use as a key in a folding set. Returns
// false if not supported, which is presently the case for compile-time
// expressions.
// TODO: Once support is more complete, in particular ensuring that various
// valid compile-time expressions are supported, it may be desirable to switch
// the default to a CARBON_FATAL error.
static auto ProfileType(Context& semantics_context, SemIR::Inst inst,
                        llvm::FoldingSetNodeID& canonical_id) -> bool {
  switch (inst.kind()) {
    case SemIR::ArrayType::Kind: {
      auto array_type = inst.As<SemIR::ArrayType>();
      canonical_id.AddInteger(
          semantics_context.sem_ir().GetArrayBoundValue(array_type.bound_id));
      canonical_id.AddInteger(array_type.element_type_id.index);
      break;
    }
    case SemIR::Builtin::Kind:
      canonical_id.AddInteger(inst.As<SemIR::Builtin>().builtin_kind.AsInt());
      break;
    case SemIR::ClassType::Kind:
      canonical_id.AddInteger(inst.As<SemIR::ClassType>().class_id.index);
      break;
    case SemIR::CrossRef::Kind: {
      // TODO: Cross-references should be canonicalized by looking at their
      // target rather than treating them as new unique types.
      auto xref = inst.As<SemIR::CrossRef>();
      canonical_id.AddInteger(xref.ir_id.index);
      canonical_id.AddInteger(xref.inst_id.index);
      break;
    }
    case SemIR::ConstType::Kind:
      canonical_id.AddInteger(
          semantics_context
              .GetUnqualifiedType(inst.As<SemIR::ConstType>().inner_id)
              .index);
      break;
    case SemIR::BindSymbolicName::Kind:
      // TODO: Use de Bruijn levels or similar to identify equivalent type
      // bindings across redeclarations.
      canonical_id.AddInteger(
          inst.As<SemIR::BindSymbolicName>().bind_name_id.index);
      break;
    case SemIR::PointerType::Kind:
      canonical_id.AddInteger(inst.As<SemIR::PointerType>().pointee_id.index);
      break;
    case SemIR::StructType::Kind: {
      auto fields = semantics_context.inst_blocks().Get(
          inst.As<SemIR::StructType>().fields_id);
      for (const auto& field_id : fields) {
        auto field =
            semantics_context.insts().GetAs<SemIR::StructTypeField>(field_id);
        canonical_id.AddInteger(field.name_id.index);
        canonical_id.AddInteger(field.field_type_id.index);
      }
      break;
    }
    case SemIR::TupleType::Kind:
      ProfileTupleType(semantics_context.type_blocks().Get(
                           inst.As<SemIR::TupleType>().elements_id),
                       canonical_id);
      break;
    case SemIR::UnboundElementType::Kind: {
      auto unbound_field_type = inst.As<SemIR::UnboundElementType>();
      canonical_id.AddInteger(unbound_field_type.class_type_id.index);
      canonical_id.AddInteger(unbound_field_type.element_type_id.index);
      break;
    }
    default: {
      // Right now, this is only expected to occur in calls from
      // ExprAsType. Diagnostics are issued there.
      return false;
    }
  }
  return true;
}

auto Context::CanonicalizeTypeAndAddInstIfNew(SemIR::Inst inst)
    -> SemIR::TypeId {
  auto profile_node = [&](llvm::FoldingSetNodeID& canonical_id) {
    return ProfileType(*this, inst, canonical_id);
  };
  auto make_inst = [&] {
    // TODO: Properly determine whether types are symbolic.
    return AddConstant(inst, /*is_symbolic=*/false).inst_id();
  };
  return CanonicalizeTypeImpl(inst.kind(), profile_node, make_inst);
}

auto Context::CanonicalizeType(SemIR::InstId inst_id) -> SemIR::TypeId {
  while (auto converted = insts().Get(inst_id).TryAs<SemIR::Converted>()) {
    inst_id = converted->result_id;
  }
  inst_id = FollowNameRefs(inst_id);

  auto it = canonical_types_.find(inst_id);
  if (it != canonical_types_.end()) {
    return it->second;
  }

  auto inst = insts().Get(inst_id);
  auto profile_node = [&](llvm::FoldingSetNodeID& canonical_id) {
    return ProfileType(*this, inst, canonical_id);
  };
  auto make_inst = [&] { return inst_id; };
  return CanonicalizeTypeImpl(inst.kind(), profile_node, make_inst);
}

auto Context::CanonicalizeStructType(SemIR::InstBlockId refs_id)
    -> SemIR::TypeId {
  return CanonicalizeTypeAndAddInstIfNew(
      SemIR::StructType{SemIR::TypeId::TypeType, refs_id});
}

auto Context::CanonicalizeTupleType(llvm::ArrayRef<SemIR::TypeId> type_ids)
    -> SemIR::TypeId {
  // Defer allocating a SemIR::TypeBlockId until we know this is a new type.
  auto profile_tuple = [&](llvm::FoldingSetNodeID& canonical_id) {
    ProfileTupleType(type_ids, canonical_id);
    return true;
  };
  auto make_tuple_inst = [&] {
    // TODO: Properly determine when types are symbolic.
    return AddConstant(SemIR::TupleType{SemIR::TypeId::TypeType,
                                        type_blocks().Add(type_ids)},
                       /*is_symbolic=*/false)
        .inst_id();
  };
  return CanonicalizeTypeImpl(SemIR::TupleType::Kind, profile_tuple,
                              make_tuple_inst);
}

auto Context::GetBuiltinType(SemIR::BuiltinKind kind) -> SemIR::TypeId {
  CARBON_CHECK(kind != SemIR::BuiltinKind::Invalid);
  auto type_id = CanonicalizeType(SemIR::InstId::ForBuiltin(kind));
  // To keep client code simpler, complete builtin types before returning them.
  bool complete = TryToCompleteType(type_id);
  CARBON_CHECK(complete) << "Failed to complete builtin type";
  return type_id;
}

auto Context::GetPointerType(SemIR::TypeId pointee_type_id) -> SemIR::TypeId {
  return CanonicalizeTypeAndAddInstIfNew(
      SemIR::PointerType{SemIR::TypeId::TypeType, pointee_type_id});
}

auto Context::GetUnqualifiedType(SemIR::TypeId type_id) -> SemIR::TypeId {
  if (auto const_type = types().TryGetAs<SemIR::ConstType>(type_id)) {
    return const_type->inner_id;
  }
  return type_id;
}

auto Context::PrintForStackDump(llvm::raw_ostream& output) const -> void {
  node_stack_.PrintForStackDump(output);
  inst_block_stack_.PrintForStackDump(output);
  params_or_args_stack_.PrintForStackDump(output);
  args_type_info_stack_.PrintForStackDump(output);
}

}  // namespace Carbon::Check
