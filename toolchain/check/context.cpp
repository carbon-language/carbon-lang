// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

#include <string>
#include <utility>

#include "common/check.h"
#include "common/vlog.h"
#include "llvm/ADT/Sequence.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/decl_name_stack.h"
#include "toolchain/check/eval.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/inst_block_stack.h"
#include "toolchain/check/merge.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/sem_ir/builtin_kind.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/inst_kind.h"
#include "toolchain/sem_ir/typed_insts.h"

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
      param_and_arg_refs_stack_(sem_ir, vlog_stream, node_stack_),
      args_type_info_stack_("args_type_info_stack_", sem_ir, vlog_stream),
      decl_name_stack_(this),
      scope_stack_(sem_ir_->identifiers()) {
  // Map the builtin `<error>` and `type` type constants to their corresponding
  // special `TypeId` values.
  type_ids_for_type_constants_.insert(
      {SemIR::ConstantId::ForTemplateConstant(SemIR::InstId::BuiltinError),
       SemIR::TypeId::Error});
  type_ids_for_type_constants_.insert(
      {SemIR::ConstantId::ForTemplateConstant(SemIR::InstId::BuiltinTypeType),
       SemIR::TypeId::TypeType});
}

auto Context::TODO(SemIRLoc loc, std::string label) -> bool {
  CARBON_DIAGNOSTIC(SemanticsTodo, Error, "Semantics TODO: `{0}`.",
                    std::string);
  emitter_->Emit(loc, SemanticsTodo, std::move(label));
  return false;
}

auto Context::VerifyOnFinish() -> void {
  // Information in all the various context objects should be cleaned up as
  // various pieces of context go out of scope. At this point, nothing should
  // remain.
  // node_stack_ will still contain top-level entities.
  scope_stack_.VerifyOnFinish();
  inst_block_stack_.VerifyOnFinish();
  param_and_arg_refs_stack_.VerifyOnFinish();
}

auto Context::AddInstInNoBlock(SemIR::LocIdAndInst loc_id_and_inst)
    -> SemIR::InstId {
  auto inst_id = sem_ir().insts().AddInNoBlock(loc_id_and_inst);
  CARBON_VLOG() << "AddInst: " << loc_id_and_inst.inst << "\n";

  auto const_id = TryEvalInst(*this, inst_id, loc_id_and_inst.inst);
  if (const_id.is_constant()) {
    CARBON_VLOG() << "Constant: " << loc_id_and_inst.inst << " -> "
                  << const_id.inst_id() << "\n";
    constant_values().Set(inst_id, const_id);
  }

  return inst_id;
}

auto Context::AddInst(SemIR::LocIdAndInst loc_id_and_inst) -> SemIR::InstId {
  auto inst_id = AddInstInNoBlock(loc_id_and_inst);
  inst_block_stack_.AddInstId(inst_id);
  return inst_id;
}

auto Context::AddPlaceholderInstInNoBlock(SemIR::LocIdAndInst loc_id_and_inst)
    -> SemIR::InstId {
  auto inst_id = sem_ir().insts().AddInNoBlock(loc_id_and_inst);
  CARBON_VLOG() << "AddPlaceholderInst: " << loc_id_and_inst.inst << "\n";
  constant_values().Set(inst_id, SemIR::ConstantId::Invalid);
  return inst_id;
}

auto Context::AddPlaceholderInst(SemIR::LocIdAndInst loc_id_and_inst)
    -> SemIR::InstId {
  auto inst_id = AddPlaceholderInstInNoBlock(loc_id_and_inst);
  inst_block_stack_.AddInstId(inst_id);
  return inst_id;
}

auto Context::AddConstant(SemIR::Inst inst, bool is_symbolic)
    -> SemIR::ConstantId {
  auto const_id = constants().GetOrAdd(inst, is_symbolic);
  CARBON_VLOG() << "AddConstant: " << inst << "\n";
  return const_id;
}

auto Context::AddInstAndPush(SemIR::LocIdAndInst loc_id_and_inst) -> void {
  auto inst_id = AddInst(loc_id_and_inst);
  node_stack_.Push(loc_id_and_inst.loc_id.node_id(), inst_id);
}

auto Context::ReplaceLocIdAndInstBeforeConstantUse(
    SemIR::InstId inst_id, SemIR::LocIdAndInst loc_id_and_inst) -> void {
  sem_ir().insts().SetLocIdAndInst(inst_id, loc_id_and_inst);

  CARBON_VLOG() << "ReplaceInst: " << inst_id << " -> " << loc_id_and_inst.inst
                << "\n";

  // Redo evaluation. This is only safe to do if this instruction has not
  // already been used as a constant, which is the caller's responsibility to
  // ensure.
  auto const_id = TryEvalInst(*this, inst_id, loc_id_and_inst.inst);
  if (const_id.is_constant()) {
    CARBON_VLOG() << "Constant: " << loc_id_and_inst.inst << " -> "
                  << const_id.inst_id() << "\n";
  }
  constant_values().Set(inst_id, const_id);
}

auto Context::ReplaceInstBeforeConstantUse(SemIR::InstId inst_id,
                                           SemIR::Inst inst) -> void {
  sem_ir().insts().Set(inst_id, inst);

  CARBON_VLOG() << "ReplaceInst: " << inst_id << " -> " << inst << "\n";

  // Redo evaluation. This is only safe to do if this instruction has not
  // already been used as a constant, which is the caller's responsibility to
  // ensure.
  auto const_id = TryEvalInst(*this, inst_id, inst);
  if (const_id.is_constant()) {
    CARBON_VLOG() << "Constant: " << inst << " -> " << const_id.inst_id()
                  << "\n";
  }
  constant_values().Set(inst_id, const_id);
}

auto Context::AddImportRef(SemIR::ImportIRInst import_ir_inst)
    -> SemIR::InstId {
  auto import_ref_id = AddPlaceholderInstInNoBlock(
      SemIR::ImportRefUnloaded{import_ir_insts().Add(import_ir_inst)});

  // We can't insert this instruction into whatever block we happen to be in,
  // because this function is typically called by name lookup in the middle of
  // an otherwise unknown checking step. But we need to add the instruction
  // somewhere, because it's referenced by other instructions and needs to be
  // visible in textual IR. Adding it to the file block is arbitrary but is the
  // best place we have right now.
  //
  // TODO: Consider adding a dedicated block for import_refs.
  inst_block_stack().AddInstIdToFileBlock(import_ref_id);
  return import_ref_id;
}

auto Context::DiagnoseDuplicateName(SemIRLoc dup_def, SemIRLoc prev_def)
    -> void {
  CARBON_DIAGNOSTIC(NameDeclDuplicate, Error,
                    "Duplicate name being declared in the same scope.");
  CARBON_DIAGNOSTIC(NameDeclPrevious, Note,
                    "Name is previously declared here.");
  emitter_->Build(dup_def, NameDeclDuplicate)
      .Note(prev_def, NameDeclPrevious)
      .Emit();
}

auto Context::DiagnoseNameNotFound(SemIRLoc loc, SemIR::NameId name_id)
    -> void {
  CARBON_DIAGNOSTIC(NameNotFound, Error, "Name `{0}` not found.",
                    SemIR::NameId);
  emitter_->Emit(loc, NameNotFound, name_id);
}

auto Context::NoteIncompleteClass(SemIR::ClassId class_id,
                                  DiagnosticBuilder& builder) -> void {
  const auto& class_info = classes().Get(class_id);
  CARBON_CHECK(!class_info.is_defined()) << "Class is not incomplete";
  if (class_info.definition_id.is_valid()) {
    CARBON_DIAGNOSTIC(ClassIncompleteWithinDefinition, Note,
                      "Class is incomplete within its definition.");
    builder.Note(class_info.definition_id, ClassIncompleteWithinDefinition);
  } else {
    CARBON_DIAGNOSTIC(ClassForwardDeclaredHere, Note,
                      "Class was forward declared here.");
    builder.Note(class_info.decl_id, ClassForwardDeclaredHere);
  }
}

auto Context::NoteUndefinedInterface(SemIR::InterfaceId interface_id,
                                     DiagnosticBuilder& builder) -> void {
  const auto& interface_info = interfaces().Get(interface_id);
  CARBON_CHECK(!interface_info.is_defined()) << "Interface is not incomplete";
  if (interface_info.is_being_defined()) {
    CARBON_DIAGNOSTIC(InterfaceUndefinedWithinDefinition, Note,
                      "Interface is currently being defined.");
    builder.Note(interface_info.definition_id,
                 InterfaceUndefinedWithinDefinition);
  } else {
    CARBON_DIAGNOSTIC(InterfaceForwardDeclaredHere, Note,
                      "Interface was forward declared here.");
    builder.Note(interface_info.decl_id, InterfaceForwardDeclaredHere);
  }
}

auto Context::AddNameToLookup(SemIR::NameId name_id, SemIR::InstId target_id)
    -> void {
  if (auto existing = scope_stack().LookupOrAddName(name_id, target_id);
      existing.is_valid()) {
    DiagnoseDuplicateName(target_id, existing);
  }
}

auto Context::LookupNameInDecl(SemIR::LocId loc_id, SemIR::NameId name_id,
                               SemIR::NameScopeId scope_id,
                               bool mark_imports_used) -> SemIR::InstId {
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
    return scope_stack().LookupInCurrentScope(name_id);
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
    return LookupNameInExactScope(loc_id, name_id, name_scopes().Get(scope_id),
                                  mark_imports_used);
  }
}

auto Context::LookupUnqualifiedName(Parse::NodeId node_id,
                                    SemIR::NameId name_id) -> SemIR::InstId {
  // TODO: Check for shadowed lookup results.

  // Find the results from enclosing lexical scopes. These will be combined with
  // results from non-lexical scopes such as namespaces and classes.
  auto [lexical_result, non_lexical_scopes] =
      scope_stack().LookupInEnclosingScopes(name_id);

  // Walk the non-lexical scopes and perform lookups into each of them.
  for (auto [index, name_scope_id] : llvm::reverse(non_lexical_scopes)) {
    if (auto non_lexical_result =
            LookupQualifiedName(node_id, name_id, name_scope_id,
                                /*required=*/false);
        non_lexical_result.is_valid()) {
      return non_lexical_result;
    }
  }

  if (lexical_result.is_valid()) {
    return lexical_result;
  }

  // We didn't find anything at all.
  DiagnoseNameNotFound(node_id, name_id);
  return SemIR::InstId::BuiltinError;
}

// Handles lookup through the import_ir_scopes for LookupNameInExactScope.
static auto LookupInImportIRScopes(Context& context, SemIRLoc loc,
                                   SemIR::NameId name_id,
                                   const SemIR::NameScope& scope,
                                   bool mark_imports_used) -> SemIR::InstId {
  auto identifier_id = name_id.AsIdentifierId();
  llvm::StringRef identifier;
  if (identifier_id.is_valid()) {
    identifier = context.identifiers().Get(identifier_id);
  }

  DiagnosticAnnotationScope annotate_diagnostics(
      &context.emitter(), [&](auto& builder) {
        CARBON_DIAGNOSTIC(InNameLookup, Note, "In name lookup for `{0}`.",
                          SemIR::NameId);
        builder.Note(loc, InNameLookup, name_id);
      });

  auto result_id = SemIR::InstId::Invalid;
  for (auto [import_ir_id, import_scope_id] : scope.import_ir_scopes) {
    auto& import_ir = context.import_irs().Get(import_ir_id);

    // Determine the NameId in the import IR.
    SemIR::NameId import_name_id = name_id;
    if (identifier_id.is_valid()) {
      auto import_identifier_id =
          import_ir.sem_ir->identifiers().Lookup(identifier);
      if (!import_identifier_id.is_valid()) {
        // Name doesn't exist in the import IR.
        continue;
      }
      import_name_id = SemIR::NameId::ForIdentifier(import_identifier_id);
    }

    // Look up the name in the import scope.
    const auto& import_scope =
        import_ir.sem_ir->name_scopes().Get(import_scope_id);
    auto it = import_scope.names.find(import_name_id);
    if (it == import_scope.names.end()) {
      // Name doesn't exist in the import scope.
      continue;
    }
    auto import_inst_id =
        context.AddImportRef({.ir_id = import_ir_id, .inst_id = it->second});
    if (result_id.is_valid()) {
      MergeImportRef(context, import_inst_id, result_id);
    } else {
      LoadImportRef(context, import_inst_id,
                    mark_imports_used ? loc : SemIR::LocId::Invalid);
      result_id = import_inst_id;
    }
  }

  return result_id;
}

auto Context::LookupNameInExactScope(SemIRLoc loc, SemIR::NameId name_id,
                                     const SemIR::NameScope& scope,
                                     bool mark_imports_used) -> SemIR::InstId {
  if (auto it = scope.names.find(name_id); it != scope.names.end()) {
    LoadImportRef(*this, it->second,
                  mark_imports_used ? loc : SemIR::LocId::Invalid);
    return it->second;
  }
  if (!scope.import_ir_scopes.empty()) {
    return LookupInImportIRScopes(*this, loc, name_id, scope,
                                  mark_imports_used);
  }
  return SemIR::InstId::Invalid;
}

auto Context::LookupQualifiedName(Parse::NodeId node_id, SemIR::NameId name_id,
                                  SemIR::NameScopeId scope_id, bool required)
    -> SemIR::InstId {
  llvm::SmallVector<SemIR::NameScopeId> scope_ids = {scope_id};
  auto result_id = SemIR::InstId::Invalid;
  bool has_error = false;

  // Walk this scope and, if nothing is found here, the scopes it extends.
  while (!scope_ids.empty()) {
    const auto& scope = name_scopes().Get(scope_ids.pop_back_val());
    has_error |= scope.has_error;

    auto scope_result_id = LookupNameInExactScope(node_id, name_id, scope,
                                                  /*mark_imports_used=*/true);
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
          SemIR::NameId);
      emitter_->Emit(node_id, NameAmbiguousDueToExtend, name_id);
      // TODO: Add notes pointing to the scopes.
      return SemIR::InstId::BuiltinError;
    }

    result_id = scope_result_id;
  }

  if (required && !result_id.is_valid()) {
    if (!has_error) {
      DiagnoseNameNotFound(node_id, name_id);
    }
    return SemIR::InstId::BuiltinError;
  }

  return result_id;
}

// Returns the scope of the Core package, or Invalid if it's not found.
//
// TODO: Consider tracking the Core package in SemIR so we don't need to use
// name lookup to find it.
static auto GetCorePackage(Context& context, SemIRLoc loc)
    -> SemIR::NameScopeId {
  auto core_ident_id = context.identifiers().Add("Core");
  auto packaging = context.parse_tree().packaging_directive();
  if (packaging && packaging->names.package_id == core_ident_id) {
    return SemIR::NameScopeId::Package;
  }
  auto core_name_id = SemIR::NameId::ForIdentifier(core_ident_id);

  // Look up `package.Core`.
  auto core_inst_id = context.LookupNameInExactScope(
      loc, core_name_id, context.name_scopes().Get(SemIR::NameScopeId::Package),
      /*mark_imports_used=*/true);
  if (!core_inst_id.is_valid()) {
    context.DiagnoseNameNotFound(loc, core_name_id);
    return SemIR::NameScopeId::Invalid;
  }

  // We expect it to be a namespace.
  if (auto namespace_inst =
          context.insts().TryGetAs<SemIR::Namespace>(core_inst_id)) {
    return namespace_inst->name_scope_id;
  }
  // TODO: This should really diagnose the name issue.
  context.DiagnoseNameNotFound(loc, core_name_id);
  return SemIR::NameScopeId::Invalid;
}

auto Context::LookupNameInCore(SemIRLoc loc, llvm::StringRef name)
    -> SemIR::InstId {
  auto core_package_id = GetCorePackage(*this, loc);
  if (!core_package_id.is_valid()) {
    return SemIR::InstId::BuiltinError;
  }

  auto name_id = SemIR::NameId::ForIdentifier(identifiers().Add(name));
  auto inst_id =
      LookupNameInExactScope(loc, name_id, name_scopes().Get(core_package_id),
                             /*mark_imports_used=*/true);
  if (!inst_id.is_valid()) {
    DiagnoseNameNotFound(loc, name_id);
    return SemIR::InstId::BuiltinError;
  }

  // Look through import_refs and aliases.
  return constant_values().Get(inst_id).inst_id();
}

template <typename BranchNode, typename... Args>
static auto AddDominatedBlockAndBranchImpl(Context& context,
                                           Parse::NodeId node_id, Args... args)
    -> SemIR::InstBlockId {
  if (!context.inst_block_stack().is_current_block_reachable()) {
    return SemIR::InstBlockId::Unreachable;
  }
  auto block_id = context.inst_blocks().AddDefaultValue();
  context.AddInst({node_id, BranchNode{block_id, args...}});
  return block_id;
}

auto Context::AddDominatedBlockAndBranch(Parse::NodeId node_id)
    -> SemIR::InstBlockId {
  return AddDominatedBlockAndBranchImpl<SemIR::Branch>(*this, node_id);
}

auto Context::AddDominatedBlockAndBranchWithArg(Parse::NodeId node_id,
                                                SemIR::InstId arg_id)
    -> SemIR::InstBlockId {
  return AddDominatedBlockAndBranchImpl<SemIR::BranchWithArg>(*this, node_id,
                                                              arg_id);
}

auto Context::AddDominatedBlockAndBranchIf(Parse::NodeId node_id,
                                           SemIR::InstId cond_id)
    -> SemIR::InstBlockId {
  return AddDominatedBlockAndBranchImpl<SemIR::BranchIf>(*this, node_id,
                                                         cond_id);
}

auto Context::AddConvergenceBlockAndPush(Parse::NodeId node_id, int num_blocks)
    -> void {
  CARBON_CHECK(num_blocks >= 2) << "no convergence";

  SemIR::InstBlockId new_block_id = SemIR::InstBlockId::Unreachable;
  for ([[maybe_unused]] auto _ : llvm::seq(num_blocks)) {
    if (inst_block_stack().is_current_block_reachable()) {
      if (new_block_id == SemIR::InstBlockId::Unreachable) {
        new_block_id = inst_blocks().AddDefaultValue();
      }
      AddInst({node_id, SemIR::Branch{new_block_id}});
    }
    inst_block_stack().Pop();
  }
  inst_block_stack().Push(new_block_id);
}

auto Context::AddConvergenceBlockWithArgAndPush(
    Parse::NodeId node_id, std::initializer_list<SemIR::InstId> block_args)
    -> SemIR::InstId {
  CARBON_CHECK(block_args.size() >= 2) << "no convergence";

  SemIR::InstBlockId new_block_id = SemIR::InstBlockId::Unreachable;
  for (auto arg_id : block_args) {
    if (inst_block_stack().is_current_block_reachable()) {
      if (new_block_id == SemIR::InstBlockId::Unreachable) {
        new_block_id = inst_blocks().AddDefaultValue();
      }
      AddInst({node_id, SemIR::BranchWithArg{new_block_id, arg_id}});
    }
    inst_block_stack().Pop();
  }
  inst_block_stack().Push(new_block_id);

  // Acquire the result value.
  SemIR::TypeId result_type_id = insts().Get(*block_args.begin()).type_id();
  return AddInst({node_id, SemIR::BlockArg{result_type_id, new_block_id}});
}

auto Context::SetBlockArgResultBeforeConstantUse(SemIR::InstId select_id,
                                                 SemIR::InstId cond_id,
                                                 SemIR::InstId if_true,
                                                 SemIR::InstId if_false)
    -> void {
  CARBON_CHECK(insts().Is<SemIR::BlockArg>(select_id));

  // Determine the constant result based on the condition value.
  SemIR::ConstantId const_id = SemIR::ConstantId::NotConstant;
  auto cond_const_id = constant_values().Get(cond_id);
  if (!cond_const_id.is_template()) {
    // Symbolic or non-constant condition means a non-constant result.
  } else if (auto literal = insts().TryGetAs<SemIR::BoolLiteral>(
                 cond_const_id.inst_id())) {
    const_id = constant_values().Get(literal.value().value.ToBool() ? if_true
                                                                    : if_false);
  } else {
    CARBON_CHECK(cond_const_id == SemIR::ConstantId::Error)
        << "Unexpected constant branch condition.";
    const_id = SemIR::ConstantId::Error;
  }

  if (const_id.is_constant()) {
    CARBON_VLOG() << "Constant: " << insts().Get(select_id) << " -> "
                  << const_id.inst_id() << "\n";
    constant_values().Set(select_id, const_id);
  }
}

// Add the current code block to the enclosing function.
auto Context::AddCurrentCodeBlockToFunction(Parse::NodeId node_id) -> void {
  CARBON_CHECK(!inst_block_stack().empty()) << "no current code block";

  if (return_scope_stack().empty()) {
    CARBON_CHECK(node_id.is_valid())
        << "No current function, but node_id not provided";
    TODO(node_id,
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

auto Context::FinalizeGlobalInit() -> void {
  inst_block_stack().PushGlobalInit();
  if (!inst_block_stack().PeekCurrentBlockContents().empty()) {
    AddInst({Parse::NodeId::Invalid, SemIR::Return{}});
    // Pop the GlobalInit block here to finalize it.
    inst_block_stack().Pop();

    // __global_init is only added if there are initialization instructions.
    auto name_id = sem_ir().identifiers().Add("__global_init");
    sem_ir().functions().Add(
        {.name_id = SemIR::NameId::ForIdentifier(name_id),
         .enclosing_scope_id = SemIR::NameScopeId::Package,
         .decl_id = SemIR::InstId::Invalid,
         .implicit_param_refs_id = SemIR::InstBlockId::Empty,
         .param_refs_id = SemIR::InstBlockId::Empty,
         .return_type_id = SemIR::TypeId::Invalid,
         .return_storage_id = SemIR::InstId::Invalid,
         .is_extern = false,
         .return_slot = SemIR::Function::ReturnSlot::Absent,
         .body_block_ids = {SemIR::InstBlockId::GlobalInit}});
  } else {
    inst_block_stack().PopGlobalInit();
  }
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
          if (value_rep.type_id == SemIR::TypeId::Error) {
            break;
          }
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
    CARBON_KIND_SWITCH(type_inst) {
      case CARBON_KIND(SemIR::ArrayType inst): {
        Push(inst.element_type_id);
        break;
      }
      case CARBON_KIND(SemIR::StructType inst): {
        for (auto field_id : context_.inst_blocks().Get(inst.fields_id)) {
          Push(context_.insts()
                   .GetAs<SemIR::StructTypeField>(field_id)
                   .field_type_id);
        }
        break;
      }
      case CARBON_KIND(SemIR::TupleType inst): {
        for (auto element_type_id :
             context_.type_blocks().Get(inst.elements_id)) {
          Push(element_type_id);
        }
        break;
      }
      case CARBON_KIND(SemIR::ClassType inst): {
        auto& class_info = context_.classes().Get(inst.class_id);
        if (!class_info.is_defined()) {
          if (diagnoser_) {
            auto builder = (*diagnoser_)();
            context_.NoteIncompleteClass(inst.class_id, builder);
            builder.Emit();
          }
          return false;
        }
        Push(class_info.object_repr_id);
        break;
      }
      case CARBON_KIND(SemIR::ConstType inst): {
        Push(inst.inner_id);
        break;
      }
      default:
        break;
    }

    return true;
  }

  // Makes an empty value representation, which is used for types that have no
  // state, such as empty structs and tuples.
  auto MakeEmptyValueRepr() const -> SemIR::ValueRepr {
    return {.kind = SemIR::ValueRepr::None,
            .type_id = context_.GetTupleType({})};
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

  auto BuildBuiltinValueRepr(SemIR::TypeId type_id,
                             SemIR::Builtin builtin) const -> SemIR::ValueRepr {
    switch (builtin.builtin_kind) {
      case SemIR::BuiltinKind::TypeType:
      case SemIR::BuiltinKind::Error:
      case SemIR::BuiltinKind::Invalid:
      case SemIR::BuiltinKind::BoolType:
      case SemIR::BuiltinKind::IntType:
      case SemIR::BuiltinKind::FloatType:
      case SemIR::BuiltinKind::NamespaceType:
      case SemIR::BuiltinKind::FunctionType:
      case SemIR::BuiltinKind::BoundMethodType:
      case SemIR::BuiltinKind::WitnessType:
        return MakeCopyValueRepr(type_id);

      case SemIR::BuiltinKind::StringType:
        // TODO: Decide on string value semantics. This should probably be a
        // custom value representation carrying a pointer and size or
        // similar.
        return MakePointerValueRepr(type_id);
    }
    llvm_unreachable("All builtin kinds were handled above");
  }

  auto BuildAnyImportRefValueRepr(SemIR::TypeId type_id,
                                  SemIR::AnyImportRef import_ref) const
      -> SemIR::ValueRepr {
    auto import_ir_inst =
        context_.import_ir_insts().Get(import_ref.import_ir_inst_id);
    const auto& import_ir =
        context_.import_irs().Get(import_ir_inst.ir_id).sem_ir;
    auto import_inst = import_ir->insts().Get(import_ir_inst.inst_id);
    CARBON_CHECK(!import_inst.Is<SemIR::AnyImportRef>())
        << "If ImportRef can point at another, this would be recursive.";
    return BuildValueRepr(type_id, import_inst);
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
                         : context_.GetStructType(
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
                         : context_.GetTupleType(value_rep_elements);
    return BuildStructOrTupleValueRepr(elements.size(), value_rep,
                                       same_as_object_rep);
  }

  // Builds and returns the value representation for the given type. All nested
  // types, as found by AddNestedIncompleteTypes, are known to be complete.
  auto BuildValueRepr(SemIR::TypeId type_id, SemIR::Inst inst) const
      -> SemIR::ValueRepr {
    CARBON_KIND_SWITCH(inst) {
      case SemIR::AdaptDecl::Kind:
      case SemIR::AddrOf::Kind:
      case SemIR::AddrPattern::Kind:
      case SemIR::ArrayIndex::Kind:
      case SemIR::ArrayInit::Kind:
      case SemIR::AsCompatible::Kind:
      case SemIR::Assign::Kind:
      case SemIR::AssociatedConstantDecl::Kind:
      case SemIR::AssociatedEntity::Kind:
      case SemIR::BaseDecl::Kind:
      case SemIR::BindAlias::Kind:
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
      case SemIR::FacetTypeAccess::Kind:
      case SemIR::FieldDecl::Kind:
      case SemIR::FunctionDecl::Kind:
      case SemIR::ImplDecl::Kind:
      case SemIR::ImportRefUnloaded::Kind:
      case SemIR::InitializeFrom::Kind:
      case SemIR::InterfaceDecl::Kind:
      case SemIR::InterfaceWitness::Kind:
      case SemIR::InterfaceWitnessAccess::Kind:
      case SemIR::IntLiteral::Kind:
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

      case SemIR::ArrayType::Kind: {
        // For arrays, it's convenient to always use a pointer representation,
        // even when the array has zero or one element, in order to support
        // indexing.
        return MakePointerValueRepr(type_id, SemIR::ValueRepr::ObjectAggregate);
      }

      case SemIR::ImportRefLoaded::Kind:
      case SemIR::ImportRefUsed::Kind:
        return BuildAnyImportRefValueRepr(type_id,
                                          inst.As<SemIR::AnyImportRef>());

      case CARBON_KIND(SemIR::StructType struct_type): {
        return BuildStructTypeValueRepr(type_id, struct_type);
      }
      case CARBON_KIND(SemIR::TupleType tuple_type): {
        return BuildTupleTypeValueRepr(type_id, tuple_type);
      }
      case CARBON_KIND(SemIR::ClassType class_type): {
        auto& class_info = context_.classes().Get(class_type.class_id);
        // The value representation of an adapter is the value representation of
        // its adapted type.
        if (class_info.adapt_id.is_valid()) {
          return GetNestedValueRepr(class_info.object_repr_id);
        }
        // Otherwise, the value representation for a class is a pointer to the
        // object representation.
        // TODO: Support customized value representations for classes.
        // TODO: Pick a better value representation when possible.
        return MakePointerValueRepr(class_info.object_repr_id,
                                    SemIR::ValueRepr::ObjectAggregate);
      }
      case SemIR::InterfaceType::Kind: {
        // TODO: Should we model the value representation as a witness?
        return MakeEmptyValueRepr();
      }
      case CARBON_KIND(SemIR::Builtin builtin): {
        return BuildBuiltinValueRepr(type_id, builtin);
      }

      case SemIR::AssociatedEntityType::Kind:
      case SemIR::BindSymbolicName::Kind:
      case SemIR::IntType::Kind:
      case SemIR::PointerType::Kind:
      case SemIR::UnboundElementType::Kind:
        return MakeCopyValueRepr(type_id);

      case CARBON_KIND(SemIR::ConstType const_type): {
        // The value representation of `const T` is the same as that of `T`.
        // Objects are not modifiable through their value representations.
        return GetNestedValueRepr(const_type.inner_id);
      }
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

auto Context::GetTypeIdForTypeConstant(SemIR::ConstantId constant_id)
    -> SemIR::TypeId {
  CARBON_CHECK(constant_id.is_constant())
      << "Canonicalizing non-constant type: " << constant_id;

  auto [it, added] = type_ids_for_type_constants_.insert(
      {constant_id, SemIR::TypeId::Invalid});
  if (added) {
    it->second = types().Add({.constant_id = constant_id});
  }
  return it->second;
}

template <typename InstT, typename... EachArgT>
static auto GetTypeImpl(Context& context, EachArgT... each_arg)
    -> SemIR::TypeId {
  // TODO: Remove inst_id parameter from TryEvalInst.
  return context.GetTypeIdForTypeConstant(
      TryEvalInst(context, SemIR::InstId::Invalid,
                  InstT{SemIR::TypeId::TypeType, each_arg...}));
}

auto Context::GetStructType(SemIR::InstBlockId refs_id) -> SemIR::TypeId {
  return GetTypeImpl<SemIR::StructType>(*this, refs_id);
}

auto Context::GetTupleType(llvm::ArrayRef<SemIR::TypeId> type_ids)
    -> SemIR::TypeId {
  // TODO: Deduplicate the type block here. Currently requesting the same tuple
  // type more than once will create multiple type blocks, all but one of which
  // is unused.
  return GetTypeImpl<SemIR::TupleType>(*this, type_blocks().Add(type_ids));
}

auto Context::GetAssociatedEntityType(SemIR::InterfaceId interface_id,
                                      SemIR::TypeId entity_type_id)
    -> SemIR::TypeId {
  return GetTypeImpl<SemIR::AssociatedEntityType>(*this, interface_id,
                                                  entity_type_id);
}

auto Context::GetBuiltinType(SemIR::BuiltinKind kind) -> SemIR::TypeId {
  CARBON_CHECK(kind != SemIR::BuiltinKind::Invalid);
  auto type_id = GetTypeIdForTypeInst(SemIR::InstId::ForBuiltin(kind));
  // To keep client code simpler, complete builtin types before returning them.
  bool complete = TryToCompleteType(type_id);
  CARBON_CHECK(complete) << "Failed to complete builtin type";
  return type_id;
}

auto Context::GetPointerType(SemIR::TypeId pointee_type_id) -> SemIR::TypeId {
  return GetTypeImpl<SemIR::PointerType>(*this, pointee_type_id);
}

auto Context::GetUnboundElementType(SemIR::TypeId class_type_id,
                                    SemIR::TypeId element_type_id)
    -> SemIR::TypeId {
  return GetTypeImpl<SemIR::UnboundElementType>(*this, class_type_id,
                                                element_type_id);
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
  param_and_arg_refs_stack_.PrintForStackDump(output);
  args_type_info_stack_.PrintForStackDump(output);
}

}  // namespace Carbon::Check
