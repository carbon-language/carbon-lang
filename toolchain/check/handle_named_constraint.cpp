// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/interface.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/check/type.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/named_constraint.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context,
                     Parse::NamedConstraintIntroducerId node_id) -> bool {
  // This named constraint is potentially generic.
  StartGenericDecl(context);
  // Create an instruction block to hold the instructions created as part of the
  // named constraint signature, such as generic parameters.
  context.inst_block_stack().Push();
  // Optional modifiers and the name follow.
  context.decl_introducer_state_stack().Push<Lex::TokenKind::Constraint>();
  context.decl_name_stack().PushScopeAndStartName();

  // Push the bracketing node.
  context.node_stack().Push(node_id);
  return true;
}

static auto BuildNamedConstraintDecl(Context& context,
                                     Parse::AnyNamedConstraintDeclId node_id,
                                     bool is_definition)
    -> std::tuple<SemIR::NamedConstraintId, SemIR::InstId> {
  auto name = PopNameComponent(context);
  auto name_context = context.decl_name_stack().FinishName(name);
  context.node_stack()
      .PopAndDiscardSoloNodeId<Parse::NodeKind::NamedConstraintIntroducer>();

  // TODO: PopSoloNodeId(`template`) if it's present, and track that in the
  // NamedConstraint. Or maybe it should be a modifier, like `abstract class`?

  // Process modifiers.
  auto [_, parent_scope_inst] =
      context.name_scopes().GetInstIfValid(name_context.parent_scope_id);
  auto introducer =
      context.decl_introducer_state_stack().Pop<Lex::TokenKind::Constraint>();
  CheckAccessModifiersOnDecl(context, introducer, parent_scope_inst);
  LimitModifiersOnDecl(context, introducer, KeywordModifierSet::Access);

  auto decl_block_id = context.inst_block_stack().Pop();

  // Add the constraint declaration.
  auto constraint_decl = SemIR::NamedConstraintDecl{
      SemIR::TypeType::TypeId, SemIR::NamedConstraintId::None, decl_block_id};
  auto decl_inst_id = AddPlaceholderInst(context, node_id, constraint_decl);

  SemIR::NamedConstraint constraint_info = {
      name_context.MakeEntityWithParamsBase(name, decl_inst_id,
                                            /*is_extern=*/false,
                                            SemIR::LibraryNameId::None)};

  DiagnoseIfGenericMissingExplicitParameters(context, constraint_info);

  // Check whether this is a redeclaration.
  SemIR::ScopeLookupResult lookup_result =
      context.decl_name_stack().LookupOrAddName(
          name_context, decl_inst_id, introducer.modifier_set.GetAccessKind());
  if (auto existing_decl = TryGetExistingDecl(context, name, lookup_result,
                                              constraint_info, is_definition)) {
    auto existing_constraint_decl =
        existing_decl->As<SemIR::NamedConstraintDecl>();
    constraint_decl.named_constraint_id =
        existing_constraint_decl.named_constraint_id;
    constraint_decl.type_id = existing_constraint_decl.type_id;
    // TODO: If the new declaration is a definition, keep its parameter
    // and implicit parameter lists rather than the ones from the
    // previous declaration.

    auto prev_decl_generic_id = context.named_constraints()
                                    .Get(constraint_decl.named_constraint_id)
                                    .generic_id;
    FinishGenericRedecl(context, prev_decl_generic_id);
  } else {
    // Create a new named constraint if this isn't a valid redeclaration.
    constraint_info.generic_id = BuildGenericDecl(context, decl_inst_id);
    constraint_decl.named_constraint_id =
        context.named_constraints().Add(constraint_info);
    if (constraint_info.has_parameters()) {
      constraint_decl.type_id = GetGenericNamedConstraintType(
          context, constraint_decl.named_constraint_id,
          context.scope_stack().PeekSpecificId());
    }
  }

  // Write the completed NamedConstraintDecl instruction.
  ReplaceInstBeforeConstantUse(context, decl_inst_id, constraint_decl);

  return {constraint_decl.named_constraint_id, decl_inst_id};
}

auto HandleParseNode(Context& context, Parse::NamedConstraintDeclId node_id)
    -> bool {
  BuildNamedConstraintDecl(context, node_id, /*is_definition=*/false);
  context.decl_name_stack().PopScope();
  return true;
}

auto HandleParseNode(Context& context,
                     Parse::NamedConstraintDefinitionStartId node_id) -> bool {
  auto [named_constraint_id, decl_inst_id] =
      BuildNamedConstraintDecl(context, node_id, /*is_definition=*/true);
  auto& constraint_info = context.named_constraints().Get(named_constraint_id);

  // TODO: Support for `template constraint`.
  bool is_template = false;

  // Track that this declaration is the definition.
  CARBON_CHECK(!constraint_info.has_definition_started(),
               "Can't merge with defined named constraints.");
  constraint_info.definition_id = decl_inst_id;
  constraint_info.scope_id = context.name_scopes().Add(
      decl_inst_id, SemIR::NameId::None, constraint_info.parent_scope_id);

  auto self_specific_id =
      context.generics().GetSelfSpecific(constraint_info.generic_id);
  StartGenericDefinition(context, constraint_info.generic_id);

  context.inst_block_stack().Push();
  context.require_impls_stack().PushArray();

  // Declare and introduce `Self`. We model `Self` as a symbolic binding whose
  // type is the named constraint, excluding any other interfaces mentioned by
  // `require` declarations. This makes it an empty facet type.
  SemIR::TypeId self_type_id =
      GetNamedConstraintType(context, named_constraint_id, self_specific_id);
  constraint_info.self_param_id = AddSelfGenericParameter(
      context, node_id, self_type_id, constraint_info.scope_id, is_template);

  // Enter the constraint scope.
  context.scope_stack().PushForEntity(decl_inst_id, constraint_info.scope_id,
                                      self_specific_id);

  constraint_info.body_block_id = context.inst_block_stack().PeekOrAdd();

  context.node_stack().Push(node_id, named_constraint_id);
  return true;
}

auto HandleParseNode(Context& context,
                     Parse::NamedConstraintDefinitionId /*node_id*/) -> bool {
  auto named_constraint_id =
      context.node_stack()
          .Pop<Parse::NodeKind::NamedConstraintDefinitionStart>();
  context.inst_block_stack().Pop();

  auto require_impls_block_id = context.require_impls_blocks().Add(
      context.require_impls_stack().PeekArray());
  context.require_impls_stack().PopArray();

  auto& constraint_info = context.named_constraints().Get(named_constraint_id);
  if (!constraint_info.complete) {
    constraint_info.require_impls_block_id = require_impls_block_id;
    // TODO: Do something with `alias` statements in the body of the
    // constraint.
    constraint_info.complete = true;
  }

  FinishGenericDefinition(context, constraint_info.generic_id);

  // The decl_name_stack and scopes are popped by `ProcessNodeIds`.
  return true;
}

}  // namespace Carbon::Check
