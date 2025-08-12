// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/implicit_type_impls.h"

#include "toolchain/check/convert.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/impl.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/check/name_ref.h"
#include "toolchain/check/pattern.h"
#include "toolchain/check/pattern_match.h"
#include "toolchain/check/type.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Produces an `impl <self_type_id> as <interface_id>` declaration. The caller
// should inspect the resulting `impl` to ensure it's incomplete before
// proceeding to define it.
static auto TryDeclareImpl(Context& context, SemIR::LocId loc_id,
                           SemIR::NameScopeId parent_scope_id,
                           SemIR::TypeId self_type_id,
                           SemIR::InstId interface_id)
    -> std::pair<SemIR::ImplId, SemIR::InstId> {
  StartGenericDecl(context);

  // Build the implicit access to the enclosing `Self`.
  // TODO: This mirrors code in handle_impl that also suggests using
  // BuildNameRef.
  auto self_inst_id = AddTypeInst(
      context, loc_id,
      SemIR::NameRef{.type_id = SemIR::TypeType::TypeId,
                     .name_id = SemIR::NameId::SelfType,
                     .value_id = context.types().GetInstId(self_type_id)});
  AddNameToLookup(context, SemIR::NameId::SelfType, self_inst_id);

  auto impl_decl_id = AddPlaceholderInst(
      context,
      SemIR::LocIdAndInst::UncheckedLoc(
          loc_id, SemIR::ImplDecl{.impl_id = SemIR::ImplId::None,
                                  .decl_block_id = SemIR::InstBlockId::Empty}));

  auto constraint_id = ExprAsType(context, loc_id, interface_id).inst_id;

  SemIR::Impl impl = {
      {
          .name_id = SemIR::NameId::None,
          .parent_scope_id = parent_scope_id,
          .generic_id = SemIR::GenericId::None,
          .first_param_node_id = Parse::NodeId::None,
          .last_param_node_id = Parse::NodeId::None,
          .pattern_block_id = SemIR::InstBlockId::None,
          .implicit_param_patterns_id = SemIR::InstBlockId::None,
          .param_patterns_id = SemIR::InstBlockId::None,
          .is_extern = false,
          .extern_library_id = SemIR::LibraryNameId::None,
          .non_owning_decl_id = SemIR::InstId::None,
          .first_owning_decl_id = impl_decl_id,
      },
      {
          .self_id = self_inst_id,
          .constraint_id = constraint_id,
          .interface =
              CheckConstraintIsInterface(context, impl_decl_id, constraint_id),
          .is_final = true,
      }};

  return StartImplDecl(context, loc_id,
                       /*implicit_params_loc_id=*/SemIR::LocId::None, impl,
                       /*is_definition=*/true, /*extend_impl=*/std::nullopt);
}

// Constructs the implicit params for the `Op` function. Returns the block and
// the `self` pattern.
static auto MakeImplicitParams(Context& context, SemIR::LocId loc_id)
    -> std::pair<SemIR::InstBlockId, SemIR::InstId> {
  BeginSubpattern(context);

  auto result = LookupUnqualifiedName(context, loc_id, SemIR::NameId::SelfType);
  auto self_id =
      BuildNameRef(context, loc_id, SemIR::NameId::SelfType,
                   result.scope_result.target_inst_id(), result.specific_id);
  auto self_type_expr = ExprAsType(context, loc_id, self_id);

  SemIR::ExprRegionId type_expr_region_id =
      EndSubpatternAsExpr(context, self_type_expr.inst_id);

  auto self_pattern_id = AddAddrSelfParamPattern(
      context, loc_id, type_expr_region_id, self_type_expr.inst_id);

  auto implicit_param_patterns_id =
      context.inst_blocks().Add({self_pattern_id});
  return {implicit_param_patterns_id, self_pattern_id};
}

// Defines the `Op` function for the `impl`.
static auto DeclareImplOpFunction(Context& context, SemIR::LocId loc_id,
                                  const SemIR::Impl& impl)
    -> std::pair<SemIR::FunctionId, SemIR::InstId> {
  StartGenericDecl(context);

  auto name_id = SemIR::NameId::ForIdentifier(context.identifiers().Add("Op"));

  context.inst_block_stack().Push();

  context.pattern_block_stack().Push();
  auto [implicit_param_patterns_id, self_pattern_id] =
      MakeImplicitParams(context, loc_id);
  constexpr auto NoRegularParams = SemIR::InstBlockId::Empty;
  constexpr auto NoReturnSlot = SemIR::InstId::None;
  auto pattern_block_id = context.pattern_block_stack().Pop();

  // Perform callee-side pattern matching to rebuild the parameter list.
  auto call_params_id = CalleePatternMatch(context, implicit_param_patterns_id,
                                           NoRegularParams, NoReturnSlot);
  auto decl_block_id = context.inst_block_stack().Pop();

  // Create the `FunctionDecl` instruction.
  SemIR::FunctionDecl function_decl = {SemIR::TypeId::None,
                                       SemIR::FunctionId::None, decl_block_id};
  auto decl_id = AddPlaceholderInst(
      context, SemIR::LocIdAndInst::UncheckedLoc(loc_id, function_decl));
  auto generic_id = BuildGenericDecl(context, decl_id);

  // Create the `Function` object.
  function_decl.function_id = context.functions().Add(SemIR::Function{
      {
          .name_id = name_id,
          .parent_scope_id = impl.scope_id,
          .generic_id = generic_id,
          .first_param_node_id = Parse::NodeId::None,
          .last_param_node_id = Parse::NodeId::None,
          .pattern_block_id = pattern_block_id,
          .implicit_param_patterns_id = implicit_param_patterns_id,
          .param_patterns_id = NoRegularParams,
          .is_extern = false,
          .extern_library_id = SemIR::LibraryNameId::None,
          .non_owning_decl_id = SemIR::InstId::None,
          .first_owning_decl_id = decl_id,
      },
      {
          .call_params_id = call_params_id,
          .return_slot_pattern_id = NoReturnSlot,
          .virtual_modifier = SemIR::FunctionFields::VirtualModifier::None,
          .virtual_index = -1,
          .self_param_id = self_pattern_id,
      }});
  function_decl.type_id =
      GetFunctionType(context, function_decl.function_id,
                      context.scope_stack().PeekSpecificId());
  ReplaceInstBeforeConstantUse(context, decl_id, function_decl);
  context.name_scopes().AddRequiredName(impl.scope_id, name_id, decl_id);

  return {function_decl.function_id, decl_id};
}

auto MakeClassDestroyImpl(Context& context, SemIR::ClassId class_id) -> void {
  if (!context.gen_implicit_type_impls()) {
    return;
  }

  // Identify the type and interface for implementation.
  auto& class_info = context.classes().Get(class_id);
  auto loc_id = context.insts().GetLocIdForDesugaring(
      SemIR::LocId(class_info.latest_decl_id()));

  auto destroy_id = LookupNameInCore(context, loc_id, "Destroy");
  if (destroy_id == SemIR::ErrorInst::InstId) {
    return;
  }

  // Declare the `impl`.
  auto [impl_id, impl_decl_id] =
      TryDeclareImpl(context, loc_id, class_info.scope_id,
                     class_info.self_type_id, destroy_id);
  auto& impl = context.impls().Get(impl_id);
  if (impl.is_complete()) {
    return;
  }

  // Define the `impl`.
  impl.definition_id = impl_decl_id;
  impl.scope_id = context.name_scopes().Add(impl_decl_id, SemIR::NameId::None,
                                            class_info.scope_id);

  context.scope_stack().PushForEntity(
      impl_decl_id, impl.scope_id,
      context.generics().GetSelfSpecific(impl.generic_id));
  StartGenericDefinition(context, impl.generic_id);
  context.inst_block_stack().Push();

  // Declare the `Op` function.
  auto [fn_id, fn_decl_id] = DeclareImplOpFunction(context, loc_id, impl);

  // Define the `Op` function.
  // TODO: Add an actual definition.
  context.scope_stack().PushForFunctionBody(fn_decl_id);
  auto& function = context.functions().Get(fn_id);
  function.SetBuiltinFunction(SemIR::BuiltinFunctionKind::NoOp);
  StartGenericDefinition(context, function.generic_id);
  FinishGenericDefinition(context, function.generic_id);
  context.scope_stack().Pop();

  // Close the `impl` definition.
  FinishImplWitness(context, impl_id);
  impl.defined = true;
  FinishGenericDefinition(context, impl.generic_id);
  context.scope_stack().Pop();
  impl.body_block_id = context.inst_block_stack().Pop();
}

}  // namespace Carbon::Check
