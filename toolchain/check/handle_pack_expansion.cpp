// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/control_flow.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/member_access.h"
#include "toolchain/check/type.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// A `...` statement is modeled as follows:
//
// - The body of the statement is a generic, nested within any enclosing
//   generic, whose last compile-time binding is the variadic index.
// - The first block of the body declares the index and branches to the rest
//   of the body. Instances of this branch in specifics of the generic represent
//   executing the body for particular index values.
// - The body ends with `BranchNextIndex`.
// - The `PackExpansion` instruction evaluates to a tuple of `InstValue`s,
//   one for each iteration, and `SpliceBranches` executes them in sequence
//   before continuing to its exit block.
//
// TODO: Support `break`, `continue`, and `return` within the body.

auto HandleParseNode(Context& context, Parse::PackExpansionStartId node_id)
    -> bool {
  if (context.generic_region_stack().IsInPackExpansion()) {
    context.TODO(node_id, "nested pack expansion");
  }

  context.scope_stack().PushForSameRegion(ScopeStack::CleanupScopeKind::Owned);

  // Create the pack expanded region and a placeholder `PackExpansion`
  // instruction for it. The instruction is filled in once we know the arity.
  auto region_id =
      context.pack_expanded_regions().Add({.expansion_id = SemIR::InstId::None,
                                           .generic_id = SemIR::GenericId::None,
                                           .index_id = SemIR::InstId::None});
  auto expansion_id = AddPlaceholderInstInNoBlock(
      context, node_id,
      SemIR::PackExpansion{.type_id = SemIR::TypeId::None,
                           .region_id = region_id,
                           .inst_id = SemIR::InstId::None});

  // Start the entry block of the body, which is the start of a new region.
  context.inst_block_stack().Push();
  context.region_stack().PushRegion(context.inst_block_stack().PeekOrAdd());

  // Declare the variadic index as a compile-time binding of a new generic.
  StartGenericDecl(context);
  auto entity_name_id = context.entity_names().AddSymbolicBindingName(
      SemIR::NameId::PackIndex, SemIR::NameScopeId::None,
      context.scope_stack().AddCompileTimeBinding(), /*is_template=*/false,
      /*is_unused=*/false, /*is_frozen_period_self=*/false);
  auto index_id = AddInst<SemIR::SymbolicBinding>(
      context, node_id,
      {.type_id = GetSingletonType(context, SemIR::IntLiteralType::TypeInstId),
       .entity_name_id = entity_name_id,
       .value_id = SemIR::InstId::None});
  context.scope_stack().PushCompileTimeBinding(index_id);
  auto generic_id = BuildGeneric(context, expansion_id);
  FinishGenericDecl(context, node_id, generic_id);
  StartGenericDefinition(context, generic_id);
  context.generic_region_stack().SetPackExpansionIndex(index_id);

  // Branch from the entry block to the body.
  auto body_id = context.inst_blocks().AddPlaceholder();
  AddInst<SemIR::Branch>(context, node_id, {.target_id = body_id});
  context.inst_block_stack().Pop();
  context.inst_block_stack().Push(body_id);
  context.region_stack().AddToRegion(body_id, node_id);

  auto& region = context.pack_expanded_regions().Get(region_id);
  region.expansion_id = expansion_id;
  region.generic_id = generic_id;
  region.index_id = index_id;

  context.node_stack().Push(node_id, expansion_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::PackExpansionStatementId node_id)
    -> bool {
  auto expansion_id =
      context.node_stack().Pop<Parse::NodeKind::PackExpansionStart>();
  auto expansion = context.insts().GetAs<SemIR::PackExpansion>(expansion_id);
  auto region_id = expansion.region_id;
  auto generic_id = context.pack_expanded_regions().Get(region_id).generic_id;

  auto* info = context.generic_region_stack().PeekPackExpansion();
  CARBON_CHECK(info, "Pack expansion body is not a pack expansion region");
  auto arity = info->arity;
  if (arity == GenericRegionStack::PackExpansionInfo::UnknownArity) {
    CARBON_DIAGNOSTIC(PackExpansionWithoutExpand, Error,
                      "pack expansion does not contain an `expand` "
                      "expression");
    context.emitter().Emit(node_id, PackExpansionWithoutExpand);
    arity = GenericRegionStack::PackExpansionInfo::ErrorArity;
  }

  // Finish the body by moving on to the next index.
  if (context.inst_block_stack().is_current_block_reachable()) {
    AddAndDiscardScopeCleanups(context);
    AddInst<SemIR::BranchNextIndex>(context, node_id, {});
  }
  context.inst_block_stack().Pop();
  FinishGenericDefinition(context, generic_id);
  auto body_block_ids = context.region_stack().PopRegion();
  context.scope_stack().Pop(/*check_unused=*/true);
  auto& region = context.pack_expanded_regions().Get(region_id);
  region.body_block_ids = std::move(body_block_ids);

  // Now we know the arity, fill in the `PackExpansion`. Its type is a tuple of
  // `InstValue`s, one per iteration, and the instruction it expands is the
  // branch at the end of the entry block.
  auto type_id = SemIR::ErrorInst::TypeId;
  if (arity >= 0) {
    llvm::SmallVector<SemIR::InstId> element_type_ids(
        arity, SemIR::InstType::TypeInstId);
    type_id = GetTupleType(context, element_type_ids);
  }
  expansion.type_id = type_id;
  expansion.inst_id =
      context.inst_blocks().Get(region.body_block_ids.front()).back();
  ReplaceInstBeforeConstantUse(context, expansion_id, expansion);
  context.inst_block_stack().AddInstId(expansion_id);

  // Execute the body for each index, then continue in the exit block.
  auto exit_id = context.inst_blocks().AddPlaceholder();
  AddInst<SemIR::SpliceBranches>(
      context, node_id, {.insts_id = expansion_id, .exit_id = exit_id});
  context.inst_block_stack().Pop();
  context.inst_block_stack().Push(exit_id);
  context.region_stack().AddToRegion(exit_id, node_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::PrefixOperatorExpandId node_id)
    -> bool {
  auto operand_id = context.node_stack().PopExpr();

  auto* info = context.generic_region_stack().PeekPackExpansion();
  if (!info) {
    CARBON_DIAGNOSTIC(ExpandOutsidePackExpansion, Error,
                      "`expand` can only be used in a pack expansion");
    context.emitter().Emit(node_id, ExpandOutsidePackExpansion);
    context.node_stack().Push(node_id, SemIR::ErrorInst::InstId);
    return true;
  }

  operand_id = ConvertToValueOrRefExpr(context, operand_id);
  auto operand_type_id = context.insts().Get(operand_id).type_id();
  if (operand_type_id == SemIR::ErrorInst::TypeId) {
    info->arity = GenericRegionStack::PackExpansionInfo::ErrorArity;
    context.node_stack().Push(node_id, SemIR::ErrorInst::InstId);
    return true;
  }

  // TODO: Support other kinds of operand, such as variadic parameters.
  auto tuple_type = context.types().TryGetAs<SemIR::TupleType>(operand_type_id);
  if (!tuple_type) {
    CARBON_DIAGNOSTIC(ExpandOperandNotTuple, Error,
                      "operand of `expand` has type {0}, which is not a "
                      "tuple type",
                      TypeOfInstId);
    context.emitter().Emit(node_id, ExpandOperandNotTuple, operand_id);
    info->arity = GenericRegionStack::PackExpansionInfo::ErrorArity;
    context.node_stack().Push(node_id, SemIR::ErrorInst::InstId);
    return true;
  }

  // Check the arity is consistent with any previous `expand`.
  auto arity = static_cast<int32_t>(
      context.inst_blocks().Get(tuple_type->type_elements_id).size());
  if (info->arity == GenericRegionStack::PackExpansionInfo::UnknownArity) {
    info->arity = arity;
    info->arity_source_id = operand_id;
  } else if (info->arity >= 0 && info->arity != arity) {
    CARBON_DIAGNOSTIC(ExpandArityMismatch, Error,
                      "operand of `expand` has {0} element{0:s}, but pack "
                      "expansion has {1} element{1:s}",
                      Diagnostics::IntAsSelect, Diagnostics::IntAsSelect);
    CARBON_DIAGNOSTIC(ExpandArityMismatchPrevious, Note,
                      "number of elements of pack expansion determined here");
    context.emitter()
        .Build(node_id, ExpandArityMismatch, arity, info->arity)
        .Note(info->arity_source_id, ExpandArityMismatchPrevious)
        .Emit();
    info->arity = GenericRegionStack::PackExpansionInfo::ErrorArity;
    context.node_stack().Push(node_id, SemIR::ErrorInst::InstId);
    return true;
  }

  context.node_stack().Push(
      node_id,
      PerformVariadicTupleAccess(context, node_id, operand_id, info->index_id));
  return true;
}

}  // namespace Carbon::Check
