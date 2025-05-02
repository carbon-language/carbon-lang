// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/pattern.h"

#include "toolchain/check/inst.h"
#include "toolchain/check/type.h"

namespace Carbon::Check {

auto BeginSubpattern(Context& context) -> void {
  context.inst_block_stack().Push();
  context.region_stack().PushRegion(context.inst_block_stack().PeekOrAdd());
}

auto EndSubpatternAsExpr(Context& context, SemIR::InstId result_id)
    -> SemIR::ExprRegionId {
  if (context.region_stack().PeekRegion().size() > 1) {
    // End the exit block with a branch to a successor block, whose contents
    // will be determined later.
    AddInst(context,
            SemIR::LocIdAndInst::NoLoc<SemIR::Branch>(
                {.target_id = context.inst_blocks().AddPlaceholder()}));
  } else {
    // This single-block region will be inserted as a SpliceBlock, so we don't
    // need control flow out of it.
  }
  auto block_id = context.inst_block_stack().Pop();
  CARBON_CHECK(block_id == context.region_stack().PeekRegion().back());

  // TODO: Is it possible to validate that this region is genuinely
  // single-entry, single-exit?
  return context.sem_ir().expr_regions().Add(
      {.block_ids = context.region_stack().PopRegion(),
       .result_id = result_id});
}

auto EndSubpatternAsNonExpr(Context& context) -> void {
  auto block_id = context.inst_block_stack().Pop();
  CARBON_CHECK(block_id == context.region_stack().PeekRegion().back());
  CARBON_CHECK(context.region_stack().PeekRegion().size() == 1);
  // TODO: Add `CARBON_CHECK(inst_blocks().Get(block_id).empty())`.
  // Currently that can fail when ending a tuple pattern in a name binding
  // decl in a class or interface.
  context.region_stack().PopAndDiscardRegion();
}

auto AddBindingPattern(Context& context, SemIR::LocId name_loc,
                       SemIR::NameId name_id, SemIR::TypeId type_id,
                       SemIR::ExprRegionId type_region_id, bool is_generic,
                       bool is_template) -> BindingPatternInfo {
  auto entity_name_id = context.entity_names().AddSymbolicBindingName(
      name_id, context.scope_stack().PeekNameScopeId(),
      is_generic ? context.scope_stack().AddCompileTimeBinding()
                 : SemIR::CompileTimeBindIndex::None,
      is_template);

  auto bind_id = SemIR::InstId::None;
  if (is_generic) {
    bind_id = AddInstInNoBlock<SemIR::BindSymbolicName>(
        context, name_loc,
        {.type_id = type_id,
         .entity_name_id = entity_name_id,
         .value_id = SemIR::InstId::None});
  } else {
    bind_id =
        AddInstInNoBlock<SemIR::BindName>(context, name_loc,
                                          {.type_id = type_id,
                                           .entity_name_id = entity_name_id,
                                           .value_id = SemIR::InstId::None});
  }

  auto pattern_type_id = GetPatternType(context, type_id);
  auto binding_pattern_id = SemIR::InstId::None;
  if (is_generic) {
    binding_pattern_id = AddPatternInst<SemIR::SymbolicBindingPattern>(
        context, name_loc,
        {.type_id = pattern_type_id, .entity_name_id = entity_name_id});
  } else {
    binding_pattern_id = AddPatternInst<SemIR::BindingPattern>(
        context, name_loc,
        {.type_id = pattern_type_id, .entity_name_id = entity_name_id});
  }

  if (is_generic) {
    context.scope_stack().PushCompileTimeBinding(bind_id);
  }

  bool inserted =
      context.bind_name_map()
          .Insert(binding_pattern_id, {.bind_name_id = bind_id,
                                       .type_expr_region_id = type_region_id})
          .is_inserted();
  CARBON_CHECK(inserted);
  return {.pattern_id = binding_pattern_id, .bind_id = bind_id};
}

}  // namespace Carbon::Check
