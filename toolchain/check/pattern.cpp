// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/pattern.h"

#include "toolchain/check/control_flow.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/return.h"
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
  CARBON_CHECK(context.inst_blocks().Get(block_id).empty());
  context.region_stack().PopAndDiscardRegion();
}

auto AddBindingPattern(Context& context, SemIR::LocId name_loc,
                       SemIR::NameId name_id, SemIR::TypeId type_id,
                       SemIR::ExprRegionId type_region_id,
                       SemIR::InstKind pattern_kind, bool is_template)
    -> BindingPatternInfo {
  SemIR::InstKind bind_name_kind;
  switch (pattern_kind) {
    case SemIR::InstKind::RefBindingPattern:
      bind_name_kind = SemIR::InstKind::RefBinding;
      break;
    case SemIR::InstKind::SymbolicBindingPattern:
      bind_name_kind = SemIR::InstKind::SymbolicBinding;
      break;
    case SemIR::InstKind::ValueBindingPattern:
      bind_name_kind = SemIR::InstKind::ValueBinding;
      break;
    default:
      CARBON_FATAL("pattern_kind is not a binding pattern kind");
  }
  bool is_generic = pattern_kind == SemIR::SymbolicBindingPattern::Kind;

  auto entity_name_id = context.entity_names().AddSymbolicBindingName(
      name_id, context.scope_stack().PeekNameScopeId(),
      is_generic ? context.scope_stack().AddCompileTimeBinding()
                 : SemIR::CompileTimeBindIndex::None,
      is_template);

  auto bind_id = AddInstInNoBlock(
      context,
      SemIR::LocIdAndInst::UncheckedLoc(
          name_loc, SemIR::AnyBinding{.kind = bind_name_kind,
                                      .type_id = type_id,
                                      .entity_name_id = entity_name_id,
                                      .value_id = SemIR::InstId::None}));

  auto pattern_type_id = GetPatternType(context, type_id);
  auto binding_pattern_id = AddPatternInst(
      context, SemIR::LocIdAndInst::UncheckedLoc(
                   name_loc,
                   SemIR::AnyBindingPattern{.kind = pattern_kind,
                                            .type_id = pattern_type_id,
                                            .entity_name_id = entity_name_id}));

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

// Returns a VarStorage inst for the given `var` pattern. If the pattern
// is the body of a returned var, this reuses the return slot, and otherwise it
// adds a new inst.
static auto GetOrAddVarStorage(Context& context, SemIR::InstId var_pattern_id,
                               bool is_returned_var) -> SemIR::InstId {
  if (is_returned_var) {
    auto& function = GetCurrentFunctionForReturn(context);
    auto return_info =
        SemIR::ReturnTypeInfo::ForFunction(context.sem_ir(), function);
    if (return_info.has_return_slot()) {
      return GetCurrentReturnSlot(context);
    }
  }
  auto pattern = context.insts().GetWithLocId(var_pattern_id);

  return AddInstWithCleanup(
      context, pattern.loc_id,
      SemIR::VarStorage{.type_id = ExtractScrutineeType(context.sem_ir(),
                                                        pattern.inst.type_id()),
                        .pattern_id = var_pattern_id});
}

auto AddPatternVarStorage(Context& context, SemIR::InstBlockId pattern_block_id,
                          bool is_returned_var) -> void {
  // We need to emit the VarStorage insts early, because they may be output
  // arguments for the initializer. However, we can't emit them when we emit
  // the corresponding `VarPattern`s because they're part of the pattern match,
  // not part of the pattern.
  // TODO: Find a way to do this without walking the whole pattern block.
  for (auto inst_id : context.inst_blocks().Get(pattern_block_id)) {
    if (context.insts().Is<SemIR::VarPattern>(inst_id)) {
      context.var_storage_map().Insert(
          inst_id, GetOrAddVarStorage(context, inst_id, is_returned_var));
    }
  }
}

auto AddParamPattern(Context& context, SemIR::LocId loc_id,
                     SemIR::NameId name_id,
                     SemIR::ExprRegionId type_expr_region_id,
                     SemIR::TypeId type_id, bool is_ref) -> SemIR::InstId {
  const auto& binding_pattern_kind = is_ref ? SemIR::RefBindingPattern::Kind
                                            : SemIR::ValueBindingPattern::Kind;
  SemIR::InstId pattern_id =
      AddBindingPattern(context, loc_id, name_id, type_id, type_expr_region_id,
                        binding_pattern_kind,
                        /*is_template=*/false)
          .pattern_id;

  const auto& param_pattern_kind =
      is_ref ? SemIR::RefParamPattern::Kind : SemIR::ValueParamPattern::Kind;
  pattern_id = AddPatternInst(
      context,
      SemIR::LocIdAndInst::UncheckedLoc(
          loc_id, SemIR::AnyParamPattern{
                      .kind = param_pattern_kind,
                      .type_id = context.insts().Get(pattern_id).type_id(),
                      .subpattern_id = pattern_id,
                      .index = SemIR::CallParamIndex::None}));

  return pattern_id;
}

}  // namespace Carbon::Check
