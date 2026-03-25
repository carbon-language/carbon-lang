// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/pattern.h"

#include "toolchain/base/kind_switch.h"
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

auto AddBindingEntityName(Context& context, SemIR::NameId name_id,
                          SemIR::ConstantId form_id, bool is_unused,
                          BindingPhase phase) -> SemIR::EntityNameId {
  SemIR::EntityName entity_name = {
      .name_id = name_id,
      .parent_scope_id = context.scope_stack().PeekNameScopeId(),
      .is_unused = is_unused || name_id == SemIR::NameId::Underscore};
  if (phase != BindingPhase::Runtime) {
    entity_name.bind_index_value =
        context.scope_stack().AddCompileTimeBinding().index;
    entity_name.is_template = phase == BindingPhase::Template;
  }
  entity_name.form_id = form_id;
  return context.entity_names().Add(entity_name);
}

auto AddBindingPattern(Context& context, SemIR::LocId name_loc,
                       SemIR::ExprRegionId type_region_id,
                       SemIR::AnyBindingPattern pattern) -> BindingPatternInfo {
  SemIR::InstKind bind_name_kind;
  switch (pattern.kind) {
    case SemIR::FormBindingPattern::Kind:
      bind_name_kind = SemIR::FormBinding::Kind;
      break;
    case SemIR::RefBindingPattern::Kind:
      bind_name_kind = SemIR::RefBinding::Kind;
      break;
    case SemIR::SymbolicBindingPattern::Kind:
      bind_name_kind = SemIR::SymbolicBinding::Kind;
      break;
    case SemIR::ValueBindingPattern::Kind:
      bind_name_kind = SemIR::ValueBinding::Kind;
      break;
    case SemIR::WrapperBindingPattern::Kind: {
      auto subpattern = context.insts().Get(pattern.subpattern_id);
      CARBON_KIND_SWITCH(subpattern) {
        case SemIR::FormParamPattern::Kind:
          bind_name_kind = SemIR::FormBinding::Kind;
          break;
        case SemIR::RefParamPattern::Kind:
        case SemIR::VarPattern::Kind:
          bind_name_kind = SemIR::RefBinding::Kind;
          break;
        case SemIR::ValueParamPattern::Kind:
          bind_name_kind = SemIR::ValueBinding::Kind;
          break;
        default:
          CARBON_FATAL("Unexpected subpattern kind for at_binding_pattern: {0}",
                       subpattern);
      }
      break;
    }
    default:
      CARBON_FATAL("pattern_kind {0} is not a binding pattern kind",
                   pattern.kind);
  }
  auto type_id = SemIR::ExtractScrutineeType(context.sem_ir(), pattern.type_id);

  auto bind_id = AddInstInNoBlock(
      context, SemIR::LocIdAndInst::RuntimeVerified(
                   context.sem_ir(), name_loc,
                   SemIR::AnyBinding{.kind = bind_name_kind,
                                     .type_id = type_id,
                                     .entity_name_id = pattern.entity_name_id,
                                     .value_id = SemIR::InstId::None}));

  auto binding_pattern_id =
      AddPatternInst(context, SemIR::LocIdAndInst::RuntimeVerified(
                                  context.sem_ir(), name_loc, pattern));

  if (pattern.kind == SemIR::SymbolicBindingPattern::Kind) {
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
// is the body of a returned var, this reuses the return parameter, and
// otherwise it adds a new inst.
static auto GetOrAddVarStorage(Context& context, SemIR::InstId var_pattern_id,
                               bool is_returned_var) -> SemIR::InstId {
  if (is_returned_var) {
    if (auto return_param_id =
            GetReturnedVarParam(context, GetCurrentFunctionForReturn(context));
        return_param_id.has_value()) {
      return return_param_id;
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
  // the corresponding `AnyVarPattern`s because they're part of the pattern
  // match, not part of the pattern.
  // TODO: Find a way to do this without walking the whole pattern block.
  for (auto inst_id : context.inst_blocks().Get(pattern_block_id)) {
    if (context.insts().Is<SemIR::AnyVarPattern>(inst_id)) {
      context.var_storage_map().Insert(
          inst_id, GetOrAddVarStorage(context, inst_id, is_returned_var));
    }
  }
}

auto AddParamPattern(Context& context, SemIR::LocId loc_id,
                     SemIR::NameId name_id,
                     SemIR::ExprRegionId type_expr_region_id,
                     SemIR::TypeId type_id, bool is_ref) -> SemIR::InstId {
  auto pattern_type_id = GetPatternType(context, type_id);
  const auto& param_pattern_kind =
      is_ref ? SemIR::RefParamPattern::Kind : SemIR::ValueParamPattern::Kind;
  auto pattern_id = AddPatternInst(
      context, SemIR::LocIdAndInst::RuntimeVerified(
                   context.sem_ir(), loc_id,
                   SemIR::AnyLeafParamPattern{.kind = param_pattern_kind,
                                              .type_id = pattern_type_id,
                                              .pretty_name_id = name_id}));

  auto entity_name_id =
      AddBindingEntityName(context, name_id,
                           /*form_id=*/SemIR::ConstantId::None,
                           /*is_unused=*/false,
                           /*phase=*/BindingPhase::Runtime);
  return AddBindingPattern(context, loc_id, type_expr_region_id,
                           {.kind = SemIR::WrapperBindingPattern::Kind,
                            .type_id = GetPatternType(context, type_id),
                            .entity_name_id = entity_name_id,
                            .subpattern_id = pattern_id})
      .pattern_id;
}

}  // namespace Carbon::Check
