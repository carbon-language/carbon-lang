// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/pattern.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/action.h"
#include "toolchain/check/class.h"
#include "toolchain/check/control_flow.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/return.h"
#include "toolchain/check/type.h"
#include "toolchain/diagnostics/emitter.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

auto BeginSubpattern(Context& context) -> void {
  context.inst_block_stack().Push();
  // TODO: This allocates an InstBlockId even in the case where the pattern has
  // no associated expression. Find a way to avoid this.
  context.region_stack().PushRegion(context.inst_block_stack().PeekOrAdd());
}

static auto PopSubpatternExpr(Context& context, SemIR::InstId result_id)
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

auto ConsumeSubpatternExpr(Context& context, SemIR::InstId result_id)
    -> SemIR::ExprRegionId {
  auto region_id = PopSubpatternExpr(context, result_id);
  // Push an empty, unreachable region so that we can later detect the region
  // has been consumed.
  context.region_stack().PushUnreachableRegion();
  return region_id;
}

auto EndEmptySubpattern(Context& context) -> void {
  if (!context.region_stack().PeekRegion().empty()) {
    CARBON_CHECK(context.inst_block_stack().PeekCurrentBlockContents().empty());
    auto block_id = context.inst_block_stack().Pop();
    CARBON_CHECK(block_id == context.region_stack().PeekRegion().back());
    CARBON_CHECK(context.region_stack().PeekRegion().size() == 1);
  }
  context.region_stack().PopAndDiscardRegion();
}

auto EndSubpattern(Context& context, NodeStack& node_stack) -> void {
  auto [node_id, maybe_expr_id] =
      node_stack.PopWithNodeIdIf<Parse::NodeCategory::Expr>();
  if (maybe_expr_id) {
    // We formed an expression, not a pattern, so convert it to an expression
    // pattern now.
    auto expr_region_id = PopSubpatternExpr(context, *maybe_expr_id);
    auto pattern_type_id =
        GetPatternType(context, context.insts().Get(*maybe_expr_id).type_id());
    node_stack.Push(node_id, AddInst<SemIR::ExprPattern>(
                                 context, node_id,
                                 {.type_id = pattern_type_id,
                                  .expr_region_id = expr_region_id}));
  } else {
    // The expression region should have been consumed when forming the pattern
    // instruction, so should now effectively be empty.
    EndEmptySubpattern(context);
  }
}

auto AddBindingEntityName(Context& context, SemIR::NameId name_id,
                          SemIR::InstId form_id, bool is_unused,
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

  // Handle non-static `var` decls in a class by creating a `FieldDecl`.
  if (InNonStaticFieldDecl(context)) {
    auto class_decl =
        context.scope_stack().TryGetCurrentScopeAs<SemIR::ClassDecl>();
    auto name_id = context.entity_names().Get(pattern.entity_name_id).name_id;
    auto& class_info = context.classes().Get(class_decl->class_id);
    auto field_type_id = GetUnboundElementType(
        context, context.types().GetTypeInstId(class_info.self_type_id),
        context.types().GetTypeInstId(type_id));

    if (name_id == SemIR::NameId::Underscore) {
      CARBON_DIAGNOSTIC(FieldNamedUnderscore, Error,
                        "expected identifier in field declaration");
      context.emitter().Emit(name_loc, FieldNamedUnderscore);
    }

    auto field_id =
        AddInst<SemIR::FieldDecl>(context, name_loc,
                                  {.type_id = field_type_id,
                                   .name_id = name_id,
                                   .index = SemIR::ElementIndex::None});
    context.field_decls_stack().AppendToTop(field_id);

    return {.pattern_id = field_id, .bind_id = field_id};
  }

  auto bind_id = AddInstInNoBlock(
      context, SemIR::LocIdAndInst::RuntimeVerified(
                   context.sem_ir(), name_loc,
                   SemIR::AnyBinding{.kind = bind_name_kind,
                                     .type_id = type_id,
                                     .entity_name_id = pattern.entity_name_id,
                                     .value_id = SemIR::InstId::None}));

  auto binding_pattern_id =
      AddInst(context, SemIR::LocIdAndInst::RuntimeVerified(context.sem_ir(),
                                                            name_loc, pattern));

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
                     SemIR::TypeId type_id, ParamPatternKind kind)
    -> SemIR::InstId {
  auto param_pattern_kind = [kind]() -> SemIR::InstKind {
    switch (kind) {
      case ParamPatternKind::Value:
        return SemIR::ValueParamPattern::Kind;
      case ParamPatternKind::Ref:
        return SemIR::RefParamPattern::Kind;
      case ParamPatternKind::Var:
        return SemIR::VarParamPattern::Kind;
    }
  }();

  auto entity_name_id = AddBindingEntityName(context, name_id,
                                             /*form_id=*/SemIR::InstId::None,
                                             /*is_unused=*/false,
                                             /*phase=*/BindingPhase::Runtime);

  auto pattern_type_id = GetPatternType(context, type_id);
  if (kind == ParamPatternKind::Var) {
    auto pattern_id = AddBindingPattern(context, loc_id, type_expr_region_id,
                                        {.kind = SemIR::RefBindingPattern::Kind,
                                         .type_id = pattern_type_id,
                                         .entity_name_id = entity_name_id,
                                         .subpattern_id = SemIR::InstId::None});
    return AddInst(context, SemIR::LocIdAndInst::RuntimeVerified(
                                context.sem_ir(), loc_id,
                                SemIR::VarParamPattern{
                                    .type_id = pattern_type_id,
                                    .subpattern_id = pattern_id.pattern_id}));
  } else {
    auto pattern_id = AddInst(
        context, SemIR::LocIdAndInst::RuntimeVerified(
                     context.sem_ir(), loc_id,
                     SemIR::AnyLeafParamPattern{.kind = param_pattern_kind,
                                                .type_id = pattern_type_id,
                                                .pretty_name_id = name_id}));

    return AddBindingPattern(context, loc_id, type_expr_region_id,
                             {.kind = SemIR::WrapperBindingPattern::Kind,
                              .type_id = GetPatternType(context, type_id),
                              .entity_name_id = entity_name_id,
                              .subpattern_id = pattern_id})
        .pattern_id;
  }
}

auto PerformAction(Context& context, SemIR::LocId loc_id,
                   SemIR::FormParamPatternAction action) -> SemIR::InstId {
  auto form_inst = context.insts().Get(
      context.constant_values().GetConstantInstId(action.form_id));
  auto type_id =
      GetPatternType(context, GetTypeComponent(context, action.form_id));
  CARBON_KIND_SWITCH(form_inst) {
    case SemIR::InitForm::Kind: {
      auto ref_param_pattern_id = AddInst(
          context,
          SemIR::LocIdAndInst::RuntimeVerified(
              context.sem_ir(), loc_id,
              SemIR::RefParamPattern{.type_id = type_id,
                                     .pretty_name_id = action.pretty_name_id}));
      return AddInst(context, SemIR::LocIdAndInst::RuntimeVerified(
                                  context.sem_ir(), loc_id,
                                  SemIR::VarPattern{
                                      .type_id = type_id,
                                      .subpattern_id = ref_param_pattern_id}));
    }
    case SemIR::RefForm::Kind: {
      return AddInst(
          context,
          SemIR::LocIdAndInst::RuntimeVerified(
              context.sem_ir(), loc_id,
              SemIR::RefParamPattern{.type_id = type_id,
                                     .pretty_name_id = action.pretty_name_id}));
    }
    case SemIR::ValueForm::Kind: {
      return AddInst(context,
                     SemIR::LocIdAndInst::RuntimeVerified(
                         context.sem_ir(), loc_id,
                         SemIR::ValueParamPattern{
                             .type_id = type_id,
                             .pretty_name_id = action.pretty_name_id}));
    }
    case SemIR::ErrorInst::Kind: {
      return SemIR::ErrorInst::InstId;
    }
    default:
      CARBON_FATAL("Unexpected param pattern form: {0}", form_inst);
  }
}

// TODO: can we share code with FormParamPatternAction?
auto PerformAction(Context& context, SemIR::LocId /*loc_id*/,
                   SemIR::OutFormParamPatternAction action) -> SemIR::InstId {
  auto form_inst = context.insts().Get(
      context.constant_values().GetConstantInstId(action.form_id));
  auto type_id =
      GetPatternType(context, GetTypeComponent(context, action.form_id));
  CARBON_KIND_SWITCH(form_inst) {
    case SemIR::ValueForm::Kind: {
      return AddInst<SemIR::ValueReturnPattern>(
          context, SemIR::LocId(action.form_id), {.type_id = type_id});
    }
    case SemIR::RefForm::Kind: {
      return AddInst<SemIR::RefReturnPattern>(
          context, SemIR::LocId(action.form_id), {.type_id = type_id});
    }
    case CARBON_KIND(SemIR::InitForm _): {
      return AddInst<SemIR::OutParamPattern>(
          context, SemIR::LocId(action.form_id),
          {.type_id = type_id, .pretty_name_id = SemIR::NameId::ReturnSlot});
    }
    case SemIR::ErrorInst::Kind: {
      return SemIR::ErrorInst::InstId;
    }
    default:
      CARBON_FATAL("unexpected inst kind: {0}", form_inst);
  }
}

}  // namespace Carbon::Check
