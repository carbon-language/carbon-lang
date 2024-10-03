// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/pattern_match.h"

#include <functional>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"

namespace Carbon::Check {
namespace {

class MatchContext {
 public:
  struct StackEntry {
    SemIR::InstId pattern_id;
    // Invalid when processing the callee side.
    SemIR::InstId scrutinee_id;
  };

  // FIXME make private

  llvm::SmallVector<StackEntry> stack_;

  // FIXME gross hack
  SemIR::InstId result_ = SemIR::InstId::Invalid;

  SemIR::SpecificId callee_specific_id = SemIR::SpecificId::Invalid;
  // TODO: SemIR::InstBlockId on_failure_;
};

// FIXME make this a method, and loc_id and access_kind be data members?
// FIXME Maybe eliminate loc_id and use the pattern and scrutinee locs, to avoid
// the invalid-loc hack for the function case?
auto ProcessPattern(Context& context, MatchContext& match,
                    MatchContext::StackEntry entry) {
  auto pattern = context.insts().GetWithLocId(entry.pattern_id);
  CARBON_KIND_SWITCH(pattern.inst) {
    case SemIR::BindingPattern::Kind:
    case SemIR::SymbolicBindingPattern::Kind: {
      CARBON_CHECK(!entry.scrutinee_id.is_valid());
      auto binding_pattern = pattern.inst.As<SemIR::AnyBindingPattern>();
      auto bind_name = context.insts().GetAs<SemIR::AnyBindName>(
          binding_pattern.bind_name_id);
      context.inst_block_stack().AddInstId(bind_name.value_id);
      context.inst_block_stack().AddInstId(binding_pattern.bind_name_id);
      match.result_ = binding_pattern.bind_name_id;
      break;
    }
    case CARBON_KIND(SemIR::AddrPattern addr_pattern): {
      if (!entry.scrutinee_id.is_valid()) {
        // We're still on the caller side of the pattern, so we traverse without
        // emitting any insts.
        match.stack_.push_back({.pattern_id = addr_pattern.inner_id,
                                .scrutinee_id = SemIR::InstId::Invalid});
        break;
      }
      auto scrutinee_ref_id =
          ConvertToValueOrRefExpr(context, entry.scrutinee_id);
      switch (SemIR::GetExprCategory(context.sem_ir(), scrutinee_ref_id)) {
        case SemIR::ExprCategory::Error:
        case SemIR::ExprCategory::DurableRef:
        case SemIR::ExprCategory::EphemeralRef:
          break;
        default:
          CARBON_DIAGNOSTIC(AddrSelfIsNonRef, Error,
                            "`addr self` method cannot be invoked on a value");
          context.emitter().Emit(
              TokenOnly(context.insts().GetLocId(entry.scrutinee_id)),
              AddrSelfIsNonRef);
          match.result_ = SemIR::InstId::BuiltinError;
          break;
      }
      auto scrutinee_ref = context.insts().Get(scrutinee_ref_id);
      auto new_scrutinee = context.AddInst<SemIR::AddrOf>(
          context.insts().GetLocId(scrutinee_ref_id),
          {.type_id = context.GetPointerType(scrutinee_ref.type_id()),
           .lvalue_id = scrutinee_ref_id});
      match.stack_.push_back(
          {.pattern_id = addr_pattern.inner_id, .scrutinee_id = new_scrutinee});
      break;
    }
    case CARBON_KIND(SemIR::ParamPattern param_pattern): {
      if (entry.scrutinee_id.is_valid()) {
        match.result_ = ConvertToValueOfType(
            context, context.insts().GetLocId(entry.scrutinee_id),
            entry.scrutinee_id,
            SemIR::GetTypeInSpecific(context.sem_ir(), match.callee_specific_id,
                                     param_pattern.type_id));
        // Do not traverse farther, because the caller side of the pattern ends
        // here.
      } else {
        match.stack_.push_back({.pattern_id = param_pattern.subpattern_id,
                                .scrutinee_id = SemIR::InstId::Invalid});
      }
      break;
    }
    default: {
      CARBON_FATAL("Inst kind not handled: {0}", pattern.inst.kind());
    }
  }
}

auto ProcessParameters(Context& context,
                       llvm::ArrayRef<SemIR::InstId> pattern_ids)
    -> SemIR::InstBlockId {
  MatchContext match;
  std::vector<SemIR::InstId> inner_param_insts;
  inner_param_insts.reserve(pattern_ids.size());
  for (SemIR::InstId inst_id : pattern_ids) {
    match.stack_.push_back(
        {.pattern_id = inst_id, .scrutinee_id = SemIR::InstId::Invalid});
    while (!match.stack_.empty()) {
      auto entry = match.stack_.pop_back_val();
      ProcessPattern(context, match, entry);
    }
    // FIXME Should we break instead, if match.result_ is an error?
    inner_param_insts.push_back(match.result_);
  }

  return context.inst_blocks().Add(inner_param_insts);
}

}  // namespace

auto ProcessSignature(Context& context,
                      SemIR::InstBlockId implicit_param_patterns_id,
                      SemIR::InstBlockId param_patterns_id) -> ParameterBlocks {
  auto params_id = SemIR::InstBlockId::Invalid;
  auto implicit_params_id = SemIR::InstBlockId::Invalid;

  if (implicit_param_patterns_id.is_valid()) {
    implicit_params_id = ProcessParameters(
        context, context.inst_blocks().Get(implicit_param_patterns_id));
  }

  if (param_patterns_id.is_valid()) {
    params_id = ProcessParameters(context,
                                  context.inst_blocks().Get(param_patterns_id));
  }

  return {.implicit_params_id = implicit_params_id, .params_id = params_id};
}

auto PatternMatchArg(Context& context, SemIR::SpecificId specific_id,
                     SemIR::InstId param, SemIR::InstId arg) -> SemIR::InstId {
  MatchContext match;
  match.callee_specific_id = specific_id;
  match.stack_.push_back({.pattern_id = param, .scrutinee_id = arg});
  while (!match.stack_.empty() &&
         match.result_ != SemIR::InstId::BuiltinError) {
    auto entry = match.stack_.pop_back_val();
    ProcessPattern(context, match, entry);
  }
  return match.result_;
}

}  // namespace Carbon::Check
