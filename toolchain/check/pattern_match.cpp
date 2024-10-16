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

// Selects between the different kinds of pattern matching.
enum class MatchKind {
  // Caller pattern matching occurs on the caller side of a function call, and
  // is responsible for matching the argument expression against the portion
  // of the pattern above the ParamPattern insts.
  Caller,

  // Callee pattern matching occurs in the function decl block, and is
  // responsible for matching the function's calling-convention parameters
  // against the portion of the pattern below the ParamPattern insts.
  Callee,

  // TODO: add enumerator for non-function-call pattern match
};

// The collected state of a pattern-matching operation.
class MatchContext {
 public:
  struct WorkItem {
    SemIR::InstId pattern_id;
    // Invalid when processing the callee side.
    SemIR::InstId scrutinee_id;
  };

  // Constructs a MatchContext. If `callee_specific_id` is valid, this pattern
  // match operation is part of implementing the signature of the given
  // specific.
  explicit MatchContext(MatchKind kind, SemIR::SpecificId callee_specific_id =
                                            SemIR::SpecificId::Invalid)
      : result_(SemIR::InstId::Invalid),
        kind_(kind),
        callee_specific_id_(callee_specific_id) {}

  // Returns whether there are any work items to process.
  auto HasWork() const -> bool {
    return !stack_.empty() && !result_.is_valid();
  }

  // Adds a work item to the stack. Cannot be called after Finish().
  auto AddWork(WorkItem work_item) -> void {
    CARBON_CHECK(!result_.is_valid());
    stack_.push_back(work_item);
  }

  // Returns the next work item to process.
  auto NextWorkItem() -> WorkItem { return stack_.pop_back_val(); }

  // Sets the result of this pattern matching operation. Must not be called
  // when there is still pending work, except to report an error.
  auto Finish(SemIR::InstId result) -> void {
    CARBON_CHECK(!HasWork() || result == SemIR::InstId::BuiltinError);
    result_ = result;
  }

  auto result() const -> SemIR::InstId { return result_; }

  auto kind() const -> MatchKind { return kind_; }

  auto callee_specific_id() const -> SemIR::SpecificId {
    return callee_specific_id_;
  }

 private:
  llvm::SmallVector<WorkItem> stack_;

  SemIR::InstId result_;

  MatchKind kind_;

  SemIR::SpecificId callee_specific_id_;
};

// Emits the pattern-match insts necessary to match the pattern inst
// `entry.pattern_id` against the scrutinee value `entry.scrutinee_id`,
// and adds to `match` any work necessary to traverse into its subpatterns.
// This behavior is contingent on the kind of match being performed, as
// indicated by `match.kind()`. For example, when performing a callee
// pattern match, this does not emit insts for patterns on the caller side.
// However, it still traverses into subpatterns if any of their descendants
// might emit insts.
// TODO: Require that `entry.scrutinee_id` is valid if and only if insts should
// be emitted, once we start emitting `Param` insts in the `ParamPattern` case.
auto EmitPatternMatch(Context& context, MatchContext& match,
                      MatchContext::WorkItem entry) -> void {
  auto pattern = context.insts().GetWithLocId(entry.pattern_id);
  CARBON_KIND_SWITCH(pattern.inst) {
    case SemIR::BindingPattern::Kind:
    case SemIR::SymbolicBindingPattern::Kind: {
      CARBON_CHECK(match.kind() == MatchKind::Callee);
      auto binding_pattern = pattern.inst.As<SemIR::AnyBindingPattern>();
      auto bind_name = context.insts().GetAs<SemIR::AnyBindName>(
          binding_pattern.bind_name_id);
      // bind_name.value_id holds the corresponding `Param` inst.
      // TODO: emit the `Param` inst as part of processing `ParamPattern`.
      context.inst_block_stack().AddInstId(bind_name.value_id);
      context.inst_block_stack().AddInstId(binding_pattern.bind_name_id);
      match.Finish(binding_pattern.bind_name_id);
      break;
    }
    case CARBON_KIND(SemIR::AddrPattern addr_pattern): {
      if (match.kind() == MatchKind::Callee) {
        // We're emitting pattern-match IR for the callee, but we're still on
        // the caller side of the pattern, so we traverse without emitting any
        // insts.
        match.AddWork({.pattern_id = addr_pattern.inner_id,
                       .scrutinee_id = SemIR::InstId::Invalid});
        break;
      }
      CARBON_CHECK(entry.scrutinee_id.is_valid());
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
          match.Finish(SemIR::InstId::BuiltinError);
          return;
      }
      auto scrutinee_ref = context.insts().Get(scrutinee_ref_id);
      auto new_scrutinee = context.AddInst<SemIR::AddrOf>(
          context.insts().GetLocId(scrutinee_ref_id),
          {.type_id = context.GetPointerType(scrutinee_ref.type_id()),
           .lvalue_id = scrutinee_ref_id});
      match.AddWork(
          {.pattern_id = addr_pattern.inner_id, .scrutinee_id = new_scrutinee});
      break;
    }
    case CARBON_KIND(SemIR::ParamPattern param_pattern): {
      switch (match.kind()) {
        case MatchKind::Caller: {
          CARBON_CHECK(entry.scrutinee_id.is_valid());
          match.Finish(ConvertToValueOfType(
              context, context.insts().GetLocId(entry.scrutinee_id),
              entry.scrutinee_id,
              SemIR::GetTypeInSpecific(context.sem_ir(),
                                       match.callee_specific_id(),
                                       param_pattern.type_id)));
          // Do not traverse farther, because the caller side of the pattern
          // ends here.
          break;
        }
        case MatchKind::Callee: {
          match.AddWork({.pattern_id = param_pattern.subpattern_id,
                         .scrutinee_id = SemIR::InstId::Invalid});
          break;
        }
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
  std::vector<SemIR::InstId> inner_param_insts;
  inner_param_insts.reserve(pattern_ids.size());
  for (SemIR::InstId inst_id : pattern_ids) {
    MatchContext match(MatchKind::Callee);
    match.AddWork(
        {.pattern_id = inst_id, .scrutinee_id = SemIR::InstId::Invalid});
    while (match.HasWork()) {
      EmitPatternMatch(context, match, match.NextWorkItem());
    }
    // TODO: Should we break instead, if match.result_ is an error?
    inner_param_insts.push_back(match.result());
  }

  return context.inst_blocks().Add(inner_param_insts);
}

}  // namespace

auto CalleePatternMatch(Context& context,
                        SemIR::InstBlockId implicit_param_patterns_id,
                        SemIR::InstBlockId param_patterns_id)
    -> ParameterBlocks {
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

auto CallerPatternMatch(Context& context, SemIR::SpecificId specific_id,
                        SemIR::InstId param, SemIR::InstId arg)
    -> SemIR::InstId {
  MatchContext match(MatchKind::Caller, specific_id);
  match.AddWork({.pattern_id = param, .scrutinee_id = arg});
  while (match.HasWork()) {
    EmitPatternMatch(context, match, match.NextWorkItem());
  }
  return match.result();
}

}  // namespace Carbon::Check
