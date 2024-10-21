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

// Returns a best-effort name for the given ParamPattern, suitable for use in
// IR pretty-printing.
// TODO: Resolve overlap with SemIR::Function::ParamPatternInfo::GetNameId
auto GetPrettyName(Context& context, SemIR::ParamPattern param_pattern)
    -> SemIR::NameId {
  if (context.insts().Is<SemIR::ReturnSlotPattern>(
          param_pattern.subpattern_id)) {
    return SemIR::NameId::ReturnSlot;
  }
  if (auto binding_pattern = context.insts().TryGetAs<SemIR::AnyBindingPattern>(
          param_pattern.subpattern_id)) {
    return context.entity_names().Get(binding_pattern->entity_name_id).name_id;
  }
  return SemIR::NameId::Invalid;
}

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
      : next_index_(0),
        result_(SemIR::InstId::Invalid),
        kind_(kind),
        callee_specific_id_(callee_specific_id),
        return_slot_id_(SemIR::InstId::Invalid) {}

  // Returns whether there are any work items to process.
  auto HasWork() const -> bool {
    return !stack_.empty() && !result_.is_valid();
  }

  // Adds a work item to the stack. Cannot be called after Finish().
  auto AddWork(WorkItem work_item) -> void { stack_.push_back(work_item); }

  // Returns the next work item to process.
  auto NextWorkItem() -> WorkItem { return stack_.pop_back_val(); }

  // Allocates the next unallocated RuntimeParamIndex, starting from 0.
  auto NextRuntimeIndex() -> SemIR::RuntimeParamIndex {
    auto result = next_index_;
    ++next_index_.index;
    return result;
  }

  // TODO: Eliminate the caller/callee API split below, by restructuring
  // CallerPatternMatch to operate on the whole pattern.

  // Sets the result of this pattern matching operation. Must not be called when
  // there is still pending work, except to report an error, or called more than
  // once between calls to ConsumeResult. Valid only during caller matching.
  auto Finish(SemIR::InstId result) -> void {
    CARBON_CHECK(!HasWork() || result == SemIR::InstId::BuiltinError);
    CARBON_CHECK(kind_ == MatchKind::Caller);
    CARBON_CHECK(result_ == SemIR::InstId::Invalid);
    result_ = result;
  }

  // Consumes and returns the result stored by Finish. Valid only during caller
  // matching.
  auto ConsumeResult() -> SemIR::InstId {
    CARBON_CHECK(stack_.empty() || result_ == SemIR::InstId::BuiltinError);
    CARBON_CHECK(kind_ == MatchKind::Caller);
    return std::exchange(result_, SemIR::InstId::Invalid);
  }

  // Records that `bind_name_id` is the ID of an inst in the AnyBindName
  // category, emitted as part of this pattern match. Valid only during callee
  // pattern matching.
  auto RecordBindName(SemIR::InstId bind_name_id) {
    CARBON_CHECK(kind_ == MatchKind::Callee);
    bind_name_ids_.push_back(bind_name_id);
  }

  // Allocates an InstBlock containing the IDs recorded by RecordBindName since
  // the last call to this function (if any), and returns its ID. Valid only
  // during callee pattern matching.
  auto ConsumeBindNames(Context& context) -> SemIR::InstBlockId {
    CARBON_CHECK(stack_.empty());
    CARBON_CHECK(kind_ == MatchKind::Callee);
    auto block_id = context.inst_blocks().Add(bind_name_ids_);
    bind_name_ids_.clear();
    return block_id;
  }

  auto kind() const -> MatchKind { return kind_; }

  auto callee_specific_id() const -> SemIR::SpecificId {
    return callee_specific_id_;
  }

  auto return_slot_id() const -> SemIR::InstId { return return_slot_id_; }

  auto set_return_slot_id(SemIR::InstId return_slot_id) {
    return_slot_id_ = return_slot_id;
  }

 private:
  llvm::SmallVector<WorkItem> stack_;

  SemIR::RuntimeParamIndex next_index_;

  SemIR::InstId result_;

  llvm::SmallVector<SemIR::InstId> bind_name_ids_;

  MatchKind kind_;

  SemIR::SpecificId callee_specific_id_;

  SemIR::InstId return_slot_id_;
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
  if (entry.pattern_id == SemIR::InstId::BuiltinError) {
    match.RecordBindName(SemIR::InstId::BuiltinError);
    return;
  }
  auto pattern = context.insts().GetWithLocId(entry.pattern_id);
  CARBON_KIND_SWITCH(pattern.inst) {
    case SemIR::BindingPattern::Kind:
    case SemIR::SymbolicBindingPattern::Kind: {
      CARBON_CHECK(match.kind() == MatchKind::Callee);
      auto binding_pattern = pattern.inst.As<SemIR::AnyBindingPattern>();
      auto bind_name = context.insts().GetAs<SemIR::AnyBindName>(
          binding_pattern.bind_name_id);
      CARBON_CHECK(!bind_name.value_id.is_valid());
      bind_name.value_id = entry.scrutinee_id;
      context.ReplaceInstBeforeConstantUse(binding_pattern.bind_name_id,
                                           bind_name);
      context.inst_block_stack().AddInstId(binding_pattern.bind_name_id);
      match.RecordBindName(binding_pattern.bind_name_id);
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
          if (param_pattern.runtime_index ==
              SemIR::RuntimeParamIndex::Unknown) {
            param_pattern.runtime_index = match.NextRuntimeIndex();
            context.ReplaceInstBeforeConstantUse(entry.pattern_id,
                                                 param_pattern);
          }
          match.AddWork(
              {.pattern_id = param_pattern.subpattern_id,
               .scrutinee_id = context.AddInst<SemIR::Param>(
                   pattern.loc_id,
                   {.type_id = param_pattern.type_id,
                    .runtime_index = param_pattern.runtime_index,
                    .pretty_name = GetPrettyName(context, param_pattern)})});
        } break;
      }
      break;
    }
    case CARBON_KIND(SemIR::ReturnSlotPattern return_slot_pattern): {
      CARBON_CHECK(match.kind() == MatchKind::Callee);
      match.set_return_slot_id(context.AddInst<SemIR::ReturnSlot>(
          pattern.loc_id, {.type_id = return_slot_pattern.type_id,
                           .value_id = entry.scrutinee_id}));
      break;
    }
    default: {
      CARBON_FATAL("Inst kind not handled: {0}", pattern.inst.kind());
    }
  }
}

}  // namespace

auto CalleePatternMatch(Context& context,
                        SemIR::InstBlockId implicit_param_patterns_id,
                        SemIR::InstBlockId param_patterns_id,
                        SemIR::InstId return_slot_pattern_id)
    -> ParameterBlocks {
  auto params_id = SemIR::InstBlockId::Invalid;
  auto implicit_params_id = SemIR::InstBlockId::Invalid;

  MatchContext match(MatchKind::Callee);
  // TODO reserve space in bind_name_ids_

  if (implicit_param_patterns_id.is_valid()) {
    // We add work to the stack in reverse so that the results will be produced
    // in the original order.
    for (SemIR::InstId inst_id :
         llvm::reverse(context.inst_blocks().Get(implicit_param_patterns_id))) {
      match.AddWork(
          {.pattern_id = inst_id, .scrutinee_id = SemIR::InstId::Invalid});
    }
    while (match.HasWork()) {
      EmitPatternMatch(context, match, match.NextWorkItem());
    }
    implicit_params_id = match.ConsumeBindNames(context);
  }

  if (param_patterns_id.is_valid()) {
    for (SemIR::InstId inst_id :
         llvm::reverse(context.inst_blocks().Get(param_patterns_id))) {
      match.AddWork(
          {.pattern_id = inst_id, .scrutinee_id = SemIR::InstId::Invalid});
    }
    while (match.HasWork()) {
      EmitPatternMatch(context, match, match.NextWorkItem());
    }
    params_id = match.ConsumeBindNames(context);
  }

  if (return_slot_pattern_id.is_valid()) {
    match.AddWork({.pattern_id = return_slot_pattern_id,
                   .scrutinee_id = SemIR::InstId::Invalid});
    while (match.HasWork()) {
      EmitPatternMatch(context, match, match.NextWorkItem());
    }
  }

  return {.implicit_params_id = implicit_params_id,
          .params_id = params_id,
          .return_slot_id = match.return_slot_id()};
}

auto CallerPatternMatch(Context& context, SemIR::SpecificId specific_id,
                        SemIR::InstId param, SemIR::InstId arg)
    -> SemIR::InstId {
  MatchContext match(MatchKind::Caller, specific_id);
  match.AddWork({.pattern_id = param, .scrutinee_id = arg});
  while (match.HasWork()) {
    EmitPatternMatch(context, match, match.NextWorkItem());
  }
  return match.ConsumeResult();
}

}  // namespace Carbon::Check
