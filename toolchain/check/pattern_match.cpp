// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/pattern_match.h"

#include <functional>
#include <utility>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/control_flow.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/pattern.h"
#include "toolchain/check/type.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/sem_ir/expr_info.h"
#include "toolchain/sem_ir/pattern.h"

namespace Carbon::Check {

namespace {

// Selects between the different kinds of pattern matching.
enum class MatchKind : uint8_t {
  // Caller pattern matching occurs on the caller side of a function call, and
  // is responsible for matching the argument expression against the portion
  // of the pattern above the ParamPattern insts.
  Caller,

  // Callee pattern matching occurs in the function decl block, and is
  // responsible for matching the function's calling-convention parameters
  // against the portion of the pattern below the ParamPattern insts.
  Callee,

  // Local pattern matching is pattern matching outside of a function call,
  // such as in a let/var declaration.
  Local,
};

// The collected state of a pattern-matching operation.
class MatchContext {
 public:
  struct WorkItem : Printable<WorkItem> {
    SemIR::InstId pattern_id;
    // `None` when processing the callee side.
    SemIR::InstId scrutinee_id;

    // If true, disables diagnostics that would otherwise require scrutinee_id
    // to be tagged with `ref`. Only affects caller pattern matching.
    bool allow_unmarked_ref = false;
    auto Print(llvm::raw_ostream& out) const -> void {
      out << "{pattern_id: " << pattern_id << ", scrutinee_id: " << scrutinee_id
          << ", allow_unmarked_ref = " << allow_unmarked_ref << "}";
    }
  };

  // Constructs a MatchContext. If `callee_specific_id` is not `None`, this
  // pattern match operation is part of implementing the signature of the given
  // specific.
  explicit MatchContext(MatchKind kind, SemIR::SpecificId callee_specific_id =
                                            SemIR::SpecificId::None)
      : kind_(kind), callee_specific_id_(callee_specific_id) {}

  // Adds a work item to the stack.
  auto AddWork(WorkItem work_item) -> void { stack_.push_back(work_item); }

  // Processes all work items on the stack.
  auto DoWork(Context& context) -> void;

  // Returns an inst block of references to all the emitted `Call` arguments.
  // Can only be called once, at the end of Caller pattern matching.
  auto CallerResults(Context& context) && -> SemIR::InstBlockId;

  // Returns an inst block of references to all the emitted `Call` params,
  // and an inst block of references to the `Call` param patterns they were
  // emitted to match. Can only be called once, at the end of Callee pattern
  // matching.
  auto CalleeResults(Context& context) && -> CalleePatternMatchResults;

  ~MatchContext();

 private:
  // Emits the pattern-match insts necessary to match the pattern inst
  // `entry.pattern_id` against the scrutinee value `entry.scrutinee_id`, and
  // adds to `stack_` any work necessary to traverse into its subpatterns. This
  // behavior is contingent on the kind of match being performed, as indicated
  // by kind_`. For example, when performing a callee pattern match, this does
  // not emit insts for patterns on the caller side. However, it still traverses
  // into subpatterns if any of their descendants might emit insts.
  // TODO: Require that `entry.scrutinee_id` is valid if and only if insts
  // should be emitted, once we start emitting `Param` insts in the
  // `ParamPattern` case.
  auto EmitPatternMatch(Context& context, MatchContext::WorkItem entry) -> void;

  // Implementations of `EmitPatternMatch` for particular pattern inst kinds.
  // The pattern argument is always equal to
  // `context.insts().Get(entry.pattern_id)`.
  auto DoEmitPatternMatch(Context& context,
                          SemIR::AnyBindingPattern binding_pattern,
                          WorkItem entry) -> void;
  auto DoEmitPatternMatch(Context& context,
                          SemIR::ValueParamPattern param_pattern,
                          WorkItem entry) -> void;
  template <typename RefParamPatternT>
    requires std::is_same_v<RefParamPatternT, SemIR::RefParamPattern> ||
             std::is_same_v<RefParamPatternT, SemIR::VarParamPattern>
  auto DoEmitPatternMatch(Context& context, RefParamPatternT param_pattern,
                          WorkItem entry) -> void;
  auto DoEmitPatternMatch(Context& context,
                          SemIR::OutParamPattern param_pattern, WorkItem entry)
      -> void;
  auto DoEmitPatternMatch(Context& context,
                          SemIR::ReturnSlotPattern return_slot_pattern,
                          WorkItem entry) -> void;
  auto DoEmitPatternMatch(Context& context, SemIR::VarPattern var_pattern,
                          WorkItem entry) -> void;
  auto DoEmitPatternMatch(Context& context, SemIR::TuplePattern tuple_pattern,
                          WorkItem entry) -> void;

  // The stack of work to be processed.
  llvm::SmallVector<WorkItem> stack_;

  // The in-progress contents of the `Call` arguments block. This is populated
  // only when kind_ is Caller.
  llvm::SmallVector<SemIR::InstId> call_args_;

  // The in-progress contents of the `Call` parameters block. This is populated
  // only when kind_ is Callee.
  llvm::SmallVector<SemIR::InstId> call_params_;

  // The in-progress contents of the `Call` parameter patterns block. This is
  // populated only when kind_ is Callee.
  llvm::SmallVector<SemIR::InstId> call_param_patterns_;

  // The kind of pattern match being performed.
  MatchKind kind_;

  // The SpecificId of the function being called (if any).
  SemIR::SpecificId callee_specific_id_;
};

}  // namespace

auto MatchContext::DoWork(Context& context) -> void {
  CARBON_CHECK(call_args_.empty() && call_params_.empty() &&
               call_param_patterns_.empty());
  switch (kind_) {
    case MatchKind::Caller: {
      call_args_.reserve(stack_.size());
      break;
    }
    case MatchKind::Callee: {
      call_param_patterns_.reserve(stack_.size());
      call_params_.reserve(stack_.size());
      break;
    }
    case MatchKind::Local:
      break;
  }
  while (!stack_.empty()) {
    EmitPatternMatch(context, stack_.pop_back_val());
  }
}

auto MatchContext::CallerResults(Context& context) && -> SemIR::InstBlockId {
  CARBON_CHECK(kind_ == MatchKind::Caller);
  auto block_id = context.inst_blocks().Add(call_args_);
  call_args_.clear();
  return block_id;
}

auto MatchContext::CalleeResults(
    Context& context) && -> CalleePatternMatchResults {
  CARBON_CHECK(kind_ == MatchKind::Callee);
  CARBON_CHECK(call_params_.size() == call_param_patterns_.size());
  auto call_param_patterns_id = context.inst_blocks().Add(call_param_patterns_);
  call_param_patterns_.clear();
  auto call_params_id = context.inst_blocks().Add(call_params_);
  call_params_.clear();
  return {.call_param_patterns_id = call_param_patterns_id,
          .call_params_id = call_params_id};
}

MatchContext::~MatchContext() {
  CARBON_CHECK(call_args_.empty() && call_params_.empty() &&
                   call_param_patterns_.empty(),
               "Unhandled pattern matching outputs. call_args_.size(): {0}, "
               "call_params_.size(): {1}, call_param_patterns_.size(): {2}",
               call_args_.size(), call_params_.size(),
               call_param_patterns_.size());
}

// Inserts the given region into the current code block. If the region
// consists of a single block, this will be implemented as a `splice_block`
// inst. Otherwise, this will end the current block with a branch to the entry
// block of the region, and add future insts to a new block which is the
// immediate successor of the region's exit block. As a result, this cannot be
// called more than once for the same region.
static auto InsertHere(Context& context, SemIR::ExprRegionId region_id)
    -> SemIR::InstId {
  auto region = context.sem_ir().expr_regions().Get(region_id);
  auto exit_block = context.inst_blocks().Get(region.block_ids.back());
  if (region.block_ids.size() == 1) {
    // TODO: Is it possible to avoid leaving an "orphan" block in the IR in the
    // first two cases?
    if (exit_block.empty()) {
      return region.result_id;
    }
    if (exit_block.size() == 1) {
      context.inst_block_stack().AddInstId(exit_block.front());
      return region.result_id;
    }
    return AddInst<SemIR::SpliceBlock>(
        context, SemIR::LocId(region.result_id),
        {.type_id = context.insts().Get(region.result_id).type_id(),
         .block_id = region.block_ids.front(),
         .result_id = region.result_id});
  }
  if (context.region_stack().empty()) {
    context.TODO(region.result_id,
                 "Control flow expressions are currently only supported inside "
                 "functions.");
    return SemIR::ErrorInst::InstId;
  }
  AddInst(context, SemIR::LocIdAndInst::NoLoc<SemIR::Branch>(
                       {.target_id = region.block_ids.front()}));
  context.inst_block_stack().Pop();
  // TODO: this will cumulatively cost O(MN) running time for M blocks
  // at the Nth level of the stack. Figure out how to do better.
  context.region_stack().AddToRegion(region.block_ids);
  auto resume_with_block_id =
      context.insts().GetAs<SemIR::Branch>(exit_block.back()).target_id;
  CARBON_CHECK(context.inst_blocks().GetOrEmpty(resume_with_block_id).empty());
  context.inst_block_stack().Push(resume_with_block_id);
  context.region_stack().AddToRegion(resume_with_block_id,
                                     SemIR::LocId(region.result_id));
  return region.result_id;
}

auto MatchContext::DoEmitPatternMatch(Context& context,
                                      SemIR::AnyBindingPattern binding_pattern,
                                      MatchContext::WorkItem entry) -> void {
  if (kind_ == MatchKind::Caller) {
    CARBON_CHECK(
        binding_pattern.kind == SemIR::SymbolicBindingPattern::Kind,
        "Found named runtime binding pattern during caller pattern match");
    return;
  }
  // We're logically consuming this map entry, so we invalidate it in order
  // to avoid accidentally consuming it twice.
  auto [bind_name_id, type_expr_region_id] =
      std::exchange(context.bind_name_map().Lookup(entry.pattern_id).value(),
                    {.bind_name_id = SemIR::InstId::None,
                     .type_expr_region_id = SemIR::ExprRegionId::None});
  // bind_name_id doesn't have a value in the case of an unused binding pattern,
  // but type_expr_region_id should always be populated.
  CARBON_CHECK(type_expr_region_id.has_value());
  InsertHere(context, type_expr_region_id);
  auto value_id = SemIR::InstId::None;
  if (kind_ == MatchKind::Local) {
    auto conversion_kind = [&binding_pattern]() -> ConversionTarget::Kind {
      switch (binding_pattern.kind) {
        case SemIR::SymbolicBindingPattern::Kind:
        case SemIR::ValueBindingPattern::Kind:
          return ConversionTarget::Value;
        case SemIR::RefBindingPattern::Kind:
          return ConversionTarget::DurableRef;
        default:
          CARBON_FATAL("Unexpected inst kind {0}", binding_pattern.kind);
      }
    }();

    if (!bind_name_id.has_value()) {
      // TODO: Is this appropriate, or should we perform a conversion based on
      // whether the `_` binding is a value or ref binding first, and then
      // separately discard the initializer for a `_` binding?
      conversion_kind = ConversionTarget::Discarded;
    }
    value_id =
        Convert(context, SemIR::LocId(entry.scrutinee_id), entry.scrutinee_id,
                {.kind = conversion_kind,
                 .type_id = context.insts().Get(bind_name_id).type_id()});
  } else {
    // In a function call, conversion is handled while matching the enclosing
    // `*ParamPattern`.
    value_id = entry.scrutinee_id;
  }
  if (bind_name_id.has_value()) {
    auto bind_name = context.insts().GetAs<SemIR::AnyBinding>(bind_name_id);
    CARBON_CHECK(!bind_name.value_id.has_value());
    bind_name.value_id = value_id;
    ReplaceInstBeforeConstantUse(context, bind_name_id, bind_name);
    context.inst_block_stack().AddInstId(bind_name_id);
  }
}

auto MatchContext::DoEmitPatternMatch(Context& context,
                                      SemIR::ValueParamPattern param_pattern,
                                      WorkItem entry) -> void {
  switch (kind_) {
    case MatchKind::Caller: {
      CARBON_CHECK(
          static_cast<size_t>(param_pattern.index.index) == call_args_.size(),
          "Parameters out of order; expecting {0} but got {1}",
          call_args_.size(), param_pattern.index.index);
      CARBON_CHECK(entry.scrutinee_id.has_value());
      if (entry.scrutinee_id == SemIR::ErrorInst::InstId) {
        call_args_.push_back(SemIR::ErrorInst::InstId);
      } else {
        call_args_.push_back(ConvertToValueOfType(
            context, SemIR::LocId(entry.scrutinee_id), entry.scrutinee_id,
            ExtractScrutineeType(
                context.sem_ir(),
                SemIR::GetTypeOfInstInSpecific(
                    context.sem_ir(), callee_specific_id_, entry.pattern_id))));
      }
      // Do not traverse farther, because the caller side of the pattern
      // ends here.
      break;
    }
    case MatchKind::Callee: {
      auto param_id = AddInst<SemIR::ValueParam>(
          context, SemIR::LocId(entry.pattern_id),
          {.type_id =
               ExtractScrutineeType(context.sem_ir(), param_pattern.type_id),
           .index = param_pattern.index,
           .pretty_name_id = SemIR::GetPrettyNameFromPatternId(
               context.sem_ir(), entry.pattern_id)});
      AddWork({.pattern_id = param_pattern.subpattern_id,
               .scrutinee_id = param_id});
      call_params_.push_back(param_id);
      call_param_patterns_.push_back(entry.pattern_id);
      break;
    }
    case MatchKind::Local: {
      CARBON_FATAL("Found ValueParamPattern during local pattern match");
    }
  }
}

template <typename RefParamPatternT>
  requires std::is_same_v<RefParamPatternT, SemIR::RefParamPattern> ||
           std::is_same_v<RefParamPatternT, SemIR::VarParamPattern>
auto MatchContext::DoEmitPatternMatch(Context& context,
                                      RefParamPatternT param_pattern,
                                      WorkItem entry) -> void {
  switch (kind_) {
    case MatchKind::Caller: {
      CARBON_CHECK(
          static_cast<size_t>(param_pattern.index.index) == call_args_.size(),
          "Parameters out of order; expecting {0} but got {1}",
          call_args_.size(), param_pattern.index.index);
      CARBON_CHECK(entry.scrutinee_id.has_value());

      if (std::is_same_v<RefParamPatternT, SemIR::VarParamPattern>) {
        call_args_.push_back(entry.scrutinee_id);
        break;
      }
      auto scrutinee_type_id = ExtractScrutineeType(
          context.sem_ir(),
          SemIR::GetTypeOfInstInSpecific(context.sem_ir(), callee_specific_id_,
                                         entry.pattern_id));
      call_args_.push_back(Convert(
          context, SemIR::LocId(entry.scrutinee_id), entry.scrutinee_id,
          {.kind = entry.allow_unmarked_ref ? ConversionTarget::UnmarkedRefParam
                                            : ConversionTarget::RefParam,
           .type_id = scrutinee_type_id}));
      // Do not traverse farther, because the caller side of the pattern
      // ends here.
      break;
    }
    case MatchKind::Callee: {
      auto param_id = AddInst<SemIR::RefParam>(
          context, SemIR::LocId(entry.pattern_id),
          {.type_id =
               ExtractScrutineeType(context.sem_ir(), param_pattern.type_id),
           .index = param_pattern.index,
           .pretty_name_id = SemIR::GetPrettyNameFromPatternId(
               context.sem_ir(), entry.pattern_id)});
      AddWork({.pattern_id = param_pattern.subpattern_id,
               .scrutinee_id = param_id});
      call_params_.push_back(param_id);
      call_param_patterns_.push_back(entry.pattern_id);
      break;
    }
    case MatchKind::Local: {
      CARBON_FATAL("Found RefParamPattern during local pattern match");
    }
  }
}

auto MatchContext::DoEmitPatternMatch(Context& context,
                                      SemIR::OutParamPattern param_pattern,
                                      WorkItem entry) -> void {
  switch (kind_) {
    case MatchKind::Caller: {
      CARBON_CHECK(
          static_cast<size_t>(param_pattern.index.index) == call_args_.size(),
          "Parameters out of order; expecting {0} but got {1}",
          call_args_.size(), param_pattern.index.index);
      CARBON_CHECK(entry.scrutinee_id.has_value());
      CARBON_CHECK(
          context.insts().Get(entry.scrutinee_id).type_id() ==
          ExtractScrutineeType(
              context.sem_ir(),
              SemIR::GetTypeOfInstInSpecific(
                  context.sem_ir(), callee_specific_id_, entry.pattern_id)));
      call_args_.push_back(entry.scrutinee_id);
      // Do not traverse farther, because the caller side of the pattern
      // ends here.
      break;
    }
    case MatchKind::Callee: {
      // TODO: Consider ways to address near-duplication with the
      // other ParamPattern cases.
      auto param_id = AddInst<SemIR::OutParam>(
          context, SemIR::LocId(entry.pattern_id),
          {.type_id =
               ExtractScrutineeType(context.sem_ir(), param_pattern.type_id),
           .index = param_pattern.index,
           .pretty_name_id = SemIR::GetPrettyNameFromPatternId(
               context.sem_ir(), entry.pattern_id)});
      AddWork({.pattern_id = param_pattern.subpattern_id,
               .scrutinee_id = param_id});
      call_param_patterns_.push_back(entry.pattern_id);
      call_params_.push_back(param_id);
      break;
    }
    case MatchKind::Local: {
      CARBON_FATAL("Found OutParamPattern during local pattern match");
    }
  }
}

auto MatchContext::DoEmitPatternMatch(
    Context& context, SemIR::ReturnSlotPattern return_slot_pattern,
    WorkItem entry) -> void {
  CARBON_CHECK(kind_ == MatchKind::Callee);
  auto type_id =
      ExtractScrutineeType(context.sem_ir(), return_slot_pattern.type_id);
  auto return_slot_id = AddInst<SemIR::ReturnSlot>(
      context, SemIR::LocId(entry.pattern_id),
      {.type_id = type_id,
       .type_inst_id = context.types().GetTypeInstId(type_id),
       .storage_id = entry.scrutinee_id});
  bool already_in_lookup =
      context.scope_stack()
          .LookupOrAddName(SemIR::NameId::ReturnSlot, return_slot_id)
          .has_value();
  CARBON_CHECK(!already_in_lookup);
}

auto MatchContext::DoEmitPatternMatch(Context& context,
                                      SemIR::VarPattern var_pattern,
                                      WorkItem entry) -> void {
  auto storage_id = SemIR::InstId::None;
  switch (kind_) {
    case MatchKind::Callee: {
      // We're emitting pattern-match IR for the callee, but we're still on
      // the caller side of the pattern, so we traverse without emitting any
      // insts.
      AddWork({.pattern_id = var_pattern.subpattern_id,
               .scrutinee_id = SemIR::InstId::None});
      return;
    }
    case MatchKind::Local: {
      // In a `var`/`let` declaration, the `VarStorage` inst is created before
      // we start pattern matching.
      auto lookup_result = context.var_storage_map().Lookup(entry.pattern_id);
      CARBON_CHECK(lookup_result);
      storage_id = lookup_result.value();
      break;
    }
    case MatchKind::Caller: {
      storage_id = AddInst<SemIR::TemporaryStorage>(
          context, SemIR::LocId(entry.pattern_id),
          {.type_id =
               ExtractScrutineeType(context.sem_ir(), var_pattern.type_id)});
      CARBON_CHECK(entry.scrutinee_id.has_value());
      break;
    }
  }
  // TODO: Find a more efficient way to put these insts in the global_init
  // block (or drop the distinction between the global_init block and the
  // file scope?)
  if (context.scope_stack().PeekIndex() == ScopeIndex::Package) {
    context.global_init().Resume();
  }
  if (entry.scrutinee_id.has_value()) {
    auto init_id = Initialize(context, SemIR::LocId(entry.pattern_id),
                              storage_id, entry.scrutinee_id);
    // If we created a `TemporaryStorage` to hold the var, create a
    // corresponding `Temporary` to model that its initialization is complete.
    // TODO: If the subpattern is a binding, we may want to destroy the
    // parameter variable in the callee instead of the caller so that we can
    // support destructive move from it.
    if (kind_ == MatchKind::Caller) {
      storage_id = AddInstWithCleanup<SemIR::Temporary>(
          context, SemIR::LocId(entry.pattern_id),
          {.type_id = context.insts().Get(storage_id).type_id(),
           .storage_id = storage_id,
           .init_id = init_id});
    } else {
      // TODO: Consider using different instruction kinds for assignment
      // versus initialization.
      AddInst<SemIR::Assign>(context, SemIR::LocId(entry.pattern_id),
                             {.lhs_id = storage_id, .rhs_id = init_id});
    }
  }
  AddWork(
      {.pattern_id = var_pattern.subpattern_id, .scrutinee_id = storage_id});
  if (context.scope_stack().PeekIndex() == ScopeIndex::Package) {
    context.global_init().Suspend();
  }
}

auto MatchContext::DoEmitPatternMatch(Context& context,
                                      SemIR::TuplePattern tuple_pattern,
                                      WorkItem entry) -> void {
  if (tuple_pattern.type_id == SemIR::ErrorInst::TypeId) {
    return;
  }
  auto subpattern_ids = context.inst_blocks().Get(tuple_pattern.elements_id);
  auto add_all_subscrutinees =
      [&](llvm::ArrayRef<SemIR::InstId> subscrutinee_ids) {
        for (auto [subpattern_id, subscrutinee_id] :
             llvm::reverse(llvm::zip_equal(subpattern_ids, subscrutinee_ids))) {
          AddWork(
              {.pattern_id = subpattern_id, .scrutinee_id = subscrutinee_id});
        }
      };
  if (!entry.scrutinee_id.has_value()) {
    CARBON_CHECK(kind_ == MatchKind::Callee);
    // If we don't have a scrutinee yet, we're still on the caller side of the
    // pattern, so the subpatterns don't have a scrutinee either.
    for (auto subpattern_id : llvm::reverse(subpattern_ids)) {
      AddWork(
          {.pattern_id = subpattern_id, .scrutinee_id = SemIR::InstId::None});
    }
    return;
  }
  auto scrutinee = context.insts().GetWithLocId(entry.scrutinee_id);
  if (auto scrutinee_literal = scrutinee.inst.TryAs<SemIR::TupleLiteral>()) {
    auto subscrutinee_ids =
        context.inst_blocks().Get(scrutinee_literal->elements_id);
    if (subscrutinee_ids.size() != subpattern_ids.size()) {
      CARBON_DIAGNOSTIC(TuplePatternSizeDoesntMatchLiteral, Error,
                        "tuple pattern expects {0} element{0:s}, but tuple "
                        "literal has {1}",
                        Diagnostics::IntAsSelect, Diagnostics::IntAsSelect);
      context.emitter().Emit(entry.pattern_id,
                             TuplePatternSizeDoesntMatchLiteral,
                             subpattern_ids.size(), subscrutinee_ids.size());
      return;
    }
    add_all_subscrutinees(subscrutinee_ids);
    return;
  }

  auto tuple_type_id =
      ExtractScrutineeType(context.sem_ir(), tuple_pattern.type_id);
  auto converted_scrutinee_id =
      ConvertToValueOrRefOfType(context, SemIR::LocId(entry.pattern_id),
                                entry.scrutinee_id, tuple_type_id);
  if (auto scrutinee_value =
          context.insts().TryGetAs<SemIR::TupleValue>(converted_scrutinee_id)) {
    add_all_subscrutinees(
        context.inst_blocks().Get(scrutinee_value->elements_id));
    return;
  }

  auto tuple_type = context.types().GetAs<SemIR::TupleType>(tuple_type_id);
  auto element_type_inst_ids =
      context.inst_blocks().Get(tuple_type.type_elements_id);
  llvm::SmallVector<SemIR::InstId> subscrutinee_ids;
  subscrutinee_ids.reserve(element_type_inst_ids.size());
  for (auto [i, element_type_id] : llvm::enumerate(
           context.types().GetBlockAsTypeIds(element_type_inst_ids))) {
    subscrutinee_ids.push_back(
        AddInst<SemIR::TupleAccess>(context, scrutinee.loc_id,
                                    {.type_id = element_type_id,
                                     .tuple_id = converted_scrutinee_id,
                                     .index = SemIR::ElementIndex(i)}));
  }
  add_all_subscrutinees(subscrutinee_ids);
}

auto MatchContext::EmitPatternMatch(Context& context,
                                    MatchContext::WorkItem entry) -> void {
  if (entry.pattern_id == SemIR::ErrorInst::InstId) {
    return;
  }
  Diagnostics::AnnotationScope annotate_diagnostics(
      &context.emitter(), [&](auto& builder) {
        if (kind_ == MatchKind::Caller) {
          CARBON_DIAGNOSTIC(InCallToFunctionParam, Note,
                            "initializing function parameter");
          builder.Note(entry.pattern_id, InCallToFunctionParam);
        }
      });
  auto pattern = context.insts().Get(entry.pattern_id);
  CARBON_KIND_SWITCH(pattern) {
    case SemIR::RefBindingPattern::Kind:
    case SemIR::SymbolicBindingPattern::Kind:
    case SemIR::ValueBindingPattern::Kind: {
      DoEmitPatternMatch(context, pattern.As<SemIR::AnyBindingPattern>(),
                         entry);
      break;
    }
    case CARBON_KIND(SemIR::ValueParamPattern param_pattern): {
      DoEmitPatternMatch(context, param_pattern, entry);
      break;
    }
    case CARBON_KIND(SemIR::RefParamPattern param_pattern): {
      DoEmitPatternMatch(context, param_pattern, entry);
      break;
    }
    case CARBON_KIND(SemIR::VarParamPattern param_pattern): {
      DoEmitPatternMatch(context, param_pattern, entry);
      break;
    }
    case CARBON_KIND(SemIR::OutParamPattern param_pattern): {
      DoEmitPatternMatch(context, param_pattern, entry);
      break;
    }
    case CARBON_KIND(SemIR::ReturnSlotPattern return_slot_pattern): {
      DoEmitPatternMatch(context, return_slot_pattern, entry);
      break;
    }
    case CARBON_KIND(SemIR::VarPattern var_pattern): {
      DoEmitPatternMatch(context, var_pattern, entry);
      break;
    }
    case CARBON_KIND(SemIR::TuplePattern tuple_pattern): {
      DoEmitPatternMatch(context, tuple_pattern, entry);
      break;
    }
    default: {
      CARBON_FATAL("Inst kind not handled: {0}", pattern.kind());
    }
  }
}

auto CalleePatternMatch(Context& context,
                        SemIR::InstBlockId implicit_param_patterns_id,
                        SemIR::InstBlockId param_patterns_id,
                        SemIR::InstBlockId return_patterns_id)
    -> CalleePatternMatchResults {
  if (!return_patterns_id.has_value() && !param_patterns_id.has_value() &&
      !implicit_param_patterns_id.has_value()) {
    return {.call_param_patterns_id = SemIR::InstBlockId::None,
            .call_params_id = SemIR::InstBlockId::None};
  }

  MatchContext match(MatchKind::Callee);

  // We add work to the stack in reverse so that the results will be produced
  // in the original order.
  for (auto return_pattern_id :
       context.inst_blocks().GetOrEmpty(return_patterns_id)) {
    match.AddWork(
        {.pattern_id = return_pattern_id, .scrutinee_id = SemIR::InstId::None});
  }

  if (param_patterns_id.has_value()) {
    for (SemIR::InstId inst_id :
         llvm::reverse(context.inst_blocks().Get(param_patterns_id))) {
      match.AddWork(
          {.pattern_id = inst_id, .scrutinee_id = SemIR::InstId::None});
    }
  }

  if (implicit_param_patterns_id.has_value()) {
    for (SemIR::InstId inst_id :
         llvm::reverse(context.inst_blocks().Get(implicit_param_patterns_id))) {
      match.AddWork(
          {.pattern_id = inst_id, .scrutinee_id = SemIR::InstId::None});
    }
  }

  match.DoWork(context);
  return std::move(match).CalleeResults(context);
}

auto CallerPatternMatch(Context& context, SemIR::SpecificId specific_id,
                        SemIR::InstId self_pattern_id,
                        SemIR::InstBlockId param_patterns_id,
                        SemIR::InstBlockId return_patterns_id,
                        SemIR::InstId self_arg_id,
                        llvm::ArrayRef<SemIR::InstId> arg_refs,
                        llvm::ArrayRef<SemIR::InstId> return_arg_ids,
                        bool is_operator_syntax) -> SemIR::InstBlockId {
  MatchContext match(MatchKind::Caller, specific_id);

  auto return_patterns = context.inst_blocks().GetOrEmpty(return_patterns_id);
  // Track the return storage, if present.
  for (auto [return_pattern_id, return_arg_id] :
       llvm::zip_equal(return_patterns, return_arg_ids)) {
    if (return_arg_id.has_value()) {
      match.AddWork(
          {.pattern_id = return_pattern_id, .scrutinee_id = return_arg_id});
    } else {
      CARBON_CHECK(return_arg_ids.size() == 1,
                   "TODO: do the match even if return_arg_id is None, so that "
                   "subsequent args are at the right index in the arg block");
    }
  }

  // Check type conversions per-element.
  for (auto [arg_id, param_pattern_id] : llvm::reverse(llvm::zip_equal(
           arg_refs, context.inst_blocks().GetOrEmpty(param_patterns_id)))) {
    match.AddWork({.pattern_id = param_pattern_id,
                   .scrutinee_id = arg_id,
                   .allow_unmarked_ref = is_operator_syntax});
  }

  if (self_pattern_id.has_value()) {
    match.AddWork({.pattern_id = self_pattern_id,
                   .scrutinee_id = self_arg_id,
                   .allow_unmarked_ref = true});
  }

  match.DoWork(context);
  return std::move(match).CallerResults(context);
}

auto LocalPatternMatch(Context& context, SemIR::InstId pattern_id,
                       SemIR::InstId scrutinee_id) -> void {
  MatchContext match(MatchKind::Local);
  match.AddWork({.pattern_id = pattern_id, .scrutinee_id = scrutinee_id});
  match.DoWork(context);
}

}  // namespace Carbon::Check
