// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/pattern_match.h"

#include <functional>
#include <utility>
#include <variant>
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

// State for caller-side pattern matching.
struct CallerState {
  // The in-progress contents of the `Call` arguments block.
  llvm::SmallVector<SemIR::InstId> call_args;

  // The SpecificId of the function being called (if any).
  SemIR::SpecificId callee_specific_id;
};

// State for callee-side pattern matching.
struct CalleeState {
  // The in-progress contents of the `Call` parameters block.
  llvm::SmallVector<SemIR::InstId> call_params;

  // The in-progress contents of the `Call` parameter patterns block.
  llvm::SmallVector<SemIR::InstId> call_param_patterns;
};

// State for local pattern matching.
struct LocalState {};

using State = std::variant<CallerState*, CalleeState*, LocalState*>;

// The worklist and state machine for a pattern-matching operation.
//
// Conceptually, pattern matching is a recursive traversal of the pattern inst
// tree: we match a pattern inst to a scrutinee inst by converting the scrutinee
// as needed, matching any subpatterns against corresponding parts of the
// scrutinee, and assembling the results of those sub-matches to form the result
// of the whole match.
//
// This recursive traversal is implemented as a stack of work items, each
// associated with a particular pattern inst. There are two types of work items,
// PreWork and PostWork, which correspond to the work that is done before and
// after visiting an inst's subpatterns, and are handled by DoPreWork and
// DoPostWork overloads, respectively. Note that when there are no subpatterns,
// DoPreWork may push a PostWork onto the stack, or may do the post-work (if
// any) locally.
//
// DoPostWork is primarily responsible for computing the pattern's result and
// adding it to result_stack_. However, the result of matching a pattern is
// often not needed, so to avoid emitting unnecessary SemIR, it should only do
// that if need_subpattern_results() is true.
//
// The traversal behavior depends on the kind of matching being performed. In
// particular, many parts of a function signature pattern are irrelevant to the
// caller, or to the callee, in which case no work will be done in that part of
// the traversal. If an entire subpattern is known to be irrelevant in the
// current matching context, it will not be traversed at all.
class MatchContext {
 public:
  struct PreWork : Printable<PreWork> {
    // `None` when processing the callee side.
    SemIR::InstId scrutinee_id;

    auto Print(llvm::raw_ostream& out) const -> void {
      out << "{PreWork, scrutinee_id: " << scrutinee_id << "}";
    }
  };

  struct PostWork : Printable<PostWork> {
    auto Print(llvm::raw_ostream& out) const -> void { out << "{PostWork}"; }
  };

  struct WorkItem : Printable<WorkItem> {
    SemIR::InstId pattern_id;

    std::variant<PreWork, PostWork> work;

    // If true, disables diagnostics that would otherwise require scrutinee_id
    // to be tagged with `ref`. Only affects caller pattern matching.
    bool allow_unmarked_ref = false;

    auto Print(llvm::raw_ostream& out) const -> void {
      out << "{pattern_id: " << pattern_id << ", work: ";
      std::visit([&](const auto& work) { out << work; }, work);
      out << ", allow_unmarked_ref: " << allow_unmarked_ref << "}";
    }
  };

  // Constructs a MatchContext.
  explicit MatchContext(Context& context) : context_(context) {}

  // Performs pattern matching for the given work item.
  auto Match(State state, WorkItem entry) -> void;

  // Performs pattern matching for the given work item, and returns the result.
  auto MatchWithResult(State state, WorkItem entry) -> SemIR::InstId;

 private:
  // Whether the result of the work item at the top of the stack is needed.
  auto need_subpattern_results() const -> bool {
    return !results_stack_.empty();
  }

  // Adds `entry` to the front of the worklist.
  auto AddWork(WorkItem entry) -> void { stack_.push_back(entry); }

  // Sets `entry.work` to `PostWork` and adds it to the front of the worklist.
  auto AddAsPostWork(WorkItem entry) -> void {
    entry.work = PostWork{};
    AddWork(entry);
  }

  // Dispatches `entry` to the appropriate DoWork method based on the kinds of
  // `entry.pattern_id` and `entry.work`.
  auto Dispatch(State state, WorkItem entry) -> void;

  // Do the pre-work for `entry`. `entry.work` must be a `PreWork` containing
  // `scrutinee_id`, and the pattern argument must be the value of
  // `entry.pattern_id` in `context`.
  auto DoPreWork(State state, SemIR::AnyBindingPattern binding_pattern,
                 SemIR::InstId scrutinee_id, WorkItem entry) -> void;
  auto DoPreWork(State state, SemIR::AnyParamPattern param_pattern,
                 SemIR::InstId scrutinee_id, WorkItem entry) -> void;
  auto DoPreWork(State state, SemIR::ExprPattern expr_pattern,
                 SemIR::InstId scrutinee_id, WorkItem entry) -> void;
  auto DoPreWork(State state, SemIR::ReturnSlotPattern return_slot_pattern,
                 SemIR::InstId scrutinee_id, WorkItem entry) -> void;
  auto DoPreWork(State state, SemIR::VarPattern var_pattern,
                 SemIR::InstId scrutinee_id, WorkItem entry) -> void;
  auto DoPreWork(State state, SemIR::TuplePattern tuple_pattern,
                 SemIR::InstId scrutinee_id, WorkItem entry) -> void;

  // Do the post-work for `entry`. `entry.work` must be a `PostWork`, and
  // the pattern argument must be the value of `entry.pattern_id` in `context_`.
  auto DoPostWork(State state, SemIR::AnyBindingPattern binding_pattern,
                  WorkItem entry) -> void;
  auto DoPostWork(State state, SemIR::VarPattern var_pattern, WorkItem entry)
      -> void;
  auto DoPostWork(State state, SemIR::AnyParamPattern param_pattern,
                  WorkItem entry) -> void;
  auto DoPostWork(State state, SemIR::ExprPattern expr_pattern, WorkItem entry)
      -> void;
  auto DoPostWork(State state, SemIR::ReturnSlotPattern return_slot_pattern,
                  WorkItem entry) -> void;
  auto DoPostWork(State state, SemIR::TuplePattern tuple_pattern,
                  WorkItem entry) -> void;

  // Asserts that there is a single inst in the top array in `results_stack_`,
  // pops that array, and returns the inst.
  auto PopResult() -> SemIR::InstId {
    CARBON_CHECK(results_stack_.PeekArray().size() == 1);
    auto value_id = results_stack_.PeekArray()[0];
    results_stack_.PopArray();
    return value_id;
  }

  // Performs the core logic of matching a variable pattern whose type is
  // `pattern_type_id`, but returns the scrutinee that its subpattern should be
  // matched with, rather than pushing it onto the worklist. This is factored
  // out so it can be reused when handling a `FormBindingPattern` or
  // `FormParamPattern` with an initializing form.
  auto DoVarPreWorkImpl(State state, SemIR::TypeId pattern_type_id,
                        SemIR::InstId scrutinee_id, WorkItem entry) const
      -> SemIR::InstId;

  // The stack of work to be processed.
  llvm::SmallVector<WorkItem> stack_;

  // The stack of in-progress match results. Each array in the stack represents
  // a single result, which may have multiple sub-results.
  ArrayStack<SemIR::InstId> results_stack_;

  Context& context_;
};

}  // namespace

auto MatchContext::Match(State state, WorkItem entry) -> void {
  CARBON_CHECK(stack_.empty());
  stack_.push_back(entry);
  while (!stack_.empty()) {
    Dispatch(state, stack_.pop_back_val());
  }
}

auto MatchContext::MatchWithResult(State state, WorkItem entry)
    -> SemIR::InstId {
  results_stack_.PushArray();
  Match(state, entry);
  return PopResult();
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

// Returns the kind of conversion to perform on the scrutinee when matching the
// given pattern. Note that this returns `NoOp` for `var` patterns, because
// their conversion needs special handling, prior to any general-purpose
// conversion that would use this function.
static auto ConversionKindFor(Context& context, SemIR::Inst pattern,
                              MatchContext::WorkItem entry)
    -> ConversionTarget::Kind {
  CARBON_KIND_SWITCH(pattern) {
    case SemIR::VarParamPattern::Kind:
    case SemIR::VarPattern::Kind:
      // See function comment.
    case SemIR::OutParamPattern::Kind:
      // OutParamPattern conversion is handled by the enclosing
      // ReturnSlotPattern.
    case SemIR::WrapperBindingPattern::Kind:
      // WrapperBindingPattern conversion is handled by its subpattern.
      return ConversionTarget::NoOp;
    case SemIR::RefBindingPattern::Kind:
      return ConversionTarget::DurableRef;
    case SemIR::RefParamPattern::Kind:
      return entry.allow_unmarked_ref ? ConversionTarget::UnmarkedRefParam
                                      : ConversionTarget::RefParam;
    case SemIR::SymbolicBindingPattern::Kind:
    case SemIR::ValueBindingPattern::Kind:
    case SemIR::ValueParamPattern::Kind:
      return ConversionTarget::Value;
    case CARBON_KIND(SemIR::FormBindingPattern form_binding_pattern): {
      auto form_id = context.entity_names()
                         .Get(form_binding_pattern.entity_name_id)
                         .form_id;
      auto form_inst_id = context.constant_values().GetInstId(form_id);
      auto form_inst = context.insts().Get(form_inst_id);

      switch (form_inst.kind()) {
        case SemIR::InitForm::Kind:
          context.TODO(entry.pattern_id, "Support local initializing forms");
          [[fallthrough]];
        case SemIR::RefForm::Kind:
          return ConversionTarget::DurableRef;
        case SemIR::SymbolicBinding::Kind:
          context.TODO(entry.pattern_id, "Support symbolic form bindings");
          [[fallthrough]];
        case SemIR::ValueForm::Kind:
        case SemIR::ErrorInst::Kind:
          return ConversionTarget::Value;
        default:
          CARBON_FATAL("Unexpected form {0}", form_inst);
      }
    }
    case CARBON_KIND(SemIR::FormParamPattern form_param_pattern): {
      auto form_inst_id =
          context.constant_values().GetInstId(form_param_pattern.form_id);
      auto form_inst = context.insts().Get(form_inst_id);

      switch (form_inst.kind()) {
        case SemIR::InitForm::Kind:
          return ConversionTarget::NoOp;
        case SemIR::RefForm::Kind:
          // TODO: Figure out rules for when the argument must have a `ref` tag.
          return entry.allow_unmarked_ref ? ConversionTarget::UnmarkedRefParam
                                          : ConversionTarget::RefParam;
        case SemIR::SymbolicBinding::Kind:
          context.TODO(entry.pattern_id, "Support symbolic form params");
          [[fallthrough]];
        case SemIR::ErrorInst::Kind:
        case SemIR::ValueForm::Kind:
          return ConversionTarget::Value;
        default:
          CARBON_FATAL("Unexpected form {0}", form_inst);
      }
    }
    default:
      CARBON_FATAL("Unexpected pattern kind in {0}", pattern);
  }
}

auto MatchContext::DoPreWork(State state,
                             SemIR::AnyBindingPattern binding_pattern,
                             SemIR::InstId scrutinee_id, WorkItem entry)
    -> void {
  bool scheduled_post_work = false;
  if (!std::holds_alternative<CallerState*>(state)) {
    results_stack_.PushArray();
    AddAsPostWork(entry);
    scheduled_post_work = true;
  } else {
    CARBON_CHECK(!need_subpattern_results());
  }
  if (binding_pattern.kind == SemIR::WrapperBindingPattern::Kind) {
    AddWork({.pattern_id = binding_pattern.subpattern_id,
             .work = PreWork{.scrutinee_id = scrutinee_id},
             .allow_unmarked_ref = entry.allow_unmarked_ref});
  } else if (scheduled_post_work) {
    // PostWork expects a result to bind the name to. If we scheduled PostWork,
    // but didn't schedule PreWork for a subpattern, the name should be bound to
    // the scrutinee.
    results_stack_.AppendToTop(scrutinee_id);
  }
}

auto MatchContext::DoPostWork(State /*state*/,
                              SemIR::AnyBindingPattern binding_pattern,
                              WorkItem entry) -> void {
  // We're logically consuming this map entry, so we invalidate it in order
  // to avoid accidentally consuming it twice.
  auto [bind_name_id, type_expr_region_id] =
      std::exchange(context_.bind_name_map().Lookup(entry.pattern_id).value(),
                    {.bind_name_id = SemIR::InstId::None,
                     .type_expr_region_id = SemIR::ExprRegionId::None});
  if (type_expr_region_id.has_value()) {
    InsertHere(context_, type_expr_region_id);
  }
  auto value_id = PopResult();

  if (value_id.has_value()) {
    auto conversion_kind = ConversionKindFor(context_, binding_pattern, entry);
    if (!bind_name_id.has_value()) {
      // TODO: Is this appropriate, or should we perform a conversion based on
      // the category of the `_` binding first, and then separately discard the
      // initializer for a `_` binding?
      conversion_kind = ConversionTarget::Discarded;
    }
    value_id =
        Convert(context_, SemIR::LocId(value_id), value_id,
                {.kind = conversion_kind,
                 .type_id = context_.insts().Get(bind_name_id).type_id()});
  } else {
    CARBON_CHECK(binding_pattern.kind == SemIR::SymbolicBindingPattern::Kind);
  }

  if (bind_name_id.has_value()) {
    auto bind_name = context_.insts().GetAs<SemIR::AnyBinding>(bind_name_id);
    CARBON_CHECK(!bind_name.value_id.has_value());
    bind_name.value_id = value_id;
    ReplaceInstBeforeConstantUse(context_, bind_name_id, bind_name);
    context_.inst_block_stack().AddInstId(bind_name_id);
  }
  if (need_subpattern_results()) {
    results_stack_.AppendToTop(value_id);
  }
}

// Returns the inst kind to use for the parameter corresponding to the given
// parameter pattern.
static auto ParamKindFor(Context& context, SemIR::Inst param_pattern,
                         MatchContext::WorkItem entry) -> SemIR::InstKind {
  CARBON_KIND_SWITCH(param_pattern) {
    case SemIR::OutParamPattern::Kind:
      return SemIR::OutParam::Kind;
    case SemIR::RefParamPattern::Kind:
    case SemIR::VarParamPattern::Kind:
      return SemIR::RefParam::Kind;
    case SemIR::ValueParamPattern::Kind:
      return SemIR::ValueParam::Kind;
    case CARBON_KIND(SemIR::FormParamPattern form_param_pattern): {
      auto form_inst_id =
          context.constant_values().GetInstId(form_param_pattern.form_id);
      auto form_inst = context.insts().Get(form_inst_id);
      switch (form_inst.kind()) {
        case SemIR::InitForm::Kind:
        case SemIR::RefForm::Kind:
          return SemIR::RefParam::Kind;
        case SemIR::SymbolicBinding::Kind:
          context.TODO(entry.pattern_id, "Support symbolic form params");
          [[fallthrough]];
        case SemIR::ErrorInst::Kind:
        case SemIR::ValueForm::Kind:
          return SemIR::ValueParam::Kind;
        default:
          CARBON_FATAL("Unexpected form {0}", form_inst);
      }
    }
    default:
      CARBON_FATAL("Unexpected param pattern kind: {0}", param_pattern);
  }
}

auto MatchContext::DoPreWork(State state, SemIR::AnyParamPattern param_pattern,
                             SemIR::InstId scrutinee_id, WorkItem entry)
    -> void {
  AddAsPostWork(entry);

  // If `param_pattern` has initializing form, match it as a `VarPattern`
  // before matching it as a parameter pattern.
  switch (param_pattern.kind) {
    case SemIR::FormParamPattern::Kind: {
      auto form_param_pattern =
          context_.insts().GetAs<SemIR::FormParamPattern>(entry.pattern_id);
      if (!context_.constant_values().InstIs<SemIR::InitForm>(
              form_param_pattern.form_id)) {
        break;
      }
      [[fallthrough]];
    }
    case SemIR::VarParamPattern::Kind: {
      scrutinee_id =
          DoVarPreWorkImpl(state, param_pattern.type_id, scrutinee_id, entry);
      entry.allow_unmarked_ref = true;
      break;
    }
    default:
      break;
  }

  CARBON_KIND_SWITCH(state) {
    case CARBON_KIND(CallerState* caller_state): {
      CARBON_CHECK(scrutinee_id.has_value());
      if (scrutinee_id == SemIR::ErrorInst::InstId) {
        caller_state->call_args.push_back(SemIR::ErrorInst::InstId);
      } else {
        auto scrutinee_type_id = ExtractScrutineeType(
            context_.sem_ir(),
            SemIR::GetTypeOfInstInSpecific(context_.sem_ir(),
                                           caller_state->callee_specific_id,
                                           entry.pattern_id));
        caller_state->call_args.push_back(
            Convert(context_, SemIR::LocId(scrutinee_id), scrutinee_id,
                    {.kind = ConversionKindFor(context_, param_pattern, entry),
                     .type_id = scrutinee_type_id}));
      }
      // Do not traverse farther or schedule PostWork, because the caller side
      // of the pattern ends here.
      break;
    }
    case CARBON_KIND(CalleeState* callee_state): {
      SemIR::Inst param = SemIR::AnyParam{
          .kind = ParamKindFor(context_, param_pattern, entry),
          .type_id =
              ExtractScrutineeType(context_.sem_ir(), param_pattern.type_id),
          .index = SemIR::CallParamIndex(callee_state->call_params.size()),
          .pretty_name_id = SemIR::GetPrettyNameFromPatternId(
              context_.sem_ir(), entry.pattern_id)};
      auto loc_id = SemIR::LocId(entry.pattern_id);
      auto param_id = SemIR::InstId::None;
      // TODO: find a way to avoid this boilerplate.
      switch (param.kind()) {
        case SemIR::OutParam::Kind:
          param_id = AddInst(context_, loc_id, param.As<SemIR::OutParam>());
          break;
        case SemIR::RefParam::Kind:
          param_id = AddInst(context_, loc_id, param.As<SemIR::RefParam>());
          break;
        case SemIR::ValueParam::Kind:
          param_id = AddInst(context_, loc_id, param.As<SemIR::ValueParam>());
          break;
        default:
          CARBON_FATAL("Unexpected parameter kind");
      }
      if (auto var_param_pattern =
              context_.insts().TryGetAs<SemIR::VarParamPattern>(
                  entry.pattern_id)) {
        AddWork({.pattern_id = var_param_pattern->subpattern_id,
                 .work = PreWork{.scrutinee_id = param_id},
                 .allow_unmarked_ref = entry.allow_unmarked_ref});
      } else {
        results_stack_.AppendToTop(param_id);
      }
      callee_state->call_params.push_back(param_id);
      callee_state->call_param_patterns.push_back(entry.pattern_id);
      break;
    }
    case CARBON_KIND(LocalState* _): {
      CARBON_FATAL("Found ValueParamPattern during local pattern match");
    }
  }
}

auto MatchContext::DoPostWork(State /*state*/,
                              SemIR::AnyParamPattern /*param_pattern*/,
                              WorkItem /*entry*/) -> void {
  // No-op: the subpattern's result is this pattern's result. Note that if
  // there were any post-work corresponding to DoVarPreWorkImpl, that work
  // would have to be done here.
}

auto MatchContext::DoPreWork(State /*state*/,
                             SemIR::ExprPattern /*expr_pattern*/,
                             SemIR::InstId /*scrutinee_id*/, WorkItem entry)
    -> void {
  context_.TODO(entry.pattern_id, "expression pattern");
}

auto MatchContext::DoPostWork(State /*state*/,
                              SemIR::ExprPattern /*expr_pattern*/,
                              WorkItem /*entry*/) -> void {}

auto MatchContext::DoPreWork(State state,
                             SemIR::ReturnSlotPattern return_slot_pattern,
                             SemIR::InstId scrutinee_id, WorkItem entry)
    -> void {
  if (std::holds_alternative<CalleeState*>(state)) {
    CARBON_CHECK(!scrutinee_id.has_value());
    results_stack_.PushArray();
    AddAsPostWork(entry);
  }
  AddWork({.pattern_id = return_slot_pattern.subpattern_id,
           .work = PreWork{.scrutinee_id = scrutinee_id}});
}

auto MatchContext::DoPostWork(State state,
                              SemIR::ReturnSlotPattern return_slot_pattern,
                              WorkItem entry) -> void {
  CARBON_CHECK(std::holds_alternative<CalleeState*>(state));
  auto type_id =
      ExtractScrutineeType(context_.sem_ir(), return_slot_pattern.type_id);
  auto return_slot_id = AddInst<SemIR::ReturnSlot>(
      context_, SemIR::LocId(entry.pattern_id),
      {.type_id = type_id,
       .type_inst_id = context_.types().GetTypeInstId(type_id),
       .storage_id = PopResult()});
  bool already_in_lookup =
      context_.scope_stack()
          .LookupOrAddName(SemIR::NameId::ReturnSlot, return_slot_id)
          .has_value();
  CARBON_CHECK(!already_in_lookup);
  if (need_subpattern_results()) {
    results_stack_.AppendToTop(return_slot_id);
  }
}

auto MatchContext::DoPreWork(State state, SemIR::VarPattern var_pattern,
                             SemIR::InstId scrutinee_id, WorkItem entry)
    -> void {
  auto new_scrutinee_id =
      DoVarPreWorkImpl(state, var_pattern.type_id, scrutinee_id, entry);
  if (need_subpattern_results()) {
    AddAsPostWork(entry);
  }
  AddWork({.pattern_id = var_pattern.subpattern_id,
           .work = PreWork{.scrutinee_id = new_scrutinee_id},
           .allow_unmarked_ref = true});
}

auto MatchContext::DoVarPreWorkImpl(State state, SemIR::TypeId pattern_type_id,
                                    SemIR::InstId scrutinee_id,
                                    WorkItem entry) const -> SemIR::InstId {
  auto storage_id = SemIR::InstId::None;
  CARBON_KIND_SWITCH(state) {
    case CARBON_KIND(CalleeState* _): {
      // We're emitting pattern-match IR for the callee, but we're still on
      // the caller side of the pattern, so we traverse without emitting any
      // insts.
      return scrutinee_id;
    }
    case CARBON_KIND(LocalState* _): {
      // In a `var`/`let` declaration, the `VarStorage` inst is created before
      // we start pattern matching.
      auto lookup_result = context_.var_storage_map().Lookup(entry.pattern_id);
      CARBON_CHECK(lookup_result);
      storage_id = lookup_result.value();
      break;
    }
    case CARBON_KIND(CallerState* _): {
      storage_id = AddInst<SemIR::TemporaryStorage>(
          context_, SemIR::LocId(entry.pattern_id),
          {.type_id =
               ExtractScrutineeType(context_.sem_ir(), pattern_type_id)});
      CARBON_CHECK(scrutinee_id.has_value());
      break;
    }
  }
  // TODO: Find a more efficient way to put these insts in the global_init
  // block (or drop the distinction between the global_init block and the
  // file scope?)
  if (context_.scope_stack().PeekIndex() == ScopeIndex::Package) {
    context_.global_init().Resume();
  }
  if (scrutinee_id.has_value()) {
    auto init_id = Initialize(context_, SemIR::LocId(entry.pattern_id),
                              storage_id, scrutinee_id);
    // If we created a `TemporaryStorage` to hold the var, create a
    // corresponding `Temporary` to model that its initialization is complete.
    // TODO: If the subpattern is a binding, we may want to destroy the
    // parameter variable in the callee instead of the caller so that we can
    // support destructive move from it.
    if (std::holds_alternative<CallerState*>(state)) {
      storage_id = AddInstWithCleanup<SemIR::Temporary>(
          context_, SemIR::LocId(entry.pattern_id),
          {.type_id = context_.insts().Get(storage_id).type_id(),
           .storage_id = storage_id,
           .init_id = init_id});
    } else {
      // TODO: Consider using different instruction kinds for assignment
      // versus initialization.
      AddInst<SemIR::Assign>(context_, SemIR::LocId(entry.pattern_id),
                             {.lhs_id = storage_id, .rhs_id = init_id});
    }
  }
  if (context_.scope_stack().PeekIndex() == ScopeIndex::Package) {
    context_.global_init().Suspend();
  }
  return storage_id;
}

auto MatchContext::DoPostWork(State /*state*/,
                              SemIR::VarPattern /*var_pattern*/,
                              WorkItem /*entry*/) -> void {
  // No-op: the subpattern's result is this pattern's result.
}

auto MatchContext::DoPreWork(State state, SemIR::TuplePattern tuple_pattern,
                             SemIR::InstId scrutinee_id, WorkItem entry)
    -> void {
  if (tuple_pattern.type_id == SemIR::ErrorInst::TypeId) {
    return;
  }
  auto subpattern_ids = context_.inst_blocks().Get(tuple_pattern.elements_id);
  if (need_subpattern_results()) {
    results_stack_.PushArray();
    AddAsPostWork(entry);
  }
  auto add_all_subscrutinees =
      [&](llvm::ArrayRef<SemIR::InstId> subscrutinee_ids) {
        for (auto [subpattern_id, subscrutinee_id] :
             llvm::reverse(llvm::zip_equal(subpattern_ids, subscrutinee_ids))) {
          AddWork({.pattern_id = subpattern_id,
                   .work = PreWork{.scrutinee_id = subscrutinee_id}});
        }
      };
  if (!scrutinee_id.has_value()) {
    CARBON_CHECK(std::holds_alternative<CalleeState*>(state));
    // If we don't have a scrutinee yet, we're still on the caller side of the
    // pattern, so the subpatterns don't have a scrutinee either.
    for (auto subpattern_id : llvm::reverse(subpattern_ids)) {
      AddWork({.pattern_id = subpattern_id,
               .work = PreWork{.scrutinee_id = SemIR::InstId::None}});
    }
    return;
  }
  auto scrutinee = context_.insts().GetWithLocId(scrutinee_id);
  if (auto scrutinee_literal = scrutinee.inst.TryAs<SemIR::TupleLiteral>()) {
    auto subscrutinee_ids =
        context_.inst_blocks().Get(scrutinee_literal->elements_id);
    if (subscrutinee_ids.size() != subpattern_ids.size()) {
      CARBON_DIAGNOSTIC(TuplePatternSizeDoesntMatchLiteral, Error,
                        "tuple pattern expects {0} element{0:s}, but tuple "
                        "literal has {1}",
                        Diagnostics::IntAsSelect, Diagnostics::IntAsSelect);
      context_.emitter().Emit(entry.pattern_id,
                              TuplePatternSizeDoesntMatchLiteral,
                              subpattern_ids.size(), subscrutinee_ids.size());
      return;
    }
    add_all_subscrutinees(subscrutinee_ids);
    return;
  }

  auto tuple_type_id =
      ExtractScrutineeType(context_.sem_ir(), tuple_pattern.type_id);
  auto converted_scrutinee_id = ConvertToValueOrRefOfType(
      context_, SemIR::LocId(entry.pattern_id), scrutinee_id, tuple_type_id);
  if (auto scrutinee_value = context_.insts().TryGetAs<SemIR::TupleValue>(
          converted_scrutinee_id)) {
    add_all_subscrutinees(
        context_.inst_blocks().Get(scrutinee_value->elements_id));
    return;
  }

  auto tuple_type = context_.types().GetAs<SemIR::TupleType>(tuple_type_id);
  auto element_type_inst_ids =
      context_.inst_blocks().Get(tuple_type.type_elements_id);
  llvm::SmallVector<SemIR::InstId> subscrutinee_ids;
  subscrutinee_ids.reserve(element_type_inst_ids.size());
  for (auto [i, element_type_id] : llvm::enumerate(
           context_.types().GetBlockAsTypeIds(element_type_inst_ids))) {
    subscrutinee_ids.push_back(
        AddInst<SemIR::TupleAccess>(context_, scrutinee.loc_id,
                                    {.type_id = element_type_id,
                                     .tuple_id = converted_scrutinee_id,
                                     .index = SemIR::ElementIndex(i)}));
  }
  add_all_subscrutinees(subscrutinee_ids);
}

auto MatchContext::DoPostWork(State /*state*/,
                              SemIR::TuplePattern tuple_pattern, WorkItem entry)
    -> void {
  auto elements_id = context_.inst_blocks().Add(results_stack_.PeekArray());
  results_stack_.PopArray();
  auto tuple_value_id =
      AddInst<SemIR::TupleValue>(context_, SemIR::LocId(entry.pattern_id),
                                 {.type_id = SemIR::ExtractScrutineeType(
                                      context_.sem_ir(), tuple_pattern.type_id),
                                  .elements_id = elements_id});
  results_stack_.AppendToTop(tuple_value_id);
}

auto MatchContext::Dispatch(State state, WorkItem entry) -> void {
  if (entry.pattern_id == SemIR::ErrorInst::InstId) {
    return;
  }
  Diagnostics::AnnotationScope annotate_diagnostics(
      &context_.emitter(), [&](auto& builder) {
        if (std::holds_alternative<CallerState*>(state)) {
          CARBON_DIAGNOSTIC(InCallToFunctionParam, Note,
                            "initializing function parameter");
          builder.Note(entry.pattern_id, InCallToFunctionParam);
        }
      });
  auto pattern = context_.insts().Get(entry.pattern_id);
  CARBON_KIND_SWITCH(entry.work) {
    case CARBON_KIND(PreWork work): {
      // TODO: Require that `work.scrutinee_id` is valid if and only if insts
      // should be emitted, once we start emitting `Param` insts in the
      // `ParamPattern` case.
      CARBON_KIND_SWITCH(pattern) {
        case CARBON_KIND_ANY(SemIR::AnyBindingPattern, any_binding_pattern): {
          DoPreWork(state, any_binding_pattern, work.scrutinee_id, entry);
          break;
        }
        case CARBON_KIND_ANY(SemIR::AnyParamPattern, any_param_pattern): {
          DoPreWork(state, any_param_pattern, work.scrutinee_id, entry);
          break;
        }
        case CARBON_KIND(SemIR::ExprPattern expr_pattern): {
          DoPreWork(state, expr_pattern, work.scrutinee_id, entry);
          break;
        }
        case CARBON_KIND(SemIR::ReturnSlotPattern return_slot_pattern): {
          DoPreWork(state, return_slot_pattern, work.scrutinee_id, entry);
          break;
        }
        case CARBON_KIND(SemIR::VarPattern var_pattern): {
          DoPreWork(state, var_pattern, work.scrutinee_id, entry);
          break;
        }
        case CARBON_KIND(SemIR::TuplePattern tuple_pattern): {
          DoPreWork(state, tuple_pattern, work.scrutinee_id, entry);
          break;
        }
        default: {
          CARBON_FATAL("Inst kind not handled: {0}", pattern.kind());
        }
      }
      break;
    }
    case CARBON_KIND(PostWork _): {
      CARBON_KIND_SWITCH(pattern) {
        case CARBON_KIND_ANY(SemIR::AnyBindingPattern, any_binding_pattern): {
          DoPostWork(state, any_binding_pattern, entry);
          break;
        }
        case CARBON_KIND_ANY(SemIR::AnyParamPattern, any_param_pattern): {
          DoPostWork(state, any_param_pattern, entry);
          break;
        }
        case CARBON_KIND(SemIR::ExprPattern expr_pattern): {
          DoPostWork(state, expr_pattern, entry);
          break;
        }
        case CARBON_KIND(SemIR::ReturnSlotPattern return_slot_pattern): {
          DoPostWork(state, return_slot_pattern, entry);
          break;
        }
        case CARBON_KIND(SemIR::VarPattern var_pattern): {
          DoPostWork(state, var_pattern, entry);
          break;
        }
        case CARBON_KIND(SemIR::TuplePattern tuple_pattern): {
          DoPostWork(state, tuple_pattern, entry);
          break;
        }
        default: {
          CARBON_FATAL("Inst kind not handled: {0}", pattern.kind());
        }
      }
      break;
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
            .call_params_id = SemIR::InstBlockId::None,
            .param_ranges = SemIR::Function::CallParamIndexRanges::Empty};
  }

  CalleeState state;
  MatchContext match(context);

  // We add work to the stack in reverse so that the results will be produced
  // in the original order.
  if (implicit_param_patterns_id.has_value()) {
    for (SemIR::InstId inst_id :
         context.inst_blocks().Get(implicit_param_patterns_id)) {
      match.Match(
          &state,
          {.pattern_id = inst_id,
           .work = MatchContext::PreWork{.scrutinee_id = SemIR::InstId::None}});
    }
  }
  auto implicit_end = SemIR::CallParamIndex(state.call_params.size());

  if (param_patterns_id.has_value()) {
    for (SemIR::InstId inst_id : context.inst_blocks().Get(param_patterns_id)) {
      match.Match(
          &state,
          {.pattern_id = inst_id,
           .work = MatchContext::PreWork{.scrutinee_id = SemIR::InstId::None}});
    }
  }
  auto explicit_end = SemIR::CallParamIndex(state.call_params.size());

  for (auto return_pattern_id :
       context.inst_blocks().GetOrEmpty(return_patterns_id)) {
    match.Match(
        &state,
        {.pattern_id = return_pattern_id,
         .work = MatchContext::PreWork{.scrutinee_id = SemIR::InstId::None}});
  }
  auto return_end = SemIR::CallParamIndex(state.call_params.size());
  CARBON_CHECK(state.call_params.size() == state.call_param_patterns.size());

  return {.call_param_patterns_id =
              context.inst_blocks().Add(state.call_param_patterns),
          .call_params_id = context.inst_blocks().Add(state.call_params),
          .param_ranges = {implicit_end, explicit_end, return_end}};
}

auto CallerPatternMatch(Context& context, SemIR::SpecificId specific_id,
                        SemIR::InstId self_pattern_id,
                        SemIR::InstBlockId param_patterns_id,
                        SemIR::InstBlockId return_patterns_id,
                        SemIR::InstId self_arg_id,
                        llvm::ArrayRef<SemIR::InstId> arg_refs,
                        llvm::ArrayRef<SemIR::InstId> return_arg_ids,
                        bool is_operator_syntax) -> SemIR::InstBlockId {
  CallerState state = {.callee_specific_id = specific_id};
  MatchContext match(context);

  if (self_pattern_id.has_value()) {
    match.Match(&state,
                {.pattern_id = self_pattern_id,
                 .work = MatchContext::PreWork{.scrutinee_id = self_arg_id},
                 .allow_unmarked_ref = true});
  }

  for (auto [arg_id, param_pattern_id] : llvm::zip_equal(
           arg_refs, context.inst_blocks().GetOrEmpty(param_patterns_id))) {
    match.Match(&state, {.pattern_id = param_pattern_id,
                         .work = MatchContext::PreWork{.scrutinee_id = arg_id},
                         .allow_unmarked_ref = is_operator_syntax});
  }

  auto return_patterns = context.inst_blocks().GetOrEmpty(return_patterns_id);
  // Track the return storage, if present.
  for (auto [return_pattern_id, return_arg_id] :
       llvm::zip_equal(return_patterns, return_arg_ids)) {
    if (return_arg_id.has_value()) {
      match.Match(&state, {.pattern_id = return_pattern_id,
                           .work = MatchContext::PreWork{.scrutinee_id =
                                                             return_arg_id}});
    } else {
      CARBON_CHECK(return_arg_ids.size() == 1,
                   "TODO: do the match even if return_arg_id is None, so that "
                   "subsequent args are at the right index in the arg block");
    }
  }

  return context.inst_blocks().Add(state.call_args);
}

auto LocalPatternMatch(Context& context, SemIR::InstId pattern_id,
                       SemIR::InstId scrutinee_id) -> void {
  LocalState state;
  MatchContext match(context);
  match.Match(&state,
              {.pattern_id = pattern_id,
               .work = MatchContext::PreWork{.scrutinee_id = scrutinee_id}});
}

}  // namespace Carbon::Check
