// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/eval.h"

#include <algorithm>
#include <array>
#include <optional>
#include <utility>

#include "common/raw_string_ostream.h"
#include "llvm/Support/ConvertUTF.h"
#include "toolchain/base/canonical_value_store.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/action.h"
#include "toolchain/check/cpp/constant.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/check/eval_inst.h"
#include "toolchain/check/facet_type.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/diagnostics/emitter.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/sem_ir/builtin_function_kind.h"
#include "toolchain/sem_ir/constant.h"
#include "toolchain/sem_ir/facet_type_info.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/generic.h"
#include "toolchain/sem_ir/id_kind.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/impl.h"
#include "toolchain/sem_ir/inst_categories.h"
#include "toolchain/sem_ir/inst_kind.h"
#include "toolchain/sem_ir/specific_named_constraint.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

namespace {
// Information about an eval block of a specific that we are currently building.
struct SpecificEvalInfo {
  // The region within the specific whose eval block we are building.
  SemIR::GenericInstIndex::Region region;
  // The work-in-progress contents of the eval block.
  llvm::ArrayRef<SemIR::InstId> values;
};

// Information about a local scope that we're currently evaluating, such as a
// call to an `eval fn`. In this scope, instructions with runtime phase may
// locally have constant values, for example values that are computed from the
// arguments to the call. These values are specific to the current evaluation
// and not global properties of the instruction.
struct LocalEvalInfo {
  // A mapping from instructions with runtime phase within the local scope to
  // the values that they have in the current evaluation. This is populated as
  // the local scope is evaluated, and due to control flow, the same instruction
  // may have its value set multiple times. This map tracks the most recent
  // value that the instruction had, which is the one that a reference to it in
  // well-formed SemIR should refer to.
  Map<SemIR::InstId, SemIR::ConstantId>* locals;
};

// Information about the context within which we are performing evaluation.
// `context` must not be null.
class EvalContext {
 public:
  explicit EvalContext(
      Context* context, SemIR::LocId fallback_loc_id,
      SemIR::SpecificId specific_id = SemIR::SpecificId::None,
      std::optional<SpecificEvalInfo> specific_eval_info = std::nullopt)
      : context_(context),
        fallback_loc_id_(fallback_loc_id),
        specific_id_(specific_id),
        specific_eval_info_(specific_eval_info) {}

  EvalContext(const EvalContext&) = delete;
  auto operator=(const EvalContext&) -> EvalContext& = delete;

  // Gets the location to use for diagnostics if a better location is
  // unavailable.
  // TODO: This is also sometimes unavailable.
  auto fallback_loc_id() const -> SemIR::LocId { return fallback_loc_id_; }

  // Returns a location to use to point at an instruction in a diagnostic, given
  // a list of instructions that might have an attached location. This is the
  // location of the first instruction in the list that has a location if there
  // is one, and otherwise the fallback location.
  auto GetDiagnosticLoc(llvm::ArrayRef<SemIR::InstId> inst_ids)
      -> SemIR::LocId {
    for (auto inst_id : inst_ids) {
      if (inst_id.has_value()) {
        auto loc_id = context_->insts().GetCanonicalLocId(inst_id);
        if (loc_id.has_value()) {
          return loc_id;
        }
      }
    }
    return fallback_loc_id_;
  }

  // Gets the value of the specified compile-time binding in this context.
  // Returns `None` if the value is not fixed in this context.
  auto GetCompileTimeBindValue(SemIR::CompileTimeBindIndex bind_index)
      -> SemIR::ConstantId {
    if (!bind_index.has_value() || !specific_id_.has_value()) {
      return SemIR::ConstantId::None;
    }

    const auto& specific = specifics().Get(specific_id_);
    auto args = inst_blocks().Get(specific.args_id);

    // Bindings past the ones with known arguments can appear as local
    // bindings of entities declared within this generic.
    if (static_cast<size_t>(bind_index.index) >= args.size()) {
      return SemIR::ConstantId::None;
    }
    return constant_values().Get(args[bind_index.index]);
  }

  // Given information about a symbolic constant, determine its value in the
  // currently-being-evaluated eval block, if it refers to that eval block. If
  // we can't find a value in this way, returns `None`.
  auto GetInEvaluatedSpecific(const SemIR::SymbolicConstant& symbolic_info)
      -> SemIR::ConstantId {
    if (!specific_eval_info_ || !symbolic_info.index.has_value()) {
      return SemIR::ConstantId::None;
    }

    CARBON_CHECK(
        symbolic_info.generic_id == specifics().Get(specific_id_).generic_id,
        "Instruction has constant operand in wrong generic");
    if (symbolic_info.index.region() != specific_eval_info_->region) {
      return SemIR::ConstantId::None;
    }

    auto inst_id = specific_eval_info_->values[symbolic_info.index.index()];
    CARBON_CHECK(inst_id.has_value(),
                 "Forward reference in eval block: index {0} referenced "
                 "before evaluation",
                 symbolic_info.index.index());
    return constant_values().Get(inst_id);
  }

  // Gets the constant value of the specified instruction in this context.
  auto GetConstantValue(SemIR::InstId inst_id) -> SemIR::ConstantId {
    auto const_id = constant_values().GetAttached(inst_id);

    // While evaluating a function, map from local non-constant instructions to
    // their earlier-evaluated values.
    if (!const_id.is_constant()) {
      if (local_eval_info_) {
        if (auto local = local_eval_info_->locals->Lookup(inst_id)) {
          return local.value();
        }
      }
      return const_id;
    }

    if (!const_id.is_symbolic()) {
      return const_id;
    }

    // While resolving a specific, map from previous instructions in the eval
    // block into their evaluated values. These values won't be present on the
    // specific itself yet, so `GetConstantValueInSpecific` won't be able to
    // find them.
    const auto& symbolic_info = constant_values().GetSymbolicConstant(const_id);
    if (auto eval_block_const_id = GetInEvaluatedSpecific(symbolic_info);
        eval_block_const_id.has_value()) {
      return eval_block_const_id;
    }

    return GetConstantValueInSpecific(sem_ir(), specific_id_, inst_id);
  }

  // Gets the type of the specified instruction in this context.
  auto GetTypeOfInst(SemIR::InstId inst_id) -> SemIR::TypeId {
    auto type_id = insts().GetAttachedType(inst_id);
    if (!type_id.is_symbolic()) {
      return type_id;
    }

    // While resolving a specific, map from previous instructions in the eval
    // block into their evaluated values. These values won't be present on the
    // specific itself yet, so `GetTypeOfInstInSpecific` won't be able to
    // find them.
    const auto& symbolic_info =
        constant_values().GetSymbolicConstant(types().GetConstantId(type_id));
    if (auto eval_block_const_id = GetInEvaluatedSpecific(symbolic_info);
        eval_block_const_id.has_value()) {
      return types().GetTypeIdForTypeConstantId(eval_block_const_id);
    }

    return GetTypeOfInstInSpecific(sem_ir(), specific_id_, inst_id);
  }

  auto ints() -> SharedValueStores::IntStore& { return sem_ir().ints(); }
  auto floats() -> SharedValueStores::FloatStore& { return sem_ir().floats(); }
  auto entity_names() -> SemIR::EntityNameStore& {
    return sem_ir().entity_names();
  }
  auto functions() -> const SemIR::FunctionStore& {
    return sem_ir().functions();
  }
  auto classes() -> const SemIR::ClassStore& { return sem_ir().classes(); }
  auto interfaces() -> const SemIR::InterfaceStore& {
    return sem_ir().interfaces();
  }
  auto specific_interfaces() -> SemIR::SpecificInterfaceStore& {
    return sem_ir().specific_interfaces();
  }
  auto facet_types() -> SemIR::FacetTypeInfoStore& {
    return sem_ir().facet_types();
  }
  auto generics() -> const SemIR::GenericStore& { return sem_ir().generics(); }
  auto specifics() -> const SemIR::SpecificStore& {
    return sem_ir().specifics();
  }
  auto insts() -> const SemIR::InstStore& { return sem_ir().insts(); }
  auto inst_blocks() -> SemIR::InstBlockStore& {
    return sem_ir().inst_blocks();
  }

  // Gets the constant value store. Note that this does not provide the constant
  // values that should be used from this evaluation context, and so should be
  // used with caution.
  auto constant_values() -> const SemIR::ConstantValueStore& {
    return sem_ir().constant_values();
  }

  // Gets the types store. Note that this does not provide the type values that
  // should be used from this evaluation context, and so should be used with
  // caution.
  auto types() -> const SemIR::TypeStore& { return sem_ir().types(); }

  auto context() -> Context& { return *context_; }

  auto sem_ir() -> SemIR::File& { return context().sem_ir(); }

  auto emitter() -> DiagnosticEmitterBase& { return context().emitter(); }

 protected:
  explicit EvalContext(Context* context, SemIR::LocId fallback_loc_id,
                       SemIR::SpecificId specific_id,
                       std::optional<LocalEvalInfo> local_eval_info)
      : context_(context),
        fallback_loc_id_(fallback_loc_id),
        specific_id_(specific_id),
        local_eval_info_(local_eval_info) {}

  // Returns the current locals map, which is assumed to exist.
  auto locals() -> Map<SemIR::InstId, SemIR::ConstantId>& {
    return *local_eval_info_->locals;
  }

 private:
  // The type-checking context in which we're performing evaluation.
  Context* context_;
  // The location to use for diagnostics when a better location isn't available.
  SemIR::LocId fallback_loc_id_;
  // The specific that we are evaluating within.
  SemIR::SpecificId specific_id_;
  // If we are currently evaluating an eval block for `specific_id_`,
  // information about that evaluation.
  std::optional<SpecificEvalInfo> specific_eval_info_;
  // If we are currently evaluating within a local scope, values of local
  // instructions that have already been evaluated. This is here rather than in
  // `FunctionEvalContext` so we can reference it from `GetConstantValue`.
  std::optional<LocalEvalInfo> local_eval_info_;
};
}  // namespace

namespace {
// The evaluation phase for an expression, computed by evaluation. These are
// ordered so that the phase of an expression is the numerically highest phase
// of its constituent evaluations. Note that an expression with any runtime
// component is known to have Runtime phase even if it involves an evaluation
// with UnknownDueToError phase.
enum class Phase : uint8_t {
  // Value could be entirely and concretely computed.
  Concrete,
  // Evaluation phase is symbolic because the expression involves specifically a
  // reference to `.Self`.
  PeriodSelfSymbolic,
  // Evaluation phase is symbolic because the expression involves a reference to
  // a non-template symbolic binding other than `.Self`.
  CheckedSymbolic,
  // Evaluation phase is symbolic because the expression involves a reference to
  // a template parameter, or otherwise depends on something template dependent.
  // The expression might also reference non-template symbolic bindings.
  TemplateSymbolic,
  // The evaluation phase is unknown because evaluation encountered an
  // already-diagnosed semantic or syntax error. This is treated as being
  // potentially constant, but with an unknown phase.
  UnknownDueToError,
  // The expression has runtime phase because of a non-constant subexpression.
  Runtime,
};
}  // namespace

static auto IsConstantOrError(Phase phase) -> bool {
  return phase != Phase::Runtime;
}

// Gets the phase in which the value of a constant will become available.
static auto GetPhase(const SemIR::ConstantValueStore& constant_values,
                     SemIR::ConstantId constant_id) -> Phase {
  if (!constant_id.is_constant()) {
    return Phase::Runtime;
  } else if (constant_id == SemIR::ErrorInst::ConstantId) {
    return Phase::UnknownDueToError;
  }
  switch (constant_values.GetDependence(constant_id)) {
    case SemIR::ConstantDependence::None:
      return Phase::Concrete;
    case SemIR::ConstantDependence::PeriodSelf:
      return Phase::PeriodSelfSymbolic;
    case SemIR::ConstantDependence::Checked:
      return Phase::CheckedSymbolic;
    case SemIR::ConstantDependence::Template:
      return Phase::TemplateSymbolic;
  }
}

// Returns the later of two phases.
static auto LatestPhase(Phase a, Phase b) -> Phase {
  return static_cast<Phase>(
      std::max(static_cast<uint8_t>(a), static_cast<uint8_t>(b)));
}

// Forms a `constant_id` describing a given evaluation result.
static auto MakeConstantResult(Context& context, SemIR::Inst inst, Phase phase)
    -> SemIR::ConstantId {
  switch (phase) {
    case Phase::Concrete:
      return context.constants().GetOrAdd(inst,
                                          SemIR::ConstantDependence::None);
    case Phase::PeriodSelfSymbolic:
      return context.constants().GetOrAdd(
          inst, SemIR::ConstantDependence::PeriodSelf);
    case Phase::CheckedSymbolic:
      return context.constants().GetOrAdd(inst,
                                          SemIR::ConstantDependence::Checked);
    case Phase::TemplateSymbolic:
      return context.constants().GetOrAdd(inst,
                                          SemIR::ConstantDependence::Template);
    case Phase::UnknownDueToError:
      return SemIR::ErrorInst::ConstantId;
    case Phase::Runtime:
      return SemIR::ConstantId::NotConstant;
  }
}

// Forms a `constant_id` describing why an evaluation was not constant.
static auto MakeNonConstantResult(Phase phase) -> SemIR::ConstantId {
  return phase == Phase::UnknownDueToError ? SemIR::ErrorInst::ConstantId
                                           : SemIR::ConstantId::NotConstant;
}

// Forms a constant for an empty tuple value.
static auto MakeEmptyTupleResult(EvalContext& eval_context)
    -> SemIR::ConstantId {
  auto type_id = GetTupleType(eval_context.context(), {});
  return MakeConstantResult(
      eval_context.context(),
      SemIR::TupleValue{.type_id = type_id,
                        .elements_id = SemIR::InstBlockId::Empty},
      Phase::Concrete);
}

// Converts a bool value into a ConstantId.
static auto MakeBoolResult(Context& context, SemIR::TypeId bool_type_id,
                           bool result) -> SemIR::ConstantId {
  return MakeConstantResult(
      context,
      SemIR::BoolLiteral{.type_id = bool_type_id,
                         .value = SemIR::BoolValue::From(result)},
      Phase::Concrete);
}

// Converts an APInt value into a ConstantId.
static auto MakeIntResult(Context& context, SemIR::TypeId type_id,
                          bool is_signed, llvm::APInt value)
    -> SemIR::ConstantId {
  CARBON_CHECK(is_signed == context.types().IsSignedInt(type_id));
  auto result = is_signed ? context.ints().AddSigned(std::move(value))
                          : context.ints().AddUnsigned(std::move(value));
  return MakeConstantResult(
      context, SemIR::IntValue{.type_id = type_id, .int_id = result},
      Phase::Concrete);
}

// Converts an APFloat value into a ConstantId.
static auto MakeFloatResult(Context& context, SemIR::TypeId type_id,
                            llvm::APFloat value) -> SemIR::ConstantId {
  auto result = context.floats().Add(std::move(value));
  return MakeConstantResult(
      context, SemIR::FloatValue{.type_id = type_id, .float_id = result},
      Phase::Concrete);
}

// Creates a FacetType constant.
static auto MakeFacetTypeResult(Context& context,
                                const SemIR::FacetTypeInfo& info, Phase phase)
    -> SemIR::ConstantId {
  SemIR::FacetTypeId facet_type_id = context.facet_types().Add(info);
  return MakeConstantResult(context,
                            SemIR::FacetType{.type_id = SemIR::TypeType::TypeId,
                                             .facet_type_id = facet_type_id},
                            phase);
}

// `GetConstantValue` checks to see whether the provided ID describes a value
// with constant phase, and if so, returns the corresponding constant value.
// Overloads are provided for different kinds of ID. `RequireConstantValue` does
// the same, but produces an error diagnostic if the input is not constant.

// AbsoluteInstId can not have its values substituted, so this overload is
// deleted. This prevents conversion to InstId.
static auto GetConstantValue(EvalContext& eval_context,
                             SemIR::AbsoluteInstId inst_id, Phase* phase)
    -> SemIR::InstId = delete;

// If the given instruction is constant, returns its constant value.
static auto GetConstantValue(EvalContext& eval_context, SemIR::InstId inst_id,
                             Phase* phase) -> SemIR::InstId {
  if (!inst_id.has_value()) {
    return SemIR::InstId::None;
  }
  auto const_id = eval_context.GetConstantValue(inst_id);
  *phase =
      LatestPhase(*phase, GetPhase(eval_context.constant_values(), const_id));
  return eval_context.constant_values().GetInstId(const_id);
}

// Issue a suitable diagnostic for an instruction that evaluated to a
// non-constant value but was required to evaluate to a constant.
static auto DiagnoseNonConstantValue(Context& context, SemIR::LocId loc_id)
    -> void {
  CARBON_DIAGNOSTIC(EvalRequiresConstantValue, Error,
                    "expression is runtime; expected constant");
  context.emitter().Emit(loc_id, EvalRequiresConstantValue);
}

// Gets a constant value for an `inst_id`, diagnosing when the input is not a
// constant value.
static auto RequireConstantValue(EvalContext& eval_context,
                                 SemIR::InstId inst_id, Phase* phase)
    -> SemIR::InstId {
  if (!inst_id.has_value()) {
    return SemIR::InstId::None;
  }
  if (inst_id == SemIR::ErrorInst::InstId) {
    *phase = Phase::UnknownDueToError;
    return SemIR::ErrorInst::InstId;
  }

  auto const_id = eval_context.GetConstantValue(inst_id);
  *phase =
      LatestPhase(*phase, GetPhase(eval_context.constant_values(), const_id));
  if (const_id.is_constant()) {
    return eval_context.constant_values().GetInstId(const_id);
  }

  DiagnoseNonConstantValue(eval_context.context(),
                           eval_context.GetDiagnosticLoc({inst_id}));
  *phase = Phase::UnknownDueToError;
  return SemIR::ErrorInst::InstId;
}

// If the given instruction is constant, returns its constant value. Otherwise,
// produces an error diagnostic. When determining the phase of the result,
// ignore any dependence on `.Self`.
//
// This is used when evaluating facet types, for which `where` expressions using
// `.Self` should not be considered symbolic
// - `Interface where .Self impls I and .A = bool` -> concrete
// - `T:! type` ... `Interface where .A = T` -> symbolic, since uses `T` which
//   is symbolic and not due to `.Self`.
static auto RequireConstantValueIgnoringPeriodSelf(EvalContext& eval_context,
                                                   SemIR::InstId inst_id,
                                                   Phase* phase)
    -> SemIR::InstId {
  if (!inst_id.has_value()) {
    return SemIR::InstId::None;
  }
  Phase constant_phase = *phase;
  auto const_inst_id =
      RequireConstantValue(eval_context, inst_id, &constant_phase);
  // Since LatestPhase(x, Phase::Concrete) == x, this is equivalent to replacing
  // Phase::PeriodSelfSymbolic with Phase::Concrete.
  if (constant_phase != Phase::PeriodSelfSymbolic) {
    *phase = LatestPhase(*phase, constant_phase);
  }
  return const_inst_id;
}

// Gets a constant value for an `inst_id`, diagnosing when the input is not
// constant, and CHECKing that it is concrete. Should only be used in contexts
// where non-concrete constants cannot appear.
static auto CheckConcreteValue(EvalContext& eval_context, SemIR::InstId inst_id)
    -> SemIR::InstId {
  auto phase = Phase::Concrete;
  auto value_inst_id = RequireConstantValue(eval_context, inst_id, &phase);
  if (phase == Phase::UnknownDueToError) {
    return SemIR::ErrorInst::InstId;
  }
  CARBON_CHECK(phase == Phase::Concrete,
               "expression evaluates to symbolic value {0}",
               eval_context.insts().Get(value_inst_id));
  return value_inst_id;
}

// Find the instruction that the given instruction instantiates to, and return
// that.
static auto GetConstantValue(EvalContext& eval_context,
                             SemIR::MetaInstId inst_id, Phase* phase)
    -> SemIR::MetaInstId {
  Phase inner_phase = Phase::Concrete;
  if (auto const_inst_id =
          GetConstantValue(eval_context, SemIR::InstId(inst_id), &inner_phase);
      const_inst_id.has_value()) {
    // The instruction has a constant value. Use that as the operand of the
    // action.
    *phase = LatestPhase(*phase, inner_phase);
    return const_inst_id;
  }

  // If this instruction is splicing in an action result, that action result is
  // our operand.
  if (auto splice = eval_context.insts().TryGetAs<SemIR::SpliceInst>(inst_id)) {
    if (auto spliced_inst_id =
            GetConstantValue(eval_context, splice->inst_id, phase);
        spliced_inst_id.has_value()) {
      if (auto inst_value_id = eval_context.insts().TryGetAs<SemIR::InstValue>(
              spliced_inst_id)) {
        return inst_value_id->inst_id;
      }
    }
  }

  // Otherwise, this is a normal instruction.
  if (OperandDependence(eval_context.context(), inst_id) ==
      SemIR::ConstantDependence::Template) {
    *phase = LatestPhase(*phase, Phase::TemplateSymbolic);
  }
  return inst_id;
}

static auto GetConstantValue(EvalContext& eval_context,
                             SemIR::TypeInstId inst_id, Phase* phase)
    -> SemIR::TypeInstId {
  // The input instruction is a TypeInstId, and eval does not change concrete
  // types (like TypeType which TypeInstId implies), so the result is also a
  // valid TypeInstId.
  return SemIR::TypeInstId::UnsafeMake(GetConstantValue(
      eval_context, static_cast<SemIR::InstId>(inst_id), phase));
}

// Explicitly discard a `DestInstId`, because we should not be using the
// destination as part of evaluation.
static auto GetConstantValue(EvalContext& /*eval_context*/,
                             SemIR::DestInstId /*inst_id*/, Phase* /*phase*/)
    -> SemIR::DestInstId {
  return SemIR::InstId::None;
}

// Given an instruction whose type may refer to a generic parameter, returns the
// corresponding type in the evaluation context.
//
// If the `InstId` is not provided, the instruction is assumed to be new and
// therefore unattached, and the type of the given instruction is returned
// unchanged, but the phase is still updated.
static auto GetTypeOfInst(EvalContext& eval_context, SemIR::InstId inst_id,
                          SemIR::Inst inst, Phase* phase) -> SemIR::TypeId {
  auto type_id = inst_id.has_value() ? eval_context.GetTypeOfInst(inst_id)
                                     : inst.type_id();
  *phase = LatestPhase(*phase,
                       GetPhase(eval_context.constant_values(),
                                eval_context.types().GetConstantId(type_id)));
  return type_id;
}

// AbsoluteInstBlockId can not have its values substituted, so this overload is
// deleted. This prevents conversion to InstBlockId.
static auto GetConstantValue(EvalContext& eval_context,
                             SemIR::AbsoluteInstBlockId inst_block_id,
                             Phase* phase) -> SemIR::InstBlockId = delete;

// If the given instruction block contains only constants, returns a
// corresponding block of those values. Ignores the instructions in the
// specified range of indexes, replacing those elements with `None`.
static auto GetConstantBlockValueIgnoringIndexRange(
    EvalContext& eval_context, SemIR::InstBlockId inst_block_id, Phase* phase,
    std::pair<int, int> ignored_range) -> SemIR::InstBlockId {
  if (!inst_block_id.has_value()) {
    return SemIR::InstBlockId::None;
  }
  auto insts = eval_context.inst_blocks().Get(inst_block_id);
  llvm::SmallVector<SemIR::InstId> const_insts;
  for (auto inst_id : insts) {
    auto const_inst_id = SemIR::InstId::None;
    if (static_cast<int>(const_insts.size()) < ignored_range.first ||
        static_cast<int>(const_insts.size()) >= ignored_range.second) {
      const_inst_id = GetConstantValue(eval_context, inst_id, phase);
      if (!const_inst_id.has_value()) {
        return SemIR::InstBlockId::None;
      }
    }

    // Once we leave the small buffer, we know the first few elements are all
    // constant, so it's likely that the entire block is constant. Resize to
    // the target size given that we're going to allocate memory now anyway.
    if (const_insts.size() == const_insts.capacity()) {
      const_insts.reserve(insts.size());
    }

    const_insts.push_back(const_inst_id);
  }
  // TODO: If the new block is identical to the original block, and we know the
  // old ID was canonical, return the original ID.
  return eval_context.inst_blocks().AddCanonical(const_insts);
}

// If the given instruction block contains only constants, returns a
// corresponding block of those values.
static auto GetConstantValue(EvalContext& eval_context,
                             SemIR::InstBlockId inst_block_id, Phase* phase)
    -> SemIR::InstBlockId {
  return GetConstantBlockValueIgnoringIndexRange(eval_context, inst_block_id,
                                                 phase, {0, 0});
}

// Compute the constant value of a type block. This may be different from the
// input type block if we have known generic arguments.
static auto GetConstantValue(EvalContext& eval_context,
                             SemIR::StructTypeFieldsId fields_id, Phase* phase)
    -> SemIR::StructTypeFieldsId {
  if (!fields_id.has_value()) {
    return SemIR::StructTypeFieldsId::None;
  }
  auto fields = eval_context.context().struct_type_fields().Get(fields_id);
  llvm::SmallVector<SemIR::StructTypeField> new_fields;
  for (auto field : fields) {
    auto new_type_inst_id =
        GetConstantValue(eval_context, field.type_inst_id, phase);
    if (!new_type_inst_id.has_value()) {
      return SemIR::StructTypeFieldsId::None;
    }

    // Once we leave the small buffer, we know the first few elements are all
    // constant, so it's likely that the entire block is constant. Resize to the
    // target size given that we're going to allocate memory now anyway.
    if (new_fields.size() == new_fields.capacity()) {
      new_fields.reserve(fields.size());
    }

    new_fields.push_back(
        {.name_id = field.name_id, .type_inst_id = new_type_inst_id});
  }
  // TODO: If the new block is identical to the original block, and we know the
  // old ID was canonical, return the original ID.
  return eval_context.context().struct_type_fields().AddCanonical(new_fields);
}

// The constant value of a specific is the specific with the corresponding
// constant values for its arguments.
static auto GetConstantValue(EvalContext& eval_context,
                             SemIR::SpecificId specific_id, Phase* phase)
    -> SemIR::SpecificId {
  if (!specific_id.has_value()) {
    return SemIR::SpecificId::None;
  }

  const auto& specific = eval_context.specifics().Get(specific_id);
  auto args_id = GetConstantValue(eval_context, specific.args_id, phase);
  if (!args_id.has_value()) {
    return SemIR::SpecificId::None;
  }

  // Generally, when making a new specific, it's done through MakeSpecific(),
  // which will ensure the declaration is resolved.
  //
  // However, the SpecificId returned here is intentionally left without its
  // declaration resolved. Imported instructions with SpecificIds should not
  // have the specific's declaration resolved, but other instructions which
  // include a new SpecificId should.
  //
  // The resolving of the specific's declaration will be ensured later when
  // evaluating the instruction containing the SpecificId.
  if (args_id == specific.args_id) {
    return specific_id;
  }
  return eval_context.context().specifics().GetOrAdd(specific.generic_id,
                                                     args_id);
}

static auto GetConstantValue(EvalContext& eval_context,
                             SemIR::SpecificInterfaceId specific_interface_id,
                             Phase* phase) -> SemIR::SpecificInterfaceId {
  const auto& interface =
      eval_context.specific_interfaces().Get(specific_interface_id);
  if (!interface.specific_id.has_value()) {
    return specific_interface_id;
  }
  return eval_context.specific_interfaces().Add(
      {.interface_id = interface.interface_id,
       .specific_id =
           GetConstantValue(eval_context, interface.specific_id, phase)});
}

// Like `GetConstantValue` but for a `FacetTypeInfo`.
static auto GetConstantFacetTypeInfo(EvalContext& eval_context,
                                     SemIR::LocId loc_id,
                                     const SemIR::FacetTypeInfo& orig,
                                     Phase* phase) -> SemIR::FacetTypeInfo {
  SemIR::FacetTypeInfo info = {};

  info.extend_constraints.reserve(orig.extend_constraints.size());
  for (const auto& extend : orig.extend_constraints) {
    // TODO: Add GetConstantValue for SpecificInterface.
    info.extend_constraints.push_back(
        {.interface_id = extend.interface_id,
         .specific_id =
             GetConstantValue(eval_context, extend.specific_id, phase)});
  }

  info.self_impls_constraints.reserve(orig.self_impls_constraints.size());
  for (const auto& self_impls : orig.self_impls_constraints) {
    // TODO: Add GetConstantValue for SpecificInterface.
    info.self_impls_constraints.push_back(
        {.interface_id = self_impls.interface_id,
         .specific_id =
             GetConstantValue(eval_context, self_impls.specific_id, phase)});
  }

  info.extend_named_constraints.reserve(orig.extend_named_constraints.size());
  for (const auto& extend : orig.extend_named_constraints) {
    // TODO: Add GetConstantValue for SpecificNamedConstraint.
    info.extend_named_constraints.push_back(
        {.named_constraint_id = extend.named_constraint_id,
         .specific_id =
             GetConstantValue(eval_context, extend.specific_id, phase)});
  }

  info.self_impls_named_constraints.reserve(
      orig.self_impls_named_constraints.size());
  for (const auto& self_impls : orig.self_impls_named_constraints) {
    // TODO: Add GetConstantValue for SpecificNamedConstraint.
    info.self_impls_named_constraints.push_back(
        {.named_constraint_id = self_impls.named_constraint_id,
         .specific_id =
             GetConstantValue(eval_context, self_impls.specific_id, phase)});
  }

  info.type_impls_interfaces.reserve(orig.type_impls_interfaces.size());
  for (const auto& type_impls : orig.type_impls_interfaces) {
    info.type_impls_interfaces.push_back(
        {.self_type =
             GetConstantValue(eval_context, type_impls.self_type, phase),
         // TODO: Add GetConstantValue for SpecificInterface.
         .specific_interface = {
             .interface_id = type_impls.specific_interface.interface_id,
             .specific_id = GetConstantValue(
                 eval_context, type_impls.specific_interface.specific_id,
                 phase)}});
  }

  info.type_impls_named_constraints.reserve(
      orig.type_impls_named_constraints.size());
  for (const auto& type_impls : orig.type_impls_named_constraints) {
    info.type_impls_named_constraints.push_back(
        {.self_type =
             GetConstantValue(eval_context, type_impls.self_type, phase),
         // TODO: Add GetConstantValue for SpecificNamedConstraint.
         .specific_named_constraint = {
             .named_constraint_id =
                 type_impls.specific_named_constraint.named_constraint_id,
             .specific_id = GetConstantValue(
                 eval_context, type_impls.specific_named_constraint.specific_id,
                 phase)}});
  }

  // Rewrite constraints are resolved first before replacing them with their
  // canonical instruction, so that in a `WhereExpr` we can work with the
  // `ImplWitnessAccess` references to `.Self` on the LHS of the constraints
  // rather than the value of the associated constant they reference.
  //
  // This also implies that we may find `ImplWitnessAccessSubstituted`
  // instructions in the LHS and RHS of these constraints, which are preserved
  // to maintain them as an unresolved reference to an associated constant, but
  // which must be handled gracefully during resolution. They will be replaced
  // with the constant value of the `ImplWitnessAccess` below when they are
  // substituted with a constant value.
  info.rewrite_constraints = orig.rewrite_constraints;
  if (!ResolveFacetTypeRewriteConstraints(eval_context.context(), loc_id,
                                          info.rewrite_constraints)) {
    *phase = Phase::UnknownDueToError;
  }

  for (auto& rewrite : info.rewrite_constraints) {
    // `where` requirements using `.Self` should not be considered symbolic.
    auto lhs_id = RequireConstantValueIgnoringPeriodSelf(eval_context,
                                                         rewrite.lhs_id, phase);
    auto rhs_id = RequireConstantValueIgnoringPeriodSelf(eval_context,
                                                         rewrite.rhs_id, phase);
    rewrite = {.lhs_id = lhs_id, .rhs_id = rhs_id};
  }

  // TODO: Process other requirements.
  info.other_requirements = orig.other_requirements;

  info.Canonicalize();
  return info;
}

static auto GetConstantValue(EvalContext& eval_context,
                             SemIR::FacetTypeId facet_type_id, Phase* phase)
    -> SemIR::FacetTypeId {
  SemIR::FacetTypeInfo info = GetConstantFacetTypeInfo(
      eval_context, SemIR::LocId::None,
      eval_context.facet_types().Get(facet_type_id), phase);
  return eval_context.facet_types().Add(info);
}

static auto GetConstantValue(EvalContext& eval_context,
                             SemIR::EntityNameId entity_name_id, Phase* phase)
    -> SemIR::EntityNameId {
  const auto& bind_name = eval_context.entity_names().Get(entity_name_id);
  Phase name_phase;
  if (bind_name.name_id == SemIR::NameId::PeriodSelf) {
    name_phase = Phase::PeriodSelfSymbolic;
  } else if (!bind_name.bind_index().has_value()) {
    name_phase = Phase::Concrete;
  } else if (bind_name.is_template) {
    name_phase = Phase::TemplateSymbolic;
  } else {
    name_phase = Phase::CheckedSymbolic;
  }
  *phase = LatestPhase(*phase, name_phase);
  return eval_context.entity_names().MakeCanonical(entity_name_id);
}

// Replaces the specified field of the given typed instruction with its constant
// value, if it has constant phase. Returns true on success, false if the value
// has runtime phase.
template <typename InstT, typename FieldIdT>
static auto ReplaceFieldWithConstantValue(EvalContext& eval_context,
                                          InstT* inst, FieldIdT InstT::* field,
                                          Phase* phase) -> bool {
  auto unwrapped = GetConstantValue(eval_context, inst->*field, phase);
  if (!unwrapped.has_value() && (inst->*field).has_value()) {
    return false;
  }
  inst->*field = unwrapped;
  return IsConstantOrError(*phase);
}

// Function template that can be called with an argument of type `T`. Used below
// to detect which overloads of `GetConstantValue` exist.
template <typename T>
static void Accept(T /*arg*/) {}

// Determines whether a `GetConstantValue` overload exists for a given ID type.
// Note that we do not check whether `GetConstantValue` is *callable* with a
// given ID type, because that would use the `InstId` overload for
// `AbsoluteInstId` and similar wrapper types, which should be left alone.
template <typename IdT>
static constexpr bool HasGetConstantValueOverload = requires {
  Accept<auto (*)(EvalContext&, IdT, Phase*)->IdT>(GetConstantValue);
};

using ArgHandlerFnT = auto(EvalContext& context, int32_t arg, Phase* phase)
    -> int32_t;

// Returns the arg handler for an `IdKind`.
template <typename... Types>
static auto GetArgHandlerFn(TypeEnum<Types...> id_kind) -> ArgHandlerFnT* {
  static constexpr std::array<ArgHandlerFnT*, SemIR::IdKind::NumValues> Table =
      {
          [](EvalContext& eval_context, int32_t arg, Phase* phase) -> int32_t {
            auto id = SemIR::Inst::FromRaw<Types>(arg);
            if constexpr (HasGetConstantValueOverload<Types>) {
              // If we have a custom `GetConstantValue` overload, call it.
              return SemIR::Inst::ToRaw(
                  GetConstantValue(eval_context, id, phase));
            } else {
              // Otherwise, we assume the value is already constant.
              return arg;
            }
          }...,
          // Invalid and None handling (ordering-sensitive).
          [](auto...) -> int32_t { CARBON_FATAL("Unexpected invalid IdKind"); },
          [](EvalContext& /*context*/, int32_t arg,
             Phase* /*phase*/) -> int32_t { return arg; },
      };
  return Table[id_kind.ToIndex()];
}

// Given the stored value `arg` of an instruction field and its corresponding
// kind `kind`, returns the constant value to use for that field, if it has a
// constant phase. `*phase` is updated to include the new constant value. If
// the resulting phase is not constant, the returned value is not useful and
// will typically be `NoneIndex`.
static auto GetConstantValueForArg(EvalContext& eval_context,
                                   SemIR::Inst::ArgAndKind arg_and_kind,
                                   Phase* phase) -> int32_t {
  return GetArgHandlerFn(arg_and_kind.kind())(eval_context,
                                              arg_and_kind.value(), phase);
}

// Given an instruction, replaces its operands with their constant values from
// the specified evaluation context. `*phase` is updated to describe the
// constant phase of the result. Returns whether `*phase` is a constant phase;
// if not, `inst` may not be fully updated and should not be used.
static auto ReplaceAllFieldsWithConstantValues(EvalContext& eval_context,
                                               SemIR::Inst* inst, Phase* phase)
    -> bool {
  auto arg0 =
      GetConstantValueForArg(eval_context, inst->arg0_and_kind(), phase);
  if (!IsConstantOrError(*phase)) {
    return false;
  }
  auto arg1 =
      GetConstantValueForArg(eval_context, inst->arg1_and_kind(), phase);
  if (!IsConstantOrError(*phase)) {
    return false;
  }
  inst->SetArgs(arg0, arg1);
  return true;
}

// Given an instruction and its ID, replaces its type with the corresponding
// value in this evaluation context. Updates `*phase` to describe the phase of
// the result, and returns whether `*phase` is a constant phase.
static auto ReplaceTypeWithConstantValue(EvalContext& eval_context,
                                         SemIR::InstId inst_id,
                                         SemIR::Inst* inst, Phase* phase)
    -> bool {
  inst->SetType(GetTypeOfInst(eval_context, inst_id, *inst, phase));
  return IsConstantOrError(*phase);
}

template <typename InstT>
static auto ReplaceTypeWithConstantValue(EvalContext& eval_context,
                                         SemIR::InstId inst_id, InstT* inst,
                                         Phase* phase) -> bool {
  inst->type_id = GetTypeOfInst(eval_context, inst_id, *inst, phase);
  return IsConstantOrError(*phase);
}

template <typename... Types>
static auto KindHasGetConstantValueOverload(TypeEnum<Types...> e) -> bool {
  static constexpr std::array<bool, SemIR::IdKind::NumTypes> Values = {
      (HasGetConstantValueOverload<Types>)...};
  return Values[e.ToIndex()];
}

static auto ResolveSpecificDeclForSpecificId(EvalContext& eval_context,
                                             SemIR::SpecificId specific_id)
    -> void {
  if (!specific_id.has_value()) {
    return;
  }

  const auto& specific = eval_context.specifics().Get(specific_id);
  const auto& generic = eval_context.generics().Get(specific.generic_id);
  if (specific_id == generic.self_specific_id) {
    // Impl witness table construction happens before its generic decl is
    // finish, in order to make the table's instructions dependent
    // instructions of the Impl's generic. But those instructions can refer to
    // the generic's self specific. We can not resolve the specific
    // declaration for the self specific until the generic is finished, but it
    // is explicitly resolved at that time in `FinishGenericDecl()`.
    return;
  }
  ResolveSpecificDecl(eval_context.context(), eval_context.fallback_loc_id(),
                      specific_id);
}

// Resolves the specific declarations for a specific id in any field of the
// `inst` instruction.
static auto ResolveSpecificDeclForInst(EvalContext& eval_context,
                                       const SemIR::Inst& inst) -> void {
  for (auto arg_and_kind : {inst.arg0_and_kind(), inst.arg1_and_kind()}) {
    // This switch must handle any field type that has a GetConstantValue()
    // overload which canonicalizes a specific (and thus potentially forms a new
    // specific) as part of forming its constant value.
    CARBON_KIND_SWITCH(arg_and_kind) {
      case CARBON_KIND(SemIR::FacetTypeId facet_type_id): {
        const auto& info =
            eval_context.context().facet_types().Get(facet_type_id);
        for (const auto& interface : info.extend_constraints) {
          ResolveSpecificDeclForSpecificId(eval_context, interface.specific_id);
        }
        for (const auto& interface : info.self_impls_constraints) {
          ResolveSpecificDeclForSpecificId(eval_context, interface.specific_id);
        }
        for (const auto& constraint : info.extend_named_constraints) {
          ResolveSpecificDeclForSpecificId(eval_context,
                                           constraint.specific_id);
        }
        for (const auto& constraint : info.self_impls_named_constraints) {
          ResolveSpecificDeclForSpecificId(eval_context,
                                           constraint.specific_id);
        }
        for (const auto& type_impls : info.type_impls_interfaces) {
          ResolveSpecificDeclForSpecificId(
              eval_context, type_impls.specific_interface.specific_id);
        }
        for (const auto& type_impls : info.type_impls_named_constraints) {
          ResolveSpecificDeclForSpecificId(
              eval_context, type_impls.specific_named_constraint.specific_id);
        }
        break;
      }
      case CARBON_KIND(SemIR::SpecificId specific_id): {
        ResolveSpecificDeclForSpecificId(eval_context, specific_id);
        break;
      }
      case CARBON_KIND(SemIR::SpecificInterfaceId specific_interface_id): {
        ResolveSpecificDeclForSpecificId(eval_context,
                                         eval_context.specific_interfaces()
                                             .Get(specific_interface_id)
                                             .specific_id);
        break;
      }

        // These id types have a GetConstantValue() overload but that overload
        // does not canonicalize any SpecificId in the value type.
      case SemIR::IdKind::For<SemIR::DestInstId>:
      case SemIR::IdKind::For<SemIR::EntityNameId>:
      case SemIR::IdKind::For<SemIR::InstBlockId>:
      case SemIR::IdKind::For<SemIR::InstId>:
      case SemIR::IdKind::For<SemIR::MetaInstId>:
      case SemIR::IdKind::For<SemIR::StructTypeFieldsId>:
      case SemIR::IdKind::For<SemIR::TypeInstId>:
        break;

      case SemIR::IdKind::None:
        // No arg.
        break;

      default:
        CARBON_CHECK(
            !KindHasGetConstantValueOverload(arg_and_kind.kind()),
            "Missing case for {0} which has a GetConstantValue() overload",
            arg_and_kind.kind());
        break;
    }
  }
}

auto AddImportedConstant(Context& context, SemIR::Inst inst)
    -> SemIR::ConstantId {
  EvalContext eval_context(&context, SemIR::LocId::None);
  CARBON_CHECK(inst.kind().has_type(), "Can't import untyped instructions: {0}",
               inst.kind());
  Phase phase = GetPhase(context.constant_values(),
                         context.types().GetConstantId(inst.type_id()));
  // We ignore the return value of ReplaceAllFieldsWithConstantValues and just
  // propagate runtime and error constant values into the resulting ConstantId.
  ReplaceAllFieldsWithConstantValues(eval_context, &inst, &phase);
  return MakeConstantResult(context, inst, phase);
}

// Performs an index into a homogeneous aggregate, retrieving the specified
// element.
static auto PerformArrayIndex(EvalContext& eval_context, SemIR::ArrayIndex inst)
    -> SemIR::ConstantId {
  Phase phase = Phase::Concrete;
  auto index_id = GetConstantValue(eval_context, inst.index_id, &phase);

  if (!index_id.has_value()) {
    return MakeNonConstantResult(phase);
  }
  auto index = eval_context.insts().TryGetAs<SemIR::IntValue>(index_id);
  if (!index) {
    CARBON_CHECK(phase != Phase::Concrete,
                 "Concrete constant integer should be a literal");
    return MakeNonConstantResult(phase);
  }

  // Array indexing is invalid if the index is constant and out of range,
  // regardless of whether the array itself is constant.
  const auto& index_val = eval_context.ints().Get(index->int_id);
  auto aggregate_type_id = eval_context.GetTypeOfInst(inst.array_id);
  if (auto array_type =
          eval_context.types().TryGetAs<SemIR::ArrayType>(aggregate_type_id)) {
    if (auto bound = eval_context.insts().TryGetAs<SemIR::IntValue>(
            array_type->bound_id)) {
      // This awkward call to `getZExtValue` is a workaround for APInt not
      // supporting comparisons between integers of different bit widths.
      if (index_val.getActiveBits() > 64 ||
          eval_context.ints()
              .Get(bound->int_id)
              .ule(index_val.getZExtValue())) {
        CARBON_DIAGNOSTIC(ArrayIndexOutOfBounds, Error,
                          "array index `{0}` is past the end of type {1}",
                          TypedInt, SemIR::TypeId);
        eval_context.emitter().Emit(
            eval_context.GetDiagnosticLoc(inst.index_id), ArrayIndexOutOfBounds,
            {.type = index->type_id, .value = index_val}, aggregate_type_id);
        return SemIR::ErrorInst::ConstantId;
      }
    }
  }

  auto aggregate_id = GetConstantValue(eval_context, inst.array_id, &phase);
  if (!aggregate_id.has_value()) {
    return MakeNonConstantResult(phase);
  }
  auto aggregate =
      eval_context.insts().TryGetAs<SemIR::AnyAggregateValue>(aggregate_id);
  if (!aggregate) {
    // TODO: Consider forming a symbolic constant or reference constant array
    // index in this case.
    return MakeNonConstantResult(phase);
  }

  auto elements = eval_context.inst_blocks().Get(aggregate->elements_id);
  return eval_context.GetConstantValue(elements[index_val.getZExtValue()]);
}

// Performs a conversion between character types, diagnosing if the value
// doesn't fit in the destination type.
static auto PerformCheckedCharConvert(Context& context, SemIR::LocId loc_id,
                                      SemIR::InstId arg_id,
                                      SemIR::TypeId dest_type_id)
    -> SemIR::ConstantId {
  auto arg = context.insts().GetAs<SemIR::CharLiteralValue>(arg_id);

  // Values over 0x80 require multiple code units in UTF-8.
  if (arg.value.index >= 0x80) {
    CARBON_DIAGNOSTIC(CharTooLargeForType, Error,
                      "character value {0} too large for type {1}",
                      SemIR::CharId, SemIR::TypeId);
    context.emitter().Emit(loc_id, CharTooLargeForType, arg.value,
                           dest_type_id);
    return SemIR::ErrorInst::ConstantId;
  }

  llvm::APInt int_val(8, arg.value.index, /*isSigned=*/false);
  return MakeIntResult(context, dest_type_id, /*is_signed=*/false, int_val);
}

// Forms a constant int type as an evaluation result. Requires that width_id is
// constant.
static auto MakeIntTypeResult(Context& context, SemIR::LocId loc_id,
                              SemIR::IntKind int_kind, SemIR::InstId width_id,
                              Phase phase) -> SemIR::ConstantId {
  auto result = SemIR::IntType{.type_id = SemIR::TypeType::TypeId,
                               .int_kind = int_kind,
                               .bit_width_id = width_id};
  if (!ValidateIntType(context, loc_id, result)) {
    return SemIR::ErrorInst::ConstantId;
  }
  return MakeConstantResult(context, result, phase);
}

// Forms a constant float type as an evaluation result. Requires that width_id
// is constant.
static auto MakeFloatTypeResult(Context& context, SemIR::LocId loc_id,
                                SemIR::InstId width_id, Phase phase)
    -> SemIR::ConstantId {
  auto result = SemIR::FloatType{.type_id = SemIR::TypeType::TypeId,
                                 .bit_width_id = width_id,
                                 .float_kind = SemIR::FloatKind::None};
  if (!ValidateFloatTypeAndSetKind(context, loc_id, result)) {
    return SemIR::ErrorInst::ConstantId;
  }
  return MakeConstantResult(context, result, phase);
}

// Performs a conversion between integer types, truncating if the value doesn't
// fit in the destination type.
static auto PerformIntConvert(Context& context, SemIR::InstId arg_id,
                              SemIR::TypeId dest_type_id) -> SemIR::ConstantId {
  auto arg_val =
      context.ints().Get(context.insts().GetAs<SemIR::IntValue>(arg_id).int_id);
  auto [dest_is_signed, bit_width_id] =
      context.sem_ir().types().GetIntTypeInfo(dest_type_id);
  if (bit_width_id.has_value()) {
    // TODO: If the value fits in the destination type, reuse the existing
    // int_id rather than recomputing it. This is probably the most common case.
    bool src_is_signed = context.sem_ir().types().IsSignedInt(
        context.insts().Get(arg_id).type_id());
    unsigned width = context.ints().Get(bit_width_id).getZExtValue();
    arg_val =
        src_is_signed ? arg_val.sextOrTrunc(width) : arg_val.zextOrTrunc(width);
  }
  return MakeIntResult(context, dest_type_id, dest_is_signed, arg_val);
}

// Performs a conversion between integer types, diagnosing if the value doesn't
// fit in the destination type.
static auto PerformCheckedIntConvert(Context& context, SemIR::LocId loc_id,
                                     SemIR::InstId arg_id,
                                     SemIR::TypeId dest_type_id)
    -> SemIR::ConstantId {
  auto arg = context.insts().GetAs<SemIR::IntValue>(arg_id);
  auto arg_val = context.ints().Get(arg.int_id);

  auto [is_signed, bit_width_id] =
      context.sem_ir().types().GetIntTypeInfo(dest_type_id);
  auto width = bit_width_id.has_value()
                   ? context.ints().Get(bit_width_id).getZExtValue()
                   : arg_val.getBitWidth();

  if (!is_signed && arg_val.isNegative()) {
    CARBON_DIAGNOSTIC(
        NegativeIntInUnsignedType, Error,
        "negative integer value {0} converted to unsigned type {1}", TypedInt,
        SemIR::TypeId);
    context.emitter().Emit(loc_id, NegativeIntInUnsignedType,
                           {.type = arg.type_id, .value = arg_val},
                           dest_type_id);
  }

  unsigned arg_non_sign_bits = arg_val.getSignificantBits() - 1;
  if (arg_non_sign_bits + is_signed > width) {
    CARBON_DIAGNOSTIC(IntTooLargeForType, Error,
                      "integer value {0} too large for type {1}", TypedInt,
                      SemIR::TypeId);
    context.emitter().Emit(loc_id, IntTooLargeForType,
                           {.type = arg.type_id, .value = arg_val},
                           dest_type_id);
  }

  return MakeConstantResult(
      context, SemIR::IntValue{.type_id = dest_type_id, .int_id = arg.int_id},
      Phase::Concrete);
}

// Performs a conversion between floating-point types, diagnosing if the value
// doesn't fit in the destination type.
static auto PerformCheckedFloatConvert(Context& context, SemIR::LocId loc_id,
                                       SemIR::InstId arg_id,
                                       SemIR::TypeId dest_type_id)
    -> SemIR::ConstantId {
  auto dest_type_object_rep_id = context.types().GetObjectRepr(dest_type_id);
  CARBON_CHECK(dest_type_object_rep_id.has_value(),
               "Conversion to incomplete type");
  auto dest_float_type =
      context.types().TryGetAs<SemIR::FloatType>(dest_type_object_rep_id);
  CARBON_CHECK(dest_float_type || context.types().Is<SemIR::FloatLiteralType>(
                                      dest_type_object_rep_id));

  if (auto literal =
          context.insts().TryGetAs<SemIR::FloatLiteralValue>(arg_id)) {
    if (!dest_float_type) {
      return MakeConstantResult(
          context,
          SemIR::FloatLiteralValue{.type_id = dest_type_id,
                                   .real_id = literal->real_id},
          Phase::Concrete);
    }

    // Convert the real literal to an llvm::APFloat and add it to the floats
    // ValueStore. In the future this would use an arbitrary precision Rational
    // type.
    //
    // TODO: Implement Carbon's actual implicit conversion rules for
    // floating-point constants, as per the design
    // docs/design/expressions/implicit_conversions.md
    auto real_value = context.sem_ir().reals().Get(literal->real_id);

    // Convert the real value to a string.
    llvm::SmallString<64> str;
    real_value.mantissa.toString(str, real_value.is_decimal ? 10 : 16,
                                 /*signed=*/false, /*formatAsCLiteral=*/true);
    str += real_value.is_decimal ? "e" : "p";
    real_value.exponent.toStringSigned(str);

    // Convert the string to an APFloat.
    llvm::APFloat result(dest_float_type->float_kind.Semantics());
    // TODO: The implementation of this conversion effectively converts back to
    // APInts, but unfortunately the conversion from integer mantissa and
    // exponent in IEEEFloat::roundSignificandWithExponent is not part of the
    // public API.
    auto status =
        result.convertFromString(str, llvm::APFloat::rmNearestTiesToEven);
    if (auto error = status.takeError()) {
      // The literal we create should always successfully parse.
      CARBON_FATAL("Float literal parsing failed: {0}",
                   toString(std::move(error)));
    }
    if (status.get() & llvm::APFloat::opOverflow) {
      CARBON_DIAGNOSTIC(FloatLiteralTooLargeForType, Error,
                        "value {0} too large for floating-point type {1}",
                        RealId, SemIR::TypeId);
      context.emitter().Emit(loc_id, FloatLiteralTooLargeForType,
                             literal->real_id, dest_type_id);
      return SemIR::ErrorInst::ConstantId;
    }
    return MakeFloatResult(context, dest_type_id, std::move(result));
  }

  if (!dest_float_type) {
    context.TODO(loc_id, "conversion from float to float literal");
    return SemIR::ErrorInst::ConstantId;
  }

  // Convert to the destination float semantics.
  auto arg = context.insts().GetAs<SemIR::FloatValue>(arg_id);
  llvm::APFloat result = context.floats().Get(arg.float_id);
  bool loses_info;
  auto status = result.convert(dest_float_type->float_kind.Semantics(),
                               llvm::APFloat::rmNearestTiesToEven, &loses_info);
  if (status & llvm::APFloat::opOverflow) {
    CARBON_DIAGNOSTIC(FloatTooLargeForType, Error,
                      "value {0} too large for floating-point type {1}",
                      llvm::APFloat, SemIR::TypeId);
    context.emitter().Emit(loc_id, FloatTooLargeForType,
                           context.floats().Get(arg.float_id), dest_type_id);
    return SemIR::ErrorInst::ConstantId;
  }

  return MakeFloatResult(context, dest_type_id, std::move(result));
}

// Issues a diagnostic for a compile-time division by zero.
static auto DiagnoseDivisionByZero(Context& context, SemIR::LocId loc_id)
    -> void {
  CARBON_DIAGNOSTIC(CompileTimeDivisionByZero, Error, "division by zero");
  context.emitter().Emit(loc_id, CompileTimeDivisionByZero);
}

// Get an integer at a suitable bit-width: either `bit_width_id` if it has a
// value, or the canonical width from the value store if not.
static auto GetIntAtSuitableWidth(Context& context, IntId int_id,
                                  IntId bit_width_id) -> llvm::APInt {
  return bit_width_id.has_value()
             ? context.ints().GetAtWidth(int_id, bit_width_id)
             : context.ints().Get(int_id);
}

// Performs a builtin unary integer -> integer operation.
static auto PerformBuiltinUnaryIntOp(Context& context, SemIR::LocId loc_id,
                                     SemIR::BuiltinFunctionKind builtin_kind,
                                     SemIR::InstId arg_id)
    -> SemIR::ConstantId {
  auto op = context.insts().GetAs<SemIR::IntValue>(arg_id);
  auto [is_signed, bit_width_id] =
      context.sem_ir().types().GetIntTypeInfo(op.type_id);
  llvm::APInt op_val = GetIntAtSuitableWidth(context, op.int_id, bit_width_id);

  switch (builtin_kind) {
    case SemIR::BuiltinFunctionKind::IntSNegate:
      if (op_val.isMinSignedValue()) {
        if (bit_width_id.has_value()) {
          CARBON_DIAGNOSTIC(CompileTimeIntegerNegateOverflow, Error,
                            "integer overflow in negation of {0}", TypedInt);
          context.emitter().Emit(loc_id, CompileTimeIntegerNegateOverflow,
                                 {.type = op.type_id, .value = op_val});
        } else {
          // Widen the integer so we don't overflow into the sign bit.
          op_val = op_val.sext(op_val.getBitWidth() +
                               llvm::APInt::APINT_BITS_PER_WORD);
        }
      }
      op_val.negate();
      break;
    case SemIR::BuiltinFunctionKind::IntUNegate:
      CARBON_CHECK(bit_width_id.has_value(), "Unsigned negate on unsized int");
      op_val.negate();
      break;
    case SemIR::BuiltinFunctionKind::IntComplement:
      // TODO: Should we have separate builtins for signed and unsigned
      // complement? Like with signed/unsigned negate, these operations do
      // different things to the integer value, even though they do the same
      // thing to the bits. We treat IntLiteral complement as signed complement,
      // given that the result of unsigned complement depends on the bit width.
      op_val.flipAllBits();
      break;
    default:
      CARBON_FATAL("Unexpected builtin kind");
  }

  return MakeIntResult(context, op.type_id, is_signed, std::move(op_val));
}

namespace {
// A pair of APInts that are the operands of a binary operator. We use an
// aggregate rather than `std::pair` to allow RVO of the individual ints.
struct APIntBinaryOperands {
  llvm::APInt lhs;
  llvm::APInt rhs;
};
}  // namespace

// Get a pair of integers at the same suitable bit-width: either their actual
// width if they have a fixed width, or the smallest canonical width in which
// they both fit otherwise.
static auto GetIntsAtSuitableWidth(Context& context, IntId lhs_id, IntId rhs_id,
                                   IntId bit_width_id) -> APIntBinaryOperands {
  // Unsized operands: take the wider of the bit widths.
  if (!bit_width_id.has_value()) {
    APIntBinaryOperands result = {.lhs = context.ints().Get(lhs_id),
                                  .rhs = context.ints().Get(rhs_id)};
    if (result.lhs.getBitWidth() != result.rhs.getBitWidth()) {
      if (result.lhs.getBitWidth() > result.rhs.getBitWidth()) {
        result.rhs = result.rhs.sext(result.lhs.getBitWidth());
      } else {
        result.lhs = result.lhs.sext(result.rhs.getBitWidth());
      }
    }
    return result;
  }

  return {.lhs = context.ints().GetAtWidth(lhs_id, bit_width_id),
          .rhs = context.ints().GetAtWidth(rhs_id, bit_width_id)};
}

namespace {
// The result of performing a binary int operation.
struct BinaryIntOpResult {
  llvm::APInt result_val;
  bool overflow;
  Lex::TokenKind op_token;
};
}  // namespace

// Computes the result of a homogeneous binary (int, int) -> int operation.
static auto ComputeBinaryIntOpResult(SemIR::BuiltinFunctionKind builtin_kind,
                                     const llvm::APInt& lhs_val,
                                     const llvm::APInt& rhs_val)
    -> BinaryIntOpResult {
  llvm::APInt result_val;
  bool overflow = false;
  Lex::TokenKind op_token = Lex::TokenKind::Not;

  switch (builtin_kind) {
    // Arithmetic.
    case SemIR::BuiltinFunctionKind::IntSAdd:
      result_val = lhs_val.sadd_ov(rhs_val, overflow);
      op_token = Lex::TokenKind::Plus;
      break;
    case SemIR::BuiltinFunctionKind::IntSSub:
      result_val = lhs_val.ssub_ov(rhs_val, overflow);
      op_token = Lex::TokenKind::Minus;
      break;
    case SemIR::BuiltinFunctionKind::IntSMul:
      result_val = lhs_val.smul_ov(rhs_val, overflow);
      op_token = Lex::TokenKind::Star;
      break;
    case SemIR::BuiltinFunctionKind::IntSDiv:
      result_val = lhs_val.sdiv_ov(rhs_val, overflow);
      op_token = Lex::TokenKind::Slash;
      break;
    case SemIR::BuiltinFunctionKind::IntSMod:
      result_val = lhs_val.srem(rhs_val);
      // LLVM weirdly lacks `srem_ov`, so we work it out for ourselves:
      // <signed min> % -1 overflows because <signed min> / -1 overflows.
      overflow = lhs_val.isMinSignedValue() && rhs_val.isAllOnes();
      op_token = Lex::TokenKind::Percent;
      break;
    case SemIR::BuiltinFunctionKind::IntUAdd:
      result_val = lhs_val + rhs_val;
      op_token = Lex::TokenKind::Plus;
      break;
    case SemIR::BuiltinFunctionKind::IntUSub:
      result_val = lhs_val - rhs_val;
      op_token = Lex::TokenKind::Minus;
      break;
    case SemIR::BuiltinFunctionKind::IntUMul:
      result_val = lhs_val * rhs_val;
      op_token = Lex::TokenKind::Star;
      break;
    case SemIR::BuiltinFunctionKind::IntUDiv:
      result_val = lhs_val.udiv(rhs_val);
      op_token = Lex::TokenKind::Slash;
      break;
    case SemIR::BuiltinFunctionKind::IntUMod:
      result_val = lhs_val.urem(rhs_val);
      op_token = Lex::TokenKind::Percent;
      break;

    // Bitwise.
    case SemIR::BuiltinFunctionKind::IntAnd:
      result_val = lhs_val & rhs_val;
      op_token = Lex::TokenKind::And;
      break;
    case SemIR::BuiltinFunctionKind::IntOr:
      result_val = lhs_val | rhs_val;
      op_token = Lex::TokenKind::Pipe;
      break;
    case SemIR::BuiltinFunctionKind::IntXor:
      result_val = lhs_val ^ rhs_val;
      op_token = Lex::TokenKind::Caret;
      break;

    case SemIR::BuiltinFunctionKind::IntLeftShift:
    case SemIR::BuiltinFunctionKind::IntRightShift:
      CARBON_FATAL("Non-homogeneous operation handled separately.");

    default:
      CARBON_FATAL("Unexpected operation kind.");
  }
  return {.result_val = std::move(result_val),
          .overflow = overflow,
          .op_token = op_token};
}

// Performs a builtin integer bit shift operation.
static auto PerformBuiltinIntShiftOp(Context& context, SemIR::LocId loc_id,
                                     SemIR::BuiltinFunctionKind builtin_kind,
                                     SemIR::InstId lhs_id, SemIR::InstId rhs_id)
    -> SemIR::ConstantId {
  auto lhs = context.insts().GetAs<SemIR::IntValue>(lhs_id);
  auto rhs = context.insts().GetAs<SemIR::IntValue>(rhs_id);

  auto [lhs_is_signed, lhs_bit_width_id] =
      context.sem_ir().types().GetIntTypeInfo(lhs.type_id);

  llvm::APInt lhs_val =
      GetIntAtSuitableWidth(context, lhs.int_id, lhs_bit_width_id);
  const auto& rhs_orig_val = context.ints().Get(rhs.int_id);
  if (lhs_bit_width_id.has_value() && rhs_orig_val.uge(lhs_val.getBitWidth())) {
    CARBON_DIAGNOSTIC(
        CompileTimeShiftOutOfRange, Error,
        "shift distance >= type width of {0} in `{1} {2:<<|>>} {3}`", unsigned,
        TypedInt, Diagnostics::BoolAsSelect, TypedInt);
    context.emitter().Emit(
        loc_id, CompileTimeShiftOutOfRange, lhs_val.getBitWidth(),
        {.type = lhs.type_id, .value = lhs_val},
        builtin_kind == SemIR::BuiltinFunctionKind::IntLeftShift,
        {.type = rhs.type_id, .value = rhs_orig_val});
    // TODO: Is it useful to recover by returning 0 or -1?
    return SemIR::ErrorInst::ConstantId;
  }

  if (rhs_orig_val.isNegative() &&
      context.sem_ir().types().IsSignedInt(rhs.type_id)) {
    CARBON_DIAGNOSTIC(CompileTimeShiftNegative, Error,
                      "shift distance negative in `{0} {1:<<|>>} {2}`",
                      TypedInt, Diagnostics::BoolAsSelect, TypedInt);
    context.emitter().Emit(
        loc_id, CompileTimeShiftNegative,
        {.type = lhs.type_id, .value = lhs_val},
        builtin_kind == SemIR::BuiltinFunctionKind::IntLeftShift,
        {.type = rhs.type_id, .value = rhs_orig_val});
    // TODO: Is it useful to recover by returning 0 or -1?
    return SemIR::ErrorInst::ConstantId;
  }

  llvm::APInt result_val;
  if (builtin_kind == SemIR::BuiltinFunctionKind::IntLeftShift) {
    if (!lhs_bit_width_id.has_value() && !lhs_val.isZero()) {
      // Ensure we don't generate a ridiculously large integer through a bit
      // shift.
      auto width = rhs_orig_val.trySExtValue();
      if (!width ||
          *width > IntStore::MaxIntWidth - lhs_val.getSignificantBits()) {
        CARBON_DIAGNOSTIC(CompileTimeUnsizedShiftOutOfRange, Error,
                          "shift distance of {0} would result in an "
                          "integer whose width is greater than the "
                          "maximum supported width of {1}",
                          TypedInt, int);
        context.emitter().Emit(loc_id, CompileTimeUnsizedShiftOutOfRange,
                               {.type = rhs.type_id, .value = rhs_orig_val},
                               IntStore::MaxIntWidth);
        return SemIR::ErrorInst::ConstantId;
      }
      lhs_val = lhs_val.sext(
          IntStore::CanonicalBitWidth(lhs_val.getSignificantBits() + *width));
    }

    result_val =
        lhs_val.shl(rhs_orig_val.getLimitedValue(lhs_val.getBitWidth()));
  } else if (lhs_is_signed) {
    result_val =
        lhs_val.ashr(rhs_orig_val.getLimitedValue(lhs_val.getBitWidth()));
  } else {
    CARBON_CHECK(lhs_bit_width_id.has_value(), "Logical shift on unsized int");
    result_val =
        lhs_val.lshr(rhs_orig_val.getLimitedValue(lhs_val.getBitWidth()));
  }
  return MakeIntResult(context, lhs.type_id, lhs_is_signed,
                       std::move(result_val));
}

// Performs a homogeneous builtin binary integer -> integer operation.
static auto PerformBuiltinBinaryIntOp(Context& context, SemIR::LocId loc_id,
                                      SemIR::BuiltinFunctionKind builtin_kind,
                                      SemIR::InstId lhs_id,
                                      SemIR::InstId rhs_id)
    -> SemIR::ConstantId {
  auto lhs = context.insts().GetAs<SemIR::IntValue>(lhs_id);
  auto rhs = context.insts().GetAs<SemIR::IntValue>(rhs_id);

  CARBON_CHECK(rhs.type_id == lhs.type_id, "Heterogeneous builtin integer op!");
  auto type_id = lhs.type_id;
  auto [is_signed, bit_width_id] =
      context.sem_ir().types().GetIntTypeInfo(type_id);
  auto [lhs_val, rhs_val] =
      GetIntsAtSuitableWidth(context, lhs.int_id, rhs.int_id, bit_width_id);

  // Check for division by zero.
  switch (builtin_kind) {
    case SemIR::BuiltinFunctionKind::IntSDiv:
    case SemIR::BuiltinFunctionKind::IntSMod:
    case SemIR::BuiltinFunctionKind::IntUDiv:
    case SemIR::BuiltinFunctionKind::IntUMod:
      if (rhs_val.isZero()) {
        DiagnoseDivisionByZero(context, loc_id);
        return SemIR::ErrorInst::ConstantId;
      }
      break;
    default:
      break;
  }

  BinaryIntOpResult result =
      ComputeBinaryIntOpResult(builtin_kind, lhs_val, rhs_val);

  if (result.overflow && !bit_width_id.has_value()) {
    // Retry with a larger bit width. Most operations can only overflow by one
    // bit, but signed n-bit multiplication can overflow to 2n-1 bits. We don't
    // need to handle unsigned multiplication here because it's not permitted
    // for unsized integers.
    //
    // Note that we speculatively first perform the calculation in the width of
    // the wider operand: smaller operations are faster and overflow to a wider
    // integer is unlikely to be needed, especially given that the width will
    // have been rounded up to a multiple of 64 bits by the int store.
    CARBON_CHECK(builtin_kind != SemIR::BuiltinFunctionKind::IntUMul,
                 "Unsigned arithmetic requires a fixed bitwidth");
    int new_width =
        builtin_kind == SemIR::BuiltinFunctionKind::IntSMul
            ? lhs_val.getBitWidth() * 2
            : IntStore::CanonicalBitWidth(lhs_val.getBitWidth() + 1);
    new_width = std::min(new_width, IntStore::MaxIntWidth);
    lhs_val = context.ints().GetAtWidth(lhs.int_id, new_width);
    rhs_val = context.ints().GetAtWidth(rhs.int_id, new_width);

    // Note that this can in theory still overflow if we limited `new_width` to
    // `MaxIntWidth`. In that case we fall through to the signed overflow
    // diagnostic below.
    result = ComputeBinaryIntOpResult(builtin_kind, lhs_val, rhs_val);
    CARBON_CHECK(!result.overflow || new_width == IntStore::MaxIntWidth);
  }

  if (result.overflow) {
    CARBON_DIAGNOSTIC(CompileTimeIntegerOverflow, Error,
                      "integer overflow in calculation `{0} {1} {2}`", TypedInt,
                      Lex::TokenKind, TypedInt);
    context.emitter().Emit(loc_id, CompileTimeIntegerOverflow,
                           {.type = type_id, .value = lhs_val}, result.op_token,
                           {.type = type_id, .value = rhs_val});
  }

  return MakeIntResult(context, type_id, is_signed,
                       std::move(result.result_val));
}

// Performs a builtin integer comparison.
static auto PerformBuiltinIntComparison(Context& context,
                                        SemIR::BuiltinFunctionKind builtin_kind,
                                        SemIR::InstId lhs_id,
                                        SemIR::InstId rhs_id,
                                        SemIR::TypeId bool_type_id)
    -> SemIR::ConstantId {
  auto lhs = context.insts().GetAs<SemIR::IntValue>(lhs_id);
  auto rhs = context.insts().GetAs<SemIR::IntValue>(rhs_id);
  llvm::APInt lhs_val = context.ints().Get(lhs.int_id);
  llvm::APInt rhs_val = context.ints().Get(rhs.int_id);

  bool result;
  switch (builtin_kind) {
    case SemIR::BuiltinFunctionKind::IntEq:
      result = (lhs_val == rhs_val);
      break;
    case SemIR::BuiltinFunctionKind::IntNeq:
      result = (lhs_val != rhs_val);
      break;
    case SemIR::BuiltinFunctionKind::IntLess:
      result = lhs_val.slt(rhs_val);
      break;
    case SemIR::BuiltinFunctionKind::IntLessEq:
      result = lhs_val.sle(rhs_val);
      break;
    case SemIR::BuiltinFunctionKind::IntGreater:
      result = lhs_val.sgt(rhs_val);
      break;
    case SemIR::BuiltinFunctionKind::IntGreaterEq:
      result = lhs_val.sge(rhs_val);
      break;
    default:
      CARBON_FATAL("Unexpected operation kind.");
  }

  return MakeBoolResult(context, bool_type_id, result);
}

// Performs a builtin unary float -> float operation.
static auto PerformBuiltinUnaryFloatOp(Context& context,
                                       SemIR::BuiltinFunctionKind builtin_kind,
                                       SemIR::InstId arg_id)
    -> SemIR::ConstantId {
  auto op = context.insts().GetAs<SemIR::FloatValue>(arg_id);
  auto op_val = context.floats().Get(op.float_id);

  switch (builtin_kind) {
    case SemIR::BuiltinFunctionKind::FloatNegate:
      op_val.changeSign();
      break;
    default:
      CARBON_FATAL("Unexpected builtin kind");
  }

  return MakeFloatResult(context, op.type_id, std::move(op_val));
}

// Performs a builtin binary float -> float operation.
static auto PerformBuiltinBinaryFloatOp(Context& context,
                                        SemIR::BuiltinFunctionKind builtin_kind,
                                        SemIR::InstId lhs_id,
                                        SemIR::InstId rhs_id)
    -> SemIR::ConstantId {
  auto lhs = context.insts().GetAs<SemIR::FloatValue>(lhs_id);
  auto rhs = context.insts().GetAs<SemIR::FloatValue>(rhs_id);
  auto lhs_val = context.floats().Get(lhs.float_id);
  auto rhs_val = context.floats().Get(rhs.float_id);

  llvm::APFloat result_val(lhs_val.getSemantics());

  switch (builtin_kind) {
    case SemIR::BuiltinFunctionKind::FloatAdd:
      result_val = lhs_val + rhs_val;
      break;
    case SemIR::BuiltinFunctionKind::FloatSub:
      result_val = lhs_val - rhs_val;
      break;
    case SemIR::BuiltinFunctionKind::FloatMul:
      result_val = lhs_val * rhs_val;
      break;
    case SemIR::BuiltinFunctionKind::FloatDiv:
      result_val = lhs_val / rhs_val;
      break;
    default:
      CARBON_FATAL("Unexpected operation kind.");
  }

  return MakeFloatResult(context, lhs.type_id, std::move(result_val));
}

// Performs a builtin float comparison.
static auto PerformBuiltinFloatComparison(
    Context& context, SemIR::BuiltinFunctionKind builtin_kind,
    SemIR::InstId lhs_id, SemIR::InstId rhs_id, SemIR::TypeId bool_type_id)
    -> SemIR::ConstantId {
  auto lhs = context.insts().GetAs<SemIR::FloatValue>(lhs_id);
  auto rhs = context.insts().GetAs<SemIR::FloatValue>(rhs_id);
  const auto& lhs_val = context.floats().Get(lhs.float_id);
  const auto& rhs_val = context.floats().Get(rhs.float_id);

  bool result;
  switch (builtin_kind) {
    case SemIR::BuiltinFunctionKind::FloatEq:
      result = (lhs_val == rhs_val);
      break;
    case SemIR::BuiltinFunctionKind::FloatNeq:
      result = (lhs_val != rhs_val);
      break;
    case SemIR::BuiltinFunctionKind::FloatLess:
      result = lhs_val < rhs_val;
      break;
    case SemIR::BuiltinFunctionKind::FloatLessEq:
      result = lhs_val <= rhs_val;
      break;
    case SemIR::BuiltinFunctionKind::FloatGreater:
      result = lhs_val > rhs_val;
      break;
    case SemIR::BuiltinFunctionKind::FloatGreaterEq:
      result = lhs_val >= rhs_val;
      break;
    default:
      CARBON_FATAL("Unexpected operation kind.");
  }

  return MakeBoolResult(context, bool_type_id, result);
}

// Performs a builtin boolean comparison.
static auto PerformBuiltinBoolComparison(
    Context& context, SemIR::BuiltinFunctionKind builtin_kind,
    SemIR::InstId lhs_id, SemIR::InstId rhs_id, SemIR::TypeId bool_type_id) {
  bool lhs = context.insts().GetAs<SemIR::BoolLiteral>(lhs_id).value.ToBool();
  bool rhs = context.insts().GetAs<SemIR::BoolLiteral>(rhs_id).value.ToBool();
  return MakeBoolResult(context, bool_type_id,
                        builtin_kind == SemIR::BuiltinFunctionKind::BoolEq
                            ? lhs == rhs
                            : lhs != rhs);
}

// Converts a call argument to a FacetTypeId.
static auto ArgToFacetTypeId(Context& context, SemIR::LocId loc_id,
                             SemIR::InstId arg_id) -> SemIR::FacetTypeId {
  auto type_arg_id = context.types().GetAsTypeInstId(arg_id);
  if (auto facet_type =
          context.insts().TryGetAs<SemIR::FacetType>(type_arg_id)) {
    return facet_type->facet_type_id;
  }
  CARBON_DIAGNOSTIC(FacetTypeRequiredForTypeAndOperator, Error,
                    "non-facet type {0} combined with `&` operator",
                    SemIR::TypeId);
  // TODO: Find a location for the lhs or rhs specifically, instead of
  // the whole thing. If that's not possible we can change the text to
  // say if it's referring to the left or the right side for the error.
  // The `arg_id` instruction has no location in it for some reason.
  context.emitter().Emit(loc_id, FacetTypeRequiredForTypeAndOperator,
                         context.types().GetTypeIdForTypeInstId(type_arg_id));
  return SemIR::FacetTypeId::None;
}

// Returns a constant for a call to a builtin function.
static auto MakeConstantForBuiltinCall(EvalContext& eval_context,
                                       SemIR::LocId loc_id, SemIR::Call call,
                                       SemIR::BuiltinFunctionKind builtin_kind,
                                       llvm::ArrayRef<SemIR::InstId> arg_ids,
                                       Phase phase) -> SemIR::ConstantId {
  auto& context = eval_context.context();
  switch (builtin_kind) {
    case SemIR::BuiltinFunctionKind::None:
      CARBON_FATAL("Not a builtin function.");

    case SemIR::BuiltinFunctionKind::NoOp: {
      return MakeEmptyTupleResult(eval_context);
    }

    case SemIR::BuiltinFunctionKind::PrimitiveCopy: {
      return context.constant_values().Get(arg_ids[0]);
    }

    case SemIR::BuiltinFunctionKind::StringAt: {
      Phase phase = Phase::Concrete;
      auto str_id = GetConstantValue(eval_context, arg_ids[0], &phase);
      auto index_id = GetConstantValue(eval_context, arg_ids[1], &phase);

      if (phase != Phase::Concrete) {
        return MakeNonConstantResult(phase);
      }

      auto str_struct = eval_context.insts().GetAs<SemIR::StructValue>(str_id);
      auto elements = eval_context.inst_blocks().Get(str_struct.elements_id);
      // String struct has two fields: a pointer to the string data and the
      // length.
      CARBON_CHECK(elements.size() == 2, "String struct should have 2 fields.");

      auto string_literal = eval_context.insts().GetAs<SemIR::StringLiteral>(
          eval_context.constant_values().GetConstantInstId(elements[0]));

      const auto& string_value =
          eval_context.sem_ir().string_literal_values().Get(
              string_literal.string_literal_id);

      auto index_inst = eval_context.insts().GetAs<SemIR::IntValue>(index_id);
      const auto& index_val = eval_context.ints().Get(index_inst.int_id);

      if (index_val.isNegative()) {
        CARBON_DIAGNOSTIC(StringAtIndexNegative, Error,
                          "index `{0}` is negative.", TypedInt);
        context.emitter().Emit(
            loc_id, StringAtIndexNegative,
            {.type = eval_context.insts().Get(index_id).type_id(),
             .value = index_val});
        return SemIR::ConstantId::NotConstant;
      }

      if (index_val.getZExtValue() >= string_value.size()) {
        CARBON_DIAGNOSTIC(
            StringAtIndexOutOfBounds, Error,
            "string index `{0}` is out of bounds; string has length {1}.",
            TypedInt, size_t);
        context.emitter().Emit(
            loc_id, StringAtIndexOutOfBounds,
            {.type = eval_context.insts().Get(index_id).type_id(),
             .value = index_val},
            string_value.size());
        return SemIR::ConstantId::NotConstant;
      }

      auto char_value =
          static_cast<uint8_t>(string_value[index_val.getZExtValue()]);

      auto int_id = eval_context.ints().Add(
          llvm::APSInt(llvm::APInt(32, char_value), /*isUnsigned=*/false));
      return MakeConstantResult(
          eval_context.context(),
          SemIR::IntValue{.type_id = call.type_id, .int_id = int_id}, phase);
    }

    case SemIR::BuiltinFunctionKind::MakeUninitialized:
    case SemIR::BuiltinFunctionKind::PrintChar:
    case SemIR::BuiltinFunctionKind::PrintInt:
    case SemIR::BuiltinFunctionKind::ReadChar:
    case SemIR::BuiltinFunctionKind::FloatAddAssign:
    case SemIR::BuiltinFunctionKind::FloatSubAssign:
    case SemIR::BuiltinFunctionKind::FloatMulAssign:
    case SemIR::BuiltinFunctionKind::FloatDivAssign:
    case SemIR::BuiltinFunctionKind::IntSAddAssign:
    case SemIR::BuiltinFunctionKind::IntSSubAssign:
    case SemIR::BuiltinFunctionKind::IntSMulAssign:
    case SemIR::BuiltinFunctionKind::IntSDivAssign:
    case SemIR::BuiltinFunctionKind::IntSModAssign:
    case SemIR::BuiltinFunctionKind::IntUAddAssign:
    case SemIR::BuiltinFunctionKind::IntUSubAssign:
    case SemIR::BuiltinFunctionKind::IntUMulAssign:
    case SemIR::BuiltinFunctionKind::IntUDivAssign:
    case SemIR::BuiltinFunctionKind::IntUModAssign:
    case SemIR::BuiltinFunctionKind::IntAndAssign:
    case SemIR::BuiltinFunctionKind::IntOrAssign:
    case SemIR::BuiltinFunctionKind::IntXorAssign:
    case SemIR::BuiltinFunctionKind::IntLeftShiftAssign:
    case SemIR::BuiltinFunctionKind::IntRightShiftAssign:
    case SemIR::BuiltinFunctionKind::PointerMakeNull:
    case SemIR::BuiltinFunctionKind::PointerIsNull:
    case SemIR::BuiltinFunctionKind::PointerUnsafeConvert:
    case SemIR::BuiltinFunctionKind::CppStdInitializerListMake: {
      // These are runtime-only builtins.
      // TODO: Consider tracking this on the `BuiltinFunctionKind`.
      return SemIR::ConstantId::NotConstant;
    }

    case SemIR::BuiltinFunctionKind::TypeAnd: {
      CARBON_CHECK(arg_ids.size() == 2);
      auto lhs_facet_type_id = ArgToFacetTypeId(context, loc_id, arg_ids[0]);
      auto rhs_facet_type_id = ArgToFacetTypeId(context, loc_id, arg_ids[1]);

      // Allow errors to be diagnosed for both sides of the operator before
      // returning here if any error occurred on either side.
      if (!lhs_facet_type_id.has_value() || !rhs_facet_type_id.has_value()) {
        return SemIR::ErrorInst::ConstantId;
      }
      // Reuse one of the argument instructions if nothing has changed.
      if (lhs_facet_type_id == rhs_facet_type_id) {
        return context.types().GetConstantId(
            context.types().GetTypeIdForTypeInstId(arg_ids[0]));
      }
      auto combined_info = SemIR::FacetTypeInfo::Combine(
          context.facet_types().Get(lhs_facet_type_id),
          context.facet_types().Get(rhs_facet_type_id));
      if (!ResolveFacetTypeRewriteConstraints(
              eval_context.context(), loc_id,
              combined_info.rewrite_constraints)) {
        phase = Phase::UnknownDueToError;
      }
      combined_info.Canonicalize();
      return MakeFacetTypeResult(eval_context.context(), combined_info, phase);
    }

    case SemIR::BuiltinFunctionKind::CharLiteralMakeType: {
      return context.constant_values().Get(SemIR::CharLiteralType::TypeInstId);
    }

    case SemIR::BuiltinFunctionKind::FloatLiteralMakeType: {
      return context.constant_values().Get(SemIR::FloatLiteralType::TypeInstId);
    }

    case SemIR::BuiltinFunctionKind::IntLiteralMakeType: {
      return context.constant_values().Get(SemIR::IntLiteralType::TypeInstId);
    }

    case SemIR::BuiltinFunctionKind::IntMakeTypeSigned: {
      return MakeIntTypeResult(context, loc_id, SemIR::IntKind::Signed,
                               arg_ids[0], phase);
    }

    case SemIR::BuiltinFunctionKind::IntMakeTypeUnsigned: {
      return MakeIntTypeResult(context, loc_id, SemIR::IntKind::Unsigned,
                               arg_ids[0], phase);
    }

    case SemIR::BuiltinFunctionKind::FloatMakeType: {
      return MakeFloatTypeResult(context, loc_id, arg_ids[0], phase);
    }

    case SemIR::BuiltinFunctionKind::BoolMakeType: {
      return context.constant_values().Get(SemIR::BoolType::TypeInstId);
    }

    case SemIR::BuiltinFunctionKind::MaybeUnformedMakeType: {
      return MakeConstantResult(
          context,
          SemIR::MaybeUnformedType{
              .type_id = SemIR::TypeType::TypeId,
              .inner_id = context.types().GetAsTypeInstId(arg_ids[0])},
          phase);
    }

    case SemIR::BuiltinFunctionKind::FormMakeType: {
      return context.constant_values().Get(SemIR::FormType::TypeInstId);
    }

    // Character conversions.
    case SemIR::BuiltinFunctionKind::CharConvertChecked: {
      if (phase != Phase::Concrete) {
        return MakeConstantResult(context, call, phase);
      }
      return PerformCheckedCharConvert(context, loc_id, arg_ids[0],
                                       call.type_id);
    }

    // Integer conversions.
    case SemIR::BuiltinFunctionKind::IntConvertChar: {
      if (phase != Phase::Concrete) {
        return MakeConstantResult(context, call, phase);
      }
      return PerformIntConvert(context, arg_ids[0], call.type_id);
    }
    case SemIR::BuiltinFunctionKind::IntConvert: {
      if (phase != Phase::Concrete) {
        return MakeConstantResult(context, call, phase);
      }
      return PerformIntConvert(context, arg_ids[0], call.type_id);
    }
    case SemIR::BuiltinFunctionKind::IntConvertChecked: {
      if (phase != Phase::Concrete) {
        return MakeConstantResult(context, call, phase);
      }
      return PerformCheckedIntConvert(context, loc_id, arg_ids[0],
                                      call.type_id);
    }

    // Unary integer -> integer operations.
    case SemIR::BuiltinFunctionKind::IntSNegate:
    case SemIR::BuiltinFunctionKind::IntUNegate:
    case SemIR::BuiltinFunctionKind::IntComplement: {
      if (phase != Phase::Concrete) {
        break;
      }
      return PerformBuiltinUnaryIntOp(context, loc_id, builtin_kind,
                                      arg_ids[0]);
    }

    // Homogeneous binary integer -> integer operations.
    case SemIR::BuiltinFunctionKind::IntSAdd:
    case SemIR::BuiltinFunctionKind::IntSSub:
    case SemIR::BuiltinFunctionKind::IntSMul:
    case SemIR::BuiltinFunctionKind::IntSDiv:
    case SemIR::BuiltinFunctionKind::IntSMod:
    case SemIR::BuiltinFunctionKind::IntUAdd:
    case SemIR::BuiltinFunctionKind::IntUSub:
    case SemIR::BuiltinFunctionKind::IntUMul:
    case SemIR::BuiltinFunctionKind::IntUDiv:
    case SemIR::BuiltinFunctionKind::IntUMod:
    case SemIR::BuiltinFunctionKind::IntAnd:
    case SemIR::BuiltinFunctionKind::IntOr:
    case SemIR::BuiltinFunctionKind::IntXor: {
      if (phase != Phase::Concrete) {
        break;
      }
      return PerformBuiltinBinaryIntOp(context, loc_id, builtin_kind,
                                       arg_ids[0], arg_ids[1]);
    }

    // Bit shift operations.
    case SemIR::BuiltinFunctionKind::IntLeftShift:
    case SemIR::BuiltinFunctionKind::IntRightShift: {
      if (phase != Phase::Concrete) {
        break;
      }
      return PerformBuiltinIntShiftOp(context, loc_id, builtin_kind, arg_ids[0],
                                      arg_ids[1]);
    }

    // Integer comparisons.
    case SemIR::BuiltinFunctionKind::IntEq:
    case SemIR::BuiltinFunctionKind::IntNeq:
    case SemIR::BuiltinFunctionKind::IntLess:
    case SemIR::BuiltinFunctionKind::IntLessEq:
    case SemIR::BuiltinFunctionKind::IntGreater:
    case SemIR::BuiltinFunctionKind::IntGreaterEq: {
      if (phase != Phase::Concrete) {
        break;
      }
      return PerformBuiltinIntComparison(context, builtin_kind, arg_ids[0],
                                         arg_ids[1], call.type_id);
    }

    // Floating-point conversions.
    case SemIR::BuiltinFunctionKind::FloatConvertChecked: {
      if (phase != Phase::Concrete) {
        return MakeConstantResult(context, call, phase);
      }
      return PerformCheckedFloatConvert(context, loc_id, arg_ids[0],
                                        call.type_id);
    }

    // Unary float -> float operations.
    case SemIR::BuiltinFunctionKind::FloatNegate: {
      if (phase != Phase::Concrete) {
        break;
      }

      return PerformBuiltinUnaryFloatOp(context, builtin_kind, arg_ids[0]);
    }

    // Binary float -> float operations.
    case SemIR::BuiltinFunctionKind::FloatAdd:
    case SemIR::BuiltinFunctionKind::FloatSub:
    case SemIR::BuiltinFunctionKind::FloatMul:
    case SemIR::BuiltinFunctionKind::FloatDiv: {
      if (phase != Phase::Concrete) {
        break;
      }
      return PerformBuiltinBinaryFloatOp(context, builtin_kind, arg_ids[0],
                                         arg_ids[1]);
    }

    // Float comparisons.
    case SemIR::BuiltinFunctionKind::FloatEq:
    case SemIR::BuiltinFunctionKind::FloatNeq:
    case SemIR::BuiltinFunctionKind::FloatLess:
    case SemIR::BuiltinFunctionKind::FloatLessEq:
    case SemIR::BuiltinFunctionKind::FloatGreater:
    case SemIR::BuiltinFunctionKind::FloatGreaterEq: {
      if (phase != Phase::Concrete) {
        break;
      }
      return PerformBuiltinFloatComparison(context, builtin_kind, arg_ids[0],
                                           arg_ids[1], call.type_id);
    }

    // Bool comparisons.
    case SemIR::BuiltinFunctionKind::BoolEq:
    case SemIR::BuiltinFunctionKind::BoolNeq: {
      if (phase != Phase::Concrete) {
        break;
      }
      return PerformBuiltinBoolComparison(context, builtin_kind, arg_ids[0],
                                          arg_ids[1], call.type_id);
    }
  }

  return SemIR::ConstantId::NotConstant;
}

static auto TryEvalCall(EvalContext& outer_eval_context, SemIR::LocId loc_id,
                        const SemIR::Function& function,
                        SemIR::SpecificId specific_id,
                        SemIR::InstBlockId args_id) -> SemIR::ConstantId;

// Returns the range of parameter indexes that contain the return storage for
// this function call.
static auto GetReturnStorageParamIndexRange(EvalContext& eval_context,
                                            const SemIR::Callee& callee)
    -> std::pair<int, int> {
  if (const auto* callee_function =
          std::get_if<SemIR::CalleeFunction>(&callee)) {
    const auto& function =
        eval_context.functions().Get(callee_function->function_id);
    return {function.call_param_ranges.return_begin().index,
            function.call_param_ranges.return_end().index};
  }

  return {0, 0};
}

// Replace the `args_id` field of a call with its constant value. The return
// storage argument, if any, is instead replaced with `None`.
static auto ReplaceCallArgsFieldWithConstantValue(EvalContext& eval_context,
                                                  const SemIR::Callee& callee,
                                                  SemIR::Call* call,
                                                  Phase* phase) -> bool {
  auto return_storage_param_index_range =
      GetReturnStorageParamIndexRange(eval_context, callee);
  auto args_id = GetConstantBlockValueIgnoringIndexRange(
      eval_context, call->args_id, phase, return_storage_param_index_range);
  if (!args_id.has_value() && call->args_id.has_value()) {
    return false;
  }
  call->args_id = args_id;
  return IsConstantOrError(*phase);
}

// Makes a constant for a call instruction.
static auto MakeConstantForCall(EvalContext& eval_context,
                                SemIR::InstId inst_id, SemIR::Call call)
    -> SemIR::ConstantId {
  Phase phase = Phase::Concrete;

  // A call with an invalid argument list is used to represent an erroneous
  // call.
  //
  // TODO: Use a better representation for this.
  if (call.args_id == SemIR::InstBlockId::None) {
    return SemIR::ErrorInst::ConstantId;
  }

  // If the callee is a C++ thunk, modify the `call` to directly call
  // the thunk's callee.
  MaybeModifyCppThunkCallForConstEval(eval_context.context(), &call);

  // Find the constant value of the callee.
  bool has_constant_callee = ReplaceFieldWithConstantValue(
      eval_context, &call, &SemIR::Call::callee_id, &phase);

  auto callee = SemIR::GetCallee(eval_context.sem_ir(), call.callee_id);
  const SemIR::Function* function = nullptr;
  auto builtin_kind = SemIR::BuiltinFunctionKind::None;
  auto evaluation_mode = SemIR::Function::EvaluationMode::None;
  if (auto* callee_function = std::get_if<SemIR::CalleeFunction>(&callee)) {
    function = &eval_context.functions().Get(callee_function->function_id);
    builtin_kind = function->builtin_function_kind();
    evaluation_mode = function->evaluation_mode;
    // Calls to builtins and to `eval` or `musteval` functions might be
    // constant.
    if (builtin_kind == SemIR::BuiltinFunctionKind::None &&
        evaluation_mode == SemIR::Function::EvaluationMode::None) {
      return SemIR::ConstantId::NotConstant;
    }
  } else {
    // Calls to non-functions, such as calls to generic entity names, might be
    // constant.
  }

  // Find the argument values and the return type.
  bool has_constant_operands =
      has_constant_callee &&
      ReplaceTypeWithConstantValue(eval_context, inst_id, &call, &phase) &&
      ReplaceCallArgsFieldWithConstantValue(eval_context, callee, &call,
                                            &phase);
  if (phase == Phase::UnknownDueToError) {
    return SemIR::ErrorInst::ConstantId;
  }

  // If any operand of the call is non-constant, the call is non-constant.
  // TODO: Some builtin calls might allow some operands to be non-constant.
  if (!has_constant_operands) {
    if (builtin_kind.IsCompTimeOnly(
            eval_context.sem_ir(), eval_context.inst_blocks().Get(call.args_id),
            call.type_id) ||
        evaluation_mode == SemIR::Function::EvaluationMode::MustEval) {
      CARBON_DIAGNOSTIC(NonConstantCallToCompTimeOnlyFunction, Error,
                        "non-constant call to compile-time-only function");
      CARBON_DIAGNOSTIC(CompTimeOnlyFunctionHere, Note,
                        "compile-time-only function declared here");
      const auto& function = eval_context.functions().Get(
          std::get<SemIR::CalleeFunction>(callee).function_id);
      eval_context.emitter()
          .Build(inst_id, NonConstantCallToCompTimeOnlyFunction)
          .Note(function.latest_decl_id(), CompTimeOnlyFunctionHere)
          .Emit();
    }
    return SemIR::ConstantId::NotConstant;
  }

  // Handle calls to builtins.
  if (builtin_kind != SemIR::BuiltinFunctionKind::None) {
    return MakeConstantForBuiltinCall(
        eval_context, SemIR::LocId(inst_id), call, builtin_kind,
        eval_context.inst_blocks().Get(call.args_id), phase);
  }

  // Handle calls to `eval` and `musteval` functions.
  if (evaluation_mode != SemIR::Function::EvaluationMode::None) {
    // A non-concrete call to `eval` or `musteval` is a template symbolic
    // constant, regardless of the phase of the arguments.
    if (phase != Phase::Concrete) {
      CARBON_CHECK(phase <= Phase::TemplateSymbolic);
      return MakeConstantResult(eval_context.context(), call,
                                Phase::TemplateSymbolic);
    }

    // TODO: Instead of performing the call immediately, add it to a work queue
    // and do it non-recursively.
    return TryEvalCall(
        eval_context, SemIR::LocId(inst_id), *function,
        std::get<SemIR::CalleeFunction>(callee).resolved_specific_id,
        call.args_id);
  }

  return SemIR::ConstantId::NotConstant;
}

// Given an instruction, compute its phase based on its operands.
static auto ComputeInstPhase(Context& context, SemIR::Inst inst) -> Phase {
  EvalContext eval_context(&context, SemIR::LocId::None);

  auto phase = GetPhase(context.constant_values(),
                        context.types().GetConstantId(inst.type_id()));
  GetConstantValueForArg(eval_context, inst.arg0_and_kind(), &phase);
  GetConstantValueForArg(eval_context, inst.arg1_and_kind(), &phase);
  CARBON_CHECK(IsConstantOrError(phase));
  return phase;
}

// Convert a ConstantEvalResult to a ConstantId. Factored out of
// TryEvalTypedInst to avoid repeated instantiation of common code.
static auto ConvertEvalResultToConstantId(Context& context,
                                          ConstantEvalResult result,
                                          SemIR::InstKind orig_inst_kind,
                                          Phase orig_phase)
    -> SemIR::ConstantId {
  if (result.is_new()) {
    auto is_symbolic_only =
        orig_inst_kind.constant_kind() == SemIR::InstConstantKind::SymbolicOnly;
    auto new_phase = result.same_phase_as_inst()
                         ? orig_phase
                         : ComputeInstPhase(context, result.new_inst());
    CARBON_CHECK(!is_symbolic_only || new_phase > Phase::Concrete ||
                     result.new_inst().kind() != orig_inst_kind,
                 "SymbolicOnly instruction `{0}` has a concrete value",
                 orig_inst_kind);
    return MakeConstantResult(context, result.new_inst(), new_phase);
  }
  return result.existing();
}

// Evaluates an instruction of a known type in an evaluation context. The
// default behavior of this function depends on the constant kind of the
// instruction:
//
//  -  InstConstantKind::Never: returns ConstantId::NotConstant.
//  -  InstConstantKind::Indirect, SymbolicOnly, SymbolicOrReference,
//     Conditional: evaluates all the operands of the instruction, and calls
//     `EvalConstantInst` to evaluate the resulting constant instruction.
//  -  InstConstantKind::WheneverPossible, Always: evaluates all the operands of
//     the instruction, and produces the resulting constant instruction as the
//     result.
//  -  InstConstantKind::Unique: returns the `inst_id` as the resulting
//     constant.
//
// Returns an error constant ID if any of the nested evaluations fail, and
// returns NotConstant if any of the nested evaluations is non-constant.
//
// This template is explicitly specialized for instructions that need special
// handling.
template <typename InstT>
static auto TryEvalTypedInst(EvalContext& eval_context, SemIR::InstId inst_id,
                             SemIR::Inst inst) -> SemIR::ConstantId {
  constexpr auto ConstantKind = InstT::Kind.constant_kind();
  if constexpr (ConstantKind == SemIR::InstConstantKind::Never) {
    return SemIR::ConstantId::NotConstant;
  } else if constexpr (ConstantKind == SemIR::InstConstantKind::AlwaysUnique) {
    CARBON_CHECK(inst_id.has_value());
    return SemIR::ConstantId::ForConcreteConstant(inst_id);
  } else {
    // Build a constant instruction by replacing each non-constant operand with
    // its constant value.
    Phase phase = Phase::Concrete;
    if ((SemIR::Internal::HasTypeIdMember<InstT> &&
         !ReplaceTypeWithConstantValue(eval_context, inst_id, &inst, &phase)) ||
        !ReplaceAllFieldsWithConstantValues(eval_context, &inst, &phase)) {
      if constexpr (ConstantKind == SemIR::InstConstantKind::Always) {
        CARBON_FATAL("{0} should always be constant", InstT::Kind);
      }
      return SemIR::ConstantId::NotConstant;
    }
    // If any operand of the instruction has an error in it, the instruction
    // itself evaluates to an error.
    if (phase == Phase::UnknownDueToError) {
      return SemIR::ErrorInst::ConstantId;
    }

    // When canonicalizing a SpecificId, we defer resolving the specific's
    // declaration until here, to avoid resolving declarations from imported
    // specifics. (Imported instructions are not evaluated.)
    ResolveSpecificDeclForInst(eval_context, inst);

    if constexpr (ConstantKind == SemIR::InstConstantKind::Always ||
                  ConstantKind == SemIR::InstConstantKind::WheneverPossible) {
      return MakeConstantResult(eval_context.context(), inst, phase);
    } else if constexpr (ConstantKind == SemIR::InstConstantKind::InstAction) {
      auto result_inst_id = PerformDelayedAction(
          eval_context.context(), SemIR::LocId(inst_id), inst.As<InstT>());
      if (result_inst_id.has_value()) {
        // The result is an instruction.
        return MakeConstantResult(
            eval_context.context(),
            SemIR::InstValue{
                .type_id = GetSingletonType(eval_context.context(),
                                            SemIR::InstType::TypeInstId),
                .inst_id = result_inst_id},
            Phase::Concrete);
      }
      // Couldn't perform the action because it's still dependent.
      return MakeConstantResult(eval_context.context(), inst,
                                Phase::TemplateSymbolic);
    } else if constexpr (InstT::Kind.constant_needs_inst_id() !=
                         SemIR::InstConstantNeedsInstIdKind::No) {
      CARBON_CHECK(inst_id.has_value());
      return ConvertEvalResultToConstantId(
          eval_context.context(),
          EvalConstantInst(eval_context.context(), inst_id, inst.As<InstT>()),
          InstT::Kind, phase);
    } else {
      return ConvertEvalResultToConstantId(
          eval_context.context(),
          EvalConstantInst(eval_context.context(), inst.As<InstT>()),
          InstT::Kind, phase);
    }
  }
}

// Specialize evaluation for array indexing because we want to check the index
// expression even if the array expression is non-constant.
template <>
auto TryEvalTypedInst<SemIR::ArrayIndex>(EvalContext& eval_context,
                                         SemIR::InstId /*inst_id*/,
                                         SemIR::Inst inst)
    -> SemIR::ConstantId {
  return PerformArrayIndex(eval_context, inst.As<SemIR::ArrayIndex>());
}

// Specialize evaluation for function calls because we want to check the callee
// expression even if an argument expression is non-constant, and because we
// will eventually want to perform control flow handling here.
template <>
auto TryEvalTypedInst<SemIR::Call>(EvalContext& eval_context,
                                   SemIR::InstId inst_id, SemIR::Inst inst)
    -> SemIR::ConstantId {
  return MakeConstantForCall(eval_context, inst_id, inst.As<SemIR::Call>());
}

// ImportRefLoaded can have a constant value, but it's owned and maintained by
// `import_ref.cpp`, not by us.
// TODO: Rearrange how `ImportRefLoaded` instructions are created so we never
// call this.
template <>
auto TryEvalTypedInst<SemIR::ImportRefLoaded>(EvalContext& /*eval_context*/,
                                              SemIR::InstId /*inst_id*/,
                                              SemIR::Inst /*inst*/)
    -> SemIR::ConstantId {
  return SemIR::ConstantId::NotConstant;
}

// Symbolic bindings are a special case because they can reach into the eval
// context and produce a context-specific value.
template <>
auto TryEvalTypedInst<SemIR::SymbolicBinding>(EvalContext& eval_context,
                                              SemIR::InstId inst_id,
                                              SemIR::Inst inst)
    -> SemIR::ConstantId {
  auto bind = inst.As<SemIR::SymbolicBinding>();

  // If we know which specific we're evaluating within and this is an argument
  // of that specific, its constant value is the corresponding argument value.
  const auto& bind_name = eval_context.entity_names().Get(bind.entity_name_id);
  if (bind_name.bind_index().has_value()) {
    if (auto value =
            eval_context.GetCompileTimeBindValue(bind_name.bind_index());
        value.has_value()) {
      return value;
    }
  }

  // The constant form of a symbolic binding is an idealized form of the
  // original, with no equivalent value.
  Phase phase = Phase::Concrete;
  bind.value_id = SemIR::InstId::None;
  if (!ReplaceTypeWithConstantValue(eval_context, inst_id, &bind, &phase) ||
      !ReplaceFieldWithConstantValue(eval_context, &bind,
                                     &SemIR::SymbolicBinding::entity_name_id,
                                     &phase)) {
    return SemIR::ConstantId::NotConstant;
  }
  // This correctly handles `Phase::UnknownDueToError`.
  return MakeConstantResult(eval_context.context(), bind, phase);
}

template <>
auto TryEvalTypedInst<SemIR::SymbolicBindingType>(EvalContext& eval_context,
                                                  SemIR::InstId inst_id,
                                                  SemIR::Inst inst)
    -> SemIR::ConstantId {
  // If a specific provides a new value for the binding with `entity_name_id`,
  // the SymbolicBindingType is evaluated for that new value.
  const auto& bind_name = eval_context.entity_names().Get(
      inst.As<SemIR::SymbolicBindingType>().entity_name_id);
  if (bind_name.bind_index().has_value()) {
    if (auto value =
            eval_context.GetCompileTimeBindValue(bind_name.bind_index());
        value.has_value()) {
      auto value_inst_id = eval_context.constant_values().GetInstId(value);

      // A SymbolicBindingType can evaluate to a FacetAccessType if the new
      // value of the entity is a facet value that that does not have a concrete
      // type (a FacetType) and does not have a new EntityName to point to (a
      // SymbolicBinding).
      auto access = SemIR::FacetAccessType{
          .type_id = SemIR::TypeType::TypeId,
          .facet_value_inst_id = value_inst_id,
      };
      return ConvertEvalResultToConstantId(
          eval_context.context(),
          EvalConstantInst(eval_context.context(), access),
          SemIR::SymbolicBindingType::Kind,
          ComputeInstPhase(eval_context.context(), access));
    }
  }

  Phase phase = Phase::Concrete;
  if (!ReplaceTypeWithConstantValue(eval_context, inst_id, &inst, &phase) ||
      !ReplaceAllFieldsWithConstantValues(eval_context, &inst, &phase)) {
    return SemIR::ConstantId::NotConstant;
  }
  // Propagate error phase after getting the constant value for all fields.
  if (phase == Phase::UnknownDueToError) {
    return SemIR::ErrorInst::ConstantId;
  }

  // Evaluation of SymbolicBindingType.
  //
  // Like FacetAccessType, a SymbolicBindingType of a FacetValue just evaluates
  // to the type inside.
  //
  // TODO: Look in ScopeStack with the entity_name_id to find the facet value
  // and get its constant value in the current specific context. The
  // facet_value_inst_id will go away.
  if (auto facet_value = eval_context.insts().TryGetAs<SemIR::FacetValue>(
          inst.As<SemIR::SymbolicBindingType>().facet_value_inst_id)) {
    return eval_context.constant_values().Get(facet_value->type_inst_id);
  }

  return MakeConstantResult(eval_context.context(), inst, phase);
}

template <>
auto TryEvalTypedInst<SemIR::Temporary>(EvalContext& eval_context,
                                        SemIR::InstId inst_id, SemIR::Inst inst)
    -> SemIR::ConstantId {
  auto temporary = inst.As<SemIR::Temporary>();
  temporary.storage_id = SemIR::InstId::None;

  Phase phase = Phase::Concrete;
  if (!ReplaceTypeWithConstantValue(eval_context, inst_id, &temporary,
                                    &phase) ||
      !ReplaceFieldWithConstantValue(eval_context, &temporary,
                                     &SemIR::Temporary::init_id, &phase)) {
    return SemIR::ConstantId::NotConstant;
  }

  return MakeConstantResult(eval_context.context(), temporary, phase);
}

// Returns whether `const_id` is the same constant facet value as
// `facet_value_inst_id`.
//
// Compares with the canonical facet value of `const_id`, dropping any `as type`
// conversions.
static auto IsSameFacetValue(Context& context, SemIR::ConstantId const_id,
                             SemIR::InstId facet_value_inst_id) -> bool {
  auto canon_const_id = GetCanonicalFacetOrTypeValue(context, const_id);
  return canon_const_id == context.constant_values().Get(facet_value_inst_id);
}

static auto AddRequirementBase(Context& context,
                               SemIR::RequirementBaseFacetType base,
                               SemIR::FacetTypeInfo* info, Phase* phase)
    -> void {
  auto base_type_inst_id =
      context.constant_values().GetConstantTypeInstId(base.base_type_inst_id);
  if (base_type_inst_id == SemIR::ErrorInst::TypeInstId) {
    *phase = Phase::UnknownDueToError;
    return;
  }

  if (auto base_facet_type =
          context.insts().TryGetAs<SemIR::FacetType>(base_type_inst_id)) {
    const auto& base_info =
        context.facet_types().Get(base_facet_type->facet_type_id);
    info->extend_constraints.append(base_info.extend_constraints);
    info->extend_named_constraints.append(base_info.extend_named_constraints);
    info->self_impls_constraints.append(base_info.self_impls_constraints);
    info->self_impls_named_constraints.append(
        base_info.self_impls_named_constraints);
    info->type_impls_interfaces.append(base_info.type_impls_interfaces);
    info->type_impls_named_constraints.append(
        base_info.type_impls_named_constraints);
    info->rewrite_constraints.append(base_info.rewrite_constraints);
    info->other_requirements |= base_info.other_requirements;
  }
}

static auto AddRequirementRewrite(Context& context,
                                  SemIR::RequirementRewrite rewrite,
                                  SemIR::FacetTypeInfo* info, Phase* phase)
    -> void {
  auto lhs_id = context.constant_values().GetConstantInstId(rewrite.lhs_id);
  auto rhs_id = context.constant_values().GetConstantInstId(rewrite.rhs_id);
  if (lhs_id == SemIR::ErrorInst::InstId ||
      rhs_id == SemIR::ErrorInst::InstId) {
    *phase = Phase::UnknownDueToError;
    return;
  }
  if (!rhs_id.has_value()) {
    // The RHS may be an arbitrary expression, which means it could have a
    // runtime value, which we reject since we can't evaluate that.
    DiagnoseNonConstantValue(context, SemIR::LocId(rewrite.rhs_id));
    *phase = Phase::UnknownDueToError;
    return;
  }

  // The FacetTypeInfo must hold canonical IDs for constant comparison, yet here
  // we must insert the non-canonical IDs:
  // * Rewrite constraints are resolved once the FacetTypeInfo is fully
  //   constructed in order to produce the constant value of the facet type.
  //   That resolution step needs the non-canonical insts to do its job
  //   correctly. For instance, the LHS may be a `ImplWitnessAccessSubstituted`
  //   instruction which preserves which element in the witness is being
  //   assigned to but evaluates to the RHS of some other rewrite. So the
  //   constant value would be incorrect to use.
  // * We use the id of the non-canonical RHS instruction as a hint to order
  //   diagnostics in the resolution of rewrites, so that they can usually refer
  //   to the rewrites in the same order as they are written in the code. Using
  //   the constant value of the RHS reorders the diagnostics in a worse way.
  // * The final step of constructing the facet type from the WhereExpr
  //   canonicalizes all the instructions, so we don't need to store canonical
  //   values here. We only need to use canonical values if we need to observe
  //   the constant value, such as to determine in the RHS has a runtime value
  //   above.
  info->rewrite_constraints.push_back(
      {.lhs_id = rewrite.lhs_id, .rhs_id = rewrite.rhs_id});
}

static auto AddRequirementImpls(Context& context, SemIR::RequirementImpls impls,
                                SemIR::InstId period_self_id,
                                SemIR::FacetTypeInfo* info, Phase* phase)
    -> void {
  auto lhs_id = context.constant_values().GetConstantInstId(impls.lhs_id);
  auto rhs_id = context.constant_values().GetConstantInstId(impls.rhs_id);
  if (lhs_id == SemIR::ErrorInst::InstId ||
      rhs_id == SemIR::ErrorInst::InstId) {
    *phase = Phase::UnknownDueToError;
    return;
  }

  if (rhs_id == SemIR::TypeType::TypeInstId) {
    // `<type> impls type` -> nothing to do.
    return;
  }

  auto facet_type = context.insts().GetAs<SemIR::FacetType>(rhs_id);
  const auto& rhs = context.facet_types().Get(facet_type.facet_type_id);

  if (IsSameFacetValue(context, context.constant_values().Get(lhs_id),
                       period_self_id)) {
    // A facet type with `.Self impls <RHS facet type>`. Whatever the RHS facet
    // type constrains for `.Self` gets forwarded to the output facet type to
    // also constrain `.Self`. Nothing on the RHS of `impls` can extend the
    // resulting facet type.
    llvm::append_range(info->self_impls_constraints, rhs.extend_constraints);
    llvm::append_range(info->self_impls_constraints,
                       rhs.self_impls_constraints);
    llvm::append_range(info->self_impls_named_constraints,
                       rhs.extend_named_constraints);
    llvm::append_range(info->self_impls_named_constraints,
                       rhs.self_impls_named_constraints);
    llvm::append_range(info->type_impls_interfaces, rhs.type_impls_interfaces);
    llvm::append_range(info->type_impls_named_constraints,
                       rhs.type_impls_named_constraints);
  } else {
    // Consider `I where C(.Self) impls (J(.Self) where .Self impls K(.Self))`,
    // when we are evaluating the `C(.Self) impls (<facet type>)` requirement.
    // The <facet type> is our `rhs` here, and it will contain:
    // * extend constraint: J(.Self)
    // * self impls constraint: K(.Self)
    //
    // The value of `.Self` changes where we cross a `where` operator. This
    // means extend constraints retain their original `.Self`, but self impls
    // constraints should have their `.Self` replaced by the LHS of the impls
    // requirement.
    //
    // However that is not quite enough. The view of the LHS of the impls
    // requirement should be a facet with a facet type of the RHS extend
    // constraints. In this case the LHS is C(.Self) and the RHS facet type is
    // `J(.Self) where .Self impls K(.Self)`. The RHS facet type has impls
    // constraints (which are on the RHS of a `where` operator), in which their
    // `.Self` should be replaced by `C(.Self)` converted to the RHS facet
    // type's extend constraints (which are on the LHS of a `where` operator),
    // which is `C(.Self) as J(.Self)`. It should be enough to convert the LHS
    // type to the type of the `.Self` that it is replacing, as that contains
    // the extend constraints.
    //
    // So the final RHS facet type to be merged into `info` is:
    //
    // `J(.Self) where (C(.Self) as J(.Self)) impls K(C(.Self) as J(.Self))`.

    auto lhs_facet_or_type = GetCanonicalFacetOrTypeValue(context, lhs_id);

    auto impls_interface = [&](SemIR::SpecificInterface si)
        -> SemIR::FacetTypeInfo::TypeImplsInterface {
      return {lhs_facet_or_type, si};
    };
    auto impls_constraint = [&](SemIR::SpecificNamedConstraint sc)
        -> SemIR::FacetTypeInfo::TypeImplsNamedConstraint {
      return {lhs_facet_or_type, sc};
    };

    // Extend constraints are copied over without replacing anything, but are
    // converted to type impls constraints so they apply to the LHS type.
    llvm::append_range(
        info->type_impls_interfaces,
        llvm::map_range(rhs.extend_constraints, impls_interface));
    llvm::append_range(
        info->type_impls_named_constraints,
        llvm::map_range(rhs.extend_named_constraints, impls_constraint));

    // To replace the `.Self` in `.Self impls X` we convert from a self impls
    // constraint to a type impls constraint where the type is the impls LHS
    // type. We must also replace any `.Self` references in the constraint in
    // the same way. The LHS type needs to be converted to a facet with its type
    // containing the RHS facet type's extend constraints so that the extend
    // constraints can be referenced in impls constraints.
    //
    // TODO: Convert the LHS used in the TypeImplsNamedConstraint to a facet
    // with the RHS extend constraints (interfaces and named constraints).
    //
    // TODO: Replace `.Self` with the LHS type as a facet with the RHS extend
    // constraints.
    llvm::append_range(
        info->type_impls_interfaces,
        llvm::map_range(rhs.self_impls_constraints, impls_interface));
    llvm::append_range(
        info->type_impls_named_constraints,
        llvm::map_range(rhs.self_impls_named_constraints, impls_constraint));

    // Type impls constraints are copied in, but need to have their `.Self`
    // references replaced by the impls LHS type. Like above, the LHS type
    // should be converted to a facet type containing the RHS facet type's
    // extend constraints.
    //
    // TODO: Convert the LHS used in the TypeImplsNamedConstraint to a facet
    // with the RHS extend constraints (interfaces and named constraints).
    //
    // TODO: Replace `.Self` with the LHS type as a facet with the RHS extend
    // constraints.
    llvm::append_range(info->type_impls_interfaces, rhs.type_impls_interfaces);
    llvm::append_range(info->type_impls_named_constraints,
                       rhs.type_impls_named_constraints);
  }

  // Other requirements are copied in.
  llvm::append_range(info->rewrite_constraints, rhs.rewrite_constraints);
  info->other_requirements |= rhs.other_requirements;
}

// Add the constraints from the WhereExpr instruction into a FacetTypeInfo in
// order to construct a FacetType constant value.
//
// TODO: Convert this to an EvalConstantInst function. This will require
// providing a `GetConstantValue` overload for a requirement block.
template <>
auto TryEvalTypedInst<SemIR::WhereExpr>(EvalContext& eval_context,
                                        SemIR::InstId where_inst_id,
                                        SemIR::Inst inst) -> SemIR::ConstantId {
  auto typed_inst = inst.As<SemIR::WhereExpr>();

  Phase phase = Phase::Concrete;
  SemIR::FacetTypeInfo info;

  if (typed_inst.period_self_id == SemIR::ErrorInst::InstId) {
    return SemIR::ErrorInst::ConstantId;
  }

  // Note that these requirement instructions don't have a constant value. That
  // means we have to look for errors inside them, we can't just look to see if
  // their constant value is an error.
  for (auto inst_id :
       eval_context.inst_blocks().GetOrEmpty(typed_inst.requirements_id)) {
    if (phase == Phase::UnknownDueToError) {
      // Abandon ship to save work once we've encountered an error.
      return SemIR::ErrorInst::ConstantId;
    }

    auto inst = eval_context.insts().Get(inst_id);
    CARBON_KIND_SWITCH(inst) {
      case CARBON_KIND(SemIR::RequirementBaseFacetType base): {
        AddRequirementBase(eval_context.context(), base, &info, &phase);
        break;
      }
      case CARBON_KIND(SemIR::RequirementRewrite rewrite): {
        AddRequirementRewrite(eval_context.context(), rewrite, &info, &phase);
        break;
      }
      case CARBON_KIND(SemIR::RequirementImpls impls): {
        AddRequirementImpls(eval_context.context(), impls,
                            typed_inst.period_self_id, &info, &phase);
        break;
      }
      case CARBON_KIND(SemIR::RequirementEquivalent _): {
        // TODO: Handle equality requirements.
        info.other_requirements = true;
        break;
      }
      default:
        CARBON_FATAL("unexpected inst {0} in WhereExpr requirements block",
                     inst);
    }
  }

  auto const_info = GetConstantFacetTypeInfo(
      eval_context, SemIR::LocId(where_inst_id), info, &phase);
  return MakeFacetTypeResult(eval_context.context(), const_info, phase);
}

// Implementation for `TryEvalInst`, wrapping `Context` with `EvalContext`.
static auto TryEvalInstInContext(EvalContext& eval_context,
                                 SemIR::InstId inst_id, SemIR::Inst inst)
    -> SemIR::ConstantId {
  using EvalInstFn =
      auto(EvalContext & eval_context, SemIR::InstId inst_id, SemIR::Inst inst)
          ->SemIR::ConstantId;
  static constexpr EvalInstFn* EvalInstFns[] = {
#define CARBON_SEM_IR_INST_KIND(Kind) &TryEvalTypedInst<SemIR::Kind>,
#include "toolchain/sem_ir/inst_kind.def"
  };
  [[clang::musttail]] return EvalInstFns[inst.kind().AsInt()](eval_context,
                                                              inst_id, inst);
}

auto TryEvalInstUnsafe(Context& context, SemIR::InstId inst_id,
                       SemIR::Inst inst) -> SemIR::ConstantId {
  EvalContext eval_context(&context, SemIR::LocId(inst_id));
  return TryEvalInstInContext(eval_context, inst_id, inst);
}

auto TryEvalBlockForSpecific(Context& context, SemIR::LocId loc_id,
                             SemIR::SpecificId specific_id,
                             SemIR::GenericInstIndex::Region region)
    -> std::pair<SemIR::InstBlockId, bool> {
  auto generic_id = context.specifics().Get(specific_id).generic_id;
  auto eval_block_id = context.generics().Get(generic_id).GetEvalBlock(region);
  auto eval_block = context.inst_blocks().Get(eval_block_id);

  llvm::SmallVector<SemIR::InstId> result;
  result.resize(eval_block.size(), SemIR::InstId::None);

  EvalContext eval_context(&context, loc_id, specific_id,
                           SpecificEvalInfo{
                               .region = region,
                               .values = result,
                           });

  Diagnostics::ContextScope diagnostic_context(
      &context.emitter(), [&](auto& builder) {
        CARBON_DIAGNOSTIC(ResolvingSpecificHere, SoftContext,
                          "unable to monomorphize specific {0}",
                          SemIR::SpecificId);
        builder.Context(loc_id, ResolvingSpecificHere, specific_id);
      });

  bool has_error = false;
  for (auto [i, inst_id] : llvm::enumerate(eval_block)) {
    auto const_id = TryEvalInstInContext(eval_context, inst_id,
                                         context.insts().Get(inst_id));
    if (const_id == SemIR::ErrorInst::ConstantId) {
      has_error = true;
    }
    result[i] = context.constant_values().GetInstId(const_id);
    CARBON_CHECK(result[i].has_value(), "Failed to evaluate {0} in eval block",
                 context.insts().Get(inst_id));
  }

  return {context.inst_blocks().Add(result), has_error};
}

// Information about the function call we are currently executing. Unlike
// evaluation, execution sequentially interprets instructions, and can handle
// control flow and (eventually) side effects and mutable state.
class FunctionExecContext : public EvalContext {
 public:
  // A block argument passed to `BranchWithArg`.
  struct BlockArgValue {
    SemIR::InstBlockId block_id = SemIR::InstBlockId::None;
    SemIR::ConstantId arg_id = SemIR::ConstantId::None;
  };

  FunctionExecContext(Context* context, SemIR::LocId loc_id,
                      SemIR::SpecificId specific_id,
                      Map<SemIR::InstId, SemIR::ConstantId>* locals,
                      SemIR::InstBlockId args_id)
      : EvalContext(context, loc_id, specific_id,
                    LocalEvalInfo{.locals = locals}),
        args_(context->inst_blocks().Get(args_id)) {}

  // Returns the argument values supplied in the call to the function.
  auto args() const -> llvm::ArrayRef<SemIR::InstId> { return args_; }

  using EvalContext::locals;

  // Branch control flow to the given block. This replaces the innermost block
  // in the block stack, but doesn't affect any enclosing blocks.
  auto BranchTo(SemIR::InstBlockId block_id) -> void {
    blocks_.back() = inst_blocks().Get(block_id);
  }

  // Push a new block to be executed immediately. After the block finishes,
  // control will resume after the current instruction.
  auto PushBlock(SemIR::InstBlockId block_id) -> void {
    blocks_.push_back(inst_blocks().Get(block_id));
  }

  // Pops and returns the next instruction to be executed.
  auto PopNextInstId() -> SemIR::InstId {
    while (blocks_.back().empty()) {
      blocks_.pop_back();
      CARBON_CHECK(!blocks_.empty(), "Fell off end of function");
    }
    return blocks_.back().consume_front();
  }

  // Sets the most recent block argument value provided by a `BranchWithArg`.
  // This can later be retrieved by a `BlockArg`.
  auto SetCurrentBlockArgValue(BlockArgValue arg) -> void {
    current_block_arg_value_ = arg;
  }

  // Returns the most recent block argument value provided by a `BranchWithArg`.
  auto current_block_arg_value() const -> BlockArgValue {
    return current_block_arg_value_;
  }

 private:
  // The stack of code blocks that we are currently evaluating. This is kept as
  // a stack so that we can schedule the function body to execute after the decl
  // block and so that we can handle `SpliceBlock`s. When the innermost block is
  // complete, it will be popped and the next outer block will execute.
  llvm::SmallVector<llvm::ArrayRef<SemIR::InstId>, 4> blocks_;

  // The arguments in the function call.
  llvm::ArrayRef<SemIR::InstId> args_;

  // The block argument provided by the most recently executed `BranchWithArg`.
  // We assume that we only need to track one of these, as the branch target
  // will invoke `BlockArg` before the next `BranchWithArg` happens. We will
  // need to track more than one of these if that ever changes.
  BlockArgValue current_block_arg_value_;
};

// Handles the result of executing an instruction in a function. Returns an
// error the result is not a constant, and otherwise updates the locals map to
// track the result as an input to later evaluations in this function and
// returns None.
static auto HandleExecResult(FunctionExecContext& eval_context,
                             SemIR::InstId inst_id, SemIR::ConstantId const_id)
    -> SemIR::ConstantId {
  if (const_id == SemIR::ErrorInst::ConstantId) {
    return const_id;
  }
  if (!const_id.has_value() || !const_id.is_constant()) {
    DiagnoseNonConstantValue(eval_context.context(),
                             eval_context.GetDiagnosticLoc(inst_id));
    return SemIR::ErrorInst::ConstantId;
  }
  eval_context.locals().Update(inst_id, const_id);
  return SemIR::ConstantId::None;
}

// Executes an instruction for TryEvalCall. By default, performs normal
// evaluation of the instruction within a context that supplies the values
// produced by executing prior instructions in this function execution. This is
// specialized for instructions that have special handling in function
// execution, such as those that access parameters or perform flow control. If
// execution should continue, returns `SemIR::ConstantId::None`, otherwise
// returns the result to produce for the enclosing function call, which should
// be either the returned value or an error.
template <typename InstT>
static auto TryExecTypedInst(FunctionExecContext& eval_context,
                             SemIR::InstId inst_id, SemIR::Inst inst)
    -> SemIR::ConstantId {
  if constexpr (InstT::Kind.expr_category().TryAsFixedCategory() ==
                SemIR::ExprCategory::NotExpr) {
    // Instructions in this category are assumed to not have a runtime effect.
    // This includes some kinds of declaration.
    return SemIR::ConstantId::None;
  }

  if constexpr (InstT::Kind.constant_kind() != SemIR::InstConstantKind::Never) {
    if (eval_context.constant_values().Get(inst_id).is_concrete()) {
      // Instruction has a concrete constant value that doesn't depend on the
      // context. We don't need to evaluate it again.
      return SemIR::ConstantId::None;
    }
  }

  // Evaluate the instruction in the current context.
  auto const_id = TryEvalTypedInst<InstT>(eval_context, inst_id, inst);
  return HandleExecResult(eval_context, inst_id, const_id);
}

template <>
auto TryExecTypedInst<SemIR::BlockArg>(FunctionExecContext& eval_context,
                                       SemIR::InstId inst_id, SemIR::Inst inst)
    -> SemIR::ConstantId {
  auto block_arg = inst.As<SemIR::BlockArg>();
  CARBON_CHECK(
      block_arg.block_id == eval_context.current_block_arg_value().block_id,
      "BlockArg does not refer to most recent BranchWithArg");
  eval_context.locals().Update(inst_id,
                               eval_context.current_block_arg_value().arg_id);
  return SemIR::ConstantId::None;
}

template <>
auto TryExecTypedInst<SemIR::Branch>(FunctionExecContext& eval_context,
                                     SemIR::InstId /*inst_id*/,
                                     SemIR::Inst inst) -> SemIR::ConstantId {
  auto branch = inst.As<SemIR::Branch>();
  eval_context.BranchTo(branch.target_id);
  return SemIR::ConstantId::None;
}

template <>
auto TryExecTypedInst<SemIR::BranchIf>(FunctionExecContext& eval_context,
                                       SemIR::InstId /*inst_id*/,
                                       SemIR::Inst inst) -> SemIR::ConstantId {
  auto branch_if = inst.As<SemIR::BranchIf>();
  auto cond_id = CheckConcreteValue(eval_context, branch_if.cond_id);
  if (cond_id == SemIR::ErrorInst::InstId) {
    return SemIR::ErrorInst::ConstantId;
  }
  auto cond = eval_context.insts().GetAs<SemIR::BoolLiteral>(cond_id);
  if (cond.value == SemIR::BoolValue::True) {
    eval_context.BranchTo(branch_if.target_id);
  }
  return SemIR::ConstantId::None;
}

template <>
auto TryExecTypedInst<SemIR::BranchWithArg>(FunctionExecContext& eval_context,
                                            SemIR::InstId /*inst_id*/,
                                            SemIR::Inst inst)
    -> SemIR::ConstantId {
  auto branch = inst.As<SemIR::BranchWithArg>();
  eval_context.SetCurrentBlockArgValue(
      {.block_id = branch.target_id,
       .arg_id = eval_context.GetConstantValue(branch.arg_id)});
  eval_context.BranchTo(branch.target_id);
  return SemIR::ConstantId::None;
}

template <>
auto TryExecTypedInst<SemIR::Return>(FunctionExecContext& eval_context,
                                     SemIR::InstId /*inst_id*/,
                                     SemIR::Inst /*inst*/)
    -> SemIR::ConstantId {
  return MakeEmptyTupleResult(eval_context);
}

template <>
auto TryExecTypedInst<SemIR::ReturnExpr>(FunctionExecContext& eval_context,
                                         SemIR::InstId /*inst_id*/,
                                         SemIR::Inst inst)
    -> SemIR::ConstantId {
  auto return_expr = inst.As<SemIR::ReturnExpr>();
  return eval_context.GetConstantValue(return_expr.expr_id);
}

template <>
auto TryExecTypedInst<SemIR::ReturnSlot>(FunctionExecContext& eval_context,
                                         SemIR::InstId inst_id,
                                         SemIR::Inst inst)
    -> SemIR::ConstantId {
  auto return_slot = inst.As<SemIR::ReturnSlot>();
  // In the case where the function's return type is not in-place, the return
  // slot will refer to an out parameter that doesn't have an argument. In that
  // case, we don't have a constant value for storage_id. To handle this, copy
  // the value directly from the locals map rather than using GetConstantValue.
  //
  // TODO: Remove this and use a normal call to `GetConstantValue` if we stop
  // adding out parameters with no corresponding argument.
  eval_context.locals().Insert(
      inst_id, eval_context.locals().Lookup(return_slot.storage_id).value());
  return SemIR::ConstantId::None;
}

template <>
auto TryExecTypedInst<SemIR::SpliceBlock>(FunctionExecContext& eval_context,
                                          SemIR::InstId /*inst_id*/,
                                          SemIR::Inst inst)
    -> SemIR::ConstantId {
  auto splice_block = inst.As<SemIR::SpliceBlock>();
  eval_context.PushBlock(splice_block.block_id);
  // TODO: Copy the values from the result_id instruction to the result of
  // the splice_block instruction once the spliced block finishes.
  return SemIR::ConstantId::None;
}

// Executes the introduction of a parameter into the local scope. Copies the
// argument supplied by the caller for the parameter into the locals map.
static auto TryExecTypedParam(FunctionExecContext& eval_context,
                              SemIR::InstId inst_id, SemIR::Inst inst)
    -> SemIR::ConstantId {
  auto param = inst.As<SemIR::AnyParam>();
  CARBON_CHECK(static_cast<size_t>(param.index.index) <
               eval_context.args().size());
  eval_context.locals().Insert(inst_id,
                               eval_context.constant_values().Get(
                                   eval_context.args()[param.index.index]));
  return SemIR::ConstantId::None;
}

template <>
auto TryExecTypedInst<SemIR::OutParam>(FunctionExecContext& eval_context,
                                       SemIR::InstId inst_id, SemIR::Inst inst)
    -> SemIR::ConstantId {
  auto param = inst.As<SemIR::OutParam>();
  if (static_cast<size_t>(param.index.index) >= eval_context.args().size()) {
    // For return values that have a copy initializing representation, the SemIR
    // has an OutParam with an index that has no corresponding argument. In that
    // case, we do not have a constant value for the parameter, but this doesn't
    // prevent the call from being constant.
    //
    // TODO: Remove this once we stop adding out parameters with no
    // corresponding argument.
    eval_context.locals().Insert(inst_id, SemIR::ConstantId::None);
    return SemIR::ConstantId::None;
  }

  if (!eval_context.args()[param.index.index].has_value()) {
    // The argument will be `None` for an index corresponding to a return
    // storage argument for return values that have an in-place initializing
    // representation. Produce an opaque "out parameter" variable for now, so
    // that references to it can still successfully evaluate.
    //
    // TODO: Create and track mutable storage for the return value here. This is
    // necessary to support things like `returned var`.
    eval_context.locals().Insert(
        inst_id,
        MakeConstantResult(
            eval_context.context(),
            SemIR::VarStorage{.type_id = inst.type_id(),
                              .pattern_id = SemIR::AbsoluteInstId::None},
            Phase::Concrete));
    return SemIR::ConstantId::None;
  }

  return TryExecTypedParam(eval_context, inst_id, inst);
}

template <>
auto TryExecTypedInst<SemIR::RefParam>(FunctionExecContext& eval_context,
                                       SemIR::InstId inst_id, SemIR::Inst inst)
    -> SemIR::ConstantId {
  return TryExecTypedParam(eval_context, inst_id, inst);
}

template <>
auto TryExecTypedInst<SemIR::ValueParam>(FunctionExecContext& eval_context,
                                         SemIR::InstId inst_id,
                                         SemIR::Inst inst)
    -> SemIR::ConstantId {
  return TryExecTypedParam(eval_context, inst_id, inst);
}

template <>
auto TryExecTypedInst<SemIR::ValueBinding>(FunctionExecContext& eval_context,
                                           SemIR::InstId inst_id,
                                           SemIR::Inst inst)
    -> SemIR::ConstantId {
  auto value_binding = inst.As<SemIR::ValueBinding>();
  auto local_value_id = eval_context.GetConstantValue(value_binding.value_id);
  eval_context.locals().Insert(inst_id, local_value_id);
  return SemIR::ConstantId::None;
}

static auto TryExecInst(FunctionExecContext& eval_context,
                        SemIR::InstId inst_id, SemIR::Inst inst)
    -> SemIR::ConstantId {
  using ExecInstFn = auto(FunctionExecContext & eval_context,
                          SemIR::InstId inst_id, SemIR::Inst inst)
                         ->SemIR::ConstantId;
  static constexpr ExecInstFn* ExecInstFns[] = {
#define CARBON_SEM_IR_INST_KIND(Kind) &TryExecTypedInst<SemIR::Kind>,
#include "toolchain/sem_ir/inst_kind.def"
  };
  [[clang::musttail]] return ExecInstFns[inst.kind().AsInt()](eval_context,
                                                              inst_id, inst);
}

// Evaluates a call to an `eval` or `musteval` function by executing the
// function body.
static auto TryEvalCall(EvalContext& outer_eval_context, SemIR::LocId loc_id,
                        const SemIR::Function& function,
                        SemIR::SpecificId specific_id,
                        SemIR::InstBlockId args_id) -> SemIR::ConstantId {
  if (function.clang_decl_id != SemIR::ClangDeclId::None) {
    return EvalCppCall(outer_eval_context.context(), loc_id,
                       function.clang_decl_id, args_id);
  } else if (function.body_block_ids.empty()) {
    // TODO: Diagnose this.
    return SemIR::ConstantId::NotConstant;
  }

  if (specific_id.has_value()) {
    ResolveSpecificDefinition(outer_eval_context.context(), loc_id,
                              specific_id);
  }

  // TODO: Consider tracking the lowest and highest inst_id in the function and
  // using an array instead of a map. We would still need a map for instantiated
  // portions of a function template.
  Map<SemIR::InstId, SemIR::ConstantId> locals;

  FunctionExecContext eval_context(&outer_eval_context.context(), loc_id,
                                   specific_id, &locals, args_id);

  Diagnostics::AnnotationScope annotate_diagnostics(
      &eval_context.emitter(), [&](auto& builder) {
        CARBON_DIAGNOSTIC(InCallToEvalFn, Note, "in call to {0} here",
                          SemIR::NameId);
        builder.Note(loc_id, InCallToEvalFn, function.name_id);
      });

  // Execute the function decl block followed by the body.
  eval_context.PushBlock(function.body_block_ids.front());
  eval_context.PushBlock(eval_context.insts()
                             .GetAs<SemIR::FunctionDecl>(function.definition_id)
                             .decl_block_id);

  // Execute the blocks. This is mostly expression evaluation, with special
  // handling for control flow and parameters.
  while (true) {
    auto inst_id = eval_context.PopNextInstId();
    auto inst = eval_context.context().insts().Get(inst_id);
    if (auto result = TryExecInst(eval_context, inst_id, inst);
        result.has_value()) {
      return result;
    }
  }
}

}  // namespace Carbon::Check
