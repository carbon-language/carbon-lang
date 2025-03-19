// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/eval.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/action.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/check/eval_inst.h"
#include "toolchain/check/facet_type.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/sem_ir/builtin_function_kind.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/generic.h"
#include "toolchain/sem_ir/id_kind.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst_kind.h"
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

// Information about the context within which we are performing evaluation.
class EvalContext {
 public:
  explicit EvalContext(
      Context& context, SemIRLoc fallback_loc,
      SemIR::SpecificId specific_id = SemIR::SpecificId::None,
      std::optional<SpecificEvalInfo> specific_eval_info = std::nullopt)
      : context_(context),
        fallback_loc_(fallback_loc),
        specific_id_(specific_id),
        specific_eval_info_(specific_eval_info) {}

  // Gets the location to use for diagnostics if a better location is
  // unavailable.
  // TODO: This is also sometimes unavailable.
  auto fallback_loc() const -> SemIRLoc { return fallback_loc_; }

  // Returns a location to use to point at an instruction in a diagnostic, given
  // a list of instructions that might have an attached location. This is the
  // location of the first instruction in the list that has a location if there
  // is one, and otherwise the fallback location.
  auto GetDiagnosticLoc(llvm::ArrayRef<SemIR::InstId> inst_ids) -> SemIRLoc {
    for (auto inst_id : inst_ids) {
      if (inst_id.has_value() &&
          context_.insts().GetLocId(inst_id).has_value()) {
        return inst_id;
      }
    }
    return fallback_loc_;
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

  // Given a constant value from the SemIR we're evaluating, finds the
  // corresponding constant value to use in the context of this evaluation.
  // This can be different if the original SemIR is for a generic and we are
  // evaluating with specific arguments for the generic parameters.
  auto GetInContext(SemIR::ConstantId const_id) -> SemIR::ConstantId {
    if (!const_id.is_symbolic()) {
      return const_id;
    }

    // While resolving a specific, map from previous instructions in the eval
    // block into their evaluated values. These values won't be present on the
    // specific itself yet, so `GetConstantInSpecific` won't be able to find
    // them.
    if (specific_eval_info_) {
      const auto& symbolic_info =
          constant_values().GetSymbolicConstant(const_id);
      if (symbolic_info.index.has_value() &&
          symbolic_info.generic_id ==
              specifics().Get(specific_id_).generic_id &&
          symbolic_info.index.region() == specific_eval_info_->region) {
        auto inst_id = specific_eval_info_->values[symbolic_info.index.index()];
        CARBON_CHECK(inst_id.has_value(),
                     "Forward reference in eval block: index {0} referenced "
                     "before evaluation",
                     symbolic_info.index.index());
        return constant_values().Get(inst_id);
      }
    }

    // Map from a specific constant value to the canonical value.
    return GetConstantInSpecific(sem_ir(), specific_id_, const_id);
  }

  // Gets the constant value of the specified instruction in this context.
  auto GetConstantValue(SemIR::InstId inst_id) -> SemIR::ConstantId {
    return GetInContext(constant_values().Get(inst_id));
  }

  // Gets the constant value of the specified type in this context.
  auto GetConstantValue(SemIR::TypeId type_id) -> SemIR::ConstantId {
    return GetInContext(types().GetConstantId(type_id));
  }

  // Gets the constant value of the specified type in this context.
  auto GetConstantValueAsType(SemIR::TypeId id) -> SemIR::TypeId {
    return context().types().GetTypeIdForTypeConstantId(GetConstantValue(id));
  }

  // Gets the instruction describing the constant value of the specified type in
  // this context.
  auto GetConstantValueAsInst(SemIR::TypeId id) -> SemIR::Inst {
    return insts().Get(
        context().constant_values().GetInstId(GetConstantValue(id)));
  }

  auto ints() -> SharedValueStores::IntStore& { return sem_ir().ints(); }
  auto floats() -> SharedValueStores::FloatStore& { return sem_ir().floats(); }
  auto entity_names() -> SemIR::EntityNameStore& {
    return sem_ir().entity_names();
  }
  auto functions() -> const ValueStore<SemIR::FunctionId>& {
    return sem_ir().functions();
  }
  auto classes() -> const ValueStore<SemIR::ClassId>& {
    return sem_ir().classes();
  }
  auto interfaces() -> const ValueStore<SemIR::InterfaceId>& {
    return sem_ir().interfaces();
  }
  auto facet_types() -> CanonicalValueStore<SemIR::FacetTypeId>& {
    return sem_ir().facet_types();
  }
  auto specifics() -> const SemIR::SpecificStore& {
    return sem_ir().specifics();
  }
  auto type_blocks() -> SemIR::BlockValueStore<SemIR::TypeBlockId>& {
    return sem_ir().type_blocks();
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

  auto context() -> Context& { return context_; }

  auto sem_ir() -> SemIR::File& { return context().sem_ir(); }

  auto emitter() -> DiagnosticEmitter<SemIRLoc>& { return context().emitter(); }

 private:
  // The type-checking context in which we're performing evaluation.
  Context& context_;
  // The location to use for diagnostics when a better location isn't available.
  SemIRLoc fallback_loc_;
  // The specific that we are evaluating within.
  SemIR::SpecificId specific_id_;
  // If we are currently evaluating an eval block for `specific_id_`,
  // information about that evaluation.
  std::optional<SpecificEvalInfo> specific_eval_info_;
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

// Returns whether the specified phase is a constant phase.
static auto IsConstant(Phase phase) -> bool {
  return phase < Phase::UnknownDueToError;
}

// Gets the phase in which the value of a constant will become available.
static auto GetPhase(const SemIR::ConstantValueStore& constant_values,
                     SemIR::ConstantId constant_id) -> Phase {
  if (!constant_id.is_constant()) {
    return Phase::Runtime;
  } else if (constant_id == SemIR::ErrorInst::SingletonConstantId) {
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

// `where` expressions using `.Self` should not be considered symbolic
// - `Interface where .Self impls I and .A = bool` -> concrete
// - `T:! type` ... `Interface where .A = T` -> symbolic, since uses `T` which
//   is symbolic and not due to `.Self`.
static auto UpdatePhaseIgnorePeriodSelf(EvalContext& eval_context,
                                        SemIR::ConstantId constant_id,
                                        Phase* phase) {
  Phase constant_phase = GetPhase(eval_context.constant_values(), constant_id);
  // Since LatestPhase(x, Phase::Concrete) == x, this is equivalent to replacing
  // Phase::PeriodSelfSymbolic with Phase::Concrete.
  if (constant_phase != Phase::PeriodSelfSymbolic) {
    *phase = LatestPhase(*phase, constant_phase);
  }
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
      return SemIR::ErrorInst::SingletonConstantId;
    case Phase::Runtime:
      return SemIR::ConstantId::NotConstant;
  }
}

// Forms a `constant_id` describing why an evaluation was not constant.
static auto MakeNonConstantResult(Phase phase) -> SemIR::ConstantId {
  return phase == Phase::UnknownDueToError
             ? SemIR::ErrorInst::SingletonConstantId
             : SemIR::ConstantId::NotConstant;
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
      context, SemIR::FloatLiteral{.type_id = type_id, .float_id = result},
      Phase::Concrete);
}

// Creates a FacetType constant.
static auto MakeFacetTypeResult(Context& context,
                                const SemIR::FacetTypeInfo& info, Phase phase)
    -> SemIR::ConstantId {
  SemIR::FacetTypeId facet_type_id = context.facet_types().Add(info);
  return MakeConstantResult(
      context,
      SemIR::FacetType{.type_id = SemIR::TypeType::SingletonTypeId,
                       .facet_type_id = facet_type_id},
      phase);
}

// `GetConstantValue` checks to see whether the provided ID describes a value
// with constant phase, and if so, returns the corresponding constant value.
// Overloads are provided for different kinds of ID.

// AbsoluteInstId can not have its values substituted, so this overload is
// deleted. This prevents conversion to InstId.
static auto GetConstantValue(EvalContext& eval_context,
                             SemIR::AbsoluteInstId inst_id, Phase* phase)
    -> SemIR::InstId = delete;

// If the given instruction is constant, returns its constant value.
static auto GetConstantValue(EvalContext& eval_context, SemIR::InstId inst_id,
                             Phase* phase) -> SemIR::InstId {
  auto const_id = eval_context.GetConstantValue(inst_id);
  *phase =
      LatestPhase(*phase, GetPhase(eval_context.constant_values(), const_id));
  return eval_context.constant_values().GetInstId(const_id);
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
  if (OperandIsDependent(eval_context.context(), inst_id)) {
    *phase = LatestPhase(*phase, Phase::TemplateSymbolic);
  }
  return inst_id;
}

// Explicitly discard a `DestInstId`, because we should not be using the
// destination as part of evaluation.
static auto GetConstantValue(EvalContext& /*eval_context*/,
                             SemIR::DestInstId /*inst_id*/, Phase* /*phase*/)
    -> SemIR::DestInstId {
  return SemIR::InstId::None;
}

// Given a type which may refer to a generic parameter, returns the
// corresponding type in the evaluation context.
static auto GetConstantValue(EvalContext& eval_context, SemIR::TypeId type_id,
                             Phase* phase) -> SemIR::TypeId {
  auto const_id = eval_context.GetConstantValue(type_id);
  *phase =
      LatestPhase(*phase, GetPhase(eval_context.constant_values(), const_id));
  return eval_context.context().types().GetTypeIdForTypeConstantId(const_id);
}

// AbsoluteInstBlockId can not have its values substituted, so this overload is
// deleted. This prevents conversion to InstBlockId.
static auto GetConstantValue(EvalContext& eval_context,
                             SemIR::AbsoluteInstBlockId inst_block_id,
                             Phase* phase) -> SemIR::InstBlockId = delete;

// If the given instruction block contains only constants, returns a
// corresponding block of those values.
static auto GetConstantValue(EvalContext& eval_context,
                             SemIR::InstBlockId inst_block_id, Phase* phase)
    -> SemIR::InstBlockId {
  if (!inst_block_id.has_value()) {
    return SemIR::InstBlockId::None;
  }
  auto insts = eval_context.inst_blocks().Get(inst_block_id);
  llvm::SmallVector<SemIR::InstId> const_insts;
  for (auto inst_id : insts) {
    auto const_inst_id = GetConstantValue(eval_context, inst_id, phase);
    if (!const_inst_id.has_value()) {
      return SemIR::InstBlockId::None;
    }

    // Once we leave the small buffer, we know the first few elements are all
    // constant, so it's likely that the entire block is constant. Resize to the
    // target size given that we're going to allocate memory now anyway.
    if (const_insts.size() == const_insts.capacity()) {
      const_insts.reserve(insts.size());
    }

    const_insts.push_back(const_inst_id);
  }
  // TODO: If the new block is identical to the original block, and we know the
  // old ID was canonical, return the original ID.
  return eval_context.inst_blocks().AddCanonical(const_insts);
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
    auto new_type_id = GetConstantValue(eval_context, field.type_id, phase);
    if (!new_type_id.has_value()) {
      return SemIR::StructTypeFieldsId::None;
    }

    // Once we leave the small buffer, we know the first few elements are all
    // constant, so it's likely that the entire block is constant. Resize to the
    // target size given that we're going to allocate memory now anyway.
    if (new_fields.size() == new_fields.capacity()) {
      new_fields.reserve(fields.size());
    }

    new_fields.push_back({.name_id = field.name_id, .type_id = new_type_id});
  }
  // TODO: If the new block is identical to the original block, and we know the
  // old ID was canonical, return the original ID.
  return eval_context.context().struct_type_fields().AddCanonical(new_fields);
}

// Compute the constant value of a type block. This may be different from the
// input type block if we have known generic arguments.
static auto GetConstantValue(EvalContext& eval_context,
                             SemIR::TypeBlockId type_block_id, Phase* phase)
    -> SemIR::TypeBlockId {
  if (!type_block_id.has_value()) {
    return SemIR::TypeBlockId::None;
  }
  auto types = eval_context.type_blocks().Get(type_block_id);
  llvm::SmallVector<SemIR::TypeId> new_types;
  for (auto type_id : types) {
    auto new_type_id = GetConstantValue(eval_context, type_id, phase);
    if (!new_type_id.has_value()) {
      return SemIR::TypeBlockId::None;
    }

    // Once we leave the small buffer, we know the first few elements are all
    // constant, so it's likely that the entire block is constant. Resize to the
    // target size given that we're going to allocate memory now anyway.
    if (new_types.size() == new_types.capacity()) {
      new_types.reserve(types.size());
    }

    new_types.push_back(new_type_id);
  }
  // TODO: If the new block is identical to the original block, and we know the
  // old ID was canonical, return the original ID.
  return eval_context.type_blocks().AddCanonical(new_types);
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

  if (args_id == specific.args_id) {
    const auto& specific = eval_context.specifics().Get(specific_id);
    // A constant specific_id should always have a resolved declaration. The
    // specific_id from the instruction may coincidentally be canonical, and so
    // constant evaluation gives the same value. In that case, we still need to
    // ensure its declaration is resolved.
    //
    // However, don't resolve the declaration if the generic's eval block hasn't
    // been set yet. This happens when building the eval block during import.
    //
    // TODO: Change importing of generic eval blocks to be less fragile and
    // remove this `if` so we unconditionally call `ResolveSpecificDeclaration`.
    if (!specific.decl_block_id.has_value() && eval_context.context()
                                                   .generics()
                                                   .Get(specific.generic_id)
                                                   .decl_block_id.has_value()) {
      ResolveSpecificDeclaration(eval_context.context(),
                                 eval_context.fallback_loc(), specific_id);
    }
    return specific_id;
  }
  return MakeSpecific(eval_context.context(), eval_context.fallback_loc(),
                      specific.generic_id, args_id);
}

// Like `GetConstantValue` but does a `FacetTypeId` -> `FacetTypeInfo`
// conversion. Does not perform canonicalization.
static auto GetConstantFacetTypeInfo(EvalContext& eval_context,
                                     SemIR::FacetTypeId facet_type_id,
                                     Phase* phase) -> SemIR::FacetTypeInfo {
  const auto& orig = eval_context.facet_types().Get(facet_type_id);
  SemIR::FacetTypeInfo info;
  info.impls_constraints.reserve(orig.impls_constraints.size());
  for (const auto& interface : orig.impls_constraints) {
    info.impls_constraints.push_back(
        {.interface_id = interface.interface_id,
         .specific_id =
             GetConstantValue(eval_context, interface.specific_id, phase)});
  }
  info.rewrite_constraints.reserve(orig.rewrite_constraints.size());
  for (const auto& rewrite : orig.rewrite_constraints) {
    auto lhs_const_id = eval_context.GetInContext(rewrite.lhs_const_id);
    auto rhs_const_id = eval_context.GetInContext(rewrite.rhs_const_id);
    // `where` requirements using `.Self` should not be considered symbolic
    UpdatePhaseIgnorePeriodSelf(eval_context, lhs_const_id, phase);
    UpdatePhaseIgnorePeriodSelf(eval_context, rhs_const_id, phase);
    info.rewrite_constraints.push_back(
        {.lhs_const_id = lhs_const_id, .rhs_const_id = rhs_const_id});
  }
  // TODO: Process other requirements.
  info.other_requirements = orig.other_requirements;
  return info;
}

static auto GetConstantValue(EvalContext& eval_context,
                             SemIR::FacetTypeId facet_type_id, Phase* phase)
    -> SemIR::FacetTypeId {
  SemIR::FacetTypeInfo info =
      GetConstantFacetTypeInfo(eval_context, facet_type_id, phase);
  info.Canonicalize();
  // TODO: Return `facet_type_id` if we can detect nothing has changed.
  return eval_context.facet_types().Add(info);
}

// Replaces the specified field of the given typed instruction with its constant
// value, if it has constant phase. Returns true on success, false if the value
// has runtime phase.
template <typename InstT, typename FieldIdT>
static auto ReplaceFieldWithConstantValue(EvalContext& eval_context,
                                          InstT* inst, FieldIdT InstT::*field,
                                          Phase* phase) -> bool {
  auto unwrapped = GetConstantValue(eval_context, inst->*field, phase);
  if (!unwrapped.has_value() && (inst->*field).has_value()) {
    return false;
  }
  inst->*field = unwrapped;
  return true;
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

// Given the stored value `arg` of an instruction field and its corresponding
// kind `kind`, returns the constant value to use for that field, if it has a
// constant phase. `*phase` is updated to include the new constant value. If
// the resulting phase is not constant, the returned value is not useful and
// will typically be `NoneIndex`.
template <typename... Type>
static auto GetConstantValueForArg(EvalContext& eval_context,
                                   SemIR::TypeEnum<Type...> kind, int32_t arg,
                                   Phase* phase) -> int32_t {
  using Handler = auto(EvalContext&, int32_t arg, Phase * phase)->int32_t;
  static constexpr Handler* Handlers[] = {
      [](EvalContext& eval_context, int32_t arg, Phase* phase) -> int32_t {
        auto id = SemIR::Inst::FromRaw<Type>(arg);
        if constexpr (HasGetConstantValueOverload<Type>) {
          // If we have a custom `GetConstantValue` overload, call it.
          return SemIR::Inst::ToRaw(GetConstantValue(eval_context, id, phase));
        } else {
          // Otherwise, we assume the value is already constant.
          return arg;
        }
      }...,
      [](EvalContext&, int32_t, Phase*) -> int32_t {
        // Handler for IdKind::Invalid is next.
        CARBON_FATAL("Instruction has argument with invalid IdKind");
      },
      [](EvalContext&, int32_t arg, Phase*) -> int32_t {
        // Handler for IdKind::None is last.
        return arg;
      }};
  return Handlers[kind.ToIndex()](eval_context, arg, phase);
}

// Given an instruction, replaces its type and operands with their constant
// values from the specified evaluation context. `*phase` is updated to describe
// the constant phase of the result. Returns whether `*phase` is a constant
// phase; if not, `inst` may not be fully updated and should not be used.
static auto ReplaceAllFieldsWithConstantValues(EvalContext& eval_context,
                                               SemIR::Inst* inst, Phase* phase)
    -> bool {
  auto type_id = SemIR::TypeId(
      GetConstantValueForArg(eval_context, SemIR::IdKind::For<SemIR::TypeId>,
                             inst->type_id().index, phase));
  inst->SetType(type_id);
  if (!IsConstant(*phase)) {
    return false;
  }

  auto kinds = inst->ArgKinds();
  auto arg0 =
      GetConstantValueForArg(eval_context, kinds.first, inst->arg0(), phase);
  if (!IsConstant(*phase)) {
    return false;
  }

  auto arg1 =
      GetConstantValueForArg(eval_context, kinds.second, inst->arg1(), phase);
  if (!IsConstant(*phase)) {
    return false;
  }
  inst->SetArgs(arg0, arg1);
  return true;
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
  auto aggregate_type_id = eval_context.GetConstantValueAsType(
      eval_context.insts().Get(inst.array_id).type_id());
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
        return SemIR::ErrorInst::SingletonConstantId;
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
    CARBON_CHECK(phase != Phase::Concrete,
                 "Unexpected representation for template constant aggregate");
    return MakeNonConstantResult(phase);
  }

  auto elements = eval_context.inst_blocks().Get(aggregate->elements_id);
  return eval_context.GetConstantValue(elements[index_val.getZExtValue()]);
}

// Forms a constant int type as an evaluation result. Requires that width_id is
// constant.
static auto MakeIntTypeResult(Context& context, SemIRLoc loc,
                              SemIR::IntKind int_kind, SemIR::InstId width_id,
                              Phase phase) -> SemIR::ConstantId {
  auto result = SemIR::IntType{
      .type_id = GetSingletonType(context, SemIR::TypeType::SingletonInstId),
      .int_kind = int_kind,
      .bit_width_id = width_id};
  if (!ValidateIntType(context, loc, result)) {
    return SemIR::ErrorInst::SingletonConstantId;
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
static auto PerformCheckedIntConvert(Context& context, SemIRLoc loc,
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
    context.emitter().Emit(loc, NegativeIntInUnsignedType,
                           {.type = arg.type_id, .value = arg_val},
                           dest_type_id);
  }

  unsigned arg_non_sign_bits = arg_val.getSignificantBits() - 1;
  if (arg_non_sign_bits + is_signed > width) {
    CARBON_DIAGNOSTIC(IntTooLargeForType, Error,
                      "integer value {0} too large for type {1}", TypedInt,
                      SemIR::TypeId);
    context.emitter().Emit(loc, IntTooLargeForType,
                           {.type = arg.type_id, .value = arg_val},
                           dest_type_id);
  }

  return MakeConstantResult(
      context, SemIR::IntValue{.type_id = dest_type_id, .int_id = arg.int_id},
      Phase::Concrete);
}

// Issues a diagnostic for a compile-time division by zero.
static auto DiagnoseDivisionByZero(Context& context, SemIRLoc loc) -> void {
  CARBON_DIAGNOSTIC(CompileTimeDivisionByZero, Error, "division by zero");
  context.emitter().Emit(loc, CompileTimeDivisionByZero);
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
static auto PerformBuiltinUnaryIntOp(Context& context, SemIRLoc loc,
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
          context.emitter().Emit(loc, CompileTimeIntegerNegateOverflow,
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
static auto PerformBuiltinIntShiftOp(Context& context, SemIRLoc loc,
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
        TypedInt, BoolAsSelect, TypedInt);
    context.emitter().Emit(
        loc, CompileTimeShiftOutOfRange, lhs_val.getBitWidth(),
        {.type = lhs.type_id, .value = lhs_val},
        builtin_kind == SemIR::BuiltinFunctionKind::IntLeftShift,
        {.type = rhs.type_id, .value = rhs_orig_val});
    // TODO: Is it useful to recover by returning 0 or -1?
    return SemIR::ErrorInst::SingletonConstantId;
  }

  if (rhs_orig_val.isNegative() &&
      context.sem_ir().types().IsSignedInt(rhs.type_id)) {
    CARBON_DIAGNOSTIC(CompileTimeShiftNegative, Error,
                      "shift distance negative in `{0} {1:<<|>>} {2}`",
                      TypedInt, BoolAsSelect, TypedInt);
    context.emitter().Emit(
        loc, CompileTimeShiftNegative, {.type = lhs.type_id, .value = lhs_val},
        builtin_kind == SemIR::BuiltinFunctionKind::IntLeftShift,
        {.type = rhs.type_id, .value = rhs_orig_val});
    // TODO: Is it useful to recover by returning 0 or -1?
    return SemIR::ErrorInst::SingletonConstantId;
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
        context.emitter().Emit(loc, CompileTimeUnsizedShiftOutOfRange,
                               {.type = rhs.type_id, .value = rhs_orig_val},
                               IntStore::MaxIntWidth);
        return SemIR::ErrorInst::SingletonConstantId;
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
static auto PerformBuiltinBinaryIntOp(Context& context, SemIRLoc loc,
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
        DiagnoseDivisionByZero(context, loc);
        return SemIR::ErrorInst::SingletonConstantId;
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
    context.emitter().Emit(loc, CompileTimeIntegerOverflow,
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
  auto op = context.insts().GetAs<SemIR::FloatLiteral>(arg_id);
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
  auto lhs = context.insts().GetAs<SemIR::FloatLiteral>(lhs_id);
  auto rhs = context.insts().GetAs<SemIR::FloatLiteral>(rhs_id);
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
  auto lhs = context.insts().GetAs<SemIR::FloatLiteral>(lhs_id);
  auto rhs = context.insts().GetAs<SemIR::FloatLiteral>(rhs_id);
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

// Returns a constant for a call to a builtin function.
static auto MakeConstantForBuiltinCall(EvalContext& eval_context, SemIRLoc loc,
                                       SemIR::Call call,
                                       SemIR::BuiltinFunctionKind builtin_kind,
                                       llvm::ArrayRef<SemIR::InstId> arg_ids,
                                       Phase phase) -> SemIR::ConstantId {
  auto& context = eval_context.context();
  switch (builtin_kind) {
    case SemIR::BuiltinFunctionKind::None:
      CARBON_FATAL("Not a builtin function.");

    case SemIR::BuiltinFunctionKind::PrintChar:
    case SemIR::BuiltinFunctionKind::PrintInt:
    case SemIR::BuiltinFunctionKind::ReadChar: {
      // These are runtime-only builtins.
      // TODO: Consider tracking this on the `BuiltinFunctionKind`.
      return SemIR::ConstantId::NotConstant;
    }

    case SemIR::BuiltinFunctionKind::TypeAnd: {
      CARBON_CHECK(arg_ids.size() == 2);
      auto lhs_facet_type_id = SemIR::FacetTypeId::None;
      auto rhs_facet_type_id = SemIR::FacetTypeId::None;
      for (auto [facet_type_id, arg_id] :
           llvm::zip(std::to_array({&lhs_facet_type_id, &rhs_facet_type_id}),
                     arg_ids)) {
        if (auto facet_type =
                context.insts().TryGetAs<SemIR::FacetType>(arg_id)) {
          *facet_type_id = facet_type->facet_type_id;
        } else {
          CARBON_DIAGNOSTIC(FacetTypeRequiredForTypeAndOperator, Error,
                            "non-facet type {0} combined with `&` operator",
                            SemIR::TypeId);
          // TODO: Find a location for the lhs or rhs specifically, instead of
          // the whole thing. If that's not possible we can change the text to
          // say if it's referring to the left or the right side for the error.
          // The `arg_id` instruction has no location in it for some reason.
          context.emitter().Emit(
              loc, FacetTypeRequiredForTypeAndOperator,
              context.types().GetTypeIdForTypeInstId(arg_id));
        }
      }
      // Allow errors to be diagnosed for both sides of the operator before
      // returning here if any error occurred on either side.
      if (!lhs_facet_type_id.has_value() || !rhs_facet_type_id.has_value()) {
        return SemIR::ErrorInst::SingletonConstantId;
      }
      // Reuse one of the argument instructions if nothing has changed.
      if (lhs_facet_type_id == rhs_facet_type_id) {
        return context.types().GetConstantId(
            context.types().GetTypeIdForTypeInstId(arg_ids[0]));
      }
      auto info = SemIR::FacetTypeInfo::Combine(
          context.facet_types().Get(lhs_facet_type_id),
          context.facet_types().Get(rhs_facet_type_id));
      info.Canonicalize();
      return MakeFacetTypeResult(eval_context.context(), info, phase);
    }

    case SemIR::BuiltinFunctionKind::IntLiteralMakeType: {
      return context.constant_values().Get(
          SemIR::IntLiteralType::SingletonInstId);
    }

    case SemIR::BuiltinFunctionKind::IntMakeTypeSigned: {
      return MakeIntTypeResult(context, loc, SemIR::IntKind::Signed, arg_ids[0],
                               phase);
    }

    case SemIR::BuiltinFunctionKind::IntMakeTypeUnsigned: {
      return MakeIntTypeResult(context, loc, SemIR::IntKind::Unsigned,
                               arg_ids[0], phase);
    }

    case SemIR::BuiltinFunctionKind::FloatMakeType: {
      // TODO: Support a symbolic constant width.
      if (phase != Phase::Concrete) {
        break;
      }
      if (!ValidateFloatBitWidth(context, loc, arg_ids[0])) {
        return SemIR::ErrorInst::SingletonConstantId;
      }
      return context.constant_values().Get(
          SemIR::LegacyFloatType::SingletonInstId);
    }

    case SemIR::BuiltinFunctionKind::BoolMakeType: {
      return context.constant_values().Get(SemIR::BoolType::SingletonInstId);
    }

    // Integer conversions.
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
      return PerformCheckedIntConvert(context, loc, arg_ids[0], call.type_id);
    }

    // Unary integer -> integer operations.
    case SemIR::BuiltinFunctionKind::IntSNegate:
    case SemIR::BuiltinFunctionKind::IntUNegate:
    case SemIR::BuiltinFunctionKind::IntComplement: {
      if (phase != Phase::Concrete) {
        break;
      }
      return PerformBuiltinUnaryIntOp(context, loc, builtin_kind, arg_ids[0]);
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
      return PerformBuiltinBinaryIntOp(context, loc, builtin_kind, arg_ids[0],
                                       arg_ids[1]);
    }

    // Bit shift operations.
    case SemIR::BuiltinFunctionKind::IntLeftShift:
    case SemIR::BuiltinFunctionKind::IntRightShift: {
      if (phase != Phase::Concrete) {
        break;
      }
      return PerformBuiltinIntShiftOp(context, loc, builtin_kind, arg_ids[0],
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

// Makes a constant for a call instruction.
static auto MakeConstantForCall(EvalContext& eval_context, SemIRLoc loc,
                                SemIR::Call call) -> SemIR::ConstantId {
  Phase phase = Phase::Concrete;

  // A call with an invalid argument list is used to represent an erroneous
  // call.
  //
  // TODO: Use a better representation for this.
  if (call.args_id == SemIR::InstBlockId::None) {
    return SemIR::ErrorInst::SingletonConstantId;
  }

  // Find the constant value of the callee.
  bool has_constant_callee = ReplaceFieldWithConstantValue(
      eval_context, &call, &SemIR::Call::callee_id, &phase);

  auto callee_function =
      SemIR::GetCalleeFunction(eval_context.sem_ir(), call.callee_id);
  auto builtin_kind = SemIR::BuiltinFunctionKind::None;
  if (callee_function.function_id.has_value()) {
    // Calls to builtins might be constant.
    builtin_kind = eval_context.functions()
                       .Get(callee_function.function_id)
                       .builtin_function_kind;
    if (builtin_kind == SemIR::BuiltinFunctionKind::None) {
      // TODO: Eventually we'll want to treat some kinds of non-builtin
      // functions as producing constants.
      return SemIR::ConstantId::NotConstant;
    }
  } else {
    // Calls to non-functions, such as calls to generic entity names, might be
    // constant.
  }

  // Find the argument values and the return type.
  bool has_constant_operands =
      has_constant_callee &&
      ReplaceFieldWithConstantValue(eval_context, &call, &SemIR::Call::type_id,
                                    &phase) &&
      ReplaceFieldWithConstantValue(eval_context, &call, &SemIR::Call::args_id,
                                    &phase);
  if (phase == Phase::UnknownDueToError) {
    return SemIR::ErrorInst::SingletonConstantId;
  }

  // If any operand of the call is non-constant, the call is non-constant.
  // TODO: Some builtin calls might allow some operands to be non-constant.
  if (!has_constant_operands) {
    if (builtin_kind.IsCompTimeOnly(
            eval_context.sem_ir(), eval_context.inst_blocks().Get(call.args_id),
            call.type_id)) {
      CARBON_DIAGNOSTIC(NonConstantCallToCompTimeOnlyFunction, Error,
                        "non-constant call to compile-time-only function");
      CARBON_DIAGNOSTIC(CompTimeOnlyFunctionHere, Note,
                        "compile-time-only function declared here");
      eval_context.emitter()
          .Build(loc, NonConstantCallToCompTimeOnlyFunction)
          .Note(eval_context.functions()
                    .Get(callee_function.function_id)
                    .latest_decl_id(),
                CompTimeOnlyFunctionHere)
          .Emit();
    }
    return SemIR::ConstantId::NotConstant;
  }

  // Handle calls to builtins.
  if (builtin_kind != SemIR::BuiltinFunctionKind::None) {
    return MakeConstantForBuiltinCall(
        eval_context, loc, call, builtin_kind,
        eval_context.inst_blocks().Get(call.args_id), phase);
  }

  return SemIR::ConstantId::NotConstant;
}

// Given an instruction, compute its phase based on its operands.
static auto ComputeInstPhase(Context& context, SemIR::Inst inst) -> Phase {
  EvalContext eval_context(context, SemIR::InstId::None);

  auto phase = GetPhase(context.constant_values(),
                        context.types().GetConstantId(inst.type_id()));
  auto kinds = inst.ArgKinds();
  GetConstantValueForArg(eval_context, kinds.first, inst.arg0(), &phase);
  GetConstantValueForArg(eval_context, kinds.second, inst.arg1(), &phase);
  CARBON_CHECK(IsConstant(phase));
  return phase;
}

// Convert a ConstantEvalResult to a ConstantId. Factored out of
// TryEvalTypedInst to avoid repeated instantiation of common code.
static auto ConvertEvalResultToConstantId(Context& context,
                                          ConstantEvalResult result,
                                          Phase orig_phase)
    -> SemIR::ConstantId {
  if (result.is_new()) {
    return MakeConstantResult(
        context, result.new_inst(),
        result.same_phase_as_inst()
            ? orig_phase
            : ComputeInstPhase(context, result.new_inst()));
  }
  return result.existing();
}

// Evaluates an instruction of a known type in an evaluation context. The
// default behavior of this function depends on the constant kind of the
// instruction:
//
//  -  InstConstantKind::Never: returns ConstantId::NotConstant.
//  -  InstConstantKind::Indirect, SymbolicOnly, Conditional: evaluates all the
//     operands of the instruction, and calls `EvalConstantInst` to evaluate the
//     resulting constant instruction.
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
  } else if constexpr (ConstantKind == SemIR::InstConstantKind::Unique) {
    CARBON_CHECK(inst_id.has_value());
    return SemIR::ConstantId::ForConcreteConstant(inst_id);
  } else {
    // Build a constant instruction by replacing each non-constant operand with
    // its constant value.
    Phase phase = Phase::Concrete;
    if (!ReplaceAllFieldsWithConstantValues(eval_context, &inst, &phase)) {
      if constexpr (ConstantKind == SemIR::InstConstantKind::Always) {
        CARBON_CHECK(phase == Phase::UnknownDueToError,
                     "{0} should always be constant", InstT::Kind);
      }
      return MakeNonConstantResult(phase);
    }
    if constexpr (ConstantKind == SemIR::InstConstantKind::Always ||
                  ConstantKind == SemIR::InstConstantKind::WheneverPossible) {
      return MakeConstantResult(eval_context.context(), inst, phase);
    } else if constexpr (ConstantKind == SemIR::InstConstantKind::InstAction) {
      auto result_inst_id = PerformDelayedAction(
          eval_context.context(), eval_context.insts().GetLocId(inst_id),
          inst.As<InstT>());
      if (result_inst_id.has_value()) {
        // The result is an instruction.
        return MakeConstantResult(
            eval_context.context(),
            SemIR::InstValue{.type_id = SemIR::InstType::SingletonTypeId,
                             .inst_id = result_inst_id},
            Phase::Concrete);
      }
      // Couldn't perform the action because it's still dependent.
      return MakeConstantResult(eval_context.context(), inst,
                                Phase::TemplateSymbolic);
    } else {
      return ConvertEvalResultToConstantId(
          eval_context.context(),
          EvalConstantInst(eval_context.context(),
                           eval_context.GetDiagnosticLoc({inst_id}),
                           inst.As<InstT>()),
          phase);
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
  return MakeConstantForCall(eval_context,
                             eval_context.GetDiagnosticLoc(inst_id),
                             inst.As<SemIR::Call>());
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

// TODO: Disable constant evaluation of SymbolicBindingPattern once
// DeduceGenericCallArguments no longer needs implicit params to have constant
// values.
template <>
auto TryEvalTypedInst<SemIR::SymbolicBindingPattern>(EvalContext& eval_context,
                                                     SemIR::InstId /*inst_id*/,
                                                     SemIR::Inst inst)
    -> SemIR::ConstantId {
  auto bind = inst.As<SemIR::SymbolicBindingPattern>();

  const auto& bind_name = eval_context.entity_names().Get(bind.entity_name_id);

  // If we know which specific we're evaluating within and this is an
  // argument of that specific, its constant value is the corresponding
  // argument value.
  if (auto value = eval_context.GetCompileTimeBindValue(bind_name.bind_index());
      value.has_value()) {
    // TODO: This seems incorrect: patterns don't typically evaluate to the
    // value matched by the pattern.
    return value;
  }

  // The constant form of a symbolic binding is an idealized form of the
  // original, with no equivalent value.
  bind.entity_name_id =
      eval_context.entity_names().MakeCanonical(bind.entity_name_id);
  return MakeConstantResult(
      eval_context.context(), bind,
      bind_name.is_template ? Phase::TemplateSymbolic : Phase::CheckedSymbolic);
}

// Symbolic bindings are a special case because they can reach into the eval
// context and produce a context-specific value.
template <>
auto TryEvalTypedInst<SemIR::BindSymbolicName>(EvalContext& eval_context,
                                               SemIR::InstId /*inst_id*/,
                                               SemIR::Inst inst)
    -> SemIR::ConstantId {
  auto bind = inst.As<SemIR::BindSymbolicName>();

  const auto& bind_name = eval_context.entity_names().Get(bind.entity_name_id);

  Phase phase;
  if (bind_name.name_id == SemIR::NameId::PeriodSelf) {
    phase = Phase::PeriodSelfSymbolic;
  } else {
    // If we know which specific we're evaluating within and this is an
    // argument of that specific, its constant value is the corresponding
    // argument value.
    if (auto value =
            eval_context.GetCompileTimeBindValue(bind_name.bind_index());
        value.has_value()) {
      return value;
    }
    phase = bind_name.is_template ? Phase::TemplateSymbolic
                                  : Phase::CheckedSymbolic;
  }
  // The constant form of a symbolic binding is an idealized form of the
  // original, with no equivalent value.
  bind.entity_name_id =
      eval_context.entity_names().MakeCanonical(bind.entity_name_id);
  bind.value_id = SemIR::InstId::None;
  if (!ReplaceFieldWithConstantValue(
          eval_context, &bind, &SemIR::BindSymbolicName::type_id, &phase)) {
    return MakeNonConstantResult(phase);
  }
  return MakeConstantResult(eval_context.context(), bind, phase);
}

// TODO: Convert this to an EvalConstantInst instruction. This will require
// providing a `GetConstantValue` overload for a requirement block.
template <>
auto TryEvalTypedInst<SemIR::WhereExpr>(EvalContext& eval_context,
                                        SemIR::InstId /*inst_id*/,
                                        SemIR::Inst inst) -> SemIR::ConstantId {
  auto typed_inst = inst.As<SemIR::WhereExpr>();

  Phase phase = Phase::Concrete;
  SemIR::TypeId base_facet_type_id =
      eval_context.insts().Get(typed_inst.period_self_id).type_id();
  SemIR::Inst base_facet_inst =
      eval_context.GetConstantValueAsInst(base_facet_type_id);
  SemIR::FacetTypeInfo info = {.other_requirements = false};
  // `where` provides that the base facet is an error, `type`, or a facet
  // type.
  if (auto facet_type = base_facet_inst.TryAs<SemIR::FacetType>()) {
    info = GetConstantFacetTypeInfo(eval_context, facet_type->facet_type_id,
                                    &phase);
  } else if (base_facet_type_id == SemIR::ErrorInst::SingletonTypeId) {
    return SemIR::ErrorInst::SingletonConstantId;
  } else {
    CARBON_CHECK(base_facet_type_id == SemIR::TypeType::SingletonTypeId,
                 "Unexpected type_id: {0}, inst: {1}", base_facet_type_id,
                 base_facet_inst);
  }
  if (typed_inst.requirements_id.has_value()) {
    auto insts = eval_context.inst_blocks().Get(typed_inst.requirements_id);
    for (auto inst_id : insts) {
      if (auto rewrite =
              eval_context.insts().TryGetAs<SemIR::RequirementRewrite>(
                  inst_id)) {
        SemIR::ConstantId lhs = eval_context.GetConstantValue(rewrite->lhs_id);
        SemIR::ConstantId rhs = eval_context.GetConstantValue(rewrite->rhs_id);
        // `where` requirements using `.Self` should not be considered
        // symbolic
        UpdatePhaseIgnorePeriodSelf(eval_context, lhs, &phase);
        UpdatePhaseIgnorePeriodSelf(eval_context, rhs, &phase);
        info.rewrite_constraints.push_back(
            {.lhs_const_id = lhs, .rhs_const_id = rhs});
      } else {
        // TODO: Handle other requirements
        info.other_requirements = true;
      }
    }
  }
  info.Canonicalize();
  return MakeFacetTypeResult(eval_context.context(), info, phase);
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

auto TryEvalInst(Context& context, SemIR::InstId inst_id, SemIR::Inst inst)
    -> SemIR::ConstantId {
  EvalContext eval_context(context, inst_id);
  return TryEvalInstInContext(eval_context, inst_id, inst);
}

auto TryEvalBlockForSpecific(Context& context, SemIRLoc loc,
                             SemIR::SpecificId specific_id,
                             SemIR::GenericInstIndex::Region region)
    -> SemIR::InstBlockId {
  auto generic_id = context.specifics().Get(specific_id).generic_id;
  auto eval_block_id = context.generics().Get(generic_id).GetEvalBlock(region);
  auto eval_block = context.inst_blocks().Get(eval_block_id);

  llvm::SmallVector<SemIR::InstId> result;
  result.resize(eval_block.size(), SemIR::InstId::None);

  EvalContext eval_context(context, loc, specific_id,
                           SpecificEvalInfo{
                               .region = region,
                               .values = result,
                           });

  DiagnosticAnnotationScope annotate_diagnostics(
      &context.emitter(), [&](auto& builder) {
        CARBON_DIAGNOSTIC(ResolvingSpecificHere, Note, "in {0} used here",
                          InstIdAsType);
        builder.Note(loc, ResolvingSpecificHere,
                     GetInstForSpecific(context, specific_id));
      });

  for (auto [i, inst_id] : llvm::enumerate(eval_block)) {
    auto const_id = TryEvalInstInContext(eval_context, inst_id,
                                         context.insts().Get(inst_id));
    result[i] = context.constant_values().GetInstId(const_id);
    CARBON_CHECK(result[i].has_value());
  }

  return context.inst_blocks().Add(result);
}

}  // namespace Carbon::Check
