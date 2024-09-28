// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/eval.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/check/generic.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/sem_ir/builtin_function_kind.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/generic.h"
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
      Context& context,
      SemIR::SpecificId specific_id = SemIR::SpecificId::Invalid,
      std::optional<SpecificEvalInfo> specific_eval_info = std::nullopt)
      : context_(context),
        specific_id_(specific_id),
        specific_eval_info_(specific_eval_info) {}

  // Gets the value of the specified compile-time binding in this context.
  // Returns `Invalid` if the value is not fixed in this context.
  auto GetCompileTimeBindValue(SemIR::CompileTimeBindIndex bind_index)
      -> SemIR::ConstantId {
    if (!bind_index.is_valid() || !specific_id_.is_valid()) {
      return SemIR::ConstantId::Invalid;
    }

    const auto& specific = specifics().Get(specific_id_);
    auto args = inst_blocks().Get(specific.args_id);

    // Bindings past the ones with known arguments can appear as local
    // bindings of entities declared within this generic.
    if (static_cast<size_t>(bind_index.index) >= args.size()) {
      return SemIR::ConstantId::Invalid;
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
      if (symbolic_info.index.is_valid() &&
          symbolic_info.generic_id ==
              specifics().Get(specific_id_).generic_id &&
          symbolic_info.index.region() == specific_eval_info_->region) {
        auto inst_id = specific_eval_info_->values[symbolic_info.index.index()];
        CARBON_CHECK(inst_id.is_valid(),
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
    return context().GetTypeIdForTypeConstant(GetConstantValue(id));
  }

  // Gets the instruction describing the constant value of the specified type in
  // this context.
  auto GetConstantValueAsInst(SemIR::TypeId id) -> SemIR::Inst {
    return insts().Get(
        context().constant_values().GetInstId(GetConstantValue(id)));
  }

  auto ints() -> CanonicalValueStore<IntId>& { return sem_ir().ints(); }
  auto floats() -> FloatValueStore& { return sem_ir().floats(); }
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

  auto emitter() -> Context::DiagnosticEmitter& { return context().emitter(); }

 private:
  // The type-checking context in which we're performing evaluation.
  Context& context_;
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
  Template,
  // Evaluation phase is symbolic because the expression involves a reference to
  // a symbolic binding.
  Symbolic,
  // The evaluation phase is unknown because evaluation encountered an
  // already-diagnosed semantic or syntax error. This is treated as being
  // potentially constant, but with an unknown phase.
  UnknownDueToError,
  // The expression has runtime phase because of a non-constant subexpression.
  Runtime,
};
}  // namespace

// Gets the phase in which the value of a constant will become available.
static auto GetPhase(SemIR::ConstantId constant_id) -> Phase {
  if (!constant_id.is_constant()) {
    return Phase::Runtime;
  } else if (constant_id == SemIR::ConstantId::Error) {
    return Phase::UnknownDueToError;
  } else if (constant_id.is_template()) {
    return Phase::Template;
  } else {
    CARBON_CHECK(constant_id.is_symbolic());
    return Phase::Symbolic;
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
    case Phase::Template:
      return context.AddConstant(inst, /*is_symbolic=*/false);
    case Phase::Symbolic:
      return context.AddConstant(inst, /*is_symbolic=*/true);
    case Phase::UnknownDueToError:
      return SemIR::ConstantId::Error;
    case Phase::Runtime:
      return SemIR::ConstantId::NotConstant;
  }
}

// Forms a `constant_id` describing why an evaluation was not constant.
static auto MakeNonConstantResult(Phase phase) -> SemIR::ConstantId {
  return phase == Phase::UnknownDueToError ? SemIR::ConstantId::Error
                                           : SemIR::ConstantId::NotConstant;
}

// Converts a bool value into a ConstantId.
static auto MakeBoolResult(Context& context, SemIR::TypeId bool_type_id,
                           bool result) -> SemIR::ConstantId {
  return MakeConstantResult(
      context,
      SemIR::BoolLiteral{.type_id = bool_type_id,
                         .value = SemIR::BoolValue::From(result)},
      Phase::Template);
}

// Converts an APInt value into a ConstantId.
static auto MakeIntResult(Context& context, SemIR::TypeId type_id,
                          llvm::APInt value) -> SemIR::ConstantId {
  auto result = context.ints().Add(std::move(value));
  return MakeConstantResult(
      context, SemIR::IntLiteral{.type_id = type_id, .int_id = result},
      Phase::Template);
}

// Converts an APFloat value into a ConstantId.
static auto MakeFloatResult(Context& context, SemIR::TypeId type_id,
                            llvm::APFloat value) -> SemIR::ConstantId {
  auto result = context.floats().Add(std::move(value));
  return MakeConstantResult(
      context, SemIR::FloatLiteral{.type_id = type_id, .float_id = result},
      Phase::Template);
}

// `GetConstantValue` checks to see whether the provided ID describes a value
// with constant phase, and if so, returns the corresponding constant value.
// Overloads are provided for different kinds of ID.

// If the given instruction is constant, returns its constant value.
static auto GetConstantValue(EvalContext& eval_context, SemIR::InstId inst_id,
                             Phase* phase) -> SemIR::InstId {
  auto const_id = eval_context.GetConstantValue(inst_id);
  *phase = LatestPhase(*phase, GetPhase(const_id));
  return eval_context.constant_values().GetInstId(const_id);
}

// Given a type which may refer to a generic parameter, returns the
// corresponding type in the evaluation context.
static auto GetConstantValue(EvalContext& eval_context, SemIR::TypeId type_id,
                             Phase* phase) -> SemIR::TypeId {
  auto const_id = eval_context.GetConstantValue(type_id);
  *phase = LatestPhase(*phase, GetPhase(const_id));
  return eval_context.context().GetTypeIdForTypeConstant(const_id);
}

// If the given instruction block contains only constants, returns a
// corresponding block of those values.
static auto GetConstantValue(EvalContext& eval_context,
                             SemIR::InstBlockId inst_block_id, Phase* phase)
    -> SemIR::InstBlockId {
  if (!inst_block_id.is_valid()) {
    return SemIR::InstBlockId::Invalid;
  }
  auto insts = eval_context.inst_blocks().Get(inst_block_id);
  llvm::SmallVector<SemIR::InstId> const_insts;
  for (auto inst_id : insts) {
    auto const_inst_id = GetConstantValue(eval_context, inst_id, phase);
    if (!const_inst_id.is_valid()) {
      return SemIR::InstBlockId::Invalid;
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
                             SemIR::TypeBlockId type_block_id, Phase* phase)
    -> SemIR::TypeBlockId {
  if (!type_block_id.is_valid()) {
    return SemIR::TypeBlockId::Invalid;
  }
  auto types = eval_context.type_blocks().Get(type_block_id);
  llvm::SmallVector<SemIR::TypeId> new_types;
  for (auto type_id : types) {
    auto new_type_id = GetConstantValue(eval_context, type_id, phase);
    if (!new_type_id.is_valid()) {
      return SemIR::TypeBlockId::Invalid;
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
  if (!specific_id.is_valid()) {
    return SemIR::SpecificId::Invalid;
  }

  const auto& specific = eval_context.specifics().Get(specific_id);
  auto args_id = GetConstantValue(eval_context, specific.args_id, phase);
  if (!args_id.is_valid()) {
    return SemIR::SpecificId::Invalid;
  }

  if (args_id == specific.args_id) {
    return specific_id;
  }
  return MakeSpecific(eval_context.context(), specific.generic_id, args_id);
}

// Replaces the specified field of the given typed instruction with its constant
// value, if it has constant phase. Returns true on success, false if the value
// has runtime phase.
template <typename InstT, typename FieldIdT>
static auto ReplaceFieldWithConstantValue(EvalContext& eval_context,
                                          InstT* inst, FieldIdT InstT::*field,
                                          Phase* phase) -> bool {
  auto unwrapped = GetConstantValue(eval_context, inst->*field, phase);
  if (!unwrapped.is_valid() && (inst->*field).is_valid()) {
    return false;
  }
  inst->*field = unwrapped;
  return true;
}

// If the specified fields of the given typed instruction have constant values,
// replaces the fields with their constant values and builds a corresponding
// constant value. Otherwise returns `ConstantId::NotConstant`. Returns
// `ConstantId::Error` if any subexpression is an error.
//
// The constant value is then checked by calling `validate_fn(typed_inst)`,
// which should return a `bool` indicating whether the new constant is valid. If
// validation passes, `transform_fn(typed_inst)` is called to produce the final
// constant instruction, and a corresponding ConstantId for the new constant is
// returned. If validation fails, it should produce a suitable error message.
// `ConstantId::Error` is returned.
template <typename InstT, typename ValidateFn, typename TransformFn,
          typename... EachFieldIdT>
static auto RebuildIfFieldsAreConstantImpl(
    EvalContext& eval_context, SemIR::Inst inst, ValidateFn validate_fn,
    TransformFn transform_fn, EachFieldIdT InstT::*... each_field_id)
    -> SemIR::ConstantId {
  // Build a constant instruction by replacing each non-constant operand with
  // its constant value.
  auto typed_inst = inst.As<InstT>();
  Phase phase = Phase::Template;
  if ((ReplaceFieldWithConstantValue(eval_context, &typed_inst, each_field_id,
                                     &phase) &&
       ...)) {
    if (phase == Phase::UnknownDueToError || !validate_fn(typed_inst)) {
      return SemIR::ConstantId::Error;
    }
    return MakeConstantResult(eval_context.context(), transform_fn(typed_inst),
                              phase);
  }
  return MakeNonConstantResult(phase);
}

// Same as above but with an identity transform function.
template <typename InstT, typename ValidateFn, typename... EachFieldIdT>
static auto RebuildAndValidateIfFieldsAreConstant(
    EvalContext& eval_context, SemIR::Inst inst, ValidateFn validate_fn,
    EachFieldIdT InstT::*... each_field_id) -> SemIR::ConstantId {
  return RebuildIfFieldsAreConstantImpl(eval_context, inst, validate_fn,
                                        std::identity{}, each_field_id...);
}

// Same as above but with no validation step.
template <typename InstT, typename TransformFn, typename... EachFieldIdT>
static auto TransformIfFieldsAreConstant(EvalContext& eval_context,
                                         SemIR::Inst inst,
                                         TransformFn transform_fn,
                                         EachFieldIdT InstT::*... each_field_id)
    -> SemIR::ConstantId {
  return RebuildIfFieldsAreConstantImpl(
      eval_context, inst, [](...) { return true; }, transform_fn,
      each_field_id...);
}

// Same as above but with no validation or transform step.
template <typename InstT, typename... EachFieldIdT>
static auto RebuildIfFieldsAreConstant(EvalContext& eval_context,
                                       SemIR::Inst inst,
                                       EachFieldIdT InstT::*... each_field_id)
    -> SemIR::ConstantId {
  return RebuildIfFieldsAreConstantImpl(
      eval_context, inst, [](...) { return true; }, std::identity{},
      each_field_id...);
}

// Rebuilds the given aggregate initialization instruction as a corresponding
// constant aggregate value, if its elements are all constants.
static auto RebuildInitAsValue(EvalContext& eval_context, SemIR::Inst inst,
                               SemIR::InstKind value_kind)
    -> SemIR::ConstantId {
  return TransformIfFieldsAreConstant(
      eval_context, inst,
      [&](SemIR::AnyAggregateInit result) {
        return SemIR::AnyAggregateValue{.kind = value_kind,
                                        .type_id = result.type_id,
                                        .elements_id = result.elements_id};
      },
      &SemIR::AnyAggregateInit::type_id, &SemIR::AnyAggregateInit::elements_id);
}

// Performs an access into an aggregate, retrieving the specified element.
static auto PerformAggregateAccess(EvalContext& eval_context, SemIR::Inst inst)
    -> SemIR::ConstantId {
  auto access_inst = inst.As<SemIR::AnyAggregateAccess>();
  Phase phase = Phase::Template;
  if (auto aggregate_id =
          GetConstantValue(eval_context, access_inst.aggregate_id, &phase);
      aggregate_id.is_valid()) {
    if (auto aggregate =
            eval_context.insts().TryGetAs<SemIR::AnyAggregateValue>(
                aggregate_id)) {
      auto elements = eval_context.inst_blocks().Get(aggregate->elements_id);
      auto index = static_cast<size_t>(access_inst.index.index);
      CARBON_CHECK(index < elements.size(), "Access out of bounds.");
      // `Phase` is not used here. If this element is a template constant, then
      // so is the result of indexing, even if the aggregate also contains a
      // symbolic context.
      return eval_context.GetConstantValue(elements[index]);
    } else {
      CARBON_CHECK(phase != Phase::Template,
                   "Failed to evaluate template constant {0}", inst);
    }
  }
  return MakeNonConstantResult(phase);
}

// Performs an index into a homogeneous aggregate, retrieving the specified
// element.
static auto PerformArrayIndex(EvalContext& eval_context, SemIR::Inst inst)
    -> SemIR::ConstantId {
  auto index_inst = inst.As<SemIR::ArrayIndex>();
  Phase phase = Phase::Template;
  auto index_id = GetConstantValue(eval_context, index_inst.index_id, &phase);

  if (!index_id.is_valid()) {
    return MakeNonConstantResult(phase);
  }
  auto index = eval_context.insts().TryGetAs<SemIR::IntLiteral>(index_id);
  if (!index) {
    CARBON_CHECK(phase != Phase::Template,
                 "Template constant integer should be a literal");
    return MakeNonConstantResult(phase);
  }

  // Array indexing is invalid if the index is constant and out of range,
  // regardless of whether the array itself is constant.
  const auto& index_val = eval_context.ints().Get(index->int_id);
  auto aggregate_type_id = eval_context.GetConstantValueAsType(
      eval_context.insts().Get(index_inst.array_id).type_id());
  if (auto array_type =
          eval_context.types().TryGetAs<SemIR::ArrayType>(aggregate_type_id)) {
    if (auto bound = eval_context.insts().TryGetAs<SemIR::IntLiteral>(
            array_type->bound_id)) {
      // This awkward call to `getZExtValue` is a workaround for APInt not
      // supporting comparisons between integers of different bit widths.
      if (index_val.getActiveBits() > 64 ||
          eval_context.ints()
              .Get(bound->int_id)
              .ule(index_val.getZExtValue())) {
        CARBON_DIAGNOSTIC(ArrayIndexOutOfBounds, Error,
                          "array index `{0}` is past the end of type `{1}`",
                          TypedInt, SemIR::TypeId);
        eval_context.emitter().Emit(
            index_inst.index_id, ArrayIndexOutOfBounds,
            {.type = index->type_id, .value = index_val}, aggregate_type_id);
        return SemIR::ConstantId::Error;
      }
    }
  }

  auto aggregate_id =
      GetConstantValue(eval_context, index_inst.array_id, &phase);
  if (!aggregate_id.is_valid()) {
    return MakeNonConstantResult(phase);
  }
  auto aggregate =
      eval_context.insts().TryGetAs<SemIR::AnyAggregateValue>(aggregate_id);
  if (!aggregate) {
    CARBON_CHECK(phase != Phase::Template,
                 "Unexpected representation for template constant aggregate");
    return MakeNonConstantResult(phase);
  }

  auto elements = eval_context.inst_blocks().Get(aggregate->elements_id);
  return eval_context.GetConstantValue(elements[index_val.getZExtValue()]);
}

// Enforces that an integer type has a valid bit width.
static auto ValidateIntType(Context& context, SemIRLoc loc,
                            SemIR::IntType result) -> bool {
  auto bit_width =
      context.insts().TryGetAs<SemIR::IntLiteral>(result.bit_width_id);
  if (!bit_width) {
    // Symbolic bit width.
    return true;
  }
  const auto& bit_width_val = context.ints().Get(bit_width->int_id);
  if (bit_width_val.isZero() ||
      (context.types().IsSignedInt(bit_width->type_id) &&
       bit_width_val.isNegative())) {
    CARBON_DIAGNOSTIC(IntWidthNotPositive, Error,
                      "integer type width of {0} is not positive", TypedInt);
    context.emitter().Emit(
        loc, IntWidthNotPositive,
        {.type = bit_width->type_id, .value = bit_width_val});
    return false;
  }
  // TODO: Pick a maximum size and document it in the design. For now
  // we use 2^^23, because that's the largest size that LLVM supports.
  constexpr int MaxIntWidth = 1 << 23;
  if (bit_width_val.ugt(MaxIntWidth)) {
    CARBON_DIAGNOSTIC(IntWidthTooLarge, Error,
                      "integer type width of {0} is greater than the "
                      "maximum supported width of {1}",
                      TypedInt, int);
    context.emitter().Emit(loc, IntWidthTooLarge,
                           {.type = bit_width->type_id, .value = bit_width_val},
                           MaxIntWidth);
    return false;
  }
  return true;
}

// Forms a constant int type as an evaluation result. Requires that width_id is
// constant.
static auto MakeIntTypeResult(Context& context, SemIRLoc loc,
                              SemIR::IntKind int_kind, SemIR::InstId width_id,
                              Phase phase) -> SemIR::ConstantId {
  auto result = SemIR::IntType{
      .type_id = context.GetBuiltinType(SemIR::BuiltinInstKind::TypeType),
      .int_kind = int_kind,
      .bit_width_id = width_id};
  if (!ValidateIntType(context, loc, result)) {
    return SemIR::ConstantId::Error;
  }
  return MakeConstantResult(context, result, phase);
}

// Enforces that the bit width is 64 for a float.
static auto ValidateFloatBitWidth(Context& context, SemIRLoc loc,
                                  SemIR::InstId inst_id) -> bool {
  auto inst = context.insts().GetAs<SemIR::IntLiteral>(inst_id);
  if (context.ints().Get(inst.int_id) == 64) {
    return true;
  }

  CARBON_DIAGNOSTIC(CompileTimeFloatBitWidth, Error, "bit width must be 64");
  context.emitter().Emit(loc, CompileTimeFloatBitWidth);
  return false;
}

// Enforces that a float type has a valid bit width.
static auto ValidateFloatType(Context& context, SemIRLoc loc,
                              SemIR::FloatType result) -> bool {
  auto bit_width =
      context.insts().TryGetAs<SemIR::IntLiteral>(result.bit_width_id);
  if (!bit_width) {
    // Symbolic bit width.
    return true;
  }
  return ValidateFloatBitWidth(context, loc, result.bit_width_id);
}

// Issues a diagnostic for a compile-time division by zero.
static auto DiagnoseDivisionByZero(Context& context, SemIRLoc loc) -> void {
  CARBON_DIAGNOSTIC(CompileTimeDivisionByZero, Error, "division by zero");
  context.emitter().Emit(loc, CompileTimeDivisionByZero);
}

// Performs a builtin unary integer -> integer operation.
static auto PerformBuiltinUnaryIntOp(Context& context, SemIRLoc loc,
                                     SemIR::BuiltinFunctionKind builtin_kind,
                                     SemIR::InstId arg_id)
    -> SemIR::ConstantId {
  auto op = context.insts().GetAs<SemIR::IntLiteral>(arg_id);
  auto op_val = context.ints().Get(op.int_id);

  switch (builtin_kind) {
    case SemIR::BuiltinFunctionKind::IntSNegate:
      if (context.types().IsSignedInt(op.type_id) &&
          op_val.isMinSignedValue()) {
        CARBON_DIAGNOSTIC(CompileTimeIntegerNegateOverflow, Error,
                          "integer overflow in negation of {0}", TypedInt);
        context.emitter().Emit(loc, CompileTimeIntegerNegateOverflow,
                               {.type = op.type_id, .value = op_val});
      }
      op_val.negate();
      break;
    case SemIR::BuiltinFunctionKind::IntUNegate:
      op_val.negate();
      break;
    case SemIR::BuiltinFunctionKind::IntComplement:
      op_val.flipAllBits();
      break;
    default:
      CARBON_FATAL("Unexpected builtin kind");
  }

  return MakeIntResult(context, op.type_id, std::move(op_val));
}

// Performs a builtin binary integer -> integer operation.
static auto PerformBuiltinBinaryIntOp(Context& context, SemIRLoc loc,
                                      SemIR::BuiltinFunctionKind builtin_kind,
                                      SemIR::InstId lhs_id,
                                      SemIR::InstId rhs_id)
    -> SemIR::ConstantId {
  auto lhs = context.insts().GetAs<SemIR::IntLiteral>(lhs_id);
  auto rhs = context.insts().GetAs<SemIR::IntLiteral>(rhs_id);
  const auto& lhs_val = context.ints().Get(lhs.int_id);
  const auto& rhs_val = context.ints().Get(rhs.int_id);

  // Check for division by zero.
  switch (builtin_kind) {
    case SemIR::BuiltinFunctionKind::IntSDiv:
    case SemIR::BuiltinFunctionKind::IntSMod:
    case SemIR::BuiltinFunctionKind::IntUDiv:
    case SemIR::BuiltinFunctionKind::IntUMod:
      if (rhs_val.isZero()) {
        DiagnoseDivisionByZero(context, loc);
        return SemIR::ConstantId::Error;
      }
      break;
    default:
      break;
  }

  bool overflow = false;
  llvm::APInt result_val;
  llvm::StringLiteral op_str = "<error>";
  switch (builtin_kind) {
    // Arithmetic.
    case SemIR::BuiltinFunctionKind::IntSAdd:
      result_val = lhs_val.sadd_ov(rhs_val, overflow);
      op_str = "+";
      break;
    case SemIR::BuiltinFunctionKind::IntSSub:
      result_val = lhs_val.ssub_ov(rhs_val, overflow);
      op_str = "-";
      break;
    case SemIR::BuiltinFunctionKind::IntSMul:
      result_val = lhs_val.smul_ov(rhs_val, overflow);
      op_str = "*";
      break;
    case SemIR::BuiltinFunctionKind::IntSDiv:
      result_val = lhs_val.sdiv_ov(rhs_val, overflow);
      op_str = "/";
      break;
    case SemIR::BuiltinFunctionKind::IntSMod:
      result_val = lhs_val.srem(rhs_val);
      // LLVM weirdly lacks `srem_ov`, so we work it out for ourselves:
      // <signed min> % -1 overflows because <signed min> / -1 overflows.
      overflow = lhs_val.isMinSignedValue() && rhs_val.isAllOnes();
      op_str = "%";
      break;
    case SemIR::BuiltinFunctionKind::IntUAdd:
      result_val = lhs_val + rhs_val;
      op_str = "+";
      break;
    case SemIR::BuiltinFunctionKind::IntUSub:
      result_val = lhs_val - rhs_val;
      op_str = "-";
      break;
    case SemIR::BuiltinFunctionKind::IntUMul:
      result_val = lhs_val * rhs_val;
      op_str = "*";
      break;
    case SemIR::BuiltinFunctionKind::IntUDiv:
      result_val = lhs_val.udiv(rhs_val);
      op_str = "/";
      break;
    case SemIR::BuiltinFunctionKind::IntUMod:
      result_val = lhs_val.urem(rhs_val);
      op_str = "%";
      break;

    // Bitwise.
    case SemIR::BuiltinFunctionKind::IntAnd:
      result_val = lhs_val & rhs_val;
      op_str = "&";
      break;
    case SemIR::BuiltinFunctionKind::IntOr:
      result_val = lhs_val | rhs_val;
      op_str = "|";
      break;
    case SemIR::BuiltinFunctionKind::IntXor:
      result_val = lhs_val ^ rhs_val;
      op_str = "^";
      break;

    // Bit shift.
    case SemIR::BuiltinFunctionKind::IntLeftShift:
    case SemIR::BuiltinFunctionKind::IntRightShift:
      op_str = (builtin_kind == SemIR::BuiltinFunctionKind::IntLeftShift)
                   ? llvm::StringLiteral("<<")
                   : llvm::StringLiteral(">>");
      if (rhs_val.uge(lhs_val.getBitWidth()) ||
          (rhs_val.isNegative() && context.types().IsSignedInt(rhs.type_id))) {
        CARBON_DIAGNOSTIC(CompileTimeShiftOutOfRange, Error,
                          "shift distance not in range [0, {0}) in {1} {2} {3}",
                          unsigned, TypedInt, llvm::StringLiteral, TypedInt);
        context.emitter().Emit(loc, CompileTimeShiftOutOfRange,
                               lhs_val.getBitWidth(),
                               {.type = lhs.type_id, .value = lhs_val}, op_str,
                               {.type = rhs.type_id, .value = rhs_val});
        // TODO: Is it useful to recover by returning 0 or -1?
        return SemIR::ConstantId::Error;
      }

      if (builtin_kind == SemIR::BuiltinFunctionKind::IntLeftShift) {
        result_val = lhs_val.shl(rhs_val);
      } else if (context.types().IsSignedInt(lhs.type_id)) {
        result_val = lhs_val.ashr(rhs_val);
      } else {
        result_val = lhs_val.lshr(rhs_val);
      }
      break;

    default:
      CARBON_FATAL("Unexpected operation kind.");
  }

  if (overflow) {
    CARBON_DIAGNOSTIC(CompileTimeIntegerOverflow, Error,
                      "integer overflow in calculation {0} {1} {2}", TypedInt,
                      llvm::StringLiteral, TypedInt);
    context.emitter().Emit(loc, CompileTimeIntegerOverflow,
                           {.type = lhs.type_id, .value = lhs_val}, op_str,
                           {.type = rhs.type_id, .value = rhs_val});
  }

  return MakeIntResult(context, lhs.type_id, std::move(result_val));
}

// Performs a builtin integer comparison.
static auto PerformBuiltinIntComparison(Context& context,
                                        SemIR::BuiltinFunctionKind builtin_kind,
                                        SemIR::InstId lhs_id,
                                        SemIR::InstId rhs_id,
                                        SemIR::TypeId bool_type_id)
    -> SemIR::ConstantId {
  auto lhs = context.insts().GetAs<SemIR::IntLiteral>(lhs_id);
  const auto& lhs_val = context.ints().Get(lhs.int_id);
  const auto& rhs_val = context.ints().Get(
      context.insts().GetAs<SemIR::IntLiteral>(rhs_id).int_id);
  bool is_signed = context.types().IsSignedInt(lhs.type_id);

  bool result;
  switch (builtin_kind) {
    case SemIR::BuiltinFunctionKind::IntEq:
      result = (lhs_val == rhs_val);
      break;
    case SemIR::BuiltinFunctionKind::IntNeq:
      result = (lhs_val != rhs_val);
      break;
    case SemIR::BuiltinFunctionKind::IntLess:
      result = is_signed ? lhs_val.slt(rhs_val) : lhs_val.ult(rhs_val);
      break;
    case SemIR::BuiltinFunctionKind::IntLessEq:
      result = is_signed ? lhs_val.sle(rhs_val) : lhs_val.ule(rhs_val);
      break;
    case SemIR::BuiltinFunctionKind::IntGreater:
      result = is_signed ? lhs_val.sgt(rhs_val) : lhs_val.sgt(rhs_val);
      break;
    case SemIR::BuiltinFunctionKind::IntGreaterEq:
      result = is_signed ? lhs_val.sge(rhs_val) : lhs_val.sge(rhs_val);
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

// Returns a constant for a call to a builtin function.
static auto MakeConstantForBuiltinCall(Context& context, SemIRLoc loc,
                                       SemIR::Call call,
                                       SemIR::BuiltinFunctionKind builtin_kind,
                                       llvm::ArrayRef<SemIR::InstId> arg_ids,
                                       Phase phase) -> SemIR::ConstantId {
  switch (builtin_kind) {
    case SemIR::BuiltinFunctionKind::None:
      CARBON_FATAL("Not a builtin function.");

    case SemIR::BuiltinFunctionKind::PrintInt: {
      // Providing a constant result would allow eliding the function call.
      return SemIR::ConstantId::NotConstant;
    }

    case SemIR::BuiltinFunctionKind::IntMakeType32: {
      return context.constant_values().Get(SemIR::InstId::BuiltinIntType);
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
      if (phase != Phase::Template) {
        break;
      }
      if (!ValidateFloatBitWidth(context, loc, arg_ids[0])) {
        return SemIR::ConstantId::Error;
      }
      return context.constant_values().Get(SemIR::InstId::BuiltinFloatType);
    }

    case SemIR::BuiltinFunctionKind::BoolMakeType: {
      return context.constant_values().Get(SemIR::InstId::BuiltinBoolType);
    }

    // Unary integer -> integer operations.
    case SemIR::BuiltinFunctionKind::IntSNegate:
    case SemIR::BuiltinFunctionKind::IntUNegate:
    case SemIR::BuiltinFunctionKind::IntComplement: {
      if (phase != Phase::Template) {
        break;
      }
      return PerformBuiltinUnaryIntOp(context, loc, builtin_kind, arg_ids[0]);
    }

    // Binary integer -> integer operations.
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
    case SemIR::BuiltinFunctionKind::IntXor:
    case SemIR::BuiltinFunctionKind::IntLeftShift:
    case SemIR::BuiltinFunctionKind::IntRightShift: {
      if (phase != Phase::Template) {
        break;
      }
      return PerformBuiltinBinaryIntOp(context, loc, builtin_kind, arg_ids[0],
                                       arg_ids[1]);
    }

    // Integer comparisons.
    case SemIR::BuiltinFunctionKind::IntEq:
    case SemIR::BuiltinFunctionKind::IntNeq:
    case SemIR::BuiltinFunctionKind::IntLess:
    case SemIR::BuiltinFunctionKind::IntLessEq:
    case SemIR::BuiltinFunctionKind::IntGreater:
    case SemIR::BuiltinFunctionKind::IntGreaterEq: {
      if (phase != Phase::Template) {
        break;
      }
      return PerformBuiltinIntComparison(context, builtin_kind, arg_ids[0],
                                         arg_ids[1], call.type_id);
    }

    // Unary float -> float operations.
    case SemIR::BuiltinFunctionKind::FloatNegate: {
      if (phase != Phase::Template) {
        break;
      }

      return PerformBuiltinUnaryFloatOp(context, builtin_kind, arg_ids[0]);
    }

    // Binary float -> float operations.
    case SemIR::BuiltinFunctionKind::FloatAdd:
    case SemIR::BuiltinFunctionKind::FloatSub:
    case SemIR::BuiltinFunctionKind::FloatMul:
    case SemIR::BuiltinFunctionKind::FloatDiv: {
      if (phase != Phase::Template) {
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
      if (phase != Phase::Template) {
        break;
      }
      return PerformBuiltinFloatComparison(context, builtin_kind, arg_ids[0],
                                           arg_ids[1], call.type_id);
    }
  }

  return SemIR::ConstantId::NotConstant;
}

// Makes a constant for a call instruction.
static auto MakeConstantForCall(EvalContext& eval_context, SemIRLoc loc,
                                SemIR::Call call) -> SemIR::ConstantId {
  Phase phase = Phase::Template;

  // A call with an invalid argument list is used to represent an erroneous
  // call.
  //
  // TODO: Use a better representation for this.
  if (call.args_id == SemIR::InstBlockId::Invalid) {
    return SemIR::ConstantId::Error;
  }

  // If the callee or return type isn't constant, this is not a constant call.
  if (!ReplaceFieldWithConstantValue(eval_context, &call,
                                     &SemIR::Call::callee_id, &phase) ||
      !ReplaceFieldWithConstantValue(eval_context, &call, &SemIR::Call::type_id,
                                     &phase)) {
    return SemIR::ConstantId::NotConstant;
  }

  auto callee_function =
      SemIR::GetCalleeFunction(eval_context.sem_ir(), call.callee_id);
  auto builtin_kind = SemIR::BuiltinFunctionKind::None;
  if (callee_function.function_id.is_valid()) {
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

  // If the arguments aren't constant, this is not a constant call.
  if (!ReplaceFieldWithConstantValue(eval_context, &call, &SemIR::Call::args_id,
                                     &phase)) {
    return SemIR::ConstantId::NotConstant;
  }
  if (phase == Phase::UnknownDueToError) {
    return SemIR::ConstantId::Error;
  }

  // Handle calls to builtins.
  if (builtin_kind != SemIR::BuiltinFunctionKind::None) {
    return MakeConstantForBuiltinCall(
        eval_context.context(), loc, call, builtin_kind,
        eval_context.inst_blocks().Get(call.args_id), phase);
  }

  return SemIR::ConstantId::NotConstant;
}

// Implementation for `TryEvalInst`, wrapping `Context` with `EvalContext`.
static auto TryEvalInstInContext(EvalContext& eval_context,
                                 SemIR::InstId inst_id, SemIR::Inst inst)
    -> SemIR::ConstantId {
  // TODO: Ensure we have test coverage for each of these cases that can result
  // in a constant, once those situations are all reachable.
  CARBON_KIND_SWITCH(inst) {
    // These cases are constants if their operands are.
    case SemIR::AddrOf::Kind:
      return RebuildIfFieldsAreConstant(eval_context, inst,
                                        &SemIR::AddrOf::type_id,
                                        &SemIR::AddrOf::lvalue_id);
    case CARBON_KIND(SemIR::ArrayType array_type): {
      return RebuildAndValidateIfFieldsAreConstant(
          eval_context, inst,
          [&](SemIR::ArrayType result) {
            auto bound_id = array_type.bound_id;
            auto int_bound = eval_context.insts().TryGetAs<SemIR::IntLiteral>(
                result.bound_id);
            if (!int_bound) {
              // TODO: Permit symbolic array bounds. This will require fixing
              // callers of `GetArrayBoundValue`.
              eval_context.context().TODO(bound_id, "symbolic array bound");
              return false;
            }
            // TODO: We should check that the size of the resulting array type
            // fits in 64 bits, not just that the bound does. Should we use a
            // 32-bit limit for 32-bit targets?
            const auto& bound_val = eval_context.ints().Get(int_bound->int_id);
            if (eval_context.types().IsSignedInt(int_bound->type_id) &&
                bound_val.isNegative()) {
              CARBON_DIAGNOSTIC(ArrayBoundNegative, Error,
                                "array bound of {0} is negative", TypedInt);
              eval_context.emitter().Emit(
                  bound_id, ArrayBoundNegative,
                  {.type = int_bound->type_id, .value = bound_val});
              return false;
            }
            if (bound_val.getActiveBits() > 64) {
              CARBON_DIAGNOSTIC(ArrayBoundTooLarge, Error,
                                "array bound of {0} is too large", TypedInt);
              eval_context.emitter().Emit(
                  bound_id, ArrayBoundTooLarge,
                  {.type = int_bound->type_id, .value = bound_val});
              return false;
            }
            return true;
          },
          &SemIR::ArrayType::bound_id, &SemIR::ArrayType::element_type_id);
    }
    case SemIR::AssociatedEntity::Kind:
      return RebuildIfFieldsAreConstant(eval_context, inst,
                                        &SemIR::AssociatedEntity::type_id);
    case SemIR::AssociatedEntityType::Kind:
      return RebuildIfFieldsAreConstant(
          eval_context, inst, &SemIR::AssociatedEntityType::interface_type_id,
          &SemIR::AssociatedEntityType::entity_type_id);
    case SemIR::BoundMethod::Kind:
      return RebuildIfFieldsAreConstant(
          eval_context, inst, &SemIR::BoundMethod::type_id,
          &SemIR::BoundMethod::object_id, &SemIR::BoundMethod::function_id);
    case SemIR::ClassType::Kind:
      return RebuildIfFieldsAreConstant(eval_context, inst,
                                        &SemIR::ClassType::specific_id);
    case SemIR::CompleteTypeWitness::Kind:
      return RebuildIfFieldsAreConstant(
          eval_context, inst, &SemIR::CompleteTypeWitness::object_repr_id);
    case SemIR::FunctionType::Kind:
      return RebuildIfFieldsAreConstant(eval_context, inst,
                                        &SemIR::FunctionType::specific_id);
    case SemIR::GenericClassType::Kind:
      return RebuildIfFieldsAreConstant(
          eval_context, inst, &SemIR::GenericClassType::enclosing_specific_id);
    case SemIR::GenericInterfaceType::Kind:
      return RebuildIfFieldsAreConstant(
          eval_context, inst,
          &SemIR::GenericInterfaceType::enclosing_specific_id);
    case SemIR::InterfaceType::Kind:
      return RebuildIfFieldsAreConstant(eval_context, inst,
                                        &SemIR::InterfaceType::specific_id);
    case SemIR::InterfaceWitness::Kind:
      return RebuildIfFieldsAreConstant(eval_context, inst,
                                        &SemIR::InterfaceWitness::elements_id);
    case CARBON_KIND(SemIR::IntType int_type): {
      return RebuildAndValidateIfFieldsAreConstant(
          eval_context, inst,
          [&](SemIR::IntType result) {
            return ValidateIntType(eval_context.context(),
                                   int_type.bit_width_id, result);
          },
          &SemIR::IntType::bit_width_id);
    }
    case SemIR::PointerType::Kind:
      return RebuildIfFieldsAreConstant(eval_context, inst,
                                        &SemIR::PointerType::pointee_id);
    case CARBON_KIND(SemIR::FloatType float_type): {
      return RebuildAndValidateIfFieldsAreConstant(
          eval_context, inst,
          [&](SemIR::FloatType result) {
            return ValidateFloatType(eval_context.context(),
                                     float_type.bit_width_id, result);
          },
          &SemIR::FloatType::bit_width_id);
    }
    case SemIR::StructType::Kind:
      return RebuildIfFieldsAreConstant(eval_context, inst,
                                        &SemIR::StructType::fields_id);
    case SemIR::StructTypeField::Kind:
      return RebuildIfFieldsAreConstant(eval_context, inst,
                                        &SemIR::StructTypeField::field_type_id);
    case SemIR::StructValue::Kind:
      return RebuildIfFieldsAreConstant(eval_context, inst,
                                        &SemIR::StructValue::type_id,
                                        &SemIR::StructValue::elements_id);
    case SemIR::TupleType::Kind:
      return RebuildIfFieldsAreConstant(eval_context, inst,
                                        &SemIR::TupleType::elements_id);
    case SemIR::TupleValue::Kind:
      return RebuildIfFieldsAreConstant(eval_context, inst,
                                        &SemIR::TupleValue::type_id,
                                        &SemIR::TupleValue::elements_id);
    case SemIR::UnboundElementType::Kind:
      return RebuildIfFieldsAreConstant(
          eval_context, inst, &SemIR::UnboundElementType::class_type_id,
          &SemIR::UnboundElementType::element_type_id);

    // Initializers evaluate to a value of the object representation.
    case SemIR::ArrayInit::Kind:
      // TODO: Add an `ArrayValue` to represent a constant array object
      // representation instead of using a `TupleValue`.
      return RebuildInitAsValue(eval_context, inst, SemIR::TupleValue::Kind);
    case SemIR::ClassInit::Kind:
      // TODO: Add a `ClassValue` to represent a constant class object
      // representation instead of using a `StructValue`.
      return RebuildInitAsValue(eval_context, inst, SemIR::StructValue::Kind);
    case SemIR::StructInit::Kind:
      return RebuildInitAsValue(eval_context, inst, SemIR::StructValue::Kind);
    case SemIR::TupleInit::Kind:
      return RebuildInitAsValue(eval_context, inst, SemIR::TupleValue::Kind);

    case SemIR::BuiltinInst::Kind:
      // Builtins are always template constants.
      return MakeConstantResult(eval_context.context(), inst, Phase::Template);

    case CARBON_KIND(SemIR::FunctionDecl fn_decl): {
      return TransformIfFieldsAreConstant(
          eval_context, fn_decl,
          [&](SemIR::FunctionDecl result) {
            return SemIR::StructValue{.type_id = result.type_id,
                                      .elements_id = SemIR::InstBlockId::Empty};
          },
          &SemIR::FunctionDecl::type_id);
    }

    case CARBON_KIND(SemIR::ClassDecl class_decl): {
      // If the class has generic parameters, we don't produce a class type, but
      // a callable whose return value is a class type.
      if (eval_context.classes().Get(class_decl.class_id).has_parameters()) {
        return TransformIfFieldsAreConstant(
            eval_context, class_decl,
            [&](SemIR::ClassDecl result) {
              return SemIR::StructValue{
                  .type_id = result.type_id,
                  .elements_id = SemIR::InstBlockId::Empty};
            },
            &SemIR::ClassDecl::type_id);
      }
      // A non-generic class declaration evaluates to the class type.
      return MakeConstantResult(
          eval_context.context(),
          SemIR::ClassType{.type_id = SemIR::TypeId::TypeType,
                           .class_id = class_decl.class_id,
                           .specific_id = SemIR::SpecificId::Invalid},
          Phase::Template);
    }

    case CARBON_KIND(SemIR::InterfaceDecl interface_decl): {
      // If the interface has generic parameters, we don't produce an interface
      // type, but a callable whose return value is an interface type.
      if (eval_context.interfaces()
              .Get(interface_decl.interface_id)
              .has_parameters()) {
        return TransformIfFieldsAreConstant(
            eval_context, interface_decl,
            [&](SemIR::InterfaceDecl result) {
              return SemIR::StructValue{
                  .type_id = result.type_id,
                  .elements_id = SemIR::InstBlockId::Empty};
            },
            &SemIR::InterfaceDecl::type_id);
      }
      // A non-generic interface declaration evaluates to the interface type.
      return MakeConstantResult(
          eval_context.context(),
          SemIR::InterfaceType{.type_id = SemIR::TypeId::TypeType,
                               .interface_id = interface_decl.interface_id,
                               .specific_id = SemIR::SpecificId::Invalid},
          Phase::Template);
    }

    case CARBON_KIND(SemIR::SpecificConstant specific): {
      // Pull the constant value out of the specific.
      return SemIR::GetConstantValueInSpecific(
          eval_context.sem_ir(), specific.specific_id, specific.inst_id);
    }

    // These cases are treated as being the unique canonical definition of the
    // corresponding constant value.
    // TODO: This doesn't properly handle redeclarations. Consider adding a
    // corresponding `Value` inst for each of these cases, or returning the
    // first declaration.
    case SemIR::AssociatedConstantDecl::Kind:
    case SemIR::BaseDecl::Kind:
    case SemIR::FieldDecl::Kind:
    case SemIR::ImplDecl::Kind:
    case SemIR::Namespace::Kind:
      return SemIR::ConstantId::ForTemplateConstant(inst_id);

    case SemIR::BoolLiteral::Kind:
    case SemIR::FloatLiteral::Kind:
    case SemIR::IntLiteral::Kind:
    case SemIR::StringLiteral::Kind:
      // Promote literals to the constant block.
      // TODO: Convert literals into a canonical form. Currently we can form two
      // different `i32` constants with the same value if they are represented
      // by `APInt`s with different bit widths.
      // TODO: Can the type of an IntLiteral or FloatLiteral be symbolic? If so,
      // we may need to rebuild.
      return MakeConstantResult(eval_context.context(), inst, Phase::Template);

    // The elements of a constant aggregate can be accessed.
    case SemIR::ClassElementAccess::Kind:
    case SemIR::InterfaceWitnessAccess::Kind:
    case SemIR::StructAccess::Kind:
    case SemIR::TupleAccess::Kind:
      return PerformAggregateAccess(eval_context, inst);
    case SemIR::ArrayIndex::Kind:
      return PerformArrayIndex(eval_context, inst);

    case CARBON_KIND(SemIR::Call call): {
      return MakeConstantForCall(eval_context, inst_id, call);
    }

    // TODO: These need special handling.
    case SemIR::BindValue::Kind:
    case SemIR::Deref::Kind:
    case SemIR::ImportRefLoaded::Kind:
    case SemIR::Temporary::Kind:
    case SemIR::TemporaryStorage::Kind:
    case SemIR::ValueAsRef::Kind:
      break;

    case CARBON_KIND(SemIR::BindSymbolicName bind): {
      const auto& bind_name =
          eval_context.entity_names().Get(bind.entity_name_id);

      // If we know which specific we're evaluating within and this is an
      // argument of that specific, its constant value is the corresponding
      // argument value.
      if (auto value =
              eval_context.GetCompileTimeBindValue(bind_name.bind_index);
          value.is_valid()) {
        return value;
      }

      // The constant form of a symbolic binding is an idealized form of the
      // original, with no equivalent value.
      bind.entity_name_id =
          eval_context.entity_names().MakeCanonical(bind.entity_name_id);
      bind.value_id = SemIR::InstId::Invalid;
      return MakeConstantResult(eval_context.context(), bind, Phase::Symbolic);
    }

    // These semantic wrappers don't change the constant value.
    case CARBON_KIND(SemIR::AsCompatible inst): {
      return eval_context.GetConstantValue(inst.source_id);
    }
    case CARBON_KIND(SemIR::BindAlias typed_inst): {
      return eval_context.GetConstantValue(typed_inst.value_id);
    }
    case CARBON_KIND(SemIR::ExportDecl typed_inst): {
      return eval_context.GetConstantValue(typed_inst.value_id);
    }
    case CARBON_KIND(SemIR::NameRef typed_inst): {
      return eval_context.GetConstantValue(typed_inst.value_id);
    }
    case CARBON_KIND(SemIR::Converted typed_inst): {
      return eval_context.GetConstantValue(typed_inst.result_id);
    }
    case CARBON_KIND(SemIR::InitializeFrom typed_inst): {
      return eval_context.GetConstantValue(typed_inst.src_id);
    }
    case CARBON_KIND(SemIR::SpliceBlock typed_inst): {
      return eval_context.GetConstantValue(typed_inst.result_id);
    }
    case CARBON_KIND(SemIR::ValueOfInitializer typed_inst): {
      return eval_context.GetConstantValue(typed_inst.init_id);
    }
    case CARBON_KIND(SemIR::FacetTypeAccess typed_inst): {
      // TODO: Once we start tracking the witness in the facet value, remove it
      // here. For now, we model a facet value as just a type.
      return eval_context.GetConstantValue(typed_inst.facet_id);
    }
    case CARBON_KIND(SemIR::WhereExpr typed_inst): {
      // TODO: This currently ignores the requirements and just produces the
      // left-hand type argument to the `where`.
      return eval_context.GetConstantValue(
          eval_context.insts().Get(typed_inst.period_self_id).type_id());
    }

    // `not true` -> `false`, `not false` -> `true`.
    // All other uses of unary `not` are non-constant.
    case CARBON_KIND(SemIR::UnaryOperatorNot typed_inst): {
      auto const_id = eval_context.GetConstantValue(typed_inst.operand_id);
      auto phase = GetPhase(const_id);
      if (phase == Phase::Template) {
        auto value = eval_context.insts().GetAs<SemIR::BoolLiteral>(
            eval_context.constant_values().GetInstId(const_id));
        return MakeBoolResult(eval_context.context(), value.type_id,
                              !value.value.ToBool());
      }
      if (phase == Phase::UnknownDueToError) {
        return SemIR::ConstantId::Error;
      }
      break;
    }

    // `const (const T)` evaluates to `const T`. Otherwise, `const T` evaluates
    // to itself.
    case CARBON_KIND(SemIR::ConstType typed_inst): {
      auto inner_id = eval_context.GetConstantValue(typed_inst.inner_id);
      if (inner_id.is_constant() &&
          eval_context.insts()
              .Get(eval_context.constant_values().GetInstId(inner_id))
              .Is<SemIR::ConstType>()) {
        return inner_id;
      }
      return MakeConstantResult(eval_context.context(), inst,
                                GetPhase(inner_id));
    }

    // These cases are either not expressions or not constant.
    case SemIR::AdaptDecl::Kind:
    case SemIR::AddrPattern::Kind:
    case SemIR::Assign::Kind:
    case SemIR::BindName::Kind:
    case SemIR::BindingPattern::Kind:
    case SemIR::BlockArg::Kind:
    case SemIR::Branch::Kind:
    case SemIR::BranchIf::Kind:
    case SemIR::BranchWithArg::Kind:
    case SemIR::ImportDecl::Kind:
    case SemIR::Param::Kind:
    case SemIR::RequirementEquivalent::Kind:
    case SemIR::RequirementImpls::Kind:
    case SemIR::RequirementRewrite::Kind:
    case SemIR::ReturnExpr::Kind:
    case SemIR::Return::Kind:
    case SemIR::StructLiteral::Kind:
    case SemIR::SymbolicBindingPattern::Kind:
    case SemIR::TupleLiteral::Kind:
    case SemIR::VarStorage::Kind:
      break;

    case SemIR::ImportRefUnloaded::Kind:
      CARBON_FATAL("ImportRefUnloaded should be loaded before TryEvalInst: {0}",
                   inst);
  }
  return SemIR::ConstantId::NotConstant;
}

auto TryEvalInst(Context& context, SemIR::InstId inst_id, SemIR::Inst inst)
    -> SemIR::ConstantId {
  EvalContext eval_context(context);
  return TryEvalInstInContext(eval_context, inst_id, inst);
}

auto TryEvalBlockForSpecific(Context& context, SemIR::SpecificId specific_id,
                             SemIR::GenericInstIndex::Region region)
    -> SemIR::InstBlockId {
  auto generic_id = context.specifics().Get(specific_id).generic_id;
  auto eval_block_id = context.generics().Get(generic_id).GetEvalBlock(region);
  auto eval_block = context.inst_blocks().Get(eval_block_id);

  llvm::SmallVector<SemIR::InstId> result;
  result.resize(eval_block.size(), SemIR::InstId::Invalid);

  EvalContext eval_context(context, specific_id,
                           SpecificEvalInfo{
                               .region = region,
                               .values = result,
                           });

  for (auto [i, inst_id] : llvm::enumerate(eval_block)) {
    auto const_id = TryEvalInstInContext(eval_context, inst_id,
                                         context.insts().Get(inst_id));
    result[i] = context.constant_values().GetInstId(const_id);

    // TODO: If this becomes possible through monomorphization failure, produce
    // a diagnostic and put `SemIR::InstId::BuiltinError` in the table entry.
    CARBON_CHECK(result[i].is_valid());
  }

  return context.inst_blocks().Add(result);
}

}  // namespace Carbon::Check
