// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/eval.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/sem_ir/builtin_function_kind.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

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
      context, SemIR::BoolLiteral{bool_type_id, SemIR::BoolValue::From(result)},
      Phase::Template);
}

// Converts an APInt value into a ConstantId.
static auto MakeIntResult(Context& context, SemIR::TypeId type_id,
                          llvm::APInt value) -> SemIR::ConstantId {
  auto result = context.ints().Add(std::move(value));
  return MakeConstantResult(context, SemIR::IntLiteral{type_id, result},
                            Phase::Template);
}

// `GetConstantValue` checks to see whether the provided ID describes a value
// with constant phase, and if so, returns the corresponding constant value.
// Overloads are provided for different kinds of ID.

// If the given instruction is constant, returns its constant value.
static auto GetConstantValue(Context& context, SemIR::InstId inst_id,
                             Phase* phase) -> SemIR::InstId {
  auto const_id = context.constant_values().Get(inst_id);
  *phase = LatestPhase(*phase, GetPhase(const_id));
  return const_id.inst_id();
}

// A type is always constant, but we still need to extract its phase.
static auto GetConstantValue(Context& context, SemIR::TypeId type_id,
                             Phase* phase) -> SemIR::TypeId {
  auto const_id = context.types().GetConstantId(type_id);
  *phase = LatestPhase(*phase, GetPhase(const_id));
  return type_id;
}

// If the given instruction block contains only constants, returns a
// corresponding block of those values.
static auto GetConstantValue(Context& context, SemIR::InstBlockId inst_block_id,
                             Phase* phase) -> SemIR::InstBlockId {
  auto insts = context.inst_blocks().Get(inst_block_id);
  llvm::SmallVector<SemIR::InstId> const_insts;
  for (auto inst_id : insts) {
    auto const_inst_id = GetConstantValue(context, inst_id, phase);
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
  // TODO: If the new block is identical to the original block, return the
  // original ID.
  return context.inst_blocks().Add(const_insts);
}

// The constant value of a type block is that type block, but we still need to
// extract its phase.
static auto GetConstantValue(Context& context, SemIR::TypeBlockId type_block_id,
                             Phase* phase) -> SemIR::TypeBlockId {
  auto types = context.type_blocks().Get(type_block_id);
  for (auto type_id : types) {
    GetConstantValue(context, type_id, phase);
  }
  return type_block_id;
}

// Replaces the specified field of the given typed instruction with its constant
// value, if it has constant phase. Returns true on success, false if the value
// has runtime phase.
template <typename InstT, typename FieldIdT>
static auto ReplaceFieldWithConstantValue(Context& context, InstT* inst,
                                          FieldIdT InstT::*field, Phase* phase)
    -> bool {
  auto unwrapped = GetConstantValue(context, inst->*field, phase);
  if (!unwrapped.is_valid()) {
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
// validation passes, a corresponding ConstantId for the new constant is
// returned. If validation fails, it should produce a suitable error message.
// `ConstantId::Error` is returned.
template <typename InstT, typename ValidateFn, typename... EachFieldIdT>
static auto RebuildAndValidateIfFieldsAreConstant(
    Context& context, SemIR::Inst inst, ValidateFn validate_fn,
    EachFieldIdT InstT::*... each_field_id) -> SemIR::ConstantId {
  // Build a constant instruction by replacing each non-constant operand with
  // its constant value.
  auto typed_inst = inst.As<InstT>();
  Phase phase = Phase::Template;
  if ((ReplaceFieldWithConstantValue(context, &typed_inst, each_field_id,
                                     &phase) &&
       ...)) {
    if (phase == Phase::UnknownDueToError || !validate_fn(typed_inst)) {
      return SemIR::ConstantId::Error;
    }
    return MakeConstantResult(context, typed_inst, phase);
  }
  return MakeNonConstantResult(phase);
}

// Same as above but with no validation step.
template <typename InstT, typename... EachFieldIdT>
static auto RebuildIfFieldsAreConstant(Context& context, SemIR::Inst inst,
                                       EachFieldIdT InstT::*... each_field_id)
    -> SemIR::ConstantId {
  return RebuildAndValidateIfFieldsAreConstant(
      context, inst, [](...) { return true; }, each_field_id...);
}

// Rebuilds the given aggregate initialization instruction as a corresponding
// constant aggregate value, if its elements are all constants.
static auto RebuildInitAsValue(Context& context, SemIR::Inst inst,
                               SemIR::InstKind value_kind)
    -> SemIR::ConstantId {
  auto init_inst = inst.As<SemIR::AnyAggregateInit>();
  Phase phase = Phase::Template;
  auto elements_id = GetConstantValue(context, init_inst.elements_id, &phase);
  return MakeConstantResult(
      context,
      SemIR::AnyAggregateValue{.kind = value_kind,
                               .type_id = init_inst.type_id,
                               .elements_id = elements_id},
      phase);
}

// Performs an access into an aggregate, retrieving the specified element.
static auto PerformAggregateAccess(Context& context, SemIR::Inst inst)
    -> SemIR::ConstantId {
  auto access_inst = inst.As<SemIR::AnyAggregateAccess>();
  Phase phase = Phase::Template;
  if (auto aggregate_id =
          GetConstantValue(context, access_inst.aggregate_id, &phase);
      aggregate_id.is_valid()) {
    if (auto aggregate =
            context.insts().TryGetAs<SemIR::AnyAggregateValue>(aggregate_id)) {
      auto elements = context.inst_blocks().Get(aggregate->elements_id);
      auto index = static_cast<size_t>(access_inst.index.index);
      CARBON_CHECK(index < elements.size()) << "Access out of bounds.";
      // `Phase` is not used here. If this element is a template constant, then
      // so is the result of indexing, even if the aggregate also contains a
      // symbolic context.
      return context.constant_values().Get(elements[index]);
    } else {
      CARBON_CHECK(phase != Phase::Template)
          << "Failed to evaluate template constant " << inst;
    }
  }
  return MakeNonConstantResult(phase);
}

// Performs an index into a homogeneous aggregate, retrieving the specified
// element.
static auto PerformAggregateIndex(Context& context, SemIR::Inst inst)
    -> SemIR::ConstantId {
  auto index_inst = inst.As<SemIR::AnyAggregateIndex>();
  Phase phase = Phase::Template;
  auto aggregate_id =
      GetConstantValue(context, index_inst.aggregate_id, &phase);
  auto index_id = GetConstantValue(context, index_inst.index_id, &phase);

  if (!index_id.is_valid()) {
    return MakeNonConstantResult(phase);
  }
  auto index = context.insts().TryGetAs<SemIR::IntLiteral>(index_id);
  if (!index) {
    CARBON_CHECK(phase != Phase::Template)
        << "Template constant integer should be a literal";
    return MakeNonConstantResult(phase);
  }

  // Array indexing is invalid if the index is constant and out of range.
  auto aggregate_type_id =
      context.insts().Get(index_inst.aggregate_id).type_id();
  const auto& index_val = context.ints().Get(index->int_id);
  if (auto array_type =
          context.types().TryGetAs<SemIR::ArrayType>(aggregate_type_id)) {
    if (auto bound =
            context.insts().TryGetAs<SemIR::IntLiteral>(array_type->bound_id)) {
      // This awkward call to `getZExtValue` is a workaround for APInt not
      // supporting comparisons between integers of different bit widths.
      if (index_val.getActiveBits() > 64 ||
          context.ints().Get(bound->int_id).ule(index_val.getZExtValue())) {
        CARBON_DIAGNOSTIC(ArrayIndexOutOfBounds, Error,
                          "Array index `{0}` is past the end of type `{1}`.",
                          TypedInt, SemIR::TypeId);
        context.emitter().Emit(index_inst.index_id, ArrayIndexOutOfBounds,
                               TypedInt{index->type_id, index_val},
                               aggregate_type_id);
        return SemIR::ConstantId::Error;
      }
    }
  }

  if (!aggregate_id.is_valid()) {
    return MakeNonConstantResult(phase);
  }
  auto aggregate =
      context.insts().TryGetAs<SemIR::AnyAggregateValue>(aggregate_id);
  if (!aggregate) {
    CARBON_CHECK(phase != Phase::Template)
        << "Unexpected representation for template constant aggregate";
    return MakeNonConstantResult(phase);
  }

  auto elements = context.inst_blocks().Get(aggregate->elements_id);
  // We checked this for the array case above.
  CARBON_CHECK(index_val.ult(elements.size()))
      << "Index out of bounds in tuple indexing";
  return context.constant_values().Get(elements[index_val.getZExtValue()]);
}

// Enforces that an integer type has a valid bit width.
auto ValidateIntType(Context& context, SemIRLoc loc, SemIR::IntType result)
    -> bool {
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
                      "Integer type width of {0} is not positive.", TypedInt);
    context.emitter().Emit(loc, IntWidthNotPositive,
                           TypedInt{bit_width->type_id, bit_width_val});
    return false;
  }
  // TODO: Pick a maximum size and document it in the design. For now
  // we use 2^^23, because that's the largest size that LLVM supports.
  constexpr int MaxIntWidth = 1 << 23;
  if (bit_width_val.ugt(MaxIntWidth)) {
    CARBON_DIAGNOSTIC(IntWidthTooLarge, Error,
                      "Integer type width of {0} is greater than the "
                      "maximum supported width of {1}.",
                      TypedInt, int);
    context.emitter().Emit(loc, IntWidthTooLarge,
                           TypedInt{bit_width->type_id, bit_width_val},
                           MaxIntWidth);
    return false;
  }
  return true;
}

// Forms a constant int type as an evaluation result. Requires that width_id is
// constant.
auto MakeIntTypeResult(Context& context, SemIRLoc loc, SemIR::IntKind int_kind,
                       SemIR::InstId width_id, Phase phase)
    -> SemIR::ConstantId {
  auto result = SemIR::IntType{
      .type_id = context.GetBuiltinType(SemIR::BuiltinKind::TypeType),
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

  CARBON_DIAGNOSTIC(CompileTimeFloatBitWidth, Error, "Bit width must be 64.");
  context.emitter().Emit(loc, CompileTimeFloatBitWidth);
  return false;
}

// Issues a diagnostic for a compile-time division by zero.
static auto DiagnoseDivisionByZero(Context& context, SemIRLoc loc) -> void {
  CARBON_DIAGNOSTIC(CompileTimeDivisionByZero, Error, "Division by zero.");
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
                          "Integer overflow in negation of {0}.", TypedInt);
        context.emitter().Emit(loc, CompileTimeIntegerNegateOverflow,
                               TypedInt{op.type_id, op_val});
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
      CARBON_FATAL() << "Unexpected builtin kind";
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
        CARBON_DIAGNOSTIC(
            CompileTimeShiftOutOfRange, Error,
            "Shift distance not in range [0, {0}) in {1} {2} {3}.", unsigned,
            TypedInt, llvm::StringLiteral, TypedInt);
        context.emitter().Emit(loc, CompileTimeShiftOutOfRange,
                               lhs_val.getBitWidth(),
                               TypedInt{lhs.type_id, lhs_val}, op_str,
                               TypedInt{rhs.type_id, rhs_val});
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
      CARBON_FATAL() << "Unexpected operation kind.";
  }

  if (overflow) {
    CARBON_DIAGNOSTIC(CompileTimeIntegerOverflow, Error,
                      "Integer overflow in calculation {0} {1} {2}.", TypedInt,
                      llvm::StringLiteral, TypedInt);
    context.emitter().Emit(loc, CompileTimeIntegerOverflow,
                           TypedInt{lhs.type_id, lhs_val}, op_str,
                           TypedInt{rhs.type_id, rhs_val});
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
      CARBON_FATAL() << "Unexpected operation kind.";
  }

  return MakeBoolResult(context, bool_type_id, result);
}

static auto PerformBuiltinCall(Context& context, SemIRLoc loc, SemIR::Call call,
                               SemIR::BuiltinFunctionKind builtin_kind,
                               llvm::ArrayRef<SemIR::InstId> arg_ids,
                               Phase phase) -> SemIR::ConstantId {
  switch (builtin_kind) {
    case SemIR::BuiltinFunctionKind::None:
      CARBON_FATAL() << "Not a builtin function.";

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
  }

  return SemIR::ConstantId::NotConstant;
}

static auto PerformCall(Context& context, SemIRLoc loc, SemIR::Call call)
    -> SemIR::ConstantId {
  Phase phase = Phase::Template;

  // A call with an invalid argument list is used to represent an erroneous
  // call.
  //
  // TODO: Use a better representation for this.
  if (call.args_id == SemIR::InstBlockId::Invalid) {
    return SemIR::ConstantId::Error;
  }

  // If the callee isn't constant, this is not a constant call.
  if (!ReplaceFieldWithConstantValue(context, &call, &SemIR::Call::callee_id,
                                     &phase)) {
    return SemIR::ConstantId::NotConstant;
  }

  // Handle calls to builtins.
  if (auto builtin_function_kind = SemIR::BuiltinFunctionKind::ForCallee(
          context.sem_ir(), call.callee_id);
      builtin_function_kind != SemIR::BuiltinFunctionKind::None) {
    if (!ReplaceFieldWithConstantValue(context, &call, &SemIR::Call::args_id,
                                       &phase)) {
      return SemIR::ConstantId::NotConstant;
    }
    if (phase == Phase::UnknownDueToError) {
      return SemIR::ConstantId::Error;
    }
    return PerformBuiltinCall(context, loc, call, builtin_function_kind,
                              context.inst_blocks().Get(call.args_id), phase);
  }
  return SemIR::ConstantId::NotConstant;
}

auto TryEvalInst(Context& context, SemIR::InstId inst_id, SemIR::Inst inst)
    -> SemIR::ConstantId {
  // TODO: Ensure we have test coverage for each of these cases that can result
  // in a constant, once those situations are all reachable.
  CARBON_KIND_SWITCH(inst) {
    // These cases are constants if their operands are.
    case SemIR::AddrOf::Kind:
      return RebuildIfFieldsAreConstant(context, inst,
                                        &SemIR::AddrOf::lvalue_id);
    case CARBON_KIND(SemIR::ArrayType array_type): {
      return RebuildAndValidateIfFieldsAreConstant(
          context, inst,
          [&](SemIR::ArrayType result) {
            auto bound_id = array_type.bound_id;
            auto int_bound =
                context.insts().TryGetAs<SemIR::IntLiteral>(result.bound_id);
            if (!int_bound) {
              // TODO: Permit symbolic array bounds. This will require fixing
              // callers of `GetArrayBoundValue`.
              context.TODO(bound_id, "symbolic array bound");
              return false;
            }
            // TODO: We should check that the size of the resulting array type
            // fits in 64 bits, not just that the bound does. Should we use a
            // 32-bit limit for 32-bit targets?
            const auto& bound_val = context.ints().Get(int_bound->int_id);
            if (context.types().IsSignedInt(int_bound->type_id) &&
                bound_val.isNegative()) {
              CARBON_DIAGNOSTIC(ArrayBoundNegative, Error,
                                "Array bound of {0} is negative.", TypedInt);
              context.emitter().Emit(bound_id, ArrayBoundNegative,
                                     TypedInt{int_bound->type_id, bound_val});
              return false;
            }
            if (bound_val.getActiveBits() > 64) {
              CARBON_DIAGNOSTIC(ArrayBoundTooLarge, Error,
                                "Array bound of {0} is too large.", TypedInt);
              context.emitter().Emit(bound_id, ArrayBoundTooLarge,
                                     TypedInt{int_bound->type_id, bound_val});
              return false;
            }
            return true;
          },
          &SemIR::ArrayType::bound_id, &SemIR::ArrayType::element_type_id);
    }
    case SemIR::AssociatedEntityType::Kind:
      return RebuildIfFieldsAreConstant(
          context, inst, &SemIR::AssociatedEntityType::entity_type_id);
    case SemIR::BoundMethod::Kind:
      return RebuildIfFieldsAreConstant(context, inst,
                                        &SemIR::BoundMethod::object_id,
                                        &SemIR::BoundMethod::function_id);
    case SemIR::InterfaceWitness::Kind:
      return RebuildIfFieldsAreConstant(context, inst,
                                        &SemIR::InterfaceWitness::elements_id);
    case CARBON_KIND(SemIR::IntType int_type): {
      return RebuildAndValidateIfFieldsAreConstant(
          context, inst,
          [&](SemIR::IntType result) {
            return ValidateIntType(context, int_type.bit_width_id, result);
          },
          &SemIR::IntType::bit_width_id);
    }
    case SemIR::PointerType::Kind:
      return RebuildIfFieldsAreConstant(context, inst,
                                        &SemIR::PointerType::pointee_id);
    case SemIR::StructType::Kind:
      return RebuildIfFieldsAreConstant(context, inst,
                                        &SemIR::StructType::fields_id);
    case SemIR::StructTypeField::Kind:
      return RebuildIfFieldsAreConstant(context, inst,
                                        &SemIR::StructTypeField::field_type_id);
    case SemIR::StructValue::Kind:
      return RebuildIfFieldsAreConstant(context, inst,
                                        &SemIR::StructValue::elements_id);
    case SemIR::TupleType::Kind:
      return RebuildIfFieldsAreConstant(context, inst,
                                        &SemIR::TupleType::elements_id);
    case SemIR::TupleValue::Kind:
      return RebuildIfFieldsAreConstant(context, inst,
                                        &SemIR::TupleValue::elements_id);
    case SemIR::UnboundElementType::Kind:
      return RebuildIfFieldsAreConstant(
          context, inst, &SemIR::UnboundElementType::class_type_id,
          &SemIR::UnboundElementType::element_type_id);

    // Initializers evaluate to a value of the object representation.
    case SemIR::ArrayInit::Kind:
      // TODO: Add an `ArrayValue` to represent a constant array object
      // representation instead of using a `TupleValue`.
      return RebuildInitAsValue(context, inst, SemIR::TupleValue::Kind);
    case SemIR::ClassInit::Kind:
      // TODO: Add a `ClassValue` to represent a constant class object
      // representation instead of using a `StructValue`.
      return RebuildInitAsValue(context, inst, SemIR::StructValue::Kind);
    case SemIR::StructInit::Kind:
      return RebuildInitAsValue(context, inst, SemIR::StructValue::Kind);
    case SemIR::TupleInit::Kind:
      return RebuildInitAsValue(context, inst, SemIR::TupleValue::Kind);

    case SemIR::AssociatedEntity::Kind:
    case SemIR::Builtin::Kind:
      // Builtins are always template constants.
      return MakeConstantResult(context, inst, Phase::Template);

    case CARBON_KIND(SemIR::ClassDecl class_decl): {
      // TODO: Once classes have generic arguments, handle them.
      return MakeConstantResult(
          context,
          SemIR::ClassType{SemIR::TypeId::TypeType, class_decl.class_id},
          Phase::Template);
    }
    case CARBON_KIND(SemIR::InterfaceDecl interface_decl): {
      // TODO: Once interfaces have generic arguments, handle them.
      return MakeConstantResult(
          context,
          SemIR::InterfaceType{SemIR::TypeId::TypeType,
                               interface_decl.interface_id},
          Phase::Template);
    }

    case SemIR::ClassType::Kind:
    case SemIR::InterfaceType::Kind:
      CARBON_FATAL() << inst.kind()
                     << " is only created during corresponding Decl handling.";

    // These cases are treated as being the unique canonical definition of the
    // corresponding constant value.
    // TODO: This doesn't properly handle redeclarations. Consider adding a
    // corresponding `Value` inst for each of these cases.
    case SemIR::AssociatedConstantDecl::Kind:
    case SemIR::BaseDecl::Kind:
    case SemIR::FieldDecl::Kind:
    case SemIR::FunctionDecl::Kind:
    case SemIR::Namespace::Kind:
      return SemIR::ConstantId::ForTemplateConstant(inst_id);

    case SemIR::BoolLiteral::Kind:
    case SemIR::IntLiteral::Kind:
    case SemIR::RealLiteral::Kind:
    case SemIR::StringLiteral::Kind:
      // Promote literals to the constant block.
      // TODO: Convert literals into a canonical form. Currently we can form two
      // different `i32` constants with the same value if they are represented
      // by `APInt`s with different bit widths.
      return MakeConstantResult(context, inst, Phase::Template);

    // The elements of a constant aggregate can be accessed.
    case SemIR::ClassElementAccess::Kind:
    case SemIR::InterfaceWitnessAccess::Kind:
    case SemIR::StructAccess::Kind:
    case SemIR::TupleAccess::Kind:
      return PerformAggregateAccess(context, inst);
    case SemIR::ArrayIndex::Kind:
    case SemIR::TupleIndex::Kind:
      return PerformAggregateIndex(context, inst);

    case CARBON_KIND(SemIR::Call call): {
      return PerformCall(context, inst_id, call);
    }

    // TODO: These need special handling.
    case SemIR::BindValue::Kind:
    case SemIR::Deref::Kind:
    case SemIR::ImportRefLoaded::Kind:
    case SemIR::ImportRefUsed::Kind:
    case SemIR::Temporary::Kind:
    case SemIR::TemporaryStorage::Kind:
    case SemIR::ValueAsRef::Kind:
      break;

    case SemIR::BindSymbolicName::Kind:
      // TODO: Consider forming a constant value here using a de Bruijn index or
      // similar, so that corresponding symbolic parameters in redeclarations
      // are treated as the same value.
      return SemIR::ConstantId::ForSymbolicConstant(inst_id);

    // These semantic wrappers don't change the constant value.
    case CARBON_KIND(SemIR::AsCompatible inst): {
      return context.constant_values().Get(inst.source_id);
    }
    case CARBON_KIND(SemIR::BindAlias typed_inst): {
      return context.constant_values().Get(typed_inst.value_id);
    }
    case CARBON_KIND(SemIR::NameRef typed_inst): {
      return context.constant_values().Get(typed_inst.value_id);
    }
    case CARBON_KIND(SemIR::Converted typed_inst): {
      return context.constant_values().Get(typed_inst.result_id);
    }
    case CARBON_KIND(SemIR::InitializeFrom typed_inst): {
      return context.constant_values().Get(typed_inst.src_id);
    }
    case CARBON_KIND(SemIR::SpliceBlock typed_inst): {
      return context.constant_values().Get(typed_inst.result_id);
    }
    case CARBON_KIND(SemIR::ValueOfInitializer typed_inst): {
      return context.constant_values().Get(typed_inst.init_id);
    }
    case CARBON_KIND(SemIR::FacetTypeAccess typed_inst): {
      // TODO: Once we start tracking the witness in the facet value, remove it
      // here. For now, we model a facet value as just a type.
      return context.constant_values().Get(typed_inst.facet_id);
    }

    // `not true` -> `false`, `not false` -> `true`.
    // All other uses of unary `not` are non-constant.
    case CARBON_KIND(SemIR::UnaryOperatorNot typed_inst): {
      auto const_id = context.constant_values().Get(typed_inst.operand_id);
      auto phase = GetPhase(const_id);
      if (phase == Phase::Template) {
        auto value =
            context.insts().GetAs<SemIR::BoolLiteral>(const_id.inst_id());
        return MakeBoolResult(context, value.type_id, !value.value.ToBool());
      }
      if (phase == Phase::UnknownDueToError) {
        return SemIR::ConstantId::Error;
      }
      break;
    }

    // `const (const T)` evaluates to `const T`. Otherwise, `const T` evaluates
    // to itself.
    case CARBON_KIND(SemIR::ConstType typed_inst): {
      auto inner_id = context.constant_values().Get(
          context.types().GetInstId(typed_inst.inner_id));
      if (inner_id.is_constant() &&
          context.insts().Get(inner_id.inst_id()).Is<SemIR::ConstType>()) {
        return inner_id;
      }
      return MakeConstantResult(context, inst, GetPhase(inner_id));
    }

    // These cases are either not expressions or not constant.
    case SemIR::AdaptDecl::Kind:
    case SemIR::AddrPattern::Kind:
    case SemIR::Assign::Kind:
    case SemIR::BindName::Kind:
    case SemIR::BlockArg::Kind:
    case SemIR::Branch::Kind:
    case SemIR::BranchIf::Kind:
    case SemIR::BranchWithArg::Kind:
    case SemIR::ImplDecl::Kind:
    case SemIR::Param::Kind:
    case SemIR::ReturnExpr::Kind:
    case SemIR::Return::Kind:
    case SemIR::StructLiteral::Kind:
    case SemIR::TupleLiteral::Kind:
    case SemIR::VarStorage::Kind:
      break;

    case SemIR::ImportRefUnloaded::Kind:
      CARBON_FATAL()
          << "ImportRefUnloaded should be loaded before TryEvalInst.";
  }
  return SemIR::ConstantId::NotConstant;
}

}  // namespace Carbon::Check
