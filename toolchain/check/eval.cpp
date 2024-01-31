// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/eval.h"

#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"
#include "toolchain/sem_ir/value_stores.h"

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
    if (!validate_fn(typed_inst)) {
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
                          llvm::APSInt, std::string);
        context.emitter().Emit(
            index_inst.index_id, ArrayIndexOutOfBounds,
            llvm::APSInt(index_val, /*isUnsigned=*/true),
            context.sem_ir().StringifyType(aggregate_type_id));
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

auto TryEvalInst(Context& context, SemIR::InstId inst_id, SemIR::Inst inst)
    -> SemIR::ConstantId {
  // TODO: Ensure we have test coverage for each of these cases that can result
  // in a constant, once those situations are all reachable.
  switch (inst.kind()) {
    // These cases are constants if their operands are.
    case SemIR::AddrOf::Kind:
      return RebuildIfFieldsAreConstant(context, inst,
                                        &SemIR::AddrOf::lvalue_id);
    case SemIR::ArrayType::Kind:
      return RebuildAndValidateIfFieldsAreConstant(
          context, inst,
          [&](SemIR::ArrayType result) {
            auto bound_id = inst.As<SemIR::ArrayType>().bound_id;
            auto int_bound =
                context.insts().TryGetAs<SemIR::IntLiteral>(result.bound_id);
            if (!int_bound) {
              // TODO: Permit symbolic array bounds. This will require fixing
              // callers of `GetArrayBoundValue`.
              context.TODO(context.insts().GetParseNode(bound_id),
                           "symbolic array bound");
              return false;
            }
            // TODO: We should check that the size of the resulting array type
            // fits in 64 bits, not just that the bound does. Should we use a
            // 32-bit limit for 32-bit targets?
            // TODO: Also check for a negative bound, once that's something we
            // can represent.
            const auto& bound_val = context.ints().Get(int_bound->int_id);
            if (bound_val.getActiveBits() > 64) {
              CARBON_DIAGNOSTIC(ArrayBoundTooLarge, Error,
                                "Array bound of {0} is too large.",
                                llvm::APInt);
              context.emitter().Emit(bound_id, ArrayBoundTooLarge, bound_val);
              return false;
            }
            return true;
          },
          &SemIR::ArrayType::bound_id, &SemIR::ArrayType::element_type_id);
    case SemIR::BoundMethod::Kind:
      return RebuildIfFieldsAreConstant(context, inst,
                                        &SemIR::BoundMethod::object_id,
                                        &SemIR::BoundMethod::function_id);
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

    // These cases are always template constants.
    case SemIR::Builtin::Kind:
    case SemIR::ClassType::Kind:
    case SemIR::InterfaceType::Kind:
      // TODO: Once classes and interfaces have generic arguments, handle them.
      return MakeConstantResult(context, inst, Phase::Template);

    // These cases are treated as being the unique canonical definition of the
    // corresponding constant value.
    // TODO: This doesn't properly handle redeclarations. Consider adding a
    // corresponding `Value` inst for each of these cases.
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
    case SemIR::StructAccess::Kind:
    case SemIR::TupleAccess::Kind:
      return PerformAggregateAccess(context, inst);
    case SemIR::ArrayIndex::Kind:
    case SemIR::TupleIndex::Kind:
      return PerformAggregateIndex(context, inst);

    // TODO: These need special handling.
    case SemIR::BindValue::Kind:
    case SemIR::Call::Kind:
    case SemIR::Deref::Kind:
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

    // These semnatic wrappers don't change the constant value.
    case SemIR::NameRef::Kind:
      return context.constant_values().Get(inst.As<SemIR::NameRef>().value_id);
    case SemIR::Converted::Kind:
      return context.constant_values().Get(
          inst.As<SemIR::Converted>().result_id);
    case SemIR::InitializeFrom::Kind:
      return context.constant_values().Get(
          inst.As<SemIR::InitializeFrom>().src_id);
    case SemIR::SpliceBlock::Kind:
      return context.constant_values().Get(
          inst.As<SemIR::SpliceBlock>().result_id);
    case SemIR::ValueOfInitializer::Kind:
      return context.constant_values().Get(
          inst.As<SemIR::ValueOfInitializer>().init_id);

    // `not true` -> `false`, `not false` -> `true`.
    // All other uses of unary `not` are non-constant.
    case SemIR::UnaryOperatorNot::Kind: {
      auto const_id = context.constant_values().Get(
          inst.As<SemIR::UnaryOperatorNot>().operand_id);
      auto phase = GetPhase(const_id);
      if (phase == Phase::Template) {
        auto value =
            context.insts().GetAs<SemIR::BoolLiteral>(const_id.inst_id());
        value.value =
            (value.value == SemIR::BoolValue::False ? SemIR::BoolValue::True
                                                    : SemIR::BoolValue::False);
        return MakeConstantResult(context, value, Phase::Template);
      }
      if (phase == Phase::UnknownDueToError) {
        return SemIR::ConstantId::Error;
      }
      break;
    }

    // `const (const T)` evaluates to `const T`. Otherwise, `const T` evaluates
    // to itself.
    case SemIR::ConstType::Kind: {
      auto inner_id = context.constant_values().Get(
          context.types().GetInstId(inst.As<SemIR::ConstType>().inner_id));
      if (inner_id.is_constant() &&
          context.insts().Get(inner_id.inst_id()).Is<SemIR::ConstType>()) {
        return inner_id;
      }
      return MakeConstantResult(context, inst, GetPhase(inner_id));
    }

    // These cases are either not expressions or not constant.
    case SemIR::AddrPattern::Kind:
    case SemIR::Assign::Kind:
    case SemIR::BindName::Kind:
    case SemIR::BlockArg::Kind:
    case SemIR::Branch::Kind:
    case SemIR::BranchIf::Kind:
    case SemIR::BranchWithArg::Kind:
    case SemIR::ClassDecl::Kind:
    case SemIR::Import::Kind:
    case SemIR::InterfaceDecl::Kind:
    case SemIR::Param::Kind:
    case SemIR::ReturnExpr::Kind:
    case SemIR::Return::Kind:
    case SemIR::StructLiteral::Kind:
    case SemIR::TupleLiteral::Kind:
    case SemIR::VarStorage::Kind:
      break;

    case SemIR::ImportRefUnused::Kind:
      CARBON_FATAL() << "ImportRefUnused should transform to ImportRefUsed "
                        "before TryEvalInst.";
  }
  return SemIR::ConstantId::NotConstant;
}

}  // namespace Carbon::Check
