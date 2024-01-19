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
// constant value. Otherwise returns `SemIR::InstId::Invalid`.
template <typename InstT, typename... EachFieldIdT>
static auto RebuildIfFieldsAreConstant(Context& context, SemIR::Inst inst,
                                       EachFieldIdT InstT::*... each_field_id)
    -> SemIR::ConstantId {
  // Build a constant instruction by replacing each non-constant operand with
  // its constant value.
  auto typed_inst = inst.As<InstT>();
  Phase phase = Phase::Template;
  if ((ReplaceFieldWithConstantValue(context, &typed_inst, each_field_id,
                                     &phase) &&
       ...)) {
    return MakeConstantResult(context, typed_inst, phase);
  }
  return phase == Phase::UnknownDueToError ? SemIR::ConstantId::Error
                                           : SemIR::ConstantId::NotConstant;
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

auto TryEvalInst(Context& context, SemIR::InstId inst_id, SemIR::Inst inst)
    -> SemIR::ConstantId {
  // TODO: Ensure we have test coverage for each of these cases that can result
  // in a constant, once those situations are all reachable.

  // clang warns on unhandled enum values; clang-tidy is incorrect here.
  // NOLINTNEXTLINE(bugprone-switch-missing-default-case)
  switch (inst.kind()) {
    // These cases are constants if their operands are.
    case SemIR::AddrOf::Kind:
      return RebuildIfFieldsAreConstant(context, inst,
                                        &SemIR::AddrOf::lvalue_id);
    case SemIR::ArrayType::Kind:
      return RebuildIfFieldsAreConstant(context, inst,
                                        &SemIR::ArrayType::bound_id);
    case SemIR::BoundMethod::Kind:
      return RebuildIfFieldsAreConstant(context, inst,
                                        &SemIR::BoundMethod::object_id,
                                        &SemIR::BoundMethod::function_id);
    case SemIR::StructType::Kind:
      return RebuildIfFieldsAreConstant(context, inst,
                                        &SemIR::StructType::fields_id);
    case SemIR::StructValue::Kind:
      return RebuildIfFieldsAreConstant(context, inst,
                                        &SemIR::StructValue::elements_id);
    case SemIR::TupleValue::Kind:
      return RebuildIfFieldsAreConstant(context, inst,
                                        &SemIR::TupleValue::elements_id);

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

    // These cases are always constants.
    case SemIR::Builtin::Kind:
    case SemIR::ClassType::Kind:
    case SemIR::PointerType::Kind:
    case SemIR::StructTypeField::Kind:
    case SemIR::TupleType::Kind:
    case SemIR::UnboundElementType::Kind:
      // TODO: Propagate symbolic / template nature from operands.
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

    // TODO: Support subobject access.
    case SemIR::ArrayIndex::Kind:
    case SemIR::ClassElementAccess::Kind:
    case SemIR::StructAccess::Kind:
    case SemIR::TupleAccess::Kind:
    case SemIR::TupleIndex::Kind:
      break;

    // TODO: These need special handling.
    case SemIR::BindValue::Kind:
    case SemIR::Call::Kind:
    case SemIR::CrossRef::Kind:
    case SemIR::Deref::Kind:
    case SemIR::Temporary::Kind:
    case SemIR::TemporaryStorage::Kind:
    case SemIR::ValueAsRef::Kind:
      break;

    case SemIR::BindSymbolicName::Kind:
      // TODO: Consider forming a constant value here using a de Bruijn index or
      // similar, so that corresponding symbolic parameters in redeclarations
      // are treated as the same value.
      return SemIR::ConstantId::ForSymbolicConstant(inst_id);

    case SemIR::BindName::Kind:
      // TODO: We need to look through `BindName`s for member accesses naming
      // fields, where the member name is a `BindName`. Should we really be
      // creating a `BindName` in that case?
      return context.constant_values().Get(inst.As<SemIR::BindName>().value_id);

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
    case SemIR::BlockArg::Kind:
    case SemIR::Branch::Kind:
    case SemIR::BranchIf::Kind:
    case SemIR::BranchWithArg::Kind:
    case SemIR::ClassDecl::Kind:
    case SemIR::Import::Kind:
    case SemIR::InterfaceDecl::Kind:
    case SemIR::LazyImportRef::Kind:
    case SemIR::Param::Kind:
    case SemIR::ReturnExpr::Kind:
    case SemIR::Return::Kind:
    case SemIR::StructLiteral::Kind:
    case SemIR::TupleLiteral::Kind:
    case SemIR::VarStorage::Kind:
      break;
  }
  return SemIR::ConstantId::NotConstant;
}

}  // namespace Carbon::Check
