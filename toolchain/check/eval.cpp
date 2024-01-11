// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/eval.h"

#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// `GetConstantValue` checks to see whether the provided ID describes a value
// with constant phase, and if so, returns the corresponding constant value.
// Overloads are provided for different kinds of ID.

// If the given instruction is constant, returns its constant value.
static auto GetConstantValue(Context& context, SemIR::InstId inst_id)
    -> SemIR::InstId {
  return context.constant_values().Get(inst_id);
}

// If the given instruction block contains only constants, returns a
// corresponding block of those values.
static auto GetConstantValue(Context& context, SemIR::InstBlockId inst_block_id)
    -> SemIR::InstBlockId {
  auto insts = context.inst_blocks().Get(inst_block_id);
  llvm::SmallVector<SemIR::InstId> const_insts;
  for (auto inst_id : insts) {
    auto const_inst_id = GetConstantValue(context, inst_id);
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
                                          FieldIdT InstT::*field) -> bool {
  auto unwrapped = GetConstantValue(context, inst->*field);
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
    -> SemIR::InstId {
  // Build a constant instruction by replacing each non-constant operand with
  // its constant value.
  auto typed_inst = inst.As<InstT>();
  if ((ReplaceFieldWithConstantValue(context, &typed_inst, each_field_id) &&
       ...)) {
    return context.AddConstantInst(typed_inst);
  }
  return SemIR::InstId::Invalid;
}

auto TryEvalInst(Context& context, SemIR::InstId inst_id, SemIR::Inst inst)
    -> SemIR::InstId {
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
    case SemIR::StructValue::Kind:
      return RebuildIfFieldsAreConstant(context, inst,
                                        &SemIR::StructValue::elements_id);
    case SemIR::Temporary::Kind:
      return RebuildIfFieldsAreConstant(context, inst,
                                        &SemIR::Temporary::init_id);
    case SemIR::TupleValue::Kind:
      return RebuildIfFieldsAreConstant(context, inst,
                                        &SemIR::TupleValue::elements_id);

    // These cases are constants already.
    case SemIR::Builtin::Kind:
    case SemIR::ClassType::Kind:
    case SemIR::ConstType::Kind:
    case SemIR::PointerType::Kind:
    case SemIR::StructType::Kind:
    case SemIR::TupleType::Kind:
    case SemIR::UnboundElementType::Kind:
      return inst_id;

    case SemIR::BaseDecl::Kind:
    case SemIR::FieldDecl::Kind:
    case SemIR::FunctionDecl::Kind:
      // TODO: Consider adding a corresponding `Value` inst.
      return inst_id;

    case SemIR::BoolLiteral::Kind:
    case SemIR::IntLiteral::Kind:
    case SemIR::RealLiteral::Kind:
    case SemIR::StringLiteral::Kind:
      // Promote literals to the constant block.
      return context.AddConstantInst(inst);

    // TODO: These need special handling.
    case SemIR::ArrayIndex::Kind:
    case SemIR::ArrayInit::Kind:
    case SemIR::BindValue::Kind:
    case SemIR::Call::Kind:
    case SemIR::ClassElementAccess::Kind:
    case SemIR::ClassInit::Kind:
    case SemIR::CrossRef::Kind:
    case SemIR::Deref::Kind:
    case SemIR::InitializeFrom::Kind:
    case SemIR::SpliceBlock::Kind:
    case SemIR::StructAccess::Kind:
    case SemIR::StructInit::Kind:
    case SemIR::TemporaryStorage::Kind:
    case SemIR::TupleAccess::Kind:
    case SemIR::TupleIndex::Kind:
    case SemIR::TupleInit::Kind:
    case SemIR::ValueAsRef::Kind:
    case SemIR::ValueOfInitializer::Kind:
      break;

    case SemIR::BindName::Kind:
    case SemIR::BindSymbolicName::Kind:
      // TODO: Should we really be looking through runtime and symbolic `let`
      // bindings?
      return GetConstantValue(context, inst.As<SemIR::AnyBindName>().value_id);

    case SemIR::NameRef::Kind:
      return GetConstantValue(context, inst.As<SemIR::NameRef>().value_id);

    case SemIR::Converted::Kind:
      return GetConstantValue(context, inst.As<SemIR::Converted>().result_id);

    case SemIR::UnaryOperatorNot::Kind: {
      auto const_id = GetConstantValue(
          context, inst.As<SemIR::UnaryOperatorNot>().operand_id);
      if (!const_id.is_valid()) {
        break;
      }
      auto value = context.insts().TryGetAs<SemIR::BoolLiteral>(const_id);
      if (!value) {
        // TODO: Can we CHECK this instead?
        break;
      }
      value->value =
          (value->value == SemIR::BoolValue::False ? SemIR::BoolValue::True
                                                   : SemIR::BoolValue::False);
      return context.AddConstantInst(*value);
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
    case SemIR::Namespace::Kind:
    case SemIR::Param::Kind:
    case SemIR::ReturnExpr::Kind:
    case SemIR::Return::Kind:
    case SemIR::StructLiteral::Kind:
    case SemIR::StructTypeField::Kind:
    case SemIR::TupleLiteral::Kind:
    case SemIR::VarStorage::Kind:
      break;
  }
  return SemIR::InstId::Invalid;
}

}  // namespace Carbon::Check
