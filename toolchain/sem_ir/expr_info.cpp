// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/expr_info.h"

#include <concepts>

#include "common/check.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst_kind.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

// Returns the InstId represented by an instruction operand.
static auto AsAnyInstId(Inst::ArgAndKind arg) -> InstId {
  if (auto inst_id = arg.TryAs<SemIR::InstId>()) {
    return *inst_id;
  }
  return arg.As<SemIR::AbsoluteInstId>();
}

auto GetExprCategory(const File& file, InstId inst_id) -> ExprCategory {
  const File* ir = &file;

  // The overall expression category if the current instruction is a value
  // expression.
  ExprCategory value_category = ExprCategory::Value;

  while (true) {
    auto untyped_inst = ir->insts().Get(inst_id);
    auto category_from_kind = untyped_inst.kind().expr_category();

    // If this instruction kind has a fixed category, return it.
    if (auto fixed_category = category_from_kind.TryAsFixedCategory()) {
      return *fixed_category == ExprCategory::Value ? value_category
                                                    : *fixed_category;
    }

    // Handle any special cases that use
    // ComputedExprCategory::DependsOnOperands.
    auto handle_special_case = [&]<typename TypedInstT>(TypedInstT inst) {
      if constexpr (std::same_as<TypedInstT, ClassElementAccess>) {
        inst_id = inst.base_id;
        // A value of class type is a pointer to an object representation.
        // Therefore, if the base is a value, the result is an ephemeral
        // reference.
        value_category = ExprCategory::EphemeralRef;
      } else if constexpr (std::same_as<TypedInstT, ImportRefLoaded> ||
                           std::same_as<TypedInstT, ImportRefUnloaded>) {
        auto import_ir_inst = ir->import_ir_insts().Get(inst.import_ir_inst_id);
        ir = ir->import_irs().Get(import_ir_inst.ir_id()).sem_ir;
        inst_id = import_ir_inst.inst_id();
      } else {
        static_assert(
            TypedInstT::Kind.expr_category().TryAsComputedCategory() !=
                ComputedExprCategory::DependsOnOperands,
            "Missing expression category computation for type");
      }
    };

    // If the category depends on the operands of the instruction, determine it.
    // Usually this means the category is the same as the category of an
    // operand.
    switch (*category_from_kind.TryAsComputedCategory()) {
      case ComputedExprCategory::ValueIfHasType: {
        return untyped_inst.kind().has_type() ? value_category
                                              : ExprCategory::NotExpr;
      }

      case ComputedExprCategory::SameAsFirstOperand: {
        inst_id = AsAnyInstId(untyped_inst.arg0_and_kind());
        break;
      }

      case ComputedExprCategory::SameAsSecondOperand: {
        inst_id = AsAnyInstId(untyped_inst.arg1_and_kind());
        break;
      }

      case ComputedExprCategory::DependsOnOperands: {
        switch (untyped_inst.kind()) {
#define CARBON_SEM_IR_INST_KIND(TypedInstT)             \
  case TypedInstT::Kind:                                \
    handle_special_case(untyped_inst.As<TypedInstT>()); \
    break;
#include "toolchain/sem_ir/inst_kind.def"
        }
      }
    }
  }
}

auto FindReturnSlotArgForInitializer(const File& sem_ir, InstId init_id)
    -> InstId {
  while (true) {
    Inst init_untyped = sem_ir.insts().Get(init_id);
    CARBON_KIND_SWITCH(init_untyped) {
      case CARBON_KIND(AsCompatible init): {
        init_id = init.source_id;
        continue;
      }
      case CARBON_KIND(Converted init): {
        init_id = init.result_id;
        continue;
      }
      case CARBON_KIND(ArrayInit init): {
        return init.dest_id;
      }
      case CARBON_KIND(ClassInit init): {
        return init.dest_id;
      }
      case CARBON_KIND(StructInit init): {
        return init.dest_id;
      }
      case CARBON_KIND(TupleInit init): {
        return init.dest_id;
      }
      case CARBON_KIND(InitializeFrom init): {
        return init.dest_id;
      }
      case CARBON_KIND(InPlaceInit init): {
        if (!ReturnTypeInfo::ForType(sem_ir, init.type_id).has_return_slot()) {
          return InstId::None;
        }
        return init.dest_id;
      }
      case CARBON_KIND(Call call): {
        if (!ReturnTypeInfo::ForType(sem_ir, call.type_id).has_return_slot()) {
          return InstId::None;
        }
        if (!call.args_id.has_value()) {
          // Argument initialization failed, so we have no return slot.
          return InstId::None;
        }
        return sem_ir.inst_blocks().Get(call.args_id).back();
      }
      case CARBON_KIND(ErrorInst _): {
        return InstId::None;
      }
      default:
        CARBON_FATAL("Initialization from unexpected inst {0}", init_untyped);
    }
  }
}

}  // namespace Carbon::SemIR
