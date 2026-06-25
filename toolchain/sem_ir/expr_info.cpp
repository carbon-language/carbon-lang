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
static auto AsAnyInstId(IdAndKind arg) -> InstId {
  if (auto inst_id = arg.TryAs<SemIR::InstId>()) {
    return *inst_id;
  }
  return arg.As<SemIR::AbsoluteInstId>();
}

static auto GetExprCategoryImpl(const File* ir, InstId inst_id)
    -> ExprCategory {
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
    auto handle_special_case =
        [&]<typename TypedInstT>(
            TypedInstT inst) -> std::optional<ExprCategory> {
      if constexpr (std::same_as<TypedInstT, ClassElementAccess>) {
        inst_id = inst.base_id;
        // A value of class type is a pointer to an object representation.
        // Therefore, if the base is a value, the result is an ephemeral
        // reference.
        value_category = ExprCategory::EphemeralRef;
        return std::nullopt;
      } else if constexpr (std::same_as<TypedInstT, ImportRefLoaded> ||
                           std::same_as<TypedInstT, ImportRefUnloaded>) {
        auto import_ir_inst = ir->import_ir_insts().Get(inst.import_ir_inst_id);
        ir = ir->import_irs().Get(import_ir_inst.ir_id()).sem_ir;
        inst_id = import_ir_inst.inst_id();
        return std::nullopt;
      } else if constexpr (std::same_as<TypedInstT, Call>) {
        auto callee = GetCallee(*ir, inst.callee_id);
        CARBON_KIND_SWITCH(callee) {
          case CARBON_KIND(SemIR::CalleeError _): {
            return ExprCategory::Error;
          }
          case CARBON_KIND(SemIR::CalleeFunction callee_function): {
            const auto& function =
                ir->functions().Get(callee_function.function_id);
            auto return_form_id = function.GetDeclaredReturnForm(
                *ir, callee_function.resolved_specific_id);
            if (!return_form_id.has_value()) {
              // Treat as equivalent to `-> ()`.
              return ExprCategory::ReprInitializing;
            }
            auto return_form = ir->insts().Get(return_form_id);
            CARBON_KIND_SWITCH(return_form) {
              case CARBON_KIND(InitForm _):
                return ExprCategory::ReprInitializing;
              case CARBON_KIND(RefForm _):
                return ExprCategory::DurableRef;
              case CARBON_KIND(ValueForm _):
                return ExprCategory::Value;
              case CARBON_KIND(ErrorInst _):
                return ExprCategory::Error;
              default:
                CARBON_FATAL("Unexpected inst kind: {0}", return_form);
            }
          }
          case CARBON_KIND(SemIR::CalleeNonFunction _): {
            return ExprCategory::NotExpr;
          }
          case CARBON_KIND(SemIR::CalleeCppOverloadSet _): {
            // TODO: support `ref` returns from C++.
            return ExprCategory::ReprInitializing;
          }
        }
      } else {
        static_assert(
            TypedInstT::Kind.expr_category().TryAsComputedCategory() !=
                ComputedExprCategory::DependsOnOperands,
            "Missing expression category computation for type");
      }
      CARBON_FATAL("Unreachable");
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
#define CARBON_SEM_IR_INST_KIND(TypedInstT)                             \
  case TypedInstT::Kind: {                                              \
    auto category = handle_special_case(untyped_inst.As<TypedInstT>()); \
    if (category.has_value()) {                                         \
      return *category;                                                 \
    }                                                                   \
    break;                                                              \
  }
#include "toolchain/sem_ir/inst_kind.def"
        }
      }
    }
  }
}

auto GetExprCategory(const File& file, InstId inst_id) -> ExprCategory {
  return GetExprCategoryImpl(&file, inst_id);
}

auto FindStorageArgForInitializer(const File& sem_ir, InstId init_id,
                                  bool allow_transitive) -> InstId {
  const File* ir = &sem_ir;
  while (true) {
    Inst init_untyped = ir->insts().Get(init_id);
    CARBON_KIND_SWITCH(init_untyped) {
      case CARBON_KIND(ImportRefLoaded init): {
        auto import_ir_inst = ir->import_ir_insts().Get(init.import_ir_inst_id);
        ir = ir->import_irs().Get(import_ir_inst.ir_id()).sem_ir;
        init_id = import_ir_inst.inst_id();
        continue;
      }
      case CARBON_KIND(ImportRefUnloaded init): {
        auto import_ir_inst = ir->import_ir_insts().Get(init.import_ir_inst_id);
        ir = ir->import_irs().Get(import_ir_inst.ir_id()).sem_ir;
        init_id = import_ir_inst.inst_id();
        continue;
      }
      case CARBON_KIND(AsCompatible init): {
        if (!allow_transitive) {
          return InstId::None;
        }
        init_id = init.source_id;
        continue;
      }
      case CARBON_KIND(Converted init): {
        if (!allow_transitive) {
          return InstId::None;
        }
        init_id = init.result_id;
        continue;
      }
      case CARBON_KIND(UpdateInit init): {
        if (!allow_transitive) {
          return InstId::None;
        }
        init_id = init.base_init_id;
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
      case CARBON_KIND(InPlaceInit init): {
        return init.dest_id;
      }
      case CARBON_KIND(MarkInPlaceInit init): {
        return init.dest_id;
      }
      case CARBON_KIND(Call call): {
        auto callee_function = GetCalleeAsFunction(*ir, call.callee_id);
        const auto& function = ir->functions().Get(callee_function.function_id);
        if (!function.return_form_inst_id.has_value()) {
          return InstId::None;
        }
        auto return_form_constant_id = GetConstantValueInSpecific(
            *ir, callee_function.resolved_specific_id,
            function.return_form_inst_id);
        auto return_form = ir->insts().Get(
            ir->constant_values().GetInstId(return_form_constant_id));
        CARBON_KIND_SWITCH(return_form) {
          case CARBON_KIND(InitForm init_form): {
            auto type_id = ir->types().GetTypeIdForTypeInstId(
                init_form.type_component_inst_id);
            if (!InitRepr::ForType(*ir, type_id).MightBeInPlace()) {
              return InstId::None;
            }

            if (!call.args_id.has_value()) {
              // Argument initialization failed, so we have no return slot.
              return InstId::None;
            }

            CARBON_CHECK(function.call_param_ranges.return_size() == 1,
                         "Unexpected number of output parameters on function");
            return ir->inst_blocks().Get(
                call.args_id)[function.call_param_ranges.return_begin().index];
          }
          case CARBON_KIND(RefForm _): {
            return InstId::None;
          }
          default:
            CARBON_FATAL("Unexpected inst kind: {0}", return_form);
        }
      }
      case CARBON_KIND(ErrorInst _): {
        return InstId::None;
      }
      default:
        CARBON_FATAL("Initialization from unexpected inst {0}", init_untyped);
    }
  }
}

// Given a type, determines the category of the decomposed form of an expression
// of that type. This is Primitive if the type does not support form
// decomposition.
static auto GetDecomposedFormKindForType(const File& sem_ir, TypeId type_id)
    -> FormInfo::Kind {
  if (sem_ir.types().Is<TupleType>(type_id)) {
    return FormInfo::Tuple;
  }
  if (sem_ir.types().Is<StructType>(type_id)) {
    return FormInfo::Struct;
  }
  return FormInfo::Primitive;
}

auto GetFormInfo(const File& sem_ir, InstId inst_id) -> FormInfo {
  auto inst = sem_ir.insts().Get(inst_id);

  ExprCategory category = GetExprCategory(sem_ir, inst_id);
  if (inst.type_id() == ErrorInst::TypeId) {
    // TODO: Should `GetExprCategory` do this?
    category = ExprCategory::Error;
  }

  FormInfo::Kind kind = FormInfo::Primitive;
  if (category == ExprCategory::Mixed) {
    kind = GetDecomposedFormKindForType(sem_ir, inst.type_id());
    CARBON_CHECK(kind != FormInfo::Primitive,
                 "Unexpected type {0} for mixed category",
                 sem_ir.types().GetAsInst(inst.type_id()));
  }

  return {.kind = kind,
          .category = category,
          .type_id = inst.type_id(),
          .constant_id = sem_ir.constant_values().Get(inst_id),
          .loc_id = LocId(inst_id),
          .inst_id = inst_id};
}

auto DecomposeForm(const File& sem_ir, FormInfo form) -> FormInfo {
  if (form.kind == FormInfo::Primitive) {
    form.kind = GetDecomposedFormKindForType(sem_ir, form.type_id);
    // TODO: Should we replace a category of Initializing with
    // EphemeralReference here to model temporary materialization if we
    // performed decomposition?
  }
  return form;
}

// Gets information about the forms of the instructions in a block.
static auto VisitFormInfos(const File& sem_ir, InstBlockId inst_block_id,
                           FormVisitor visitor) -> void {
  auto inst_ids = sem_ir.inst_blocks().Get(inst_block_id);
  for (auto inst_id : inst_ids) {
    visitor(GetFormInfo(sem_ir, inst_id));
  }
}

auto VisitTupleElementForms(const File& sem_ir, FormInfo form,
                            FormVisitor visitor) -> void {
  // If we have a tuple literal, directly grab the forms of its elements.
  if (auto tuple_lit_inst =
          sem_ir.insts().TryGetAsIfValid<TupleLiteral>(form.inst_id)) {
    VisitFormInfos(sem_ir, tuple_lit_inst->elements_id, visitor);
    return;
  }

  // Otherwise, decompose the type and, if available, the constant value.
  auto tuple_type = sem_ir.types().GetAs<TupleType>(form.type_id);
  auto element_type_inst_ids =
      sem_ir.inst_blocks().Get(tuple_type.type_elements_id);

  auto tuple_const_inst = sem_ir.insts().TryGetAsIfValid<TupleValue>(
      sem_ir.constant_values().GetInstIdIfValid(form.constant_id));
  auto tuple_const_inst_ids =
      tuple_const_inst ? sem_ir.inst_blocks().Get(tuple_const_inst->elements_id)
                       : llvm::ArrayRef<InstId>();

  for (auto [type_inst_id, const_inst_id] :
       llvm::zip_longest(element_type_inst_ids, tuple_const_inst_ids)) {
    visitor({.kind = FormInfo::Primitive,
             .category = form.category,
             .type_id = sem_ir.types().GetTypeIdForTypeInstId(*type_inst_id),
             .constant_id = const_inst_id
                                ? sem_ir.constant_values().Get(*const_inst_id)
                                : ConstantId::NotConstant,
             .loc_id = form.loc_id,
             .inst_id = InstId::None});
  }
}

auto VisitStructElementForms(const File& sem_ir, FormInfo form,
                             FormVisitor visitor) -> void {
  // If we have a struct literal, directly grab the forms of its elements.
  if (auto struct_lit_inst =
          sem_ir.insts().TryGetAsIfValid<StructLiteral>(form.inst_id)) {
    VisitFormInfos(sem_ir, struct_lit_inst->elements_id, visitor);
    return;
  }

  // Otherwise, decompose the type and, if available, the constant value.
  auto struct_type = sem_ir.types().GetAs<StructType>(form.type_id);
  auto fields = sem_ir.struct_type_fields().Get(struct_type.fields_id);

  auto struct_const_inst = sem_ir.insts().TryGetAsIfValid<StructValue>(
      sem_ir.constant_values().GetInstIdIfValid(form.constant_id));
  auto struct_const_inst_ids =
      struct_const_inst
          ? sem_ir.inst_blocks().Get(struct_const_inst->elements_id)
          : llvm::ArrayRef<InstId>();

  for (auto [field, const_inst_id] :
       llvm::zip_longest(fields, struct_const_inst_ids)) {
    visitor(
        {.kind = FormInfo::Primitive,
         .category = form.category,
         .type_id = sem_ir.types().GetTypeIdForTypeInstId(field->type_inst_id),
         .constant_id = const_inst_id
                            ? sem_ir.constant_values().Get(*const_inst_id)
                            : ConstantId::NotConstant,
         .loc_id = form.loc_id,
         .inst_id = InstId::None});
  }
}

}  // namespace Carbon::SemIR
