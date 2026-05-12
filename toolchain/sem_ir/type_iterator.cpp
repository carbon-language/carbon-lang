// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/type_iterator.h"

#include <utility>

#include "toolchain/base/kind_switch.h"
#include "toolchain/sem_ir/constant.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

auto TypeIterator::Next() -> Step {
  while (!work_list_.empty()) {
    auto next = work_list_.back();
    work_list_.pop_back();

    CARBON_KIND_SWITCH(next) {
      case CARBON_KIND(ConcreteNonTypeValue value): {
        return Step::ConcreteValue{.inst_id = value.inst_id};
      }
      case CARBON_KIND(EndType _): {
        return Step::End();
      }
      case CARBON_KIND(FacetType facet_type): {
        const auto& info = sem_ir_->facet_types().Get(facet_type.facet_type_id);
        for (const auto& extend : info.extend_constraints) {
          Push(SpecificInterface{extend.interface_id, extend.specific_id});
        }
        for (const auto& extend : info.extend_named_constraints) {
          Push(SpecificNamedConstraint{extend.named_constraint_id,
                                       extend.specific_id});
        }
        for (const auto& impls : info.self_impls_constraints) {
          Push(SpecificInterface{impls.interface_id, impls.specific_id});
        }
        for (const auto& impls : info.self_impls_named_constraints) {
          Push(SpecificNamedConstraint{impls.named_constraint_id,
                                       impls.specific_id});
        }
        for (const auto& type_impls : info.type_impls_interfaces) {
          PushInstId(type_impls.self_type);
          Push(type_impls.specific_interface);
        }
        for (const auto& type_impls : info.type_impls_named_constraints) {
          PushInstId(type_impls.self_type);
          Push(type_impls.specific_named_constraint);
        }
        break;
      }
      case CARBON_KIND(SpecificInterface interface): {
        auto args = GetSpecificArgs(interface.specific_id);
        if (args.empty()) {
          return Step::InterfaceStartOnly{
              {.interface_id = interface.interface_id}};
        } else {
          Push(EndType());
          PushArgs(args);
          return Step::InterfaceStart{.interface_id = interface.interface_id};
        }
      }
      case CARBON_KIND(SpecificNamedConstraint constraint): {
        auto args = GetSpecificArgs(constraint.specific_id);
        if (args.empty()) {
          return Step::NamedConstraintStartOnly{
              {.named_constraint_id = constraint.named_constraint_id}};
        } else {
          Push(EndType());
          PushArgs(args);
          return Step::NamedConstraintStart{.named_constraint_id =
                                                constraint.named_constraint_id};
        }
      }
      case CARBON_KIND(StructFieldName value): {
        return Step::StructFieldName{.name_id = value.name_id};
      }
      case CARBON_KIND(SymbolicNonTypeValue value): {
        return Step::SymbolicValue{.inst_id = value.inst_id};
      }
      case CARBON_KIND(TypeValue value): {
        if (auto step = ProcessType(value.inst_id)) {
          return *step;
        }
      }
    }
  }

  return Step::Done();
}

auto TypeIterator::ProcessType(InstId inst_id) -> std::optional<Step> {
  auto inst = sem_ir_->insts().Get(inst_id);
  // TODO: This categorization should mostly be driven by information in the
  // inst kind.
  CARBON_KIND_SWITCH(inst) {
      // ==== Symbolic types ====

    case CARBON_KIND(SymbolicBinding bind): {
      return Step::SymbolicType{.entity_name_id = bind.entity_name_id,
                                .facet = inst_id};
    }
    case CARBON_KIND(SymbolicBindingPattern bind): {
      return Step::SymbolicType{.entity_name_id = bind.entity_name_id,
                                .facet = inst_id};
    }

    case Call::Kind:
    case TypeOfInst::Kind: {
      return Step::TemplateType();
    }

    case TupleAccess::Kind: {
      // Tuple access of a concrete value would have evaluated to the accessed
      // value, so we only see TupleAccess in a type when it's a symbolic value.
      CARBON_CHECK(sem_ir_->constant_values().Get(inst_id).is_symbolic());
      return Step::SymbolicType{.entity_name_id = EntityNameId::None,
                                .facet = inst_id};
    }

      // ==== Concrete types ====

    case AssociatedEntityType::Kind:
    case BoolType::Kind:
    case CharLiteralType::Kind:
    case CppOverloadSetType::Kind:
    case CppTemplateNameType::Kind:
    case FacetType::Kind:
    case FloatLiteralType::Kind:
    case FloatType::Kind:
    case FormType::Kind:
    case FunctionType::Kind:
    case FunctionTypeWithSelfType::Kind:
    case GenericClassType::Kind:
    case GenericInterfaceType::Kind:
    case GenericNamedConstraintType::Kind:
    case IntLiteralType::Kind:
    case NamespaceType::Kind:
    case RequireSpecificDefinitionType::Kind:
    case TypeType::Kind:
    case UnboundElementType::Kind:
    case VtableType::Kind:
    case WitnessType::Kind: {
      return Step::ConcreteType{
          .type_id = sem_ir_->types().GetTypeIdForTypeInstId(inst_id)};
    }

    case CARBON_KIND(IntType int_type): {
      Push(EndType());
      PushArgs({int_type.bit_width_id});
      return Step::IntStart{
          .type_id = sem_ir_->types().GetTypeIdForTypeInstId(inst_id)};
    }

      // ==== Aggregate types ====

    case CARBON_KIND(ArrayType array_type): {
      Push(EndType());
      PushInstId(array_type.element_type_inst_id);
      PushInstId(array_type.bound_id);
      return Step::ArrayStart{
          .type_id = sem_ir_->types().GetTypeIdForTypeInstId(inst_id)};
    }
    case CARBON_KIND(ClassType class_type): {
      auto args = GetSpecificArgs(class_type.specific_id);
      if (args.empty()) {
        return Step::ClassStartOnly{
            {.class_id = class_type.class_id,
             .type_id = sem_ir_->types().GetTypeIdForTypeInstId(inst_id)}};
      } else {
        Push(EndType());
        PushArgs(args);
        return Step::ClassStart{
            .class_id = class_type.class_id,
            .type_id = sem_ir_->types().GetTypeIdForTypeInstId(inst_id)};
      }
    }
    case CARBON_KIND(ConstType const_type): {
      Push(EndType());
      PushInstId(const_type.inner_id);
      return Step::ConstStart();
    }
    case CARBON_KIND(ImplWitnessAssociatedConstant assoc): {
      PushTypeId(assoc.type_id);
      return std::nullopt;
    }
    case CARBON_KIND(MaybeUnformedType maybe_unformed_type): {
      Push(EndType());
      PushInstId(maybe_unformed_type.inner_id);
      return Step::MaybeUnformedStart();
    }
    case CARBON_KIND(PartialType partial_type): {
      Push(EndType());
      PushInstId(partial_type.inner_id);
      return Step::PartialStart();
    }
    case CARBON_KIND(PointerType pointer_type): {
      Push(EndType());
      PushInstId(pointer_type.pointee_id);
      return Step::PointerStart();
    }
    case CARBON_KIND(TupleType tuple_type): {
      auto inner_types =
          sem_ir_->inst_blocks().Get(tuple_type.type_elements_id);
      if (inner_types.empty()) {
        return Step::TupleStartOnly{
            {.type_id = sem_ir_->types().GetTypeIdForTypeInstId(inst_id)}};
      } else {
        Push(EndType());
        PushArgs(sem_ir_->inst_blocks().Get(tuple_type.type_elements_id));
        return Step::TupleStart{
            .type_id = sem_ir_->types().GetTypeIdForTypeInstId(inst_id)};
      }
    }
    case CARBON_KIND(StructType struct_type): {
      auto fields = sem_ir_->struct_type_fields().Get(struct_type.fields_id);
      if (fields.empty()) {
        return Step::StructStartOnly{
            {.type_id = sem_ir_->types().GetTypeIdForTypeInstId(inst_id)}};
      } else {
        Push(EndType());
        for (const auto& field : llvm::reverse(fields)) {
          Push(StructFieldName{.name_id = field.name_id});
          PushInstId(field.type_inst_id);
        }
        return Step::StructStart{
            .type_id = sem_ir_->types().GetTypeIdForTypeInstId(inst_id)};
      }
    }

      // ==== Dependent instructions ====

    case CARBON_KIND(FacetAccessType access): {
      if (sem_ir_->constant_values().Get(inst_id).is_concrete()) {
        return Step::ConcreteType{
            .type_id = sem_ir_->types().GetTypeIdForTypeInstId(inst_id)};
      }
      PushInstId(access.facet_value_inst_id);
      return std::nullopt;
    }

    case CARBON_KIND(FacetValue value): {
      // We return FacetValues as a separate iterative step, then also recurse
      // into them.
      PushInstId(value.type_inst_id);
      return Step::TypeWrapper{.kind = Step::TypeWrapper::FacetValue,
                               .inst_id = inst_id};
    }

    case CARBON_KIND(ImplWitnessAccess access): {
      // We return FacetValues as a separate iterative step, then also recurse
      // into the the self value being accessed.
      //
      // Witness access of a concrete value would have evaluated to the accessed
      // value, so we only see ImplWitnessAccess in a type when it's a symbolic
      // value, which implies it contains a LookupImplWitness.
      CARBON_CHECK(sem_ir_->constant_values().Get(inst_id).is_symbolic());
      auto witness =
          sem_ir_->insts().GetAs<LookupImplWitness>(access.witness_id);
      // Recurse into symbolic ImplWitnessAccess, replacing it with the self
      // value for the iteration. If there are nested accesses, this replaces
      // them all with the root self.
      PushInstId(witness.query_self_inst_id);
      return Step::TypeWrapper{.kind = Step::TypeWrapper::ImplWitnessAccess,
                               .inst_id = inst_id};
    }

    case ErrorInst::Kind:
      return Step::Error();

    default:
      // TODO: Rearrange this so that missing instruction kinds are detected
      // at compile-time not runtime.
      CARBON_FATAL("Unhandled type instruction {0}", inst);
  }
}

auto TypeIterator::GetSpecificArgs(SpecificId specific_id) const
    -> llvm::ArrayRef<InstId> {
  if (specific_id == SpecificId::None) {
    return {};
  }
  auto specific = sem_ir_->specifics().Get(specific_id);
  return sem_ir_->inst_blocks().Get(specific.args_id);
}

// Push all arguments from the array into the work queue.
auto TypeIterator::PushArgs(llvm::ArrayRef<InstId> args) -> void {
  for (auto arg_id : llvm::reverse(args)) {
    PushInstId(arg_id);
  }
}

// Push an instruction's type value into the work queue, or a marker if the
// instruction has a symbolic value.
auto TypeIterator::PushInstId(InstId inst_id) -> void {
  // Work with canonical instructions only. Types always have a constant value.
  // Do this here instead of in Add(InstId) to also handle the user providing a
  // non-canonical input through other Add() methods.
  inst_id = sem_ir_->constant_values().GetConstantInstId(inst_id);

  if (sem_ir_->types().IsFacetType(sem_ir_->insts().Get(inst_id).type_id())) {
    Push(TypeValue{.inst_id = inst_id});
  } else if (sem_ir_->constant_values().Get(inst_id).is_symbolic()) {
    Push(SymbolicNonTypeValue{.inst_id = inst_id});
  } else {
    Push(ConcreteNonTypeValue{.inst_id = inst_id});
  }
}

auto TypeIterator::PushTypeId(TypeId type_id) -> void {
  Push(TypeValue{.inst_id = sem_ir_->types().GetTypeInstId(type_id)});
}

// Push the next step into the work queue.
auto TypeIterator::Push(WorkItem item) -> void { work_list_.push_back(item); }

}  // namespace Carbon::SemIR
