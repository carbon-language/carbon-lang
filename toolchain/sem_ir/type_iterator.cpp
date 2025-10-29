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
      case CARBON_KIND(StructFieldName value): {
        return Step::StructFieldName{.name_id = value.name_id};
      }
      case CARBON_KIND(SymbolicNonTypeValue value): {
        return Step::SymbolicValue{.inst_id = value.inst_id};
      }
      case CARBON_KIND(SymbolicType symbolic): {
        return Step::SymbolicType{.facet_type_id = symbolic.facet_type_id};
      }
      case CARBON_KIND(TypeId type_id): {
        if (auto step = ProcessTypeId(type_id)) {
          return *step;
        }
      }
    }
  }

  return Step::Done();
}

auto TypeIterator::ProcessTypeId(TypeId type_id) -> std::optional<Step> {
  auto inst_id = sem_ir_->types().GetInstId(type_id);
  auto inst = sem_ir_->insts().Get(inst_id);
  CARBON_KIND_SWITCH(inst) {
      // ==== Symbolic types ====

    case SymbolicBinding::Kind:
    case SymbolicBindingPattern::Kind: {
      return Step::SymbolicType{.facet_type_id = type_id};
    }
    case TypeOfInst::Kind: {
      return Step::TemplateType();
    }

    case CARBON_KIND(FacetAccessType access): {
      auto facet_type_id =
          sem_ir_->insts().Get(access.facet_value_inst_id).type_id();
      return Step::SymbolicType{.facet_type_id = facet_type_id};
    }
    case CARBON_KIND(SemIR::SymbolicBindingType bind): {
      return Step::SymbolicBinding{.entity_name_id = bind.entity_name_id};
    }

      // ==== Concrete types ====

    case AssociatedEntityType::Kind:
    case BoolType::Kind:
    case CharLiteralType::Kind:
    case CppOverloadSetType::Kind:
    case CppVoidType::Kind:
    case FacetType::Kind:
    case FloatLiteralType::Kind:
    case FloatType::Kind:
    case FunctionType::Kind:
    case FunctionTypeWithSelfType::Kind:
    case GenericClassType::Kind:
    case GenericInterfaceType::Kind:
    case ImplWitnessAccess::Kind:
    case IntLiteralType::Kind:
    case NamespaceType::Kind:
    case TypeType::Kind:
    case WitnessType::Kind: {
      return Step::ConcreteType{.type_id = type_id};
    }

    case CARBON_KIND(IntType int_type): {
      Push(EndType());
      PushArgs({int_type.bit_width_id});
      return Step::IntStart{.type_id = type_id};
    }

      // ==== Aggregate types ====

    case CARBON_KIND(ArrayType array_type): {
      Push(EndType());
      PushInstId(array_type.element_type_inst_id);
      PushInstId(array_type.bound_id);
      return Step::ArrayStart{.type_id = type_id};
    }
    case CARBON_KIND(ClassType class_type): {
      auto args = GetSpecificArgs(class_type.specific_id);
      if (args.empty()) {
        return Step::ClassStartOnly{
            {.class_id = class_type.class_id, .type_id = type_id}};
      } else {
        Push(EndType());
        PushArgs(args);
        return Step::ClassStart{.class_id = class_type.class_id,
                                .type_id = type_id};
      }
    }
    case CARBON_KIND(ConstType const_type): {
      Push(EndType());
      PushInstId(const_type.inner_id);
      return Step::ConstStart();
    }
    case CARBON_KIND(ImplWitnessAssociatedConstant assoc): {
      Push(assoc.type_id);
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
        return Step::TupleStartOnly{{.type_id = type_id}};
      } else {
        Push(EndType());
        PushArgs(sem_ir_->inst_blocks().Get(tuple_type.type_elements_id));
        return Step::TupleStart{.type_id = type_id};
      }
    }
    case CARBON_KIND(StructType struct_type): {
      auto fields = sem_ir_->struct_type_fields().Get(struct_type.fields_id);
      if (fields.empty()) {
        return Step::StructStartOnly{{.type_id = type_id}};
      } else {
        Push(EndType());
        for (const auto& field : llvm::reverse(fields)) {
          Push(StructFieldName{.name_id = field.name_id});
          PushInstId(field.type_inst_id);
        }
        return Step::StructStart{.type_id = type_id};
      }
    }

    case ErrorInst::Kind:
      return Step::Error();

    default:
      // TODO: Rearrange this so that missing instruction kinds are detected
      // at compile-time not runtime.
      CARBON_FATAL("Unhandled type instruction {0}", inst_id);
  }
}

auto TypeIterator::TryGetInstIdAsTypeId(InstId inst_id) const
    -> std::variant<TypeId, SymbolicType> {
  if (auto facet_value = sem_ir_->insts().TryGetAs<FacetValue>(inst_id)) {
    inst_id = facet_value->type_inst_id;
  }

  auto type_id_of_inst_id = sem_ir_->insts().Get(inst_id).type_id();
  // All instructions of type FacetType are symbolic except for FacetValue:
  // - In non-generic code, values of type FacetType are only created through
  //   conversion to a FacetType (e.g. `Class as Iface`), which produces a
  //   non-symbolic FacetValue.
  // - In generic code, binding values of type FacetType are symbolic as they
  //   refer to an unknown type. Non-binding values would be FacetValues like
  //   in non-generic code, but would be symbolic as well.
  // - In specifics of generic code, when deducing a value for a symbolic
  //   binding of type FacetType, we always produce a FacetValue (which may or
  //   may not itself be symbolic) through conversion.
  //
  // FacetValues are handled earlier by getting the type instruction from
  // them. That type instruction is never of type FacetType. If it refers to a
  // FacetType it does so through a FacetAccessType, which is of type TypeType
  // and thus does not match here.
  if (auto facet_type =
          sem_ir_->types().TryGetAs<FacetType>(type_id_of_inst_id)) {
    return SymbolicType{.facet_type_id = type_id_of_inst_id};
  }
  // Non-type values are concrete, only types are symbolic.
  if (type_id_of_inst_id != TypeType::TypeId) {
    return TypeId::None;
  }
  return sem_ir_->types().GetTypeIdForTypeInstId(inst_id);
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
  auto maybe_type_id = TryGetInstIdAsTypeId(inst_id);
  if (std::holds_alternative<SymbolicType>(maybe_type_id)) {
    Push(std::get<SymbolicType>(maybe_type_id));
  } else if (auto type_id = std::get<TypeId>(maybe_type_id);
             type_id.has_value()) {
    Push(type_id);
  } else if (sem_ir_->constant_values().Get(inst_id).is_symbolic()) {
    Push(SymbolicNonTypeValue{.inst_id = inst_id});
  } else {
    Push(ConcreteNonTypeValue{.inst_id = inst_id});
  }
}

// Push the next step into the work queue.
auto TypeIterator::Push(WorkItem item) -> void { work_list_.push_back(item); }

}  // namespace Carbon::SemIR
