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

    // TODO: Consider using a CARBON_KIND_SWITCH on `next` here after
    // https://github.com/carbon-language/carbon-lang/pull/5433 arrives, instead
    // of a bunch of `if` conditions.

    if (std::holds_alternative<EndType>(next)) {
      return Step::End();
    }

    if (const auto* interface = std::get_if<SemIR::SpecificInterface>(&next)) {
      auto args = GetSpecificArgs(interface->specific_id);
      if (args.empty()) {
        return Step::InterfaceStartOnly{
            {.interface_id = interface->interface_id}};
      } else {
        Push(EndType());
        PushArgs(args);
        return Step::InterfaceStart{.interface_id = interface->interface_id};
      }
    }

    if (const auto* symbolic = std::get_if<SymbolicType>(&next)) {
      return Step::SymbolicType{.facet_type_id = symbolic->facet_type_id};
    }

    if (const auto* value = std::get_if<ConcreteNonTypeValue>(&next)) {
      return Step::ConcreteValue{.inst_id = value->inst_id};
    }

    if (const auto* value = std::get_if<SymbolicNonTypeValue>(&next)) {
      return Step::SymbolicValue{.inst_id = value->inst_id};
    }

    if (const auto* value = std::get_if<StructFieldName>(&next)) {
      return Step::StructFieldName{.name_id = value->name_id};
    }

    SemIR::TypeId type_id = std::get<SemIR::TypeId>(next);
    auto inst_id = sem_ir_->types().GetInstId(type_id);
    auto inst = sem_ir_->insts().Get(inst_id);
    CARBON_KIND_SWITCH(inst) {
        // ==== Symbolic types ====

      case SemIR::BindSymbolicName::Kind:
      case SemIR::SymbolicBindingPattern::Kind: {
        return Step::SymbolicType{.facet_type_id = type_id};
      }
      case SemIR::TypeOfInst::Kind: {
        return Step::TemplateType();
      }

      case CARBON_KIND(SemIR::FacetAccessType access): {
        auto facet_type_id =
            sem_ir_->insts().Get(access.facet_value_inst_id).type_id();
        return Step::SymbolicType{.facet_type_id = facet_type_id};
      }

        // ==== Concrete types ====

      case SemIR::AssociatedEntityType::Kind:
      case SemIR::BoolType::Kind:
      case SemIR::CharLiteralType::Kind:
      case SemIR::FacetType::Kind:
      case SemIR::FloatLiteralType::Kind:
      case SemIR::FloatType::Kind:
      case SemIR::FunctionType::Kind:
      case SemIR::FunctionTypeWithSelfType::Kind:
      case SemIR::GenericClassType::Kind:
      case SemIR::GenericInterfaceType::Kind:
      case SemIR::ImplWitnessAccess::Kind:
      case SemIR::IntLiteralType::Kind:
      case SemIR::NamespaceType::Kind:
      case SemIR::TypeType::Kind:
      case SemIR::WitnessType::Kind: {
        return Step::ConcreteType{.type_id = type_id};
      }

      case CARBON_KIND(SemIR::IntType int_type): {
        Push(EndType());
        PushArgs({int_type.bit_width_id});
        return Step::IntStart{.type_id = type_id};
      }

        // ==== Aggregate types ====

      case CARBON_KIND(SemIR::ArrayType array_type): {
        Push(EndType());
        PushInstId(array_type.element_type_inst_id);
        PushInstId(array_type.bound_id);
        return Step::ArrayStart{.type_id = type_id};
      }
      case CARBON_KIND(SemIR::ClassType class_type): {
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
      case CARBON_KIND(SemIR::ConstType const_type): {
        Push(EndType());
        PushInstId(const_type.inner_id);
        return Step::ConstStart();
      }
      case CARBON_KIND(SemIR::ImplWitnessAssociatedConstant assoc): {
        Push(assoc.type_id);
        break;
      }
      case CARBON_KIND(SemIR::MaybeUnformedType maybe_unformed_type): {
        Push(EndType());
        PushInstId(maybe_unformed_type.inner_id);
        return Step::MaybeUnformedStart();
      }
      case CARBON_KIND(SemIR::PartialType partial_type): {
        Push(EndType());
        PushInstId(partial_type.inner_id);
        return Step::PartialStart();
      }
      case CARBON_KIND(SemIR::PointerType pointer_type): {
        Push(EndType());
        PushInstId(pointer_type.pointee_id);
        return Step::PointerStart();
      }
      case CARBON_KIND(SemIR::TupleType tuple_type): {
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
      case CARBON_KIND(SemIR::StructType struct_type): {
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

      case SemIR::ErrorInst::Kind:
        return Step::Error();

      default:
        // TODO: Rearrange this so that missing instruction kinds are detected
        // at compile-time not runtime.
        CARBON_FATAL("Unhandled type instruction {0}", inst_id);
    }
  }

  return Step::Done();
}

auto TypeIterator::TryGetInstIdAsTypeId(SemIR::InstId inst_id) const
    -> std::variant<SemIR::TypeId, SymbolicType> {
  if (auto facet_value =
          sem_ir_->insts().TryGetAs<SemIR::FacetValue>(inst_id)) {
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
          sem_ir_->types().TryGetAs<SemIR::FacetType>(type_id_of_inst_id)) {
    return SymbolicType{.facet_type_id = type_id_of_inst_id};
  }
  // Non-type values are concrete, only types are symbolic.
  if (type_id_of_inst_id != SemIR::TypeType::TypeId) {
    return SemIR::TypeId::None;
  }
  return sem_ir_->types().GetTypeIdForTypeInstId(inst_id);
}

auto TypeIterator::GetSpecificArgs(SemIR::SpecificId specific_id) const
    -> llvm::ArrayRef<SemIR::InstId> {
  if (specific_id == SemIR::SpecificId::None) {
    return {};
  }
  auto specific = sem_ir_->specifics().Get(specific_id);
  return sem_ir_->inst_blocks().Get(specific.args_id);
}

// Push all arguments from the array into the work queue.
auto TypeIterator::PushArgs(llvm::ArrayRef<SemIR::InstId> args) -> void {
  for (auto arg_id : llvm::reverse(args)) {
    PushInstId(arg_id);
  }
}

// Push an instruction's type value into the work queue, or a marker if the
// instruction has a symbolic value.
auto TypeIterator::PushInstId(SemIR::InstId inst_id) -> void {
  auto maybe_type_id = TryGetInstIdAsTypeId(inst_id);
  if (std::holds_alternative<SymbolicType>(maybe_type_id)) {
    Push(std::get<SymbolicType>(maybe_type_id));
  } else if (auto type_id = std::get<SemIR::TypeId>(maybe_type_id);
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
