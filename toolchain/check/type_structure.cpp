// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/type_structure.h"

#include <variant>

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/sem_ir/facet_type_info.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/impl.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto TypeStructure::IsCompatibleWith(const TypeStructure& /*other*/) const
    -> bool {
  // TODO: Return false sometimes.
  return true;
}

// A class that builds a `TypeStructure` for an `Impl` that represents its self
// type.
class TypeStructureBuilder {
 public:
  explicit TypeStructureBuilder(Context& context) : context_(context) {}

  auto Run(const SemIR::Impl& impl, int priority_ordering = 0)
      -> TypeStructure {
    CARBON_CHECK(work_list_.empty());

    position_ = 0;
    first_symbolic_distance_ = TypeStructure::InfiniteDistance;
    structure_.clear();

    Push(GetInstIdAsTypeId(impl.constraint_id));
    Push(GetInstIdAsTypeId(impl.self_id));
    BuildTypeStructure();

    return TypeStructure(std::exchange(structure_, {}),
                         first_symbolic_distance_, priority_ordering,
                         impl.interface.interface_id);
  }

 private:
  auto BuildTypeStructure() -> void {
    while (!work_list_.empty()) {
      SemIR::TypeId next_type_id = work_list_.back();
      work_list_.pop_back();

      auto inst_id = context_.types().GetInstId(next_type_id);
      auto inst = context_.insts().Get(inst_id);
      CARBON_KIND_SWITCH(inst) {
        case CARBON_KIND(SemIR::BindSymbolicName bind): {
          (void)bind;
          // We found a symbolic type.
          SetFirstSymbolic(position_);
          ++position_;
          structure_.push_back(TypeStructure::Structural::Symbolic);
          break;
        }
        case CARBON_KIND(SemIR::SymbolicBindingPattern bind): {
          (void)bind;
          // We found a symbolic type.
          SetFirstSymbolic(position_);
          ++position_;
          structure_.push_back(TypeStructure::Structural::Symbolic);
          break;
        }
        case CARBON_KIND(SemIR::FacetAccessType access): {
          (void)access;
          // We found a symbolic type (referenced by the FacetAccessType).
          SetFirstSymbolic(position_);
          ++position_;
          structure_.push_back(TypeStructure::Structural::Symbolic);
          break;
        }
        case CARBON_KIND(SemIR::TypeType type_type): {
          (void)type_type;
          ++position_;
          structure_.push_back(TypeStructure::Structural::Concrete);
          break;
        }
        case CARBON_KIND(SemIR::BoolType bool_type): {
          (void)bool_type;
          ++position_;
          structure_.push_back(TypeStructure::Structural::Concrete);
          break;
        }
        case CARBON_KIND(SemIR::IntType int_type): {
          (void)int_type;
          ++position_;  // Opening of type.
          if (context_.constant_values().Get(inst_id).is_concrete()) {
            structure_.push_back(TypeStructure::Structural::Concrete);
          } else {
            structure_.push_back(TypeStructure::Structural::ConcreteOpenParen);
            PushArgs({int_type.bit_width_id});
            ++position_;  // Closing of type.
            structure_.push_back(TypeStructure::Structural::ConcreteCloseParen);
          }
          break;
        }
        case CARBON_KIND(SemIR::IntLiteralType int_literal_type): {
          (void)int_literal_type;
          ++position_;
          structure_.push_back(TypeStructure::Structural::Concrete);
          break;
        }
        case CARBON_KIND(SemIR::GenericClassType generic_type): {
          (void)generic_type;
          ++position_;
          structure_.push_back(TypeStructure::Structural::Concrete);
          break;
        }
        case CARBON_KIND(SemIR::GenericInterfaceType generic_type): {
          (void)generic_type;
          ++position_;
          structure_.push_back(TypeStructure::Structural::Concrete);
          break;
        }
        case CARBON_KIND(SemIR::ArrayType array_type): {
          ++position_;  // Opening of type.
          structure_.push_back(TypeStructure::Structural::ConcreteOpenParen);
          Push(array_type.element_type_id);
          ++position_;  // Closing of type.
          structure_.push_back(TypeStructure::Structural::ConcreteCloseParen);
          break;
        }
        case CARBON_KIND(SemIR::ClassType class_type): {
          auto args = GetSpecificArgs(class_type.specific_id);
          ++position_;  // Opening of type.
          if (args.empty()) {
            structure_.push_back(TypeStructure::Structural::Concrete);
          } else {
            structure_.push_back(TypeStructure::Structural::ConcreteOpenParen);
            PushArgs(args);
            ++position_;  // Closing of type.
            structure_.push_back(TypeStructure::Structural::ConcreteCloseParen);
          }
          break;
        }
        case CARBON_KIND(SemIR::FacetType facet_type): {
          auto facet_type_info =
              context_.facet_types().Get(facet_type.facet_type_id);
          ++position_;  // Opening of type.
          structure_.push_back(TypeStructure::Structural::ConcreteOpenParen);
          for (const auto& i : facet_type_info.impls_constraints) {
            PushArgs(GetSpecificArgs(i.specific_id));
          }
          if (facet_type_info.other_requirements) {
            // TODO: This goes away when other_requirements does. Are there
            // other places we need to look for symbolics in FacetTypeInfo at
            // that point? For now we treat it as having a symbolic at the end
            // of the facet type, so facet types with other_requirements are
            // chosen with lower priority than those without.
            SetFirstSymbolic(position_);
            ++position_;
            structure_.push_back(TypeStructure::Structural::Symbolic);
          } else {
            ++position_;
            structure_.push_back(TypeStructure::Structural::Concrete);
          }
          ++position_;  // Closing of type.
          structure_.push_back(TypeStructure::Structural::ConcreteCloseParen);
          break;
        }
        case CARBON_KIND(SemIR::TupleType tuple_type): {
          auto inner_types = context_.type_blocks().Get(tuple_type.elements_id);
          ++position_;  // Opening of type.
          if (inner_types.empty()) {
            structure_.push_back(TypeStructure::Structural::Concrete);
          } else {
            structure_.push_back(TypeStructure::Structural::ConcreteOpenParen);
            for (auto type :
                 context_.type_blocks().Get(tuple_type.elements_id)) {
              Push(type);
            }
            ++position_;  // Closing of type.
            structure_.push_back(TypeStructure::Structural::ConcreteCloseParen);
          }
          break;
        }
        case CARBON_KIND(SemIR::StructType struct_type): {
          auto fields =
              context_.struct_type_fields().Get(struct_type.fields_id);
          ++position_;  // Opening of type.
          if (fields.empty()) {
            structure_.push_back(TypeStructure::Structural::Concrete);
          } else {
            structure_.push_back(TypeStructure::Structural::ConcreteOpenParen);
            for (const auto& field : fields) {
              Push(field.type_id);
            }
            ++position_;  // Closing of type.
            structure_.push_back(TypeStructure::Structural::ConcreteCloseParen);
          }
          break;
        }
        default:
          CARBON_FATAL("Unhandled type instruction {0}", inst_id);
      }
    }
  }

  struct SymbolicType {};

  auto TryGetInstIdAsTypeId(SemIR::InstId inst_id) const
      -> std::variant<SymbolicType, SemIR::TypeId> {
    if (auto facet_value =
            context_.insts().TryGetAs<SemIR::FacetValue>(inst_id)) {
      inst_id = facet_value->type_inst_id;
    }

    auto type_id_of_inst_id = context_.insts().Get(inst_id).type_id();
    if (context_.types().Is<SemIR::FacetType>(type_id_of_inst_id)) {
      return SymbolicType();
    }
    if (type_id_of_inst_id != SemIR::TypeType::SingletonTypeId) {
      return SemIR::TypeId::None;
    }
    return context_.types().GetTypeIdForTypeInstId(inst_id);
  }
  auto GetInstIdAsTypeId(SemIR::InstId inst_id) const -> SemIR::TypeId {
    auto maybe_type_id = TryGetInstIdAsTypeId(inst_id);
    auto type_id = std::get<SemIR::TypeId>(maybe_type_id);
    CARBON_CHECK(type_id != SemIR::TypeId::None);
    return type_id;
  }

  // Sets the `distance` in `first_symbolic_distance_` if it does not already
  // have a non-infinite value.
  auto SetFirstSymbolic(int distance) -> void {
    if (first_symbolic_distance_ == TypeStructure::InfiniteDistance) {
      first_symbolic_distance_ = distance;
    }
  }

  auto GetSpecificArgs(SemIR::SpecificId specific_id)
      -> llvm::ArrayRef<SemIR::InstId> {
    if (specific_id == SemIR::SpecificId::None) {
      return {};
    }
    auto specific = context_.specifics().Get(specific_id);
    return context_.inst_blocks().Get(specific.args_id);
  }

  // Push all arguments from the array.
  auto PushArgs(llvm::ArrayRef<SemIR::InstId> args) -> void {
    for (auto arg_id : args) {
      auto maybe_type_id = TryGetInstIdAsTypeId(arg_id);
      if (std::holds_alternative<SymbolicType>(maybe_type_id)) {
        SetFirstSymbolic(position_);
      } else {
        // We may find non-types, but these are not symbolic, so we just count
        // their position but don't push anything.
        if (auto type_id = std::get<SemIR::TypeId>(maybe_type_id);
            type_id.has_value()) {
          Push(type_id);
        }
      }
      ++position_;
    }
  }

  auto Push(SemIR::TypeId type_id) -> void { work_list_.push_back(type_id); }

  Context& context_;
  llvm::SmallVector<SemIR::TypeId> work_list_;
  int position_;
  int first_symbolic_distance_;
  std::vector<TypeStructure::Structural> structure_;
};

auto BuildTypeStructure(Context& context, const SemIR::Impl& impl,
                        int priority_ordering) -> TypeStructure {
  TypeStructureBuilder builder(context);
  return builder.Run(impl, priority_ordering);
}

}  // namespace Carbon::Check
