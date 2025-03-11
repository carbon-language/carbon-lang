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

// A class that builds a `TypeStructure` for an `Impl` that represents its self
// type.
class TypeStructureBuilder {
 public:
  explicit TypeStructureBuilder(Context& context) : context_(context) {}

  auto Run(const SemIR::Impl& impl, SemIR::InstId witness_id,
           int priority_ordering = 0) -> TypeStructure {
    CARBON_CHECK(work_list_.empty());

    int distance_to_first_symbolic_type;
    if (witness_id.has_value() && impl.generic_id.has_value()) {
      // Compute distance to the first symbolic type in the impl's self and
      // constraint types.
      Push(GetInstIdAsTypeId(impl.constraint_id));
      Push(GetInstIdAsTypeId(impl.self_id));
      distance_to_first_symbolic_type = FindDistanceToFirstSymbolicType();
    } else {
      // If there's no symbolic type in the impl's self type, then we use an
      // infinite distance.
      distance_to_first_symbolic_type = TypeStructure::InfiniteDistance;
    }

    return TypeStructure(distance_to_first_symbolic_type, priority_ordering,
                         witness_id, impl.interface.interface_id);
  }

 private:
  auto FindDistanceToFirstSymbolicType() -> int {
    int distance = 0;

    while (!work_list_.empty()) {
      SemIR::TypeId next_type_id = work_list_.back();
      work_list_.pop_back();

      auto inst_id = context_.types().GetInstId(next_type_id);
      auto inst = context_.insts().Get(inst_id);
      CARBON_KIND_SWITCH(inst) {
        case CARBON_KIND(SemIR::BindSymbolicName bind): {
          // We found a symbolic type.
          (void)bind;
          return distance;
        }
        case CARBON_KIND(SemIR::SymbolicBindingPattern bind): {
          // We found a symbolic type.
          (void)bind;
          return distance;
        }
        case CARBON_KIND(SemIR::FacetAccessType access): {
          // We found a symbolic type (referenced by the FacetAccessType).
          (void)access;
          return distance;
        }
        case CARBON_KIND(SemIR::ArrayType array_type): {
          ++distance;
          Push(array_type.element_type_id);
          break;
        }
        case CARBON_KIND(SemIR::ClassType class_type): {
          ++distance;
          if (!PushArgs(class_type.specific_id, distance)) {
            return distance;
          }
          break;
        }
        case CARBON_KIND(SemIR::FacetType facet_type): {
          ++distance;
          auto f = context_.facet_types().Get(facet_type.facet_type_id);
          for (const auto& i : f.impls_constraints) {
            if (!PushArgs(i.specific_id, distance)) {
              return distance;
            }
          }
          if (f.other_requirements) {
            // TODO: This goes away when other_requirements does. Are there
            // other places we need to look for symbolics in FacetTypeInfo at
            // that point? For now we treat it as having a symbolic at the end
            // of the facet type, so facet types with other_requirements are
            // chosen with lower priority than those without.
            return distance;
          }
          break;
        }
        case CARBON_KIND(SemIR::TupleType tuple_type): {
          ++distance;
          for (auto type : context_.type_blocks().Get(tuple_type.elements_id)) {
            Push(type);
          }
          break;
        }
        case CARBON_KIND(SemIR::StructType struct_type): {
          ++distance;
          auto fields =
              context_.struct_type_fields().Get(struct_type.fields_id);
          for (const auto& field : fields) {
            Push(field.type_id);
          }
          break;
        }
        case CARBON_KIND(SemIR::IntLiteralType int_literal): {
          (void)int_literal;
          ++distance;
          break;
        }
        default:
          CARBON_FATAL("Unhandled type instruction");
      }
    }

    return TypeStructure::InfiniteDistance;  // No symbolic type found.
  }

  struct Symbolic {};

  auto TryGetInstIdAsTypeId(SemIR::InstId inst_id) const
      -> std::variant<Symbolic, SemIR::TypeId> {
    if (auto facet_value =
            context_.insts().TryGetAs<SemIR::FacetValue>(inst_id)) {
      inst_id = facet_value->type_inst_id;
    }

    auto type_id_of_inst_id = context_.insts().Get(inst_id).type_id();
    if (context_.types().Is<SemIR::FacetType>(type_id_of_inst_id)) {
      return Symbolic();
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

  // Push all arguments from the specific. If a symbolic argument is found,
  // returns false indicating that the caller now knows the final distance to a
  // symbolic.
  auto PushArgs(SemIR::SpecificId specific_id, int& distance) -> bool {
    if (specific_id == SemIR::SpecificId::None) {
      return true;
    }
    auto specific = context_.specifics().Get(specific_id);
    for (auto param_id : context_.inst_blocks().Get(specific.args_id)) {
      auto maybe_type_id = TryGetInstIdAsTypeId(param_id);
      if (std::holds_alternative<Symbolic>(maybe_type_id)) {
        return false;
      }
      auto type_id = std::get<SemIR::TypeId>(maybe_type_id);
      ++distance;
      // If we find a non-type, which is therefore not symbolic, we count it and
      // move on.
      if (type_id != SemIR::TypeId::None) {
        Push(type_id);
      }
    }
    return true;
  }

  auto Push(SemIR::TypeId type_id) -> void { work_list_.push_back(type_id); }

  Context& context_;
  llvm::SmallVector<SemIR::TypeId> work_list_;
};

auto BuildTypeStructure(Context& context, const SemIR::Impl& impl,
                        SemIR::InstId witness_id, int priority_ordering)
    -> TypeStructure {
  TypeStructureBuilder builder(context);
  return builder.Run(impl, witness_id, priority_ordering);
}

}  // namespace Carbon::Check
