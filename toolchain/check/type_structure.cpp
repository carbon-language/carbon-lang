// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/type_structure.h"

#include <variant>

#include "llvm/ADT/ScopeExit.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/sem_ir/facet_type_info.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/impl.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto TypeStructure::IsCompatibleWith(const TypeStructure& other) const -> bool {
  const auto& lhs = structure_;
  const auto& rhs = other.structure_;

  const auto* lhs_cursor = lhs.begin();
  const auto* rhs_cursor = rhs.begin();

  while (true) {
    // If both structures end at the same time, they match.
    if (lhs_cursor == lhs.end() && rhs_cursor == rhs.end()) {
      return true;
    }
    // If one structure ends sooner than the other, they don't match.
    if (lhs_cursor == lhs.end() || rhs_cursor == rhs.end()) {
      return false;
    }
    // Same structural element on both sides, they match and both are consumed.
    if (*lhs_cursor == *rhs_cursor) {
      ++lhs_cursor;
      ++rhs_cursor;
      continue;
    }
    // If the element on each side is concrete but they not the same structural
    // shape, then the structures don't match.
    if (*lhs_cursor != Structural::Symbolic &&
        *rhs_cursor != Structural::Symbolic) {
      return false;
    }

    // From here we know one side is a Symbolic and the other is not. We can
    // match the Symbolic against either a single Concrete or a larger bracketed
    // set of Concrete structural elements.
    //
    // We move the symbolic to the RHS to make only one case to handle below,
    // and make sure the cursors are swapped back at the bottom of the loop
    // iteration, so that they compare with the correct `end()`s.
    auto unswap =
        llvm::make_scope_exit([&] { std::swap(lhs_cursor, rhs_cursor); });
    if (*lhs_cursor == Structural::Symbolic) {
      std::swap(lhs_cursor, rhs_cursor);
    } else {
      unswap.release();
    }

    // Consume the symbolic on the RHS.
    ++rhs_cursor;

    // There's a Concrete on the LHS; it matches with the Symbolic and is
    // consumed.
    if (*lhs_cursor == Structural::Concrete) {
      ++lhs_cursor;
      continue;
    }

    // The symbolic on the RHS is in the same position as a close paren on the
    // LHS, which means the structures can not match.
    //
    // Example:
    // - ((c))
    // - ((c?))
    if (*lhs_cursor == Structural::ConcreteCloseParen) {
      return false;
    }

    // There's an open paren on the LHS; the Symbolic matches with everything on
    // the LHS up to the matching closing paren.
    CARBON_CHECK(*lhs_cursor == Structural::ConcreteOpenParen);
    int depth = 0;
    do {
      switch (*lhs_cursor) {
        case Structural::ConcreteOpenParen:
          depth += 1;
          break;
        case Structural::ConcreteCloseParen:
          depth -= 1;
          break;
        case Structural::Concrete:
          break;
        case Structural::Symbolic:
          break;
      }
      ++lhs_cursor;
    } while (depth > 0);
  }

  return true;
}

// A class that builds a `TypeStructure` for an `Impl` that represents its self
// type.
class TypeStructureBuilder {
 public:
  explicit TypeStructureBuilder(Context& context) : context_(context) {}

  auto Run(SemIR::InstId self_inst_id,
           SemIR::SpecificInterface interface_constraint) -> TypeStructure {
    CARBON_CHECK(work_list_.empty());

    first_symbolic_distance_ = TypeStructure::InfiniteDistance;
    structure_.clear();

    // The self type comes first in the type structure, so we push it last, as
    // the queue works from the back.
    Push(interface_constraint);
    PushInstId(self_inst_id);
    BuildTypeStructure();

    return TypeStructure(std::exchange(structure_, {}),
                         first_symbolic_distance_);
  }

 private:
  auto BuildTypeStructure() -> void {
    while (!work_list_.empty()) {
      auto next = work_list_.back();
      work_list_.pop_back();

      if (std::holds_alternative<CloseType>(next)) {
        auto close_type = std::get<CloseType>(next);
        switch (close_type.closing) {
          case CloseOnly:
            break;
          case CloseWithConcrete:
            AppendStructural(TypeStructure::Structural::Concrete);
            break;
          case CloseWithSymbolic:
            AppendStructural(TypeStructure::Structural::Symbolic);
            break;
        }
        AppendStructural(TypeStructure::Structural::ConcreteCloseParen);
        continue;
      }

      if (std::holds_alternative<SemIR::SpecificInterface>(next)) {
        const auto& interface = std::get<SemIR::SpecificInterface>(next);
        auto args = GetSpecificArgs(interface.specific_id);
        if (args.empty()) {
          AppendStructural(TypeStructure::Structural::Concrete);
        } else {
          AppendStructural(TypeStructure::Structural::ConcreteOpenParen);
          Push(CloseType());
          PushArgs(args);
        }
        continue;
      }

      if (std::holds_alternative<SymbolicType>(next)) {
        AppendStructural(TypeStructure::Structural::Symbolic);
        continue;
      }

      if (std::holds_alternative<NonTypeValue>(next)) {
        AppendStructural(TypeStructure::Structural::Concrete);
        continue;
      }

      SemIR::TypeId next_type_id = std::get<SemIR::TypeId>(next);
      auto inst_id = context_.types().GetInstId(next_type_id);
      auto inst = context_.insts().Get(inst_id);
      CARBON_KIND_SWITCH(inst) {
          // ==== Symbolic types ====

        case CARBON_KIND(SemIR::BindSymbolicName type): {
          (void)type;
          Push(SymbolicType());
          break;
        }
        case CARBON_KIND(SemIR::SymbolicBindingPattern type): {
          (void)type;
          Push(SymbolicType());
          break;
        }
        case CARBON_KIND(SemIR::FacetAccessType type): {
          (void)type;
          Push(SymbolicType());
          break;
        }

          // ==== Concrete types ====

        case CARBON_KIND(SemIR::AssociatedEntityType type): {
          (void)type;
          AppendStructural(TypeStructure::Structural::Concrete);
          break;
        }
        case CARBON_KIND(SemIR::BoolType type): {
          (void)type;
          AppendStructural(TypeStructure::Structural::Concrete);
          break;
        }
        case CARBON_KIND(SemIR::FloatType type): {
          (void)type;
          AppendStructural(TypeStructure::Structural::Concrete);
          break;
        }
        case CARBON_KIND(SemIR::GenericClassType type): {
          (void)type;
          AppendStructural(TypeStructure::Structural::Concrete);
          break;
        }
        case CARBON_KIND(SemIR::GenericInterfaceType type): {
          (void)type;
          AppendStructural(TypeStructure::Structural::Concrete);
          break;
        }
        case CARBON_KIND(SemIR::ImplWitnessAccess type): {
          (void)type;
          AppendStructural(TypeStructure::Structural::Concrete);
          break;
        }
        case CARBON_KIND(SemIR::IntType int_type): {
          if (context_.constant_values().Get(inst_id).is_concrete()) {
            AppendStructural(TypeStructure::Structural::Concrete);
          } else {
            AppendStructural(TypeStructure::Structural::ConcreteOpenParen);
            Push(CloseType{});
            PushArgs({int_type.bit_width_id});
          }
          break;
        }
        case CARBON_KIND(SemIR::IntLiteralType type): {
          (void)type;
          AppendStructural(TypeStructure::Structural::Concrete);
          break;
        }
        case CARBON_KIND(SemIR::LegacyFloatType type): {
          (void)type;
          AppendStructural(TypeStructure::Structural::Concrete);
          break;
        }
        case CARBON_KIND(SemIR::PointerType type): {
          (void)type;
          AppendStructural(TypeStructure::Structural::Concrete);
          break;
        }
        case CARBON_KIND(SemIR::StringType type): {
          (void)type;
          AppendStructural(TypeStructure::Structural::Concrete);
          break;
        }
        case CARBON_KIND(SemIR::TypeType type): {
          (void)type;
          AppendStructural(TypeStructure::Structural::Concrete);
          break;
        }

          // ==== Aggregate types ====

        case CARBON_KIND(SemIR::ArrayType array_type): {
          AppendStructural(TypeStructure::Structural::ConcreteOpenParen);
          Push(CloseType{});
          Push(array_type.element_type_id);
          PushInstId(array_type.bound_id);
          break;
        }
        case CARBON_KIND(SemIR::ClassType class_type): {
          auto args = GetSpecificArgs(class_type.specific_id);
          if (args.empty()) {
            AppendStructural(TypeStructure::Structural::Concrete);
          } else {
            AppendStructural(TypeStructure::Structural::ConcreteOpenParen);
            Push(CloseType{});
            PushArgs(args);
          }
          break;
        }
        case CARBON_KIND(SemIR::ConstType const_type): {
          // We don't put the `const` into the type structure since it is a
          // modifier; just move to the inner type.
          Push(const_type.inner_id);
          break;
        }
        case CARBON_KIND(SemIR::FacetType facet_type): {
          auto facet_type_info =
              context_.facet_types().Get(facet_type.facet_type_id);
          AppendStructural(TypeStructure::Structural::ConcreteOpenParen);
          // TODO: The need for `.closing` goes away when other_requirements
          // does. Are there other places we need to look for symbolics in
          // FacetTypeInfo at that point? For now we treat it as having a
          // symbolic at the end of the facet type, so facet types with
          // other_requirements are chosen with lower priority than those
          // without.
          Push(CloseType{.closing = facet_type_info.other_requirements
                                        ? CloseWithSymbolic
                                        : CloseWithConcrete});
          for (const auto& i : facet_type_info.impls_constraints) {
            PushArgs(GetSpecificArgs(i.specific_id));
          }
          break;
        }
        case CARBON_KIND(SemIR::TupleType tuple_type): {
          auto inner_types = context_.type_blocks().Get(tuple_type.elements_id);
          if (inner_types.empty()) {
            AppendStructural(TypeStructure::Structural::Concrete);
          } else {
            AppendStructural(TypeStructure::Structural::ConcreteOpenParen);
            Push(CloseType{});
            for (auto type :
                 context_.type_blocks().Get(tuple_type.elements_id)) {
              Push(type);
            }
          }
          break;
        }
        case CARBON_KIND(SemIR::StructType struct_type): {
          auto fields =
              context_.struct_type_fields().Get(struct_type.fields_id);
          if (fields.empty()) {
            AppendStructural(TypeStructure::Structural::Concrete);
          } else {
            AppendStructural(TypeStructure::Structural::ConcreteOpenParen);
            Push(CloseType{});
            for (const auto& field : fields) {
              Push(field.type_id);
            }
          }
          break;
        }
        default:
          CARBON_FATAL("Unhandled type instruction {0}", inst_id);
      }
    }
  }

  enum CloseTypeClosing {
    CloseOnly,
    CloseWithConcrete,
    CloseWithSymbolic,
  };
  struct CloseType {
    CloseTypeClosing closing = CloseOnly;
  };
  struct SymbolicType {};
  struct NonTypeValue {};

  using WorkItem = std::variant<SemIR::TypeId, SymbolicType, NonTypeValue,
                                SemIR::SpecificInterface, CloseType>;

  auto TryGetInstIdAsTypeId(SemIR::InstId inst_id) const
      -> std::variant<SemIR::TypeId, SymbolicType> {
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
    for (auto arg_id : llvm::reverse(args)) {
      PushInstId(arg_id);
    }
  }

  auto PushInstId(SemIR::InstId inst_id) -> void {
    auto maybe_type_id = TryGetInstIdAsTypeId(inst_id);
    if (std::holds_alternative<SymbolicType>(maybe_type_id)) {
      Push(SymbolicType());
    } else if (auto type_id = std::get<SemIR::TypeId>(maybe_type_id);
               type_id.has_value()) {
      Push(type_id);
    } else {
      Push(NonTypeValue());
    }
  }

  auto Push(WorkItem item) -> void { work_list_.push_back(item); }

  auto AppendStructural(TypeStructure::Structural structural) -> void {
    if (structural == TypeStructure::Structural::Symbolic) {
      // Sets the `distance` in `first_symbolic_distance_` if it does not
      // already have a non-infinite value.
      if (first_symbolic_distance_ == TypeStructure::InfiniteDistance) {
        first_symbolic_distance_ = structure_.size();
      }
    }
    structure_.push_back(structural);
  }

  Context& context_;
  llvm::SmallVector<WorkItem> work_list_;
  int first_symbolic_distance_;
  llvm::SmallVector<TypeStructure::Structural> structure_;
};

auto BuildTypeStructure(Context& context, SemIR::InstId self_inst_id,
                        SemIR::SpecificInterface interface) -> TypeStructure {
  TypeStructureBuilder builder(context);
  return builder.Run(self_inst_id, interface);
}

}  // namespace Carbon::Check
