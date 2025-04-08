// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/type_structure.h"

#include <variant>

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/sem_ir/constant.h"
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
    //
    // TODO: If we kept the constant value of the concrete element in the type
    // structure, then we could compare them and use that to eliminate matching
    // impls that are not actually compatible.
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

    // Returns false if the lhs and rhs can not match, true if we should
    // continue checking for compatibility.
    auto consume_symbolic = [](const auto*& lhs_cursor,
                               const auto*& rhs_cursor) -> bool {
      // Consume the symbolic on the RHS.
      ++rhs_cursor;

      // The symbolic on the RHS is in the same position as a close paren on the
      // LHS, which means the structures can not match.
      //
      // Example:
      // - ((c))
      // - ((c?))
      if (*lhs_cursor == Structural::ConcreteCloseParen) {
        return false;
      }

      // There's either a Concrete element or an open paren on the LHS. If it's
      // the former, the Symbolic just matches with it. If it's the latter, the
      // Symbolic matches with everything on the LHS up to the matching closing
      // paren.
      CARBON_CHECK(*lhs_cursor == Structural::Concrete ||
                   *lhs_cursor == Structural::ConcreteOpenParen);
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
      return true;
    };

    // We move the symbolic to the RHS to make only one case to handle in the
    // lambda.
    if (*lhs_cursor == Structural::Symbolic) {
      if (!consume_symbolic(rhs_cursor, lhs_cursor)) {
        return false;
      }
    } else {
      if (!consume_symbolic(lhs_cursor, rhs_cursor)) {
        return false;
      }
    }
  }

  return true;
}

// A class that builds a `TypeStructure` for an `Impl`, or an impl lookup query,
// that represents its self type and interface.
class TypeStructureBuilder {
 public:
  // `context` must not be null.
  explicit TypeStructureBuilder(Context* context) : context_(context) {}

  auto Run(SemIR::InstId self_inst_id,
           SemIR::SpecificInterface interface_constraint) -> TypeStructure {
    CARBON_CHECK(work_list_.empty());

    symbolic_type_indices_.clear();
    structure_.clear();

    // The self type comes first in the type structure, so we push it last, as
    // the queue works from the back.
    Push(interface_constraint);
    PushInstId(self_inst_id);
    BuildTypeStructure();

    // TODO: This requires 4 SmallVector moves (two here and two in the
    // constructor). Find a way to reduce that.
    return TypeStructure(std::exchange(structure_, {}),
                         std::exchange(symbolic_type_indices_, {}));
  }

 private:
  auto BuildTypeStructure() -> void {
    while (!work_list_.empty()) {
      auto next = work_list_.back();
      work_list_.pop_back();

      if (std::holds_alternative<CloseType>(next)) {
        AppendStructural(TypeStructure::Structural::ConcreteCloseParen);
        continue;
      }

      if (const auto* interface =
              std::get_if<SemIR::SpecificInterface>(&next)) {
        auto args = GetSpecificArgs(interface->specific_id);
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
        // TODO: Include the value's type into the structure, with the type
        // coming first and paired together with the value, like:
        // `{TypeWithPossibleNestedTypes, Concrete}`.
        // We might want a different bracket marker than ConcreteOpenParen for
        // this so that it can look different in the type structure when dumped.
        AppendStructural(TypeStructure::Structural::Concrete);
        continue;
      }

      SemIR::TypeId next_type_id = std::get<SemIR::TypeId>(next);
      auto inst_id = context_->types().GetInstId(next_type_id);
      auto inst = context_->insts().Get(inst_id);
      CARBON_KIND_SWITCH(inst) {
          // ==== Symbolic types ====

        case SemIR::BindSymbolicName::Kind:
        case SemIR::SymbolicBindingPattern::Kind:
        case SemIR::FacetAccessType::Kind: {
          Push(SymbolicType());
          break;
        }
        case SemIR::TypeOfInst::Kind: {
          // TODO: For a template value with a fixed type, such as `template n:!
          // i32`, we could look at the type of the value to see if it's
          // template-dependent (which it's not here) and add that type to the
          // type structure?
          // https://github.com/carbon-language/carbon-lang/pull/5124#discussion_r2006617038
          Push(SymbolicType());
          break;
        }

          // ==== Concrete types ====

        case SemIR::AssociatedEntityType::Kind:
        case SemIR::BoolType::Kind:
        case SemIR::FloatType::Kind:
        case SemIR::GenericClassType::Kind:
        case SemIR::GenericInterfaceType::Kind:
        case SemIR::ImplWitnessAccess::Kind:
        case SemIR::IntLiteralType::Kind:
        case SemIR::LegacyFloatType::Kind:
        case SemIR::StringType::Kind:
        case SemIR::TypeType::Kind:
        case SemIR::WitnessType::Kind: {
          AppendStructural(TypeStructure::Structural::Concrete);
          break;
        }

        case CARBON_KIND(SemIR::FacetType facet_type): {
          (void)facet_type;
          // A `FacetType` instruction shows up in the self type of impl lookup
          // queries like `C(D)` where `C` requires its parameter to satisfy
          // some `FacetType` `Z`. The `D` argument is converted to a
          // `FacetValue` satisfying `Z`, and the type of `C` in the self type
          // has a specific with the type of that `FacetValue`, which is the
          // `FacetType` satisfying `Z` we see here.
          //
          // The `FacetValue` may still be symbolic in generic code but its
          // type, the `FacetType` here, is concrete.
          AppendStructural(TypeStructure::Structural::Concrete);
          break;
        }
        case CARBON_KIND(SemIR::IntType int_type): {
          if (context_->constant_values().Get(inst_id).is_concrete()) {
            AppendStructural(TypeStructure::Structural::Concrete);
          } else {
            AppendStructural(TypeStructure::Structural::ConcreteOpenParen);
            Push(CloseType());
            PushArgs({int_type.bit_width_id});
          }
          break;
        }

          // ==== Aggregate types ====

        case CARBON_KIND(SemIR::ArrayType array_type): {
          AppendStructural(TypeStructure::Structural::ConcreteOpenParen);
          Push(CloseType());
          PushInstId(array_type.element_type_inst_id);
          PushInstId(array_type.bound_id);
          break;
        }
        case CARBON_KIND(SemIR::ClassType class_type): {
          auto args = GetSpecificArgs(class_type.specific_id);
          if (args.empty()) {
            AppendStructural(TypeStructure::Structural::Concrete);
          } else {
            AppendStructural(TypeStructure::Structural::ConcreteOpenParen);
            Push(CloseType());
            PushArgs(args);
          }
          break;
        }
        case CARBON_KIND(SemIR::ConstType const_type): {
          // We don't put the `const` into the type structure since it is a
          // modifier; just move to the inner type.
          PushInstId(const_type.inner_id);
          break;
        }
        case CARBON_KIND(SemIR::PointerType pointer_type): {
          AppendStructural(TypeStructure::Structural::ConcreteOpenParen);
          Push(CloseType());
          PushInstId(pointer_type.pointee_id);
          break;
        }
        case CARBON_KIND(SemIR::TupleType tuple_type): {
          auto inner_types =
              context_->type_blocks().Get(tuple_type.elements_id);
          if (inner_types.empty()) {
            AppendStructural(TypeStructure::Structural::Concrete);
          } else {
            AppendStructural(TypeStructure::Structural::ConcreteOpenParen);
            Push(CloseType());
            for (auto type :
                 context_->type_blocks().Get(tuple_type.elements_id)) {
              Push(type);
            }
          }
          break;
        }
        case CARBON_KIND(SemIR::StructType struct_type): {
          auto fields =
              context_->struct_type_fields().Get(struct_type.fields_id);
          if (fields.empty()) {
            AppendStructural(TypeStructure::Structural::Concrete);
          } else {
            AppendStructural(TypeStructure::Structural::ConcreteOpenParen);
            Push(CloseType());
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

  // A work item to mark the closing paren for an aggregate concrete type.
  struct CloseType {};
  // A work item to mark a symbolic type.
  struct SymbolicType {};
  // A work item to mark a non-type value.
  struct NonTypeValue {};

  using WorkItem = std::variant<SemIR::TypeId, SymbolicType, NonTypeValue,
                                SemIR::SpecificInterface, CloseType>;

  // Get the TypeId for an instruction that is not a facet value, otherwise
  // return SymbolicType to indicate the instruction is a symbolic facet value.
  //
  // If the instruction is not a type value, the return is TypeId::None.
  //
  // We reuse the `SymbolicType` work item here to give a nice return type.
  auto TryGetInstIdAsTypeId(SemIR::InstId inst_id) const
      -> std::variant<SemIR::TypeId, SymbolicType> {
    if (auto facet_value =
            context_->insts().TryGetAs<SemIR::FacetValue>(inst_id)) {
      inst_id = facet_value->type_inst_id;
    }

    auto type_id_of_inst_id = context_->insts().Get(inst_id).type_id();
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
    if (context_->types().Is<SemIR::FacetType>(type_id_of_inst_id)) {
      return SymbolicType();
    }
    // Non-type values are concrete, only types are symbolic.
    if (type_id_of_inst_id != SemIR::TypeType::SingletonTypeId) {
      return SemIR::TypeId::None;
    }
    return context_->types().GetTypeIdForTypeInstId(inst_id);
  }

  // Get the instructions in the specific's instruction block as an ArrayRef.
  auto GetSpecificArgs(SemIR::SpecificId specific_id)
      -> llvm::ArrayRef<SemIR::InstId> {
    if (specific_id == SemIR::SpecificId::None) {
      return {};
    }
    auto specific = context_->specifics().Get(specific_id);
    return context_->inst_blocks().Get(specific.args_id);
  }

  // Push all arguments from the array into the work queue.
  auto PushArgs(llvm::ArrayRef<SemIR::InstId> args) -> void {
    for (auto arg_id : llvm::reverse(args)) {
      PushInstId(arg_id);
    }
  }

  // Push an instruction's type value into the work queue, or a marker if the
  // instruction has a symbolic value.
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

  // Push the next step into the work queue.
  auto Push(WorkItem item) -> void { work_list_.push_back(item); }

  // Append a structural element to the TypeStructure being built.
  auto AppendStructural(TypeStructure::Structural structural) -> void {
    if (structural == TypeStructure::Structural::Symbolic) {
      symbolic_type_indices_.push_back(structure_.size());
    }
    structure_.push_back(structural);
  }

  Context* context_;
  llvm::SmallVector<WorkItem> work_list_;
  llvm::SmallVector<int> symbolic_type_indices_;
  llvm::SmallVector<TypeStructure::Structural> structure_;
};

auto BuildTypeStructure(Context& context, SemIR::InstId self_inst_id,
                        SemIR::SpecificInterface interface) -> TypeStructure {
  TypeStructureBuilder builder(&context);
  return builder.Run(self_inst_id, interface);
}

}  // namespace Carbon::Check
