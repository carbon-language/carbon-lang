// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/type_completion.h"

#include "llvm/ADT/SmallVector.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/type.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

namespace {
// Worklist-based type completion mechanism.
//
// When attempting to complete a type, we may find other types that also need to
// be completed: types nested within that type, and the value representation of
// the type. In order to complete a type without recursing arbitrarily deeply,
// we use a worklist of tasks:
//
// - An `AddNestedIncompleteTypes` step adds a task for all incomplete types
//   nested within a type to the work list.
// - A `BuildInfo` step computes the `CompleteTypeInfo` for a type, once all of
//   its nested types are complete, and marks the type as complete.
class TypeCompleter {
 public:
  // `context` mut not be null.
  TypeCompleter(Context* context, SemIRLoc loc,
                MakeDiagnosticBuilderFn diagnoser)
      : context_(context), loc_(loc), diagnoser_(diagnoser) {}

  // Attempts to complete the given type. Returns true if it is now complete,
  // false if it could not be completed.
  auto Complete(SemIR::TypeId type_id) -> bool;

 private:
  enum class Phase : int8_t {
    // The next step is to add nested types to the list of types to complete.
    AddNestedIncompleteTypes,
    // The next step is to build the `CompleteTypeInfo` for the type.
    BuildInfo,
  };

  struct WorkItem {
    SemIR::TypeId type_id;
    Phase phase;
  };

  // Adds `type_id` to the work list, if it's not already complete.
  auto Push(SemIR::TypeId type_id) -> void;

  // Runs the next step.
  auto ProcessStep() -> bool;

  // Adds any types nested within `type_inst` that need to be complete for
  // `type_inst` to be complete to our work list.
  auto AddNestedIncompleteTypes(SemIR::Inst type_inst) -> bool;

  // Makes an empty value representation, which is used for types that have no
  // state, such as empty structs and tuples.
  auto MakeEmptyValueRepr() const -> SemIR::ValueRepr;

  // Makes a value representation that uses pass-by-copy, copying the given
  // type.
  auto MakeCopyValueRepr(SemIR::TypeId rep_id,
                         SemIR::ValueRepr::AggregateKind aggregate_kind =
                             SemIR::ValueRepr::NotAggregate) const
      -> SemIR::ValueRepr;

  // Makes a value representation that uses pass-by-address with the given
  // pointee type.
  auto MakePointerValueRepr(SemIR::TypeId pointee_id,
                            SemIR::ValueRepr::AggregateKind aggregate_kind =
                                SemIR::ValueRepr::NotAggregate) const
      -> SemIR::ValueRepr;

  // Gets the value representation of a nested type, which should already be
  // complete.
  auto GetNestedInfo(SemIR::TypeId nested_type_id) const
      -> SemIR::CompleteTypeInfo;

  template <typename InstT>
    requires(InstT::Kind.template IsAnyOf<
             SemIR::AutoType, SemIR::BoolType, SemIR::BoundMethodType,
             SemIR::ErrorInst, SemIR::FacetType, SemIR::FloatType,
             SemIR::IntType, SemIR::IntLiteralType, SemIR::LegacyFloatType,
             SemIR::NamespaceType, SemIR::PointerType,
             SemIR::SpecificFunctionType, SemIR::TypeType, SemIR::VtableType,
             SemIR::WitnessType>())
  auto BuildInfoForInst(SemIR::TypeId type_id, InstT /*inst*/) const
      -> SemIR::CompleteTypeInfo {
    return {.value_repr = MakeCopyValueRepr(type_id)};
  }

  auto BuildInfoForInst(SemIR::TypeId type_id, SemIR::StringType /*inst*/) const
      -> SemIR::CompleteTypeInfo;

  auto BuildStructOrTupleValueRepr(size_t num_elements,
                                   SemIR::TypeId elementwise_rep,
                                   bool same_as_object_rep) const
      -> SemIR::ValueRepr;

  auto BuildInfoForInst(SemIR::TypeId type_id,
                        SemIR::StructType struct_type) const
      -> SemIR::CompleteTypeInfo;

  auto BuildInfoForInst(SemIR::TypeId type_id,
                        SemIR::TupleType tuple_type) const
      -> SemIR::CompleteTypeInfo;

  auto BuildInfoForInst(SemIR::TypeId type_id, SemIR::ArrayType /*inst*/) const
      -> SemIR::CompleteTypeInfo;

  auto BuildInfoForInst(SemIR::TypeId /*type_id*/, SemIR::ClassType inst) const
      -> SemIR::CompleteTypeInfo;

  template <typename InstT>
    requires(InstT::Kind.template IsAnyOf<
             SemIR::AssociatedEntityType, SemIR::FunctionType,
             SemIR::FunctionTypeWithSelfType, SemIR::GenericClassType,
             SemIR::GenericInterfaceType, SemIR::InstType,
             SemIR::UnboundElementType, SemIR::WhereExpr>())
  auto BuildInfoForInst(SemIR::TypeId /*type_id*/, InstT /*inst*/) const
      -> SemIR::CompleteTypeInfo {
    // These types have no runtime operations, so we use an empty value
    // representation.
    //
    // TODO: There is information we could model here:
    // - For an interface, we could use a witness.
    // - For an associated entity, we could use an index into the witness.
    // - For an unbound element, we could use an index or offset.
    return {.value_repr = MakeEmptyValueRepr()};
  }

  auto BuildInfoForInst(SemIR::TypeId /*type_id*/, SemIR::ConstType inst) const
      -> SemIR::CompleteTypeInfo;

  template <typename InstT>
    requires(InstT::Kind.constant_kind() ==
                 SemIR::InstConstantKind::SymbolicOnly ||
             InstT::Kind.is_type() == SemIR::InstIsType::Never)
  auto BuildInfoForInst(SemIR::TypeId type_id, InstT inst) const
      -> SemIR::CompleteTypeInfo {
    if constexpr (InstT::Kind.is_type() == SemIR::InstIsType::Never) {
      CARBON_FATAL("Type refers to non-type inst {0}", inst);
    } else {
      // For symbolic types, we arbitrarily pick a copy representation.
      return {.value_repr = MakeCopyValueRepr(type_id)};
    }
  }

  // Builds and returns the `CompleteTypeInfo` for the given type. All nested
  // types, as found by AddNestedIncompleteTypes, are known to be complete.
  auto BuildInfo(SemIR::TypeId type_id, SemIR::Inst inst) const
      -> SemIR::CompleteTypeInfo;

  Context* context_;
  llvm::SmallVector<WorkItem> work_list_;
  SemIRLoc loc_;
  MakeDiagnosticBuilderFn diagnoser_;
};
}  // namespace

auto TypeCompleter::Complete(SemIR::TypeId type_id) -> bool {
  Push(type_id);
  while (!work_list_.empty()) {
    if (!ProcessStep()) {
      return false;
    }
  }
  return true;
}

auto TypeCompleter::Push(SemIR::TypeId type_id) -> void {
  if (!context_->types().IsComplete(type_id)) {
    work_list_.push_back(
        {.type_id = type_id, .phase = Phase::AddNestedIncompleteTypes});
  }
}

auto TypeCompleter::ProcessStep() -> bool {
  auto [type_id, phase] = work_list_.back();

  // We might have enqueued the same type more than once. Just skip the
  // type if it's already complete.
  if (context_->types().IsComplete(type_id)) {
    work_list_.pop_back();
    return true;
  }

  auto inst_id = context_->types().GetInstId(type_id);
  auto inst = context_->insts().Get(inst_id);
  auto old_work_list_size = work_list_.size();

  switch (phase) {
    case Phase::AddNestedIncompleteTypes:
      if (!AddNestedIncompleteTypes(inst)) {
        return false;
      }
      CARBON_CHECK(work_list_.size() >= old_work_list_size,
                   "AddNestedIncompleteTypes should not remove work items");
      work_list_[old_work_list_size - 1].phase = Phase::BuildInfo;
      break;

    case Phase::BuildInfo: {
      auto info = BuildInfo(type_id, inst);
      context_->types().SetComplete(type_id, info);
      CARBON_CHECK(old_work_list_size == work_list_.size(),
                   "BuildInfo should not change work items");
      work_list_.pop_back();

      // Also complete the value representation type, if necessary. This
      // should never fail: the value representation shouldn't require any
      // additional nested types to be complete.
      if (!context_->types().IsComplete(info.value_repr.type_id)) {
        work_list_.push_back(
            {.type_id = info.value_repr.type_id, .phase = Phase::BuildInfo});
      }
      // For a pointer representation, the pointee also needs to be complete.
      if (info.value_repr.kind == SemIR::ValueRepr::Pointer) {
        if (info.value_repr.type_id == SemIR::ErrorInst::SingletonTypeId) {
          break;
        }
        auto pointee_type_id =
            context_->sem_ir().GetPointeeType(info.value_repr.type_id);
        if (!context_->types().IsComplete(pointee_type_id)) {
          work_list_.push_back(
              {.type_id = pointee_type_id, .phase = Phase::BuildInfo});
        }
      }
      break;
    }
  }

  return true;
}

auto TypeCompleter::AddNestedIncompleteTypes(SemIR::Inst type_inst) -> bool {
  CARBON_KIND_SWITCH(type_inst) {
    case CARBON_KIND(SemIR::ArrayType inst): {
      Push(context_->types().GetTypeIdForTypeInstId(inst.element_type_inst_id));
      break;
    }
    case CARBON_KIND(SemIR::StructType inst): {
      for (auto field : context_->struct_type_fields().Get(inst.fields_id)) {
        Push(field.type_id);
      }
      break;
    }
    case CARBON_KIND(SemIR::TupleType inst): {
      for (auto element_type_id :
           context_->type_blocks().Get(inst.elements_id)) {
        Push(element_type_id);
      }
      break;
    }
    case CARBON_KIND(SemIR::ClassType inst): {
      auto& class_info = context_->classes().Get(inst.class_id);
      if (!class_info.is_complete()) {
        if (diagnoser_) {
          auto builder = diagnoser_();
          NoteIncompleteClass(*context_, inst.class_id, builder);
          builder.Emit();
        }
        return false;
      }
      if (inst.specific_id.has_value()) {
        ResolveSpecificDefinition(*context_, loc_, inst.specific_id);
      }
      if (auto adapted_type_id =
              class_info.GetAdaptedType(context_->sem_ir(), inst.specific_id);
          adapted_type_id.has_value()) {
        Push(adapted_type_id);
      } else {
        Push(class_info.GetObjectRepr(context_->sem_ir(), inst.specific_id));
      }
      break;
    }
    case CARBON_KIND(SemIR::ConstType inst): {
      Push(context_->types().GetTypeIdForTypeInstId(inst.inner_id));
      break;
    }
    case CARBON_KIND(SemIR::FacetType inst): {
      auto identified_id = RequireIdentifiedFacetType(*context_, inst);
      const auto& identified =
          context_->identified_facet_types().Get(identified_id);
      // Every mentioned interface needs to be complete.
      for (auto req_interface : identified.required_interfaces()) {
        auto interface_id = req_interface.interface_id;
        const auto& interface = context_->interfaces().Get(interface_id);
        if (!interface.is_complete()) {
          if (diagnoser_) {
            auto builder = diagnoser_();
            NoteIncompleteInterface(*context_, interface_id, builder);
            builder.Emit();
          }
          return false;
        }

        if (req_interface.specific_id.has_value()) {
          ResolveSpecificDefinition(*context_, loc_, req_interface.specific_id);
        }
      }
      break;
    }

    default:
      break;
  }

  return true;
}

auto TypeCompleter::MakeEmptyValueRepr() const -> SemIR::ValueRepr {
  return {.kind = SemIR::ValueRepr::None,
          .type_id = GetTupleType(*context_, {})};
}

auto TypeCompleter::MakeCopyValueRepr(
    SemIR::TypeId rep_id, SemIR::ValueRepr::AggregateKind aggregate_kind) const
    -> SemIR::ValueRepr {
  return {.kind = SemIR::ValueRepr::Copy,
          .aggregate_kind = aggregate_kind,
          .type_id = rep_id};
}

auto TypeCompleter::MakePointerValueRepr(
    SemIR::TypeId pointee_id,
    SemIR::ValueRepr::AggregateKind aggregate_kind) const -> SemIR::ValueRepr {
  // TODO: Should we add `const` qualification to `pointee_id`?
  return {.kind = SemIR::ValueRepr::Pointer,
          .aggregate_kind = aggregate_kind,
          .type_id = GetPointerType(*context_,
                                    context_->types().GetInstId(pointee_id))};
}

auto TypeCompleter::GetNestedInfo(SemIR::TypeId nested_type_id) const
    -> SemIR::CompleteTypeInfo {
  CARBON_CHECK(context_->types().IsComplete(nested_type_id),
               "Nested type should already be complete");
  auto info = context_->types().GetCompleteTypeInfo(nested_type_id);
  CARBON_CHECK(info.value_repr.kind != SemIR::ValueRepr::Unknown,
               "Complete type should have a value representation");
  return info;
}

auto TypeCompleter::BuildInfoForInst(SemIR::TypeId type_id,
                                     SemIR::StringType /*inst*/) const
    -> SemIR::CompleteTypeInfo {
  // TODO: Decide on string value semantics. This should probably be a
  // custom value representation carrying a pointer and size or
  // similar.
  return {.value_repr = MakePointerValueRepr(type_id)};
}

auto TypeCompleter::BuildStructOrTupleValueRepr(size_t num_elements,
                                                SemIR::TypeId elementwise_rep,
                                                bool same_as_object_rep) const
    -> SemIR::ValueRepr {
  SemIR::ValueRepr::AggregateKind aggregate_kind =
      same_as_object_rep ? SemIR::ValueRepr::ValueAndObjectAggregate
                         : SemIR::ValueRepr::ValueAggregate;

  if (num_elements == 1) {
    // The value representation for a struct or tuple with a single element
    // is a struct or tuple containing the value representation of the
    // element.
    // TODO: Consider doing the same whenever `elementwise_rep` is
    // sufficiently small.
    return MakeCopyValueRepr(elementwise_rep, aggregate_kind);
  }
  // For a struct or tuple with multiple fields, we use a pointer
  // to the elementwise value representation.
  return MakePointerValueRepr(elementwise_rep, aggregate_kind);
}

auto TypeCompleter::BuildInfoForInst(SemIR::TypeId type_id,
                                     SemIR::StructType struct_type) const
    -> SemIR::CompleteTypeInfo {
  auto fields = context_->struct_type_fields().Get(struct_type.fields_id);
  if (fields.empty()) {
    return {.value_repr = MakeEmptyValueRepr()};
  }

  // Find the value representation for each field, and construct a struct
  // of value representations.
  llvm::SmallVector<SemIR::StructTypeField> value_rep_fields;
  value_rep_fields.reserve(fields.size());
  bool same_as_object_rep = true;
  SemIR::ClassId abstract_class_id = SemIR::ClassId::None;
  for (auto field : fields) {
    auto field_info = GetNestedInfo(field.type_id);
    if (!field_info.value_repr.IsCopyOfObjectRepr(context_->sem_ir(),
                                                  field.type_id)) {
      same_as_object_rep = false;
      field.type_id = field_info.value_repr.type_id;
    }
    value_rep_fields.push_back(field);
    // Take the first non-None abstract_class_id, if any.
    if (field_info.abstract_class_id.has_value() &&
        !abstract_class_id.has_value()) {
      abstract_class_id = field_info.abstract_class_id;
    }
  }

  auto value_rep =
      same_as_object_rep
          ? type_id
          : GetStructType(
                *context_,
                context_->struct_type_fields().AddCanonical(value_rep_fields));
  return {.value_repr = BuildStructOrTupleValueRepr(fields.size(), value_rep,
                                                    same_as_object_rep),
          .abstract_class_id = abstract_class_id};
}

auto TypeCompleter::BuildInfoForInst(SemIR::TypeId type_id,
                                     SemIR::TupleType tuple_type) const
    -> SemIR::CompleteTypeInfo {
  // TODO: Share more code with structs.
  auto elements = context_->type_blocks().Get(tuple_type.elements_id);
  if (elements.empty()) {
    return {.value_repr = MakeEmptyValueRepr()};
  }

  // Find the value representation for each element, and construct a tuple
  // of value representations.
  llvm::SmallVector<SemIR::TypeId> value_rep_elements;
  value_rep_elements.reserve(elements.size());
  bool same_as_object_rep = true;
  SemIR::ClassId abstract_class_id = SemIR::ClassId::None;
  for (auto element_type_id : elements) {
    auto element_info = GetNestedInfo(element_type_id);
    if (!element_info.value_repr.IsCopyOfObjectRepr(context_->sem_ir(),
                                                    element_type_id)) {
      same_as_object_rep = false;
    }
    value_rep_elements.push_back(element_info.value_repr.type_id);
    // Take the first non-None abstract_class_id, if any.
    if (element_info.abstract_class_id.has_value() &&
        !abstract_class_id.has_value()) {
      abstract_class_id = element_info.abstract_class_id;
    }
  }

  auto value_rep = same_as_object_rep
                       ? type_id
                       : GetTupleType(*context_, value_rep_elements);
  return {.value_repr = BuildStructOrTupleValueRepr(elements.size(), value_rep,
                                                    same_as_object_rep),
          .abstract_class_id = abstract_class_id};
}

auto TypeCompleter::BuildInfoForInst(SemIR::TypeId type_id,
                                     SemIR::ArrayType /*inst*/) const
    -> SemIR::CompleteTypeInfo {
  // For arrays, it's convenient to always use a pointer representation,
  // even when the array has zero or one element, in order to support
  // indexing.
  return {.value_repr =
              MakePointerValueRepr(type_id, SemIR::ValueRepr::ObjectAggregate)};
}

auto TypeCompleter::BuildInfoForInst(SemIR::TypeId /*type_id*/,
                                     SemIR::ClassType inst) const
    -> SemIR::CompleteTypeInfo {
  auto& class_info = context_->classes().Get(inst.class_id);
  auto abstract_class_id =
      class_info.inheritance_kind == SemIR::Class::InheritanceKind::Abstract
          ? inst.class_id
          : SemIR::ClassId::None;

  // The value representation of an adapter is the value representation of
  // its adapted type.
  if (auto adapted_type_id =
          class_info.GetAdaptedType(context_->sem_ir(), inst.specific_id);
      adapted_type_id.has_value()) {
    auto info = GetNestedInfo(adapted_type_id);
    info.abstract_class_id = abstract_class_id;
    return info;
  }
  // Otherwise, the value representation for a class is a pointer to the
  // object representation.
  // TODO: Support customized value representations for classes.
  // TODO: Pick a better value representation when possible.
  return {.value_repr = MakePointerValueRepr(
              class_info.GetObjectRepr(context_->sem_ir(), inst.specific_id),
              SemIR::ValueRepr::ObjectAggregate),
          .abstract_class_id = abstract_class_id};
}

auto TypeCompleter::BuildInfoForInst(SemIR::TypeId /*type_id*/,
                                     SemIR::ConstType inst) const
    -> SemIR::CompleteTypeInfo {
  // The value representation of `const T` is the same as that of `T`.
  // Objects are not modifiable through their value representations.
  return GetNestedInfo(context_->types().GetTypeIdForTypeInstId(inst.inner_id));
}

// Builds and returns the value representation for the given type. All nested
// types, as found by AddNestedIncompleteTypes, are known to be complete.
auto TypeCompleter::BuildInfo(SemIR::TypeId type_id, SemIR::Inst inst) const
    -> SemIR::CompleteTypeInfo {
  // Use overload resolution to select the implementation, producing compile
  // errors when BuildInfoForInst isn't defined for a given instruction.
  CARBON_KIND_SWITCH(inst) {
#define CARBON_SEM_IR_INST_KIND(Name)             \
  case CARBON_KIND(SemIR::Name typed_inst): {     \
    return BuildInfoForInst(type_id, typed_inst); \
  }
#include "toolchain/sem_ir/inst_kind.def"
  }
}

auto TryToCompleteType(Context& context, SemIR::TypeId type_id, SemIRLoc loc,
                       MakeDiagnosticBuilderFn diagnoser) -> bool {
  return TypeCompleter(&context, loc, diagnoser).Complete(type_id);
}

auto CompleteTypeOrCheckFail(Context& context, SemIR::TypeId type_id) -> void {
  bool complete =
      TypeCompleter(&context, SemIR::LocId::None, nullptr).Complete(type_id);
  CARBON_CHECK(complete, "Expected {0} to be a complete type",
               context.types().GetAsInst(type_id));
}

auto RequireCompleteType(Context& context, SemIR::TypeId type_id,
                         SemIR::LocId loc_id, MakeDiagnosticBuilderFn diagnoser)
    -> bool {
  CARBON_CHECK(diagnoser);

  if (!TypeCompleter(&context, loc_id, diagnoser).Complete(type_id)) {
    return false;
  }

  // For a symbolic type, create an instruction to require the corresponding
  // specific type to be complete.
  if (type_id.is_symbolic()) {
    // TODO: Deduplicate these.
    AddInstInNoBlock(context, loc_id,
                     SemIR::RequireCompleteType{
                         .type_id = GetSingletonType(
                             context, SemIR::WitnessType::SingletonInstId),
                         .complete_type_id = type_id});
  }

  return true;
}

// Adds a note to a diagnostic explaining that a class is abstract.
static auto NoteAbstractClass(Context& context, SemIR::ClassId class_id,
                              bool direct_use, DiagnosticBuilder& builder)
    -> void {
  const auto& class_info = context.classes().Get(class_id);
  CARBON_CHECK(
      class_info.inheritance_kind == SemIR::Class::InheritanceKind::Abstract,
      "Class is not abstract");
  CARBON_DIAGNOSTIC(
      ClassAbstractHere, Note,
      "{0:=0:uses class that|=1:class} was declared abstract here",
      Diagnostics::IntAsSelect);
  builder.Note(class_info.definition_id, ClassAbstractHere,
               static_cast<int>(direct_use));
}

auto RequireConcreteType(Context& context, SemIR::TypeId type_id,
                         SemIR::LocId loc_id, MakeDiagnosticBuilderFn diagnoser,
                         MakeDiagnosticBuilderFn abstract_diagnoser) -> bool {
  // TODO: For symbolic types, should add an implicit constraint that they are
  // not abstract.
  CARBON_CHECK(abstract_diagnoser);

  // The representation of a facet type does not depend on its definition, so
  // they are considered "concrete" even when not complete.
  if (context.types().IsFacetType(type_id)) {
    return true;
  }

  if (!RequireCompleteType(context, type_id, loc_id, diagnoser)) {
    return false;
  }

  auto complete_info = context.types().GetCompleteTypeInfo(type_id);
  if (complete_info.abstract_class_id.has_value()) {
    auto builder = abstract_diagnoser();
    if (builder) {
      bool direct_use = false;
      if (auto inst = context.types().TryGetAs<SemIR::ClassType>(type_id)) {
        if (inst->class_id == complete_info.abstract_class_id) {
          direct_use = true;
        }
      }
      NoteAbstractClass(context, complete_info.abstract_class_id, direct_use,
                        builder);
      builder.Emit();
    }
    return false;
  }

  return true;
}

auto RequireIdentifiedFacetType(Context& context,
                                const SemIR::FacetType& facet_type)
    -> SemIR::IdentifiedFacetTypeId {
  if (auto identified_id =
          context.identified_facet_types().TryGetId(facet_type.facet_type_id);
      identified_id.has_value()) {
    return identified_id;
  }
  const auto& facet_type_info =
      context.facet_types().Get(facet_type.facet_type_id);

  // TODO: expand named constraints
  // TODO: Process other kinds of requirements.
  return context.identified_facet_types().Add(
      facet_type.facet_type_id, {facet_type_info.extend_constraints,
                                 facet_type_info.self_impls_constraints});
}

auto AsCompleteType(Context& context, SemIR::TypeId type_id,
                    SemIR::LocId loc_id, MakeDiagnosticBuilderFn diagnoser)
    -> SemIR::TypeId {
  return RequireCompleteType(context, type_id, loc_id, diagnoser)
             ? type_id
             : SemIR::ErrorInst::SingletonTypeId;
}

// Returns the type `type_id` if it is a concrete type, or produces an
// incomplete or abstract type error and returns an error type. This is a
// convenience wrapper around `RequireConcreteType`.
auto AsConcreteType(Context& context, SemIR::TypeId type_id,
                    SemIR::LocId loc_id, MakeDiagnosticBuilderFn diagnoser,
                    MakeDiagnosticBuilderFn abstract_diagnoser)
    -> SemIR::TypeId {
  return RequireConcreteType(context, type_id, loc_id, diagnoser,
                             abstract_diagnoser)
             ? type_id
             : SemIR::ErrorInst::SingletonTypeId;
}

auto NoteIncompleteClass(Context& context, SemIR::ClassId class_id,
                         DiagnosticBuilder& builder) -> void {
  const auto& class_info = context.classes().Get(class_id);
  CARBON_CHECK(!class_info.is_complete(), "Class is not incomplete");
  if (class_info.has_definition_started()) {
    CARBON_DIAGNOSTIC(ClassIncompleteWithinDefinition, Note,
                      "class is incomplete within its definition");
    builder.Note(class_info.definition_id, ClassIncompleteWithinDefinition);
  } else {
    CARBON_DIAGNOSTIC(ClassForwardDeclaredHere, Note,
                      "class was forward declared here");
    builder.Note(class_info.latest_decl_id(), ClassForwardDeclaredHere);
  }
}

auto NoteIncompleteInterface(Context& context, SemIR::InterfaceId interface_id,
                             DiagnosticBuilder& builder) -> void {
  const auto& interface_info = context.interfaces().Get(interface_id);
  CARBON_CHECK(!interface_info.is_complete(), "Interface is not incomplete");
  if (interface_info.is_being_defined()) {
    CARBON_DIAGNOSTIC(InterfaceIncompleteWithinDefinition, Note,
                      "interface is currently being defined");
    builder.Note(interface_info.definition_id,
                 InterfaceIncompleteWithinDefinition);
  } else {
    CARBON_DIAGNOSTIC(InterfaceForwardDeclaredHere, Note,
                      "interface was forward declared here");
    builder.Note(interface_info.latest_decl_id(), InterfaceForwardDeclaredHere);
  }
}

}  // namespace Carbon::Check
