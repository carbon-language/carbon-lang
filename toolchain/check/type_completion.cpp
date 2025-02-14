// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/type_completion.h"

#include "llvm/ADT/SmallVector.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/type.h"

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
// - A `BuildValueRepr` step computes the value representation for a
//   type, once all of its nested types are complete, and marks the type as
//   complete.
class TypeCompleter {
 public:
  TypeCompleter(Context& context, SemIRLoc loc,
                Context::BuildDiagnosticFn diagnoser)
      : context_(context), loc_(loc), diagnoser_(diagnoser) {}

  // Attempts to complete the given type. Returns true if it is now complete,
  // false if it could not be completed.
  auto Complete(SemIR::TypeId type_id) -> bool;

 private:
  enum class Phase : int8_t {
    // The next step is to add nested types to the list of types to complete.
    AddNestedIncompleteTypes,
    // The next step is to build the value representation for the type.
    BuildValueRepr,
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
  auto GetNestedValueRepr(SemIR::TypeId nested_type_id) const
      -> SemIR::ValueRepr;

  template <typename InstT>
    requires(
        InstT::Kind.template IsAnyOf<
            SemIR::AutoType, SemIR::BoolType, SemIR::BoundMethodType,
            SemIR::ErrorInst, SemIR::FloatType, SemIR::IntType,
            SemIR::IntLiteralType, SemIR::LegacyFloatType, SemIR::NamespaceType,
            SemIR::PointerType, SemIR::SpecificFunctionType, SemIR::TypeType,
            SemIR::VtableType, SemIR::WitnessType>())
  auto BuildValueReprForInst(SemIR::TypeId type_id, InstT /*inst*/) const
      -> SemIR::ValueRepr {
    return MakeCopyValueRepr(type_id);
  }

  auto BuildValueReprForInst(SemIR::TypeId type_id,
                             SemIR::StringType /*inst*/) const
      -> SemIR::ValueRepr;

  auto BuildStructOrTupleValueRepr(size_t num_elements,
                                   SemIR::TypeId elementwise_rep,
                                   bool same_as_object_rep) const
      -> SemIR::ValueRepr;

  auto BuildValueReprForInst(SemIR::TypeId type_id,
                             SemIR::StructType struct_type) const
      -> SemIR::ValueRepr;

  auto BuildValueReprForInst(SemIR::TypeId type_id,
                             SemIR::TupleType tuple_type) const
      -> SemIR::ValueRepr;

  auto BuildValueReprForInst(SemIR::TypeId type_id,
                             SemIR::ArrayType /*inst*/) const
      -> SemIR::ValueRepr;

  auto BuildValueReprForInst(SemIR::TypeId /*type_id*/,
                             SemIR::ClassType inst) const -> SemIR::ValueRepr;

  template <typename InstT>
    requires(InstT::Kind.template IsAnyOf<
             SemIR::AssociatedEntityType, SemIR::FacetType, SemIR::FunctionType,
             SemIR::FunctionTypeWithSelfType, SemIR::GenericClassType,
             SemIR::GenericInterfaceType, SemIR::UnboundElementType,
             SemIR::WhereExpr>())
  auto BuildValueReprForInst(SemIR::TypeId /*type_id*/, InstT /*inst*/) const
      -> SemIR::ValueRepr {
    // These types have no runtime operations, so we use an empty value
    // representation.
    //
    // TODO: There is information we could model here:
    // - For an interface, we could use a witness.
    // - For an associated entity, we could use an index into the witness.
    // - For an unbound element, we could use an index or offset.
    return MakeEmptyValueRepr();
  }

  auto BuildValueReprForInst(SemIR::TypeId /*type_id*/,
                             SemIR::ConstType inst) const -> SemIR::ValueRepr;

  template <typename InstT>
    requires(InstT::Kind.constant_kind() ==
                 SemIR::InstConstantKind::SymbolicOnly ||
             InstT::Kind.is_type() == SemIR::InstIsType::Never)
  auto BuildValueReprForInst(SemIR::TypeId type_id, InstT inst) const
      -> SemIR::ValueRepr {
    if constexpr (InstT::Kind.is_type() == SemIR::InstIsType::Never) {
      CARBON_FATAL("Type refers to non-type inst {0}", inst);
    } else {
      // For symbolic types, we arbitrarily pick a copy representation.
      return MakeCopyValueRepr(type_id);
    }
  }

  // Builds and returns the value representation for the given type. All nested
  // types, as found by AddNestedIncompleteTypes, are known to be complete.
  auto BuildValueRepr(SemIR::TypeId type_id, SemIR::Inst inst) const
      -> SemIR::ValueRepr;

  Context& context_;
  llvm::SmallVector<WorkItem> work_list_;
  SemIRLoc loc_;
  Context::BuildDiagnosticFn diagnoser_;
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
  if (!context_.types().IsComplete(type_id)) {
    work_list_.push_back(
        {.type_id = type_id, .phase = Phase::AddNestedIncompleteTypes});
  }
}

auto TypeCompleter::ProcessStep() -> bool {
  auto [type_id, phase] = work_list_.back();

  // We might have enqueued the same type more than once. Just skip the
  // type if it's already complete.
  if (context_.types().IsComplete(type_id)) {
    work_list_.pop_back();
    return true;
  }

  auto inst_id = context_.types().GetInstId(type_id);
  auto inst = context_.insts().Get(inst_id);
  auto old_work_list_size = work_list_.size();

  switch (phase) {
    case Phase::AddNestedIncompleteTypes:
      if (!AddNestedIncompleteTypes(inst)) {
        return false;
      }
      CARBON_CHECK(work_list_.size() >= old_work_list_size,
                   "AddNestedIncompleteTypes should not remove work items");
      work_list_[old_work_list_size - 1].phase = Phase::BuildValueRepr;
      break;

    case Phase::BuildValueRepr: {
      auto value_rep = BuildValueRepr(type_id, inst);
      context_.types().SetValueRepr(type_id, value_rep);
      CARBON_CHECK(old_work_list_size == work_list_.size(),
                   "BuildValueRepr should not change work items");
      work_list_.pop_back();

      // Also complete the value representation type, if necessary. This
      // should never fail: the value representation shouldn't require any
      // additional nested types to be complete.
      if (!context_.types().IsComplete(value_rep.type_id)) {
        work_list_.push_back(
            {.type_id = value_rep.type_id, .phase = Phase::BuildValueRepr});
      }
      // For a pointer representation, the pointee also needs to be complete.
      if (value_rep.kind == SemIR::ValueRepr::Pointer) {
        if (value_rep.type_id == SemIR::ErrorInst::SingletonTypeId) {
          break;
        }
        auto pointee_type_id =
            context_.sem_ir().GetPointeeType(value_rep.type_id);
        if (!context_.types().IsComplete(pointee_type_id)) {
          work_list_.push_back(
              {.type_id = pointee_type_id, .phase = Phase::BuildValueRepr});
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
      Push(inst.element_type_id);
      break;
    }
    case CARBON_KIND(SemIR::StructType inst): {
      for (auto field : context_.struct_type_fields().Get(inst.fields_id)) {
        Push(field.type_id);
      }
      break;
    }
    case CARBON_KIND(SemIR::TupleType inst): {
      for (auto element_type_id :
           context_.type_blocks().Get(inst.elements_id)) {
        Push(element_type_id);
      }
      break;
    }
    case CARBON_KIND(SemIR::ClassType inst): {
      auto& class_info = context_.classes().Get(inst.class_id);
      if (!class_info.is_defined()) {
        if (diagnoser_) {
          auto builder = diagnoser_();
          NoteIncompleteClass(context_, inst.class_id, builder);
          builder.Emit();
        }
        return false;
      }
      if (inst.specific_id.has_value()) {
        ResolveSpecificDefinition(context_, loc_, inst.specific_id);
      }
      if (auto adapted_type_id =
              class_info.GetAdaptedType(context_.sem_ir(), inst.specific_id);
          adapted_type_id.has_value()) {
        Push(adapted_type_id);
      } else {
        Push(class_info.GetObjectRepr(context_.sem_ir(), inst.specific_id));
      }
      break;
    }
    case CARBON_KIND(SemIR::ConstType inst): {
      Push(inst.inner_id);
      break;
    }
    default:
      break;
  }

  return true;
}

auto TypeCompleter::MakeEmptyValueRepr() const -> SemIR::ValueRepr {
  return {.kind = SemIR::ValueRepr::None,
          .type_id = GetTupleType(context_, {})};
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
          .type_id = GetPointerType(context_, pointee_id)};
}

auto TypeCompleter::GetNestedValueRepr(SemIR::TypeId nested_type_id) const
    -> SemIR::ValueRepr {
  CARBON_CHECK(context_.types().IsComplete(nested_type_id),
               "Nested type should already be complete");
  auto value_rep = context_.types().GetValueRepr(nested_type_id);
  CARBON_CHECK(value_rep.kind != SemIR::ValueRepr::Unknown,
               "Complete type should have a value representation");
  return value_rep;
}

auto TypeCompleter::BuildValueReprForInst(SemIR::TypeId type_id,
                                          SemIR::StringType /*inst*/) const
    -> SemIR::ValueRepr {
  // TODO: Decide on string value semantics. This should probably be a
  // custom value representation carrying a pointer and size or
  // similar.
  return MakePointerValueRepr(type_id);
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

auto TypeCompleter::BuildValueReprForInst(SemIR::TypeId type_id,
                                          SemIR::StructType struct_type) const
    -> SemIR::ValueRepr {
  auto fields = context_.struct_type_fields().Get(struct_type.fields_id);
  if (fields.empty()) {
    return MakeEmptyValueRepr();
  }

  // Find the value representation for each field, and construct a struct
  // of value representations.
  llvm::SmallVector<SemIR::StructTypeField> value_rep_fields;
  value_rep_fields.reserve(fields.size());
  bool same_as_object_rep = true;
  for (auto field : fields) {
    auto field_value_rep = GetNestedValueRepr(field.type_id);
    if (!field_value_rep.IsCopyOfObjectRepr(context_.sem_ir(), field.type_id)) {
      same_as_object_rep = false;
      field.type_id = field_value_rep.type_id;
    }
    value_rep_fields.push_back(field);
  }

  auto value_rep =
      same_as_object_rep
          ? type_id
          : GetStructType(context_, context_.struct_type_fields().AddCanonical(
                                        value_rep_fields));
  return BuildStructOrTupleValueRepr(fields.size(), value_rep,
                                     same_as_object_rep);
}

auto TypeCompleter::BuildValueReprForInst(SemIR::TypeId type_id,
                                          SemIR::TupleType tuple_type) const
    -> SemIR::ValueRepr {
  // TODO: Share more code with structs.
  auto elements = context_.type_blocks().Get(tuple_type.elements_id);
  if (elements.empty()) {
    return MakeEmptyValueRepr();
  }

  // Find the value representation for each element, and construct a tuple
  // of value representations.
  llvm::SmallVector<SemIR::TypeId> value_rep_elements;
  value_rep_elements.reserve(elements.size());
  bool same_as_object_rep = true;
  for (auto element_type_id : elements) {
    auto element_value_rep = GetNestedValueRepr(element_type_id);
    if (!element_value_rep.IsCopyOfObjectRepr(context_.sem_ir(),
                                              element_type_id)) {
      same_as_object_rep = false;
    }
    value_rep_elements.push_back(element_value_rep.type_id);
  }

  auto value_rep =
      same_as_object_rep ? type_id : GetTupleType(context_, value_rep_elements);
  return BuildStructOrTupleValueRepr(elements.size(), value_rep,
                                     same_as_object_rep);
}

auto TypeCompleter::BuildValueReprForInst(SemIR::TypeId type_id,
                                          SemIR::ArrayType /*inst*/) const
    -> SemIR::ValueRepr {
  // For arrays, it's convenient to always use a pointer representation,
  // even when the array has zero or one element, in order to support
  // indexing.
  return MakePointerValueRepr(type_id, SemIR::ValueRepr::ObjectAggregate);
}

auto TypeCompleter::BuildValueReprForInst(SemIR::TypeId /*type_id*/,
                                          SemIR::ClassType inst) const
    -> SemIR::ValueRepr {
  auto& class_info = context_.classes().Get(inst.class_id);
  // The value representation of an adapter is the value representation of
  // its adapted type.
  if (auto adapted_type_id =
          class_info.GetAdaptedType(context_.sem_ir(), inst.specific_id);
      adapted_type_id.has_value()) {
    return GetNestedValueRepr(adapted_type_id);
  }
  // Otherwise, the value representation for a class is a pointer to the
  // object representation.
  // TODO: Support customized value representations for classes.
  // TODO: Pick a better value representation when possible.
  return MakePointerValueRepr(
      class_info.GetObjectRepr(context_.sem_ir(), inst.specific_id),
      SemIR::ValueRepr::ObjectAggregate);
}

auto TypeCompleter::BuildValueReprForInst(SemIR::TypeId /*type_id*/,
                                          SemIR::ConstType inst) const
    -> SemIR::ValueRepr {
  // The value representation of `const T` is the same as that of `T`.
  // Objects are not modifiable through their value representations.
  return GetNestedValueRepr(inst.inner_id);
}

// Builds and returns the value representation for the given type. All nested
// types, as found by AddNestedIncompleteTypes, are known to be complete.
auto TypeCompleter::BuildValueRepr(SemIR::TypeId type_id,
                                   SemIR::Inst inst) const -> SemIR::ValueRepr {
  // Use overload resolution to select the implementation, producing compile
  // errors when BuildValueReprForInst isn't defined for a given instruction.
  CARBON_KIND_SWITCH(inst) {
#define CARBON_SEM_IR_INST_KIND(Name)                  \
  case CARBON_KIND(SemIR::Name typed_inst): {          \
    return BuildValueReprForInst(type_id, typed_inst); \
  }
#include "toolchain/sem_ir/inst_kind.def"
  }
}

auto TryToCompleteType(Context& context, SemIR::TypeId type_id, SemIRLoc loc,
                       Context::BuildDiagnosticFn diagnoser) -> bool {
  return TypeCompleter(context, loc, diagnoser).Complete(type_id);
}

auto CompleteTypeOrCheckFail(Context& context, SemIR::TypeId type_id) -> void {
  bool complete =
      TypeCompleter(context, SemIR::LocId::None, nullptr).Complete(type_id);
  CARBON_CHECK(complete, "Expected {0} to be a complete type",
               context.types().GetAsInst(type_id));
}

auto RequireCompleteType(Context& context, SemIR::TypeId type_id,
                         SemIR::LocId loc_id,
                         Context::BuildDiagnosticFn diagnoser) -> bool {
  CARBON_CHECK(diagnoser);

  if (!TypeCompleter(context, loc_id, diagnoser).Complete(type_id)) {
    return false;
  }

  // For a symbolic type, create an instruction to require the corresponding
  // specific type to be complete.
  if (type_id.AsConstantId().is_symbolic()) {
    // TODO: Deduplicate these.
    AddInstInNoBlock(
        context,
        SemIR::LocIdAndInst(
            loc_id, SemIR::RequireCompleteType{
                        .type_id = GetSingletonType(
                            context, SemIR::WitnessType::SingletonInstId),
                        .complete_type_id = type_id}));
  }

  return true;
}

// Adds a note to a diagnostic explaining that a class is abstract.
static auto NoteAbstractClass(Context& context, SemIR::ClassId class_id,
                              Context::DiagnosticBuilder& builder) -> void {
  const auto& class_info = context.classes().Get(class_id);
  CARBON_CHECK(
      class_info.inheritance_kind == SemIR::Class::InheritanceKind::Abstract,
      "Class is not abstract");
  CARBON_DIAGNOSTIC(ClassAbstractHere, Note,
                    "class was declared abstract here");
  builder.Note(class_info.definition_id, ClassAbstractHere);
}

auto RequireConcreteType(Context& context, SemIR::TypeId type_id,
                         SemIR::LocId loc_id,
                         Context::BuildDiagnosticFn diagnoser,
                         Context::BuildDiagnosticFn abstract_diagnoser)
    -> bool {
  CARBON_CHECK(abstract_diagnoser);

  if (!RequireCompleteType(context, type_id, loc_id, diagnoser)) {
    return false;
  }

  if (auto class_type = context.types().TryGetAs<SemIR::ClassType>(type_id)) {
    auto& class_info = context.classes().Get(class_type->class_id);
    if (class_info.inheritance_kind !=
        SemIR::Class::InheritanceKind::Abstract) {
      return true;
    }

    auto builder = abstract_diagnoser();
    if (!builder) {
      return false;
    }
    NoteAbstractClass(context, class_type->class_id, builder);
    builder.Emit();
    return false;
  }

  return true;
}

auto RequireDefinedType(Context& context, SemIR::TypeId type_id,
                        SemIR::LocId loc_id,
                        Context::BuildDiagnosticFn diagnoser) -> bool {
  if (!RequireCompleteType(context, type_id, loc_id, diagnoser)) {
    return false;
  }

  if (auto facet_type = context.types().TryGetAs<SemIR::FacetType>(type_id)) {
    const auto& facet_type_info =
        context.facet_types().Get(facet_type->facet_type_id);
    for (auto interface : facet_type_info.impls_constraints) {
      auto interface_id = interface.interface_id;
      if (!context.interfaces().Get(interface_id).is_defined()) {
        auto builder = diagnoser();
        NoteUndefinedInterface(context, interface_id, builder);
        builder.Emit();
        return false;
      }

      if (interface.specific_id.has_value()) {
        ResolveSpecificDefinition(context, loc_id, interface.specific_id);
      }
    }
    // TODO: Finish facet type resolution.
    //
    // Note that we will need Self to be passed into facet type resolution.
    // The `.Self` of a facet type created by `where` will then be bound to the
    // provided self type.
    //
    // For example, in `T:! X where ...`, we will bind the `.Self` of the
    // `where` facet type to `T`, and in `(X where ...) where ...`, we will bind
    // the inner `.Self` to the outer `.Self`.
    //
    // If the facet type contains a rewrite, we may have deferred converting the
    // rewritten value to the type of the associated constant. That conversion
    // should also be performed as part of resolution, and may depend on the
    // Self type.
  }

  return true;
}

auto AsCompleteType(Context& context, SemIR::TypeId type_id,
                    SemIR::LocId loc_id, Context::BuildDiagnosticFn diagnoser)
    -> SemIR::TypeId {
  return RequireCompleteType(context, type_id, loc_id, diagnoser)
             ? type_id
             : SemIR::ErrorInst::SingletonTypeId;
}

// Returns the type `type_id` if it is a concrete type, or produces an
// incomplete or abstract type error and returns an error type. This is a
// convenience wrapper around `RequireConcreteType`.
auto AsConcreteType(Context& context, SemIR::TypeId type_id,
                    SemIR::LocId loc_id, Context::BuildDiagnosticFn diagnoser,
                    Context::BuildDiagnosticFn abstract_diagnoser)
    -> SemIR::TypeId {
  return RequireConcreteType(context, type_id, loc_id, diagnoser,
                             abstract_diagnoser)
             ? type_id
             : SemIR::ErrorInst::SingletonTypeId;
}

auto NoteIncompleteClass(Context& context, SemIR::ClassId class_id,
                         Context::DiagnosticBuilder& builder) -> void {
  const auto& class_info = context.classes().Get(class_id);
  CARBON_CHECK(!class_info.is_defined(), "Class is not incomplete");
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

auto NoteUndefinedInterface(Context& context, SemIR::InterfaceId interface_id,
                            Context::DiagnosticBuilder& builder) -> void {
  const auto& interface_info = context.interfaces().Get(interface_id);
  CARBON_CHECK(!interface_info.is_defined(), "Interface is not incomplete");
  if (interface_info.is_being_defined()) {
    CARBON_DIAGNOSTIC(InterfaceUndefinedWithinDefinition, Note,
                      "interface is currently being defined");
    builder.Note(interface_info.definition_id,
                 InterfaceUndefinedWithinDefinition);
  } else {
    CARBON_DIAGNOSTIC(InterfaceForwardDeclaredHere, Note,
                      "interface was forward declared here");
    builder.Note(interface_info.latest_decl_id(), InterfaceForwardDeclaredHere);
  }
}

}  // namespace Carbon::Check
