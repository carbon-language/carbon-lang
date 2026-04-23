// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/type_completion.h"

#include "common/concepts.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/cpp/import.h"
#include "toolchain/check/facet_type.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/literal.h"
#include "toolchain/check/subst.h"
#include "toolchain/check/type.h"
#include "toolchain/diagnostics/emitter.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/sem_ir/constant.h"
#include "toolchain/sem_ir/facet_type_info.h"
#include "toolchain/sem_ir/generic.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/named_constraint.h"
#include "toolchain/sem_ir/specific_interface.h"
#include "toolchain/sem_ir/specific_named_constraint.h"
#include "toolchain/sem_ir/type_info.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto DiagnoseIncompleteClass(Context& context, SemIR::ClassId class_id)
    -> void {
  // The caller must provide context for any diagnostics in type completion.
  context.emitter().CheckHasContext();

  const auto& class_info = context.classes().Get(class_id);
  CARBON_CHECK(!class_info.is_complete(), "Class is not incomplete");
  if (class_info.has_definition_started()) {
    CARBON_DIAGNOSTIC(ClassIncompleteWithinDefinition, Error,
                      "class is incomplete within its definition");
    context.emitter().Emit(class_info.definition_id,
                           ClassIncompleteWithinDefinition);
  } else {
    CARBON_DIAGNOSTIC(ClassForwardDeclaredHere, Error,
                      "class was forward declared here");
    context.emitter().Emit(class_info.latest_decl_id(),
                           ClassForwardDeclaredHere);
  }
}

auto DiagnoseIncompleteInterface(Context& context,
                                 SemIR::InterfaceId interface_id) -> void {
  // The caller must provide context for any diagnostics in type completion.
  context.emitter().CheckHasContext();

  const auto& interface_info = context.interfaces().Get(interface_id);
  CARBON_CHECK(!interface_info.is_complete(), "Interface is not incomplete");
  if (interface_info.is_being_defined()) {
    CARBON_DIAGNOSTIC(InterfaceIncompleteWithinDefinition, Error,
                      "interface is currently being defined");
    context.emitter().Emit(interface_info.definition_id,
                           InterfaceIncompleteWithinDefinition);
  } else {
    CARBON_DIAGNOSTIC(InterfaceForwardDeclaredHere, Error,
                      "interface was forward declared here");
    context.emitter().Emit(interface_info.latest_decl_id(),
                           InterfaceForwardDeclaredHere);
  }
}

auto DiagnoseAbstractClass(Context& context, SemIR::ClassId class_id,
                           bool direct_use) -> void {
  // The caller must provide context for any diagnostics in type completion.
  context.emitter().CheckHasContext();

  const auto& class_info = context.classes().Get(class_id);
  CARBON_CHECK(
      class_info.inheritance_kind == SemIR::Class::InheritanceKind::Abstract,
      "Class is not abstract");
  CARBON_DIAGNOSTIC(
      ClassAbstractHere, Error,
      "{0:=0:uses class that|=1:class} was declared abstract here",
      Diagnostics::IntAsSelect);
  context.emitter().Emit(class_info.definition_id, ClassAbstractHere,
                         static_cast<int>(direct_use));
}

static auto DiagnoseIncompleteNamedConstraint(
    Context& context, SemIR::NamedConstraintId named_constraint_id) -> void {
  // The caller must provide context for any diagnostics in type completion.
  context.emitter().CheckHasContext();

  const auto& constraint = context.named_constraints().Get(named_constraint_id);
  CARBON_CHECK(!constraint.is_complete(), "Named constraint is not incomplete");
  if (constraint.is_being_defined()) {
    CARBON_DIAGNOSTIC(NamedConstraintIncompleteWithinDefinition, Error,
                      "constraint is currently being defined");
    context.emitter().Emit(constraint.definition_id,
                           NamedConstraintIncompleteWithinDefinition);
  } else {
    CARBON_DIAGNOSTIC(NamedConstraintForwardDeclaredHere, Error,
                      "constraint was forward declared here");
    context.emitter().Emit(constraint.latest_decl_id(),
                           NamedConstraintForwardDeclaredHere);
  }
}

// Returns true if either eval block contains an error.
static auto SpecificHasError(Context& context, SemIR::SpecificId specific_id)
    -> bool {
  return specific_id.has_value() &&
         context.specifics().Get(specific_id).HasError();
}

static auto RequireCompleteFacetType(Context& context, SemIR::LocId loc_id,
                                     const SemIR::FacetType& facet_type,
                                     bool diagnose) -> bool {
  const auto& facet_type_info =
      context.facet_types().Get(facet_type.facet_type_id);

  for (auto extends : facet_type_info.extend_constraints) {
    auto interface_id = extends.interface_id;
    const auto& interface = context.interfaces().Get(interface_id);
    if (!interface.is_complete()) {
      if (diagnose) {
        DiagnoseIncompleteInterface(context, interface_id);
      }
      return false;
    }
    if (interface.generic_id.has_value()) {
      ResolveSpecificDefinition(context, loc_id, extends.specific_id);
      if (SpecificHasError(context, extends.specific_id)) {
        return false;
      }
    }

    auto interface_with_self_self_specific_args = context.inst_blocks().Get(
        context.specifics().GetArgsOrEmpty(context.generics().GetSelfSpecific(
            interface.generic_with_self_id)));
    auto self_facet = interface_with_self_self_specific_args.back();
    auto interface_with_self_specific_id = MakeSpecificWithInnerSelf(
        context, loc_id, interface.generic_id, interface.generic_with_self_id,
        extends.specific_id, context.constant_values().Get(self_facet));
    if (SpecificHasError(context, interface_with_self_specific_id)) {
      return false;
    }
  }

  for (auto extends : facet_type_info.extend_named_constraints) {
    auto named_constraint_id = extends.named_constraint_id;
    const auto& constraint =
        context.named_constraints().Get(named_constraint_id);
    if (!constraint.is_complete()) {
      if (diagnose) {
        DiagnoseIncompleteNamedConstraint(context, named_constraint_id);
      }
      return false;
    }
    if (constraint.generic_id.has_value()) {
      ResolveSpecificDefinition(context, loc_id, extends.specific_id);
      if (SpecificHasError(context, extends.specific_id)) {
        return false;
      }
    }

    auto constraint_with_self_self_specific_args = context.inst_blocks().Get(
        context.specifics().GetArgsOrEmpty(context.generics().GetSelfSpecific(
            constraint.generic_with_self_id)));
    auto self_facet = constraint_with_self_self_specific_args.back();
    auto constraint_with_self_specific_id = MakeSpecificWithInnerSelf(
        context, loc_id, constraint.generic_id, constraint.generic_with_self_id,
        extends.specific_id, context.constant_values().Get(self_facet));
    if (SpecificHasError(context, constraint_with_self_specific_id)) {
      return false;
    }
  }

  return true;
}

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
  TypeCompleter(Context* context, SemIR::LocId loc_id, bool diagnose)
      : context_(context), loc_id_(loc_id), diagnose_(diagnose) {}

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

  // Makes type info for a type with an empty value representation, which is
  // used for types that have no state, such as empty structs and tuples.
  auto MakeEmptyTypeInfo() const -> SemIR::CompleteTypeInfo;

  // Makes type info for a type with a dependent value representation, which is
  // used for symbolic types.
  auto MakeDependentTypeInfo(SemIR::TypeId type_id) const
      -> SemIR::CompleteTypeInfo;

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

  // Gets the type info for a nested type, which should already be complete.
  auto GetNestedInfo(SemIR::TypeId nested_type_id) const
      -> SemIR::CompleteTypeInfo;

  template <typename InstT>
    requires(InstT::Kind.template IsAnyOf<
             SemIR::AutoType, SemIR::BoundMethodType, SemIR::CharLiteralType,
             SemIR::ErrorInst, SemIR::FacetType, SemIR::FloatLiteralType,
             SemIR::FormType, SemIR::IntLiteralType, SemIR::NamespaceType,
             SemIR::PatternType, SemIR::RequireSpecificDefinitionType,
             SemIR::SpecificFunctionType, SemIR::TypeType, SemIR::VtableType,
             SemIR::WitnessType>())
  auto BuildInfoForInst(SemIR::TypeId type_id, InstT /*inst*/) const
      -> SemIR::CompleteTypeInfo {
    // These types are empty at runtime but have values to copy at compile time.
    return {.value_repr = MakeCopyValueRepr(type_id),
            .object_layout = SemIR::ObjectLayout::Empty()};
  }

  auto BuildInfoForInst(SemIR::TypeId type_id, SemIR::BoolType inst) const
      -> SemIR::CompleteTypeInfo;
  auto BuildInfoForInst(SemIR::TypeId type_id, SemIR::PointerType inst) const
      -> SemIR::CompleteTypeInfo;
  auto BuildInfoForInst(SemIR::TypeId type_id, SemIR::IntType inst) const
      -> SemIR::CompleteTypeInfo;
  auto BuildInfoForInst(SemIR::TypeId type_id, SemIR::FloatType inst) const
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
             SemIR::AssociatedEntityType, SemIR::CppOverloadSetType,
             SemIR::CppTemplateNameType, SemIR::FunctionType,
             SemIR::FunctionTypeWithSelfType, SemIR::GenericClassType,
             SemIR::GenericInterfaceType, SemIR::GenericNamedConstraintType,
             SemIR::InstType, SemIR::UnboundElementType, SemIR::WhereExpr>())
  auto BuildInfoForInst(SemIR::TypeId /*type_id*/, InstT /*inst*/) const
      -> SemIR::CompleteTypeInfo {
    // These types have no runtime operations, so we use an empty value
    // representation.
    //
    // TODO: There is information we could model here:
    // - For an interface, we could use a witness.
    // - For an associated entity, we could use an index into the witness.
    // - For an unbound element, we could use an index or offset.
    return MakeEmptyTypeInfo();
  }

  auto BuildInfoForInst(SemIR::TypeId /*type_id*/, SemIR::ConstType inst) const
      -> SemIR::CompleteTypeInfo;

  auto BuildInfoForInst(SemIR::TypeId type_id,
                        SemIR::CustomLayoutType inst) const
      -> SemIR::CompleteTypeInfo;

  auto BuildInfoForInst(SemIR::TypeId /*type_id*/,
                        SemIR::MaybeUnformedType inst) const
      -> SemIR::CompleteTypeInfo;

  auto BuildInfoForInst(SemIR::TypeId /*type_id*/,
                        SemIR::PartialType inst) const
      -> SemIR::CompleteTypeInfo;

  auto BuildInfoForInst(SemIR::TypeId /*type_id*/,
                        SemIR::ImplWitnessAssociatedConstant inst) const
      -> SemIR::CompleteTypeInfo;

  auto BuildInfoForInst(SemIR::TypeId /*type_id*/,
                        SemIR::ImplWitnessAccess inst) const
      -> SemIR::CompleteTypeInfo;

  template <typename InstT>
    requires(InstT::Kind.is_type() == SemIR::InstIsType::Never)
  auto BuildInfoForInst(SemIR::TypeId /*type_id*/, InstT inst) const
      -> SemIR::CompleteTypeInfo {
    CARBON_FATAL("Type refers to non-type inst {0}", inst);
  }

  template <typename InstT>
    requires(InstT::Kind.is_symbolic_when_type())
  auto BuildInfoForInst(SemIR::TypeId type_id, InstT /*inst*/) const
      -> SemIR::CompleteTypeInfo {
    return MakeDependentTypeInfo(type_id);
  }

  // Builds and returns the `CompleteTypeInfo` for the given type. All nested
  // types, as found by AddNestedIncompleteTypes, are known to be complete.
  auto BuildInfo(SemIR::TypeId type_id, SemIR::Inst inst) const
      -> SemIR::CompleteTypeInfo;

  Context* context_;
  llvm::SmallVector<WorkItem> work_list_;
  SemIR::LocId loc_id_;
  bool diagnose_;
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

  auto inst_id = context_->types().GetTypeInstId(type_id);
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
        if (info.value_repr.type_id == SemIR::ErrorInst::TypeId) {
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
        Push(context_->types().GetTypeIdForTypeInstId(field.type_inst_id));
      }
      break;
    }
    case CARBON_KIND(SemIR::TupleType inst): {
      for (auto element_type_id : context_->types().GetBlockAsTypeIds(
               context_->inst_blocks().Get(inst.type_elements_id))) {
        Push(element_type_id);
      }
      break;
    }
    case CARBON_KIND(SemIR::ClassType inst): {
      auto& class_info = context_->classes().Get(inst.class_id);
      // If the class was imported from C++, ask Clang to try to complete it.
      if (!class_info.is_complete() && class_info.scope_id.has_value()) {
        auto& scope = context_->name_scopes().Get(class_info.scope_id);
        if (scope.clang_decl_context_id().has_value()) {
          if (!ImportClassDefinitionForClangDecl(
                  *context_, inst.class_id, scope.clang_decl_context_id())) {
            // Clang produced a diagnostic. Don't produce one of our own.
            return false;
          }
        }
      }
      if (!class_info.is_complete()) {
        if (diagnose_) {
          DiagnoseIncompleteClass(*context_, inst.class_id);
        }
        return false;
      }
      if (inst.specific_id.has_value()) {
        ResolveSpecificDefinition(*context_, loc_id_, inst.specific_id);
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
    case CARBON_KIND(SemIR::CustomLayoutType inst): {
      for (auto field : context_->struct_type_fields().Get(inst.fields_id)) {
        Push(context_->types().GetTypeIdForTypeInstId(field.type_inst_id));
      }
      break;
    }
    case CARBON_KIND(SemIR::MaybeUnformedType inst): {
      Push(context_->types().GetTypeIdForTypeInstId(inst.inner_id));
      break;
    }
    case CARBON_KIND(SemIR::PartialType inst): {
      Push(context_->types().GetTypeIdForTypeInstId(inst.inner_id));
      break;
    }
    case CARBON_KIND(SemIR::FacetType inst): {
      if (!RequireCompleteFacetType(*context_, loc_id_, inst, diagnose_)) {
        return false;
      }
      break;
    }

    default:
      break;
  }

  return true;
}

auto TypeCompleter::MakeEmptyTypeInfo() const -> SemIR::CompleteTypeInfo {
  return {.value_repr = {.kind = SemIR::ValueRepr::None,
                         .type_id = GetTupleType(*context_, {})},
          .object_layout = SemIR::ObjectLayout::Empty()};
}

auto TypeCompleter::MakeDependentTypeInfo(SemIR::TypeId type_id) const
    -> SemIR::CompleteTypeInfo {
  return {
      .value_repr = {.kind = SemIR::ValueRepr::Dependent, .type_id = type_id},
      .object_layout = SemIR::ObjectLayout()};
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
          .type_id = GetPointerType(
              *context_, context_->types().GetTypeInstId(pointee_id))};
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
    return MakeEmptyTypeInfo();
  }

  // Find the value representation for each field, and construct a struct
  // of value representations.
  llvm::SmallVector<SemIR::StructTypeField> value_rep_fields;
  value_rep_fields.reserve(fields.size());
  bool same_as_object_rep = true;
  SemIR::ClassId abstract_class_id = SemIR::ClassId::None;
  SemIR::ObjectLayout layout = SemIR::ObjectLayout::Empty();
  for (auto field : fields) {
    auto field_type_id =
        context_->types().GetTypeIdForTypeInstId(field.type_inst_id);
    auto field_info = GetNestedInfo(field_type_id);
    if (!field_info.value_repr.IsCopyOfObjectRepr(context_->sem_ir(),
                                                  field_type_id)) {
      same_as_object_rep = false;
      field.type_inst_id =
          context_->types().GetTypeInstId(field_info.value_repr.type_id);
    }
    value_rep_fields.push_back(field);
    // Take the first non-None abstract_class_id, if any.
    if (field_info.abstract_class_id.has_value() &&
        !abstract_class_id.has_value()) {
      abstract_class_id = field_info.abstract_class_id;
    }
    // Accumulate layout.
    layout.TryAppendField(field_info.object_layout);
  }

  auto value_rep =
      same_as_object_rep
          ? type_id
          : GetStructType(
                *context_,
                context_->struct_type_fields().AddCanonical(value_rep_fields));
  return {.value_repr = BuildStructOrTupleValueRepr(fields.size(), value_rep,
                                                    same_as_object_rep),
          .object_layout = layout,
          .abstract_class_id = abstract_class_id};
}

auto TypeCompleter::BuildInfoForInst(SemIR::TypeId type_id,
                                     SemIR::TupleType tuple_type) const
    -> SemIR::CompleteTypeInfo {
  // TODO: Share more code with structs.
  auto elements = context_->inst_blocks().Get(tuple_type.type_elements_id);
  if (elements.empty()) {
    return MakeEmptyTypeInfo();
  }

  // Find the value representation for each element, and construct a tuple
  // of value representations.
  llvm::SmallVector<SemIR::InstId> value_rep_elements;
  value_rep_elements.reserve(elements.size());
  bool same_as_object_rep = true;
  SemIR::ClassId abstract_class_id = SemIR::ClassId::None;
  SemIR::ObjectLayout layout = SemIR::ObjectLayout::Empty();
  for (auto element_type_id : context_->types().GetBlockAsTypeIds(elements)) {
    auto element_info = GetNestedInfo(element_type_id);
    if (!element_info.value_repr.IsCopyOfObjectRepr(context_->sem_ir(),
                                                    element_type_id)) {
      same_as_object_rep = false;
    }
    value_rep_elements.push_back(
        context_->types().GetTypeInstId(element_info.value_repr.type_id));
    // Take the first non-None abstract_class_id, if any.
    if (element_info.abstract_class_id.has_value() &&
        !abstract_class_id.has_value()) {
      abstract_class_id = element_info.abstract_class_id;
    }
    // Accumulate layout.
    layout.TryAppendField(element_info.object_layout);
  }

  auto value_rep = same_as_object_rep
                       ? type_id
                       : GetTupleType(*context_, value_rep_elements);
  return {.value_repr = BuildStructOrTupleValueRepr(elements.size(), value_rep,
                                                    same_as_object_rep),
          .object_layout = layout,
          .abstract_class_id = abstract_class_id};
}

auto TypeCompleter::BuildInfoForInst(SemIR::TypeId type_id,
                                     SemIR::ArrayType array_type) const
    -> SemIR::CompleteTypeInfo {
  // For arrays, it's convenient to always use a pointer representation,
  // even when the array has zero or one element, in order to support
  // indexing.
  auto element_type_id =
      context_->types().GetTypeIdForTypeInstId(array_type.element_type_inst_id);
  auto element_info = GetNestedInfo(element_type_id);

  SemIR::ObjectLayout layout;
  if (element_info.object_layout.has_value()) {
    if (auto bound = context_->sem_ir().GetZExtIntValue(array_type.bound_id)) {
      layout =
          SemIR::ObjectLayout::ForArray(element_info.object_layout, *bound);
    }
  }

  return {.value_repr =
              MakePointerValueRepr(type_id, SemIR::ValueRepr::ObjectAggregate),
          .object_layout = layout};
}

auto TypeCompleter::BuildInfoForInst(SemIR::TypeId type_id,
                                     SemIR::BoolType /*inst*/) const
    -> SemIR::CompleteTypeInfo {
  // TODO: Decide whether to use a size of `1` here, like `Core.Int(1)`. One
  // concern is that {.a: bool, .b: bool} would be laid out with the bools at
  // offsets 0 and 8, which on big-endian systems would imply the MSB stores the
  // value not the LSB. Consider right-aligning fields on big-endian systems
  // instead (except when bit-packing, by whatever means we support that).
  return {.value_repr = MakeCopyValueRepr(type_id),
          .object_layout = {.size = SemIR::ObjectSize::Bytes(1),
                            .alignment = SemIR::ObjectSize::Bytes(1)}};
}

auto TypeCompleter::BuildInfoForInst(SemIR::TypeId type_id,
                                     SemIR::PointerType /*inst*/) const
    -> SemIR::CompleteTypeInfo {
  return {.value_repr = MakeCopyValueRepr(type_id),
          .object_layout = context_->sem_ir().GetPointerLayout()};
}

auto TypeCompleter::BuildInfoForInst(SemIR::TypeId type_id,
                                     SemIR::IntType inst) const
    -> SemIR::CompleteTypeInfo {
  SemIR::ObjectLayout layout;
  if (auto bit_width = context_->sem_ir().GetZExtIntValue(inst.bit_width_id)) {
    auto size = SemIR::ObjectSize::Bits(*bit_width);
    // TODO: The upper bound for alignment here should be target-specific.
    auto align = SemIR::ObjectSize::Bits(
        std::clamp<int64_t>(llvm::PowerOf2Ceil(*bit_width), 8, 256));
    layout = {.size = size, .alignment = align};
  }
  return {.value_repr = MakeCopyValueRepr(type_id), .object_layout = layout};
}

auto TypeCompleter::BuildInfoForInst(SemIR::TypeId type_id,
                                     SemIR::FloatType inst) const
    -> SemIR::CompleteTypeInfo {
  SemIR::ObjectLayout layout;
  if (auto bit_width = context_->sem_ir().GetZExtIntValue(inst.bit_width_id)) {
    auto size = SemIR::ObjectSize::Bits(*bit_width);
    // TODO: Pick a suitable alignment here. For some targets, we may want to
    // use 32-bit alignment for f64 (and f80). For now we round up to a power
    // of 2.
    auto align = SemIR::ObjectSize::Bits(llvm::PowerOf2Ceil(*bit_width));
    layout = {.size = size, .alignment = align};
  }
  return {.value_repr = MakeCopyValueRepr(type_id), .object_layout = layout};
}

auto TypeCompleter::BuildInfoForInst(SemIR::TypeId /*type_id*/,
                                     SemIR::ClassType inst) const
    -> SemIR::CompleteTypeInfo {
  auto& class_info = context_->classes().Get(inst.class_id);
  auto abstract_class_id =
      class_info.inheritance_kind == SemIR::Class::InheritanceKind::Abstract
          ? inst.class_id
          : SemIR::ClassId::None;

  // The object and value representation of an adapter are the object and value
  // representation of its adapted type.
  if (auto adapted_type_id =
          class_info.GetAdaptedType(context_->sem_ir(), inst.specific_id);
      adapted_type_id.has_value()) {
    auto info = GetNestedInfo(adapted_type_id);
    info.abstract_class_id = abstract_class_id;
    return info;
  }
  // Otherwise, the value representation for a class is a pointer to the
  // object representation, which was computed when the class was defined.
  auto object_repr_type_id =
      class_info.GetObjectRepr(context_->sem_ir(), inst.specific_id);
  // TODO: Support customized value representations for classes.
  // TODO: Pick a better value representation when possible.
  return {.value_repr = MakePointerValueRepr(object_repr_type_id,
                                             SemIR::ValueRepr::ObjectAggregate),
          .object_layout = GetNestedInfo(object_repr_type_id).object_layout,
          .abstract_class_id = abstract_class_id};
}

auto TypeCompleter::BuildInfoForInst(SemIR::TypeId /*type_id*/,
                                     SemIR::ConstType inst) const
    -> SemIR::CompleteTypeInfo {
  // The object and value representation of `const T` are the same as those of
  // `T`. Objects are not modifiable through their value representations.
  return GetNestedInfo(context_->types().GetTypeIdForTypeInstId(inst.inner_id));
}

auto TypeCompleter::BuildInfoForInst(SemIR::TypeId type_id,
                                     SemIR::CustomLayoutType inst) const
    -> SemIR::CompleteTypeInfo {
  // TODO: Should we support other value representations for custom layout
  // types?
  const auto& layout = context_->custom_layouts().Get(inst.layout_id);
  return {.value_repr = MakePointerValueRepr(type_id),
          .object_layout = {
              .size = layout[SemIR::CustomLayoutId::SizeIndex],
              .alignment = layout[SemIR::CustomLayoutId::AlignIndex]}};
}

auto TypeCompleter::BuildInfoForInst(SemIR::TypeId type_id,
                                     SemIR::MaybeUnformedType inst) const
    -> SemIR::CompleteTypeInfo {
  // `MaybeUnformed(T)` has the same value representation as `T` if that value
  // representation preserves all the bytes of the value, including any padding
  // bits. Otherwise we need to use a different representation.
  auto inner_type_id = context_->types().GetTypeIdForTypeInstId(inst.inner_id);
  auto nested = GetNestedInfo(inner_type_id);
  if (nested.value_repr.kind == SemIR::ValueRepr::Custom) {
    nested.value_repr = MakePointerValueRepr(type_id);
  } else if (nested.value_repr.kind == SemIR::ValueRepr::Copy) {
    auto type_inst = context_->types().GetAsInst(nested.value_repr.type_id);
    // TODO: Should ValueRepr::IsCopyOfObjectRepr return false for `bool`?
    if (!nested.value_repr.IsCopyOfObjectRepr(context_->sem_ir(),
                                              inner_type_id) ||
        type_inst.Is<SemIR::BoolType>()) {
      nested.value_repr = MakePointerValueRepr(type_id);
    }
    // TODO: Handle any other types that we treat as having discarded padding
    // bits. For now there are no such types, as all class types and all structs
    // and tuples with more than one element are passed indirectly.
  }
  return nested;
}

auto TypeCompleter::BuildInfoForInst(SemIR::TypeId /*type_id*/,
                                     SemIR::PartialType inst) const
    -> SemIR::CompleteTypeInfo {
  // The value representation of `partial T` is the same as that of `T`.
  // Objects are not modifiable through their value representations. However,
  // `partial T` is never abstract.
  auto inner_type_id = context_->types().GetTypeIdForTypeInstId(inst.inner_id);
  auto info = GetNestedInfo(inner_type_id);
  info.abstract_class_id = SemIR::ClassId::None;
  return info;
}

auto TypeCompleter::BuildInfoForInst(
    SemIR::TypeId /*type_id*/, SemIR::ImplWitnessAssociatedConstant inst) const
    -> SemIR::CompleteTypeInfo {
  return GetNestedInfo(inst.type_id);
}

auto TypeCompleter::BuildInfoForInst(SemIR::TypeId type_id,
                                     SemIR::ImplWitnessAccess /*inst*/) const
    -> SemIR::CompleteTypeInfo {
  // An ImplWitnessAccess is typically symbolic. But even if the witness is
  // replaced by a concrete one, which does not provide a value for the
  // ImplWitnessAcess to use, then it is still a dependent value, as the actual
  // value remains unknown.
  return MakeDependentTypeInfo(type_id);
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

auto TryToCompleteType(Context& context, SemIR::TypeId type_id,
                       SemIR::LocId loc_id, bool diagnose) -> bool {
  return TypeCompleter(&context, loc_id, diagnose).Complete(type_id);
}

auto CompleteTypeOrCheckFail(Context& context, SemIR::TypeId type_id) -> void {
  bool complete =
      TypeCompleter(&context, SemIR::LocId::None, false).Complete(type_id);
  CARBON_CHECK(complete, "Expected {0} to be a complete type",
               context.types().GetAsInst(type_id));
}

auto RequireCompleteType(Context& context, SemIR::TypeId type_id,
                         SemIR::LocId loc_id,
                         DiagnosticContextFn diagnostic_context) -> bool {
  CARBON_CHECK(diagnostic_context);
  Diagnostics::ContextScope scope(&context.emitter(), diagnostic_context);

  if (!TypeCompleter(&context, loc_id, true).Complete(type_id)) {
    return false;
  }

  // For a symbolic type, create an instruction to require the corresponding
  // specific type to be complete.
  if (type_id.is_symbolic()) {
    // TODO: Deduplicate these.
    AddInstInNoBlock(
        context, loc_id,
        SemIR::RequireCompleteType{
            .type_id =
                GetSingletonType(context, SemIR::WitnessType::TypeInstId),
            .complete_type_inst_id = context.types().GetTypeInstId(type_id)});
  }

  return true;
}

auto TryIsConcreteType(Context& context, SemIR::TypeId type_id,
                       SemIR::LocId loc_id) -> bool {
  if (!TryToCompleteType(context, type_id, loc_id)) {
    return false;
  }

  auto complete_info = context.types().GetCompleteTypeInfo(type_id);
  CARBON_CHECK(complete_info.value_repr.type_id.has_value(),
               "TryIsConcreteType called for an incomplete type. Call "
               "TryToCompleteType first.");
  return !complete_info.abstract_class_id.has_value();
}

auto RequireConcreteType(Context& context, SemIR::TypeId type_id,
                         SemIR::LocId loc_id,
                         DiagnosticContextFn complete_type_diagnostic_context,
                         DiagnosticContextFn concrete_type_diagnostic_context)
    -> bool {
  if (!RequireCompleteType(context, type_id, loc_id,
                           complete_type_diagnostic_context)) {
    return false;
  }

  CARBON_CHECK(concrete_type_diagnostic_context);
  Diagnostics::ContextScope scope(&context.emitter(),
                                  concrete_type_diagnostic_context);

  // TODO: For symbolic types, should add an implicit constraint that they are
  // not abstract.
  const auto& complete_info = context.types().GetCompleteTypeInfo(type_id);
  CARBON_CHECK(complete_info.value_repr.type_id.has_value(),
               "RequireConcreteType called for an incomplete type. Call "
               "RequireCompleteType first.");
  if (!complete_info.IsAbstract()) {
    return true;
  }

  bool direct_use = false;
  if (auto inst = context.types().TryGetAs<SemIR::ClassType>(type_id)) {
    if (inst->class_id == complete_info.abstract_class_id) {
      direct_use = true;
    }
  }
  DiagnoseAbstractClass(context, complete_info.abstract_class_id, direct_use);
  return false;
}

// Given a canonical facet value, or a type value, return a facet value.
static auto GetSelfFacetValue(Context& context, SemIR::ConstantId self_const_id)
    -> SemIR::ConstantId {
  if (self_const_id == SemIR::ErrorInst::ConstantId) {
    return SemIR::ErrorInst::ConstantId;
  }

  auto self_inst_id = context.constant_values().GetInstId(self_const_id);
  auto type_id = context.insts().Get(self_inst_id).type_id();
  CARBON_CHECK(context.types().IsFacetType(type_id));

  if (context.types().Is<SemIR::FacetType>(type_id)) {
    return self_const_id;
  }

  return GetConstantFacetValueForType(
      context, context.types().GetAsTypeInstId(self_inst_id));
}

static auto IdentifyFacetType(Context& context, SemIR::LocId loc_id,
                              SemIR::ConstantId self_const_id,
                              const SemIR::FacetType& facet_type,
                              bool allow_partially_identified, bool diagnose)
    -> SemIR::IdentifiedFacetTypeId {
  // While partially identified facet types end up in the store of
  // IdentifiedFacetTypes, we don't try to construct a key to look for them
  // here, so we will only early-out here for fully identified facet types. To
  // construct the key for a partially identified facet type we need to know the
  // set of required impls that it contains, which requires us to do most of the
  // work of identifying the facet type (though we could skip the mapping of
  // constant values into specifics).
  auto key =
      SemIR::IdentifiedFacetTypeKey{.facet_type_id = facet_type.facet_type_id,
                                    .self_const_id = self_const_id};
  if (auto identified_id = context.identified_facet_types().Lookup(key);
      identified_id.has_value()) {
    return identified_id;
  }

  struct SelfImplsFacetType {
    bool extend;
    SemIR::ConstantId self;
    SemIR::FacetTypeId facet_type;
  };

  // Work queue.
  llvm::SmallVector<SelfImplsFacetType> work = {
      {true, self_const_id, facet_type.facet_type_id}};

  // Outputs for the IdentifiedFacetType.
  bool partially_identified = false;
  llvm::SmallVector<SemIR::IdentifiedFacetType::RequiredImpl> extends;
  llvm::SmallVector<SemIR::IdentifiedFacetType::RequiredImpl> impls;

  while (!work.empty()) {
    SelfImplsFacetType next_impls = work.pop_back_val();
    bool facet_type_extends = next_impls.extend;
    auto self_const_id = GetCanonicalFacetOrTypeValue(context, next_impls.self);
    const auto& facet_type_info =
        context.facet_types().Get(next_impls.facet_type);

    auto self_and_interface = [&](SemIR::SpecificInterface interface)
        -> SemIR::IdentifiedFacetType::RequiredImpl {
      return {self_const_id, interface};
    };
    auto type_and_interface =
        [&](const SemIR::FacetTypeInfo::TypeImplsInterface& impls)
        -> SemIR::IdentifiedFacetType::RequiredImpl {
      return {context.constant_values().Get(impls.self_type),
              impls.specific_interface};
    };

    if (facet_type_extends) {
      llvm::append_range(extends,
                         llvm::map_range(facet_type_info.extend_constraints,
                                         self_and_interface));
    } else {
      llvm::append_range(impls,
                         llvm::map_range(facet_type_info.extend_constraints,
                                         self_and_interface));
    }
    llvm::append_range(impls,
                       llvm::map_range(facet_type_info.self_impls_constraints,
                                       self_and_interface));
    llvm::append_range(impls,
                       llvm::map_range(facet_type_info.type_impls_interfaces,
                                       type_and_interface));

    if (facet_type_info.extend_named_constraints.empty() &&
        facet_type_info.self_impls_named_constraints.empty() &&
        facet_type_info.type_impls_named_constraints.empty()) {
      continue;
    }

    // The self may have type TypeType. But the `Self` in a generic require decl
    // has type FacetType, so we need something similar to replace it in the
    // specific.
    auto self_facet = GetSelfFacetValue(context, self_const_id);

    for (auto extends : facet_type_info.extend_named_constraints) {
      const auto& constraint =
          context.named_constraints().Get(extends.named_constraint_id);

      llvm::ArrayRef<SemIR::RequireImplsId> require_impls_ids;
      if (constraint.is_complete()) {
        require_impls_ids = context.require_impls_blocks().Get(
            constraint.require_impls_block_id);
      } else if (allow_partially_identified) {
        partially_identified = true;
        if (constraint.is_being_defined()) {
          require_impls_ids = context.require_impls_stack().PeekForScope(
              extends.named_constraint_id);
        } else {
          continue;
        }
      } else {
        if (diagnose) {
          DiagnoseIncompleteNamedConstraint(context,
                                            extends.named_constraint_id);
        }
        return SemIR::IdentifiedFacetTypeId::None;
      }

      auto constraint_with_self_specific_id = MakeSpecificWithInnerSelf(
          context, loc_id, constraint.generic_id,
          constraint.generic_with_self_id, extends.specific_id, self_facet);
      if (SpecificHasError(context, constraint_with_self_specific_id)) {
        return SemIR::IdentifiedFacetTypeId::None;
      }

      for (auto require_impls_id : llvm::reverse(require_impls_ids)) {
        const auto& require = context.require_impls().Get(require_impls_id);

        // Each require is in its own generic, with no additional bindings and
        // no definition, so that they can have their specifics independently
        // instantiated.
        auto require_specific_id = CopySpecificToGeneric(
            context, SemIR::LocId(require.decl_id),
            constraint_with_self_specific_id, require.generic_id);
        auto require_self = GetConstantValueInSpecific(
            context.sem_ir(), require_specific_id, require.self_id);
        auto require_facet_type = GetConstantValueInSpecific(
            context.sem_ir(), require_specific_id, require.facet_type_inst_id);
        if (require_self == SemIR::ErrorInst::ConstantId ||
            require_facet_type == SemIR::ErrorInst::ConstantId) {
          return SemIR::IdentifiedFacetTypeId::None;
        }

        auto facet_type_id =
            context.constant_values()
                .GetInstAs<SemIR::FacetType>(require_facet_type)
                .facet_type_id;
        bool extend = facet_type_extends && require.extend_self;
        work.push_back({extend, require_self, facet_type_id});
      }
    }

    for (auto impls : facet_type_info.self_impls_named_constraints) {
      const auto& constraint =
          context.named_constraints().Get(impls.named_constraint_id);

      llvm::ArrayRef<SemIR::RequireImplsId> require_impls_ids;
      if (constraint.is_complete()) {
        require_impls_ids = context.require_impls_blocks().Get(
            constraint.require_impls_block_id);
      } else if (allow_partially_identified) {
        partially_identified = true;
        if (constraint.is_being_defined()) {
          require_impls_ids = context.require_impls_stack().PeekForScope(
              impls.named_constraint_id);
        } else {
          continue;
        }
      } else {
        if (diagnose) {
          DiagnoseIncompleteNamedConstraint(context, impls.named_constraint_id);
        }
        return SemIR::IdentifiedFacetTypeId::None;
      }

      auto constraint_with_self_specific_id = MakeSpecificWithInnerSelf(
          context, loc_id, constraint.generic_id,
          constraint.generic_with_self_id, impls.specific_id, self_facet);
      if (SpecificHasError(context, constraint_with_self_specific_id)) {
        return SemIR::IdentifiedFacetTypeId::None;
      }

      for (auto require_impls_id : llvm::reverse(require_impls_ids)) {
        const auto& require = context.require_impls().Get(require_impls_id);

        // Each require is in its own generic, with no additional bindings and
        // no definition, so that they can have their specifics independently
        // instantiated.
        auto require_specific_id = CopySpecificToGeneric(
            context, SemIR::LocId(require.decl_id),
            constraint_with_self_specific_id, require.generic_id);
        auto require_self = GetConstantValueInSpecific(
            context.sem_ir(), require_specific_id, require.self_id);
        auto require_facet_type = GetConstantValueInSpecific(
            context.sem_ir(), require_specific_id, require.facet_type_inst_id);
        if (require_self == SemIR::ErrorInst::ConstantId ||
            require_facet_type == SemIR::ErrorInst::ConstantId) {
          return SemIR::IdentifiedFacetTypeId::None;
        }

        auto facet_type_id =
            context.constant_values()
                .GetInstAs<SemIR::FacetType>(require_facet_type)
                .facet_type_id;
        work.push_back({false, require_self, facet_type_id});
      }
    }

    for (const auto& type_impls :
         facet_type_info.type_impls_named_constraints) {
      auto [self_type_inst_id, impls] = type_impls;
      const auto& constraint =
          context.named_constraints().Get(impls.named_constraint_id);

      llvm::ArrayRef<SemIR::RequireImplsId> require_impls_ids;
      if (constraint.is_complete()) {
        require_impls_ids = context.require_impls_blocks().Get(
            constraint.require_impls_block_id);
      } else if (allow_partially_identified) {
        partially_identified = true;
        if (constraint.is_being_defined()) {
          require_impls_ids = context.require_impls_stack().PeekForScope(
              impls.named_constraint_id);
        } else {
          continue;
        }
      } else {
        if (diagnose) {
          DiagnoseIncompleteNamedConstraint(context, impls.named_constraint_id);
        }
        return SemIR::IdentifiedFacetTypeId::None;
      }

      auto self_type_facet = GetSelfFacetValue(
          context, context.constant_values().Get(self_type_inst_id));

      auto constraint_with_self_specific_id = MakeSpecificWithInnerSelf(
          context, loc_id, constraint.generic_id,
          constraint.generic_with_self_id, impls.specific_id, self_type_facet);
      if (SpecificHasError(context, constraint_with_self_specific_id)) {
        return SemIR::IdentifiedFacetTypeId::None;
      }

      for (auto require_impls_id : llvm::reverse(require_impls_ids)) {
        const auto& require = context.require_impls().Get(require_impls_id);

        // Each require is in its own generic, with no additional bindings and
        // no definition, so that they can have their specifics independently
        // instantiated.
        auto require_specific_id = CopySpecificToGeneric(
            context, SemIR::LocId(require.decl_id),
            constraint_with_self_specific_id, require.generic_id);
        auto require_self = GetConstantValueInSpecific(
            context.sem_ir(), require_specific_id, require.self_id);
        auto require_facet_type = GetConstantValueInSpecific(
            context.sem_ir(), require_specific_id, require.facet_type_inst_id);
        if (require_self == SemIR::ErrorInst::ConstantId ||
            require_facet_type == SemIR::ErrorInst::ConstantId) {
          return SemIR::IdentifiedFacetTypeId::None;
        }

        auto facet_type_id =
            context.constant_values()
                .GetInstAs<SemIR::FacetType>(require_facet_type)
                .facet_type_id;
        work.push_back({false, require_self, facet_type_id});
      }
    }
  }

  // TODO: Process other kinds of requirements.
  return context.identified_facet_types().Add(
      {key, partially_identified, extends, impls});
}

auto TryToIdentifyFacetType(Context& context, SemIR::LocId loc_id,
                            SemIR::ConstantId self_const_id,
                            const SemIR::FacetType& facet_type,
                            bool allow_partially_identified)
    -> SemIR::IdentifiedFacetTypeId {
  return IdentifyFacetType(context, loc_id, self_const_id, facet_type,
                           allow_partially_identified, /*diagnose=*/false);
}

auto RequireIdentifiedFacetType(Context& context, SemIR::LocId loc_id,
                                SemIR::ConstantId self_const_id,
                                const SemIR::FacetType& facet_type,
                                DiagnosticContextFn diagnostic_context,
                                bool diagnose) -> SemIR::IdentifiedFacetTypeId {
  CARBON_CHECK(diagnostic_context);
  Diagnostics::ContextScope scope(&context.emitter(), diagnostic_context);

  return IdentifyFacetType(context, loc_id, self_const_id, facet_type,
                           /*allow_partially_identified=*/false, diagnose);
}

}  // namespace Carbon::Check
