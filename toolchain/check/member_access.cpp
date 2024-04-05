// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/ADT/STLExtras.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/subst.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Returns the name scope corresponding to base_id, or nullopt if not a scope.
// On invalid scopes, prints a diagnostic and still returns the scope.
static auto GetAsNameScope(Context& context, Parse::NodeId node_id,
                           SemIR::ConstantId base_const_id)
    -> std::optional<SemIR::NameScopeId> {
  auto base = context.insts().Get(base_const_id.inst_id());
  if (auto base_as_namespace = base.TryAs<SemIR::Namespace>()) {
    return base_as_namespace->name_scope_id;
  }
  // TODO: Consider refactoring the near-identical class and interface support
  // below.
  if (auto base_as_class = base.TryAs<SemIR::ClassType>()) {
    auto& class_info = context.classes().Get(base_as_class->class_id);
    if (!class_info.is_defined()) {
      CARBON_DIAGNOSTIC(QualifiedExprInIncompleteClassScope, Error,
                        "Member access into incomplete class `{0}`.",
                        std::string);
      auto builder = context.emitter().Build(
          node_id, QualifiedExprInIncompleteClassScope,
          context.sem_ir().StringifyTypeExpr(base_const_id.inst_id()));
      context.NoteIncompleteClass(base_as_class->class_id, builder);
      builder.Emit();
    }
    return class_info.scope_id;
  }
  if (auto base_as_interface = base.TryAs<SemIR::InterfaceType>()) {
    auto& interface_info =
        context.interfaces().Get(base_as_interface->interface_id);
    if (!interface_info.is_defined()) {
      CARBON_DIAGNOSTIC(QualifiedExprInUndefinedInterfaceScope, Error,
                        "Member access into undefined interface `{0}`.",
                        std::string);
      auto builder = context.emitter().Build(
          node_id, QualifiedExprInUndefinedInterfaceScope,
          context.sem_ir().StringifyTypeExpr(base_const_id.inst_id()));
      context.NoteUndefinedInterface(base_as_interface->interface_id, builder);
      builder.Emit();
    }
    return interface_info.scope_id;
  }
  // TODO: Per the design, if `base_id` is any kind of type, then lookup should
  // treat it as a name scope, even if it doesn't have members. For example,
  // `(i32*).X` should fail because there's no name `X` in `i32*`, not because
  // there's no name `X` in `type`.
  return std::nullopt;
}

// Returns the index of the specified class element within the class's
// representation.
static auto GetClassElementIndex(Context& context, SemIR::InstId element_id)
    -> SemIR::ElementIndex {
  auto element_inst = context.insts().Get(element_id);
  if (auto field = element_inst.TryAs<SemIR::FieldDecl>()) {
    return field->index;
  }
  if (auto base = element_inst.TryAs<SemIR::BaseDecl>()) {
    return base->index;
  }
  CARBON_FATAL() << "Unexpected value " << element_inst
                 << " in class element name";
}

// Returns whether `function_id` is an instance method, that is, whether it has
// an implicit `self` parameter.
static auto IsInstanceMethod(const SemIR::File& sem_ir,
                             SemIR::FunctionId function_id) -> bool {
  const auto& function = sem_ir.functions().Get(function_id);
  for (auto param_id :
       sem_ir.inst_blocks().Get(function.implicit_param_refs_id)) {
    auto param =
        SemIR::Function::GetParamFromParamRefId(sem_ir, param_id).second;
    if (param.name_id == SemIR::NameId::SelfValue) {
      return true;
    }
  }

  return false;
}

// Returns whether `name_scope_id` is a scope for which impl lookup should be
// performed if we find an associated entity.
static auto ScopeNeedsImplLookup(Context& context,
                                 SemIR::NameScopeId name_scope_id) -> bool {
  auto inst_id = context.name_scopes().GetInstIdIfValid(name_scope_id);
  if (!inst_id.is_valid()) {
    return false;
  }

  auto inst = context.insts().Get(inst_id);
  if (inst.Is<SemIR::InterfaceDecl>()) {
    // Don't perform impl lookup if an associated entity is named as a member of
    // a facet type.
    return false;
  }
  if (inst.Is<SemIR::Namespace>()) {
    // Don't perform impl lookup if an associated entity is named as a namespace
    // member.
    // TODO: This case is not yet listed in the design.
    return false;
  }
  // Any other kind of scope is assumed to be a type that implements the
  // interface containing the associated entity, and impl lookup is performed.
  return true;
}

// Given a type and an interface, searches for an impl that describes how that
// type implements that interface, and returns the corresponding witness.
// Returns an invalid InstId if no matching impl is found.
static auto LookupInterfaceWitness(Context& context, SemIRLoc loc,
                                   SemIR::ConstantId type_const_id,
                                   SemIR::InterfaceId interface_id)
    -> SemIR::InstId {
  // TODO: Add a better impl lookup system. At the very least, we should only be
  // considering impls that are for the same interface we're querying. We can
  // also skip impls that mention any types that aren't part of our impl query.
  for (const auto& impl : context.impls().array_ref()) {
    if (context.types().GetInstId(impl.self_id) != type_const_id.inst_id()) {
      continue;
    }
    auto interface_type =
        context.types().TryGetAs<SemIR::InterfaceType>(impl.constraint_id);
    if (!interface_type) {
      // TODO: An impl of a constraint type should be treated as implementing
      // the constraint's interfaces.
      continue;
    }
    if (interface_type->interface_id != interface_id) {
      continue;
    }
    if (!impl.witness_id.is_valid()) {
      // TODO: Diagnose if the impl isn't defined yet?
      return SemIR::InstId::Invalid;
    }
    LoadImportRef(context, impl.witness_id, loc);
    return impl.witness_id;
  }
  return SemIR::InstId::Invalid;
}

// Performs impl lookup for a member name expression. This finds the relevant
// impl witness and extracts the corresponding impl member.
static auto PerformImplLookup(Context& context, Parse::NodeId node_id,
                              SemIR::ConstantId type_const_id,
                              SemIR::AssociatedEntityType assoc_type,
                              SemIR::InstId member_id) -> SemIR::InstId {
  auto& interface = context.interfaces().Get(assoc_type.interface_id);
  auto witness_id = LookupInterfaceWitness(context, node_id, type_const_id,
                                           assoc_type.interface_id);
  if (!witness_id.is_valid()) {
    CARBON_DIAGNOSTIC(MissingImplInMemberAccess, Error,
                      "Cannot access member of interface {0} in type {1} "
                      "that does not implement that interface.",
                      SemIR::NameId, std::string);
    context.emitter().Emit(
        node_id, MissingImplInMemberAccess, interface.name_id,
        context.sem_ir().StringifyTypeExpr(type_const_id.inst_id()));
    return SemIR::InstId::BuiltinError;
  }

  auto const_id = context.constant_values().Get(member_id);
  if (!const_id.is_constant()) {
    if (const_id != SemIR::ConstantId::Error) {
      context.TODO(member_id, "non-constant associated entity");
    }
    return SemIR::InstId::BuiltinError;
  }

  auto assoc_entity =
      context.insts().TryGetAs<SemIR::AssociatedEntity>(const_id.inst_id());
  if (!assoc_entity) {
    context.TODO(member_id, "unexpected value for associated entity");
    return SemIR::InstId::BuiltinError;
  }

  // Substitute into the type declared in the interface.
  Substitution substitutions[1] = {
      {.bind_id = interface.self_param_id, .replacement_id = type_const_id}};
  auto subst_type_id =
      SubstType(context, assoc_type.entity_type_id, substitutions);

  return context.AddInst(SemIR::InterfaceWitnessAccess{
      subst_type_id, witness_id, assoc_entity->index});
}

// Performs a member name lookup into the specified scope, including performing
// impl lookup if necessary. If the scope is invalid, assume an error has
// already been diagnosed, and return BuiltinError.
static auto LookupMemberNameInScope(Context& context, Parse::NodeId node_id,
                                    SemIR::InstId /*base_id*/,
                                    SemIR::NameId name_id,
                                    SemIR::ConstantId name_scope_const_id,
                                    SemIR::NameScopeId name_scope_id)
    -> SemIR::InstId {
  auto inst_id = name_scope_id.is_valid() ? context.LookupQualifiedName(
                                                node_id, name_id, name_scope_id)
                                          : SemIR::InstId::BuiltinError;
  auto inst = context.insts().Get(inst_id);
  // TODO: Use a different kind of instruction that also references the
  // `base_id` so that `SemIR` consumers can find it.
  auto member_id = context.AddInst(
      {node_id, SemIR::NameRef{inst.type_id(), name_id, inst_id}});

  // If member name lookup finds an associated entity name, and the scope is not
  // a facet type, perform impl lookup.
  //
  // TODO: We need to do this as part of searching extended scopes, because a
  // lookup that finds an associated entity and also finds the corresponding
  // impl member is not supposed to be treated as ambiguous.
  if (auto assoc_type = context.types().TryGetAs<SemIR::AssociatedEntityType>(
          inst.type_id())) {
    if (ScopeNeedsImplLookup(context, name_scope_id)) {
      member_id = PerformImplLookup(context, node_id, name_scope_const_id,
                                    *assoc_type, member_id);
    }
  }

  return member_id;
}

// Performs the instance binding step in member access. If the found member is a
// field, forms a class member access. If the found member is an instance
// method, forms a bound method. Otherwise, the member is returned unchanged.
static auto PerformInstanceBinding(Context& context, Parse::NodeId node_id,
                                   SemIR::InstId base_id,
                                   SemIR::InstId member_id) -> SemIR::InstId {
  auto member_type_id = context.insts().Get(member_id).type_id();
  if (auto unbound_element_type =
          context.types().TryGetAs<SemIR::UnboundElementType>(member_type_id)) {
    // Convert the base to the type of the element if necessary.
    base_id = ConvertToValueOrRefOfType(context, node_id, base_id,
                                        unbound_element_type->class_type_id);

    // Find the specified element, which could be either a field or a base
    // class, and build an element access expression.
    auto element_id = context.constant_values().Get(member_id);
    CARBON_CHECK(element_id.is_constant())
        << "Non-constant value " << context.insts().Get(member_id)
        << " of unbound element type";
    auto index = GetClassElementIndex(context, element_id.inst_id());
    auto access_id = context.AddInst(
        {node_id, SemIR::ClassElementAccess{
                      unbound_element_type->element_type_id, base_id, index}});
    if (SemIR::GetExprCategory(context.sem_ir(), base_id) ==
            SemIR::ExprCategory::Value &&
        SemIR::GetExprCategory(context.sem_ir(), access_id) !=
            SemIR::ExprCategory::Value) {
      // Class element access on a value expression produces an ephemeral
      // reference if the class's value representation is a pointer to the
      // object representation. Add a value binding in that case so that the
      // expression category of the result matches the expression category of
      // the base.
      access_id = ConvertToValueExpr(context, access_id);
    }
    return access_id;
  }

  if (member_type_id ==
      context.GetBuiltinType(SemIR::BuiltinKind::FunctionType)) {
    // Find the named function and check whether it's an instance method.
    auto function_name_id = context.constant_values().Get(member_id);
    if (function_name_id == SemIR::ConstantId::Error) {
      return SemIR::InstId::BuiltinError;
    }

    CARBON_CHECK(function_name_id.is_constant())
        << "Non-constant value " << context.insts().Get(member_id)
        << " of function type";
    auto function_decl = context.insts()
                             .Get(function_name_id.inst_id())
                             .TryAs<SemIR::FunctionDecl>();
    CARBON_CHECK(function_decl)
        << "Unexpected value "
        << context.insts().Get(function_name_id.inst_id())
        << " of function type";
    if (IsInstanceMethod(context.sem_ir(), function_decl->function_id)) {
      return context.AddInst(
          {node_id, SemIR::BoundMethod{context.GetBuiltinType(
                                           SemIR::BuiltinKind::BoundMethodType),
                                       base_id, member_id}});
    }
  }

  // Not an instance member: no instance binding.
  return member_id;
}

auto PerformMemberAccess(Context& context, Parse::NodeId node_id,
                         SemIR::InstId base_id, SemIR::NameId name_id)
    -> SemIR::InstId {
  // If the base is a name scope, such as a class or namespace, perform lookup
  // into that scope.
  if (auto base_const_id = context.constant_values().Get(base_id);
      base_const_id.is_constant()) {
    if (auto name_scope_id = GetAsNameScope(context, node_id, base_const_id)) {
      return LookupMemberNameInScope(context, node_id, base_id, name_id,
                                     base_const_id, *name_scope_id);
    }
  }

  // If the base isn't a scope, it must have a complete type.
  auto base_type_id = context.insts().Get(base_id).type_id();
  if (!context.TryToCompleteType(base_type_id, [&] {
        CARBON_DIAGNOSTIC(IncompleteTypeInMemberAccess, Error,
                          "Member access into object of incomplete type `{0}`.",
                          SemIR::TypeId);
        return context.emitter().Build(base_id, IncompleteTypeInMemberAccess,
                                       base_type_id);
      })) {
    return SemIR::InstId::BuiltinError;
  }

  // Materialize a temporary for the base expression if necessary.
  base_id = ConvertToValueOrRefExpr(context, base_id);
  base_type_id = context.insts().Get(base_id).type_id();
  auto base_type_const_id = context.types().GetConstantId(base_type_id);

  // Find the scope corresponding to the base type.
  auto name_scope_id = GetAsNameScope(context, node_id, base_type_const_id);
  if (!name_scope_id) {
    // The base type is not a name scope. Try some fallback options.
    if (auto struct_type = context.insts().TryGetAs<SemIR::StructType>(
            base_type_const_id.inst_id())) {
      // TODO: Do we need to optimize this with a lookup table for O(1)?
      for (auto [i, ref_id] :
           llvm::enumerate(context.inst_blocks().Get(struct_type->fields_id))) {
        auto field = context.insts().GetAs<SemIR::StructTypeField>(ref_id);
        if (name_id == field.name_id) {
          // TODO: Model this as producing a lookup result, and do instance
          // binding separately. Perhaps a struct type should be a name scope.
          return context.AddInst(
              {node_id, SemIR::StructAccess{field.field_type_id, base_id,
                                            SemIR::ElementIndex(i)}});
        }
      }
      CARBON_DIAGNOSTIC(QualifiedExprNameNotFound, Error,
                        "Type `{0}` does not have a member `{1}`.",
                        SemIR::TypeId, SemIR::NameId);
      context.emitter().Emit(node_id, QualifiedExprNameNotFound, base_type_id,
                             name_id);
      return SemIR::InstId::BuiltinError;
    }

    if (base_type_id != SemIR::TypeId::Error) {
      CARBON_DIAGNOSTIC(QualifiedExprUnsupported, Error,
                        "Type `{0}` does not support qualified expressions.",
                        SemIR::TypeId);
      context.emitter().Emit(node_id, QualifiedExprUnsupported, base_type_id);
    }
    return SemIR::InstId::BuiltinError;
  }

  // Perform lookup into the base type.
  auto member_id = LookupMemberNameInScope(context, node_id, base_id, name_id,
                                           base_type_const_id, *name_scope_id);

  // Perform instance binding if we found an instance member.
  member_id = PerformInstanceBinding(context, node_id, base_id, member_id);

  return member_id;
}

auto PerformCompoundMemberAccess(Context& context, Parse::NodeId node_id,
                                 SemIR::InstId base_id,
                                 SemIR::InstId member_expr_id)
    -> SemIR::InstId {
  // Materialize a temporary for the base expression if necessary.
  base_id = ConvertToValueOrRefExpr(context, base_id);
  auto base_type_id = context.insts().Get(base_id).type_id();
  auto base_type_const_id = context.types().GetConstantId(base_type_id);

  auto member_id = member_expr_id;
  auto member = context.insts().Get(member_id);

  // If the member expression names an associated entity, impl lookup is always
  // performed using the type of the base expression.
  if (auto assoc_type = context.types().TryGetAs<SemIR::AssociatedEntityType>(
          member.type_id())) {
    member_id = PerformImplLookup(context, node_id, base_type_const_id,
                                  *assoc_type, member_id);
  }

  // Perform instance binding if we found an instance member.
  member_id = PerformInstanceBinding(context, node_id, base_id, member_id);

  // If we didn't perform impl lookup or instance binding, that's an error
  // because the base expression is not used for anything.
  if (member_id == member_expr_id) {
    CARBON_DIAGNOSTIC(CompoundMemberAccessDoesNotUseBase, Error,
                      "Member name of type `{0}` in compound member access is "
                      "not an instance member or an interface member.",
                      SemIR::TypeId);
    context.emitter().Emit(node_id, CompoundMemberAccessDoesNotUseBase,
                           member.type_id());
  }

  return member_id;
}

}  // namespace Carbon::Check
