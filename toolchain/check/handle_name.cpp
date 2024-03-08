// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/ADT/STLExtras.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/eval.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Returns the name scope corresponding to base_id, or nullopt if not a scope.
// On invalid scopes, prints a diagnostic and still returns the scope.
static auto GetAsNameScope(Context& context, SemIR::InstId base_id)
    -> std::optional<SemIR::NameScopeId> {
  auto base_const_id = context.constant_values().Get(base_id);
  if (!base_const_id.is_constant()) {
    // A name scope must be a constant.
    return std::nullopt;
  }
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
      auto builder =
          context.emitter().Build(base_id, QualifiedExprInIncompleteClassScope,
                                  context.sem_ir().StringifyTypeExpr(base_id));
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
          base_id, QualifiedExprInUndefinedInterfaceScope,
          context.sem_ir().StringifyTypeExpr(base_id));
      context.NoteUndefinedInterface(base_as_interface->interface_id, builder);
      builder.Emit();
    }
    return interface_info.scope_id;
  }
  return std::nullopt;
}

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

auto HandleMemberAccessExpr(Context& context, Parse::MemberAccessExprId node_id)
    -> bool {
  SemIR::NameId name_id = context.node_stack().PopName();
  auto base_id = context.node_stack().PopExpr();

  // If the base is a name scope, such as a class or namespace, perform lookup
  // into that scope.
  if (auto name_scope_id = GetAsNameScope(context, base_id)) {
    auto inst_id =
        name_scope_id->is_valid()
            ? context.LookupQualifiedName(node_id, name_id, *name_scope_id)
            : SemIR::InstId::BuiltinError;
    auto inst = context.insts().Get(inst_id);
    // TODO: Track that this instruction was named within `base_id`.
    context.AddInstAndPush(
        {node_id, SemIR::NameRef{inst.type_id(), name_id, inst_id}});
    return true;
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
    context.node_stack().Push(node_id, SemIR::InstId::BuiltinError);
    return true;
  }

  // Materialize a temporary for the base expression if necessary.
  base_id = ConvertToValueOrRefExpr(context, base_id);
  base_type_id = context.insts().Get(base_id).type_id();

  switch (auto base_type = context.types().GetAsInst(base_type_id);
          base_type.kind()) {
    case SemIR::ClassType::Kind: {
      // Perform lookup for the name in the class scope.
      auto class_scope_id = context.classes()
                                .Get(base_type.As<SemIR::ClassType>().class_id)
                                .scope_id;
      auto member_id =
          context.LookupQualifiedName(node_id, name_id, class_scope_id);

      // Perform instance binding if we found an instance member.
      auto member_type_id = context.insts().Get(member_id).type_id();
      if (auto unbound_element_type =
              context.types().TryGetAs<SemIR::UnboundElementType>(
                  member_type_id)) {
        // Convert the base to the type of the element if necessary.
        base_id = ConvertToValueOrRefOfType(
            context, node_id, base_id, unbound_element_type->class_type_id);

        // Find the specified element, which could be either a field or a base
        // class, and build an element access expression.
        auto element_id = context.constant_values().Get(member_id);
        CARBON_CHECK(element_id.is_constant())
            << "Non-constant value " << context.insts().Get(member_id)
            << " of unbound element type";
        auto index = GetClassElementIndex(context, element_id.inst_id());
        auto access_id =
            context.AddInst({node_id, SemIR::ClassElementAccess{
                                          unbound_element_type->element_type_id,
                                          base_id, index}});
        if (SemIR::GetExprCategory(context.sem_ir(), base_id) ==
                SemIR::ExprCategory::Value &&
            SemIR::GetExprCategory(context.sem_ir(), access_id) !=
                SemIR::ExprCategory::Value) {
          // Class element access on a value expression produces an
          // ephemeral reference if the class's value representation is a
          // pointer to the object representation. Add a value binding in
          // that case so that the expression category of the result
          // matches the expression category of the base.
          access_id = ConvertToValueExpr(context, access_id);
        }
        context.node_stack().Push(node_id, access_id);
        return true;
      }
      if (member_type_id ==
          context.GetBuiltinType(SemIR::BuiltinKind::FunctionType)) {
        // Find the named function and check whether it's an instance method.
        auto function_name_id = context.constant_values().Get(member_id);
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
          context.AddInstAndPush(
              {node_id,
               SemIR::BoundMethod{
                   context.GetBuiltinType(SemIR::BuiltinKind::BoundMethodType),
                   base_id, member_id}});
          return true;
        }
      }

      // For a non-instance member, the result is that member.
      // TODO: Track that this was named within `base_id`.
      context.AddInstAndPush(
          {node_id, SemIR::NameRef{member_type_id, name_id, member_id}});
      return true;
    }
    case SemIR::StructType::Kind: {
      auto refs = context.inst_blocks().Get(
          base_type.As<SemIR::StructType>().fields_id);
      // TODO: Do we need to optimize this with a lookup table for O(1)?
      for (auto [i, ref_id] : llvm::enumerate(refs)) {
        auto field = context.insts().GetAs<SemIR::StructTypeField>(ref_id);
        if (name_id == field.name_id) {
          context.AddInstAndPush(
              {node_id, SemIR::StructAccess{field.field_type_id, base_id,
                                            SemIR::ElementIndex(i)}});
          return true;
        }
      }
      CARBON_DIAGNOSTIC(QualifiedExprNameNotFound, Error,
                        "Type `{0}` does not have a member `{1}`.",
                        SemIR::TypeId, SemIR::NameId);
      context.emitter().Emit(node_id, QualifiedExprNameNotFound, base_type_id,
                             name_id);
      break;
    }
    // TODO: `ConstType` should support member access just like the
    // corresponding non-const type, except that the result should have `const`
    // type if it creates a reference expression performing field access.
    default: {
      if (base_type_id != SemIR::TypeId::Error) {
        CARBON_DIAGNOSTIC(QualifiedExprUnsupported, Error,
                          "Type `{0}` does not support qualified expressions.",
                          SemIR::TypeId);
        context.emitter().Emit(node_id, QualifiedExprUnsupported, base_type_id);
      }
      break;
    }
  }

  // Should only be reached on error.
  context.node_stack().Push(node_id, SemIR::InstId::BuiltinError);
  return true;
}

auto HandlePointerMemberAccessExpr(Context& context,
                                   Parse::PointerMemberAccessExprId node_id)
    -> bool {
  return context.TODO(node_id, "HandlePointerMemberAccessExpr");
}

static auto GetIdentifierAsName(Context& context, Parse::NodeId node_id)
    -> std::optional<SemIR::NameId> {
  auto token = context.parse_tree().node_token(node_id);
  if (context.tokens().GetKind(token) != Lex::TokenKind::Identifier) {
    CARBON_CHECK(context.parse_tree().node_has_error(node_id));
    return std::nullopt;
  }
  return SemIR::NameId::ForIdentifier(context.tokens().GetIdentifier(token));
}

// Handle a name that is used as an expression by performing unqualified name
// lookup.
static auto HandleNameAsExpr(Context& context, Parse::NodeId node_id,
                             SemIR::NameId name_id) -> bool {
  auto value_id = context.LookupUnqualifiedName(node_id, name_id);
  auto value = context.insts().Get(value_id);
  context.AddInstAndPush(
      {node_id, SemIR::NameRef{value.type_id(), name_id, value_id}});
  return true;
}

auto HandleIdentifierName(Context& context, Parse::IdentifierNameId node_id)
    -> bool {
  // The parent is responsible for binding the name.
  auto name_id = GetIdentifierAsName(context, node_id);
  if (!name_id) {
    return context.TODO(node_id, "Error recovery from keyword name.");
  }
  context.node_stack().Push(node_id, *name_id);
  return true;
}

auto HandleIdentifierNameExpr(Context& context,
                              Parse::IdentifierNameExprId node_id) -> bool {
  auto name_id = GetIdentifierAsName(context, node_id);
  if (!name_id) {
    return context.TODO(node_id, "Error recovery from keyword name.");
  }
  return HandleNameAsExpr(context, node_id, *name_id);
}

auto HandleBaseName(Context& context, Parse::BaseNameId node_id) -> bool {
  context.node_stack().Push(node_id, SemIR::NameId::Base);
  return true;
}

auto HandleSelfTypeNameExpr(Context& context, Parse::SelfTypeNameExprId node_id)
    -> bool {
  return HandleNameAsExpr(context, node_id, SemIR::NameId::SelfType);
}

auto HandleSelfValueName(Context& context, Parse::SelfValueNameId node_id)
    -> bool {
  context.node_stack().Push(node_id, SemIR::NameId::SelfValue);
  return true;
}

auto HandleSelfValueNameExpr(Context& context,
                             Parse::SelfValueNameExprId node_id) -> bool {
  return HandleNameAsExpr(context, node_id, SemIR::NameId::SelfValue);
}

auto HandleQualifiedName(Context& context, Parse::QualifiedNameId node_id)
    -> bool {
  auto [node_id2, name_id2] = context.node_stack().PopNameWithNodeId();

  Parse::NodeId node_id1 = context.node_stack().PeekNodeId();
  switch (context.parse_tree().node_kind(node_id1)) {
    case Parse::NodeKind::QualifiedName:
      // This is the second or subsequent QualifiedName in a chain.
      // Nothing to do: the first QualifiedName remains as a
      // bracketing node for later QualifiedNames.
      break;

    case Parse::NodeKind::IdentifierName: {
      // This is the first QualifiedName in a chain, and starts with an
      // identifier name.
      auto name_id =
          context.node_stack().Pop<Parse::NodeKind::IdentifierName>();
      context.decl_name_stack().ApplyNameQualifier(node_id1, name_id);
      // Add the QualifiedName so that it can be used for bracketing.
      context.node_stack().Push(node_id);
      break;
    }

    default:
      CARBON_FATAL() << "Unexpected node kind on left side of qualified "
                        "declaration name";
  }

  context.decl_name_stack().ApplyNameQualifier(node_id2, name_id2);
  return true;
}

auto HandlePackageExpr(Context& context, Parse::PackageExprId node_id) -> bool {
  context.AddInstAndPush(
      {node_id,
       SemIR::NameRef{context.GetBuiltinType(SemIR::BuiltinKind::NamespaceType),
                      SemIR::NameId::PackageNamespace,
                      SemIR::InstId::PackageNamespace}});
  return true;
}

}  // namespace Carbon::Check
