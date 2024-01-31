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
  return std::nullopt;
}

// Given an instruction produced by a name lookup, get the value to use for that
// result in an expression.
static auto GetExprValueForLookupResult(Context& context,
                                        SemIR::InstId lookup_result_id)
    -> SemIR::InstId {
  // If lookup finds a class declaration, the value is its `Self` type.
  auto lookup_result = context.insts().Get(lookup_result_id);
  if (auto class_decl = lookup_result.TryAs<SemIR::ClassDecl>()) {
    return context.types().GetInstId(
        context.classes().Get(class_decl->class_id).self_type_id);
  }
  if (auto interface_decl = lookup_result.TryAs<SemIR::InterfaceDecl>()) {
    return TryEvalInst(context, SemIR::InstId::Invalid,
                       SemIR::InterfaceType{SemIR::TypeId::TypeType,
                                            interface_decl->interface_id})
        .inst_id();
  }

  // Anything else should be a typed value already.
  CARBON_CHECK(lookup_result.kind().value_kind() == SemIR::InstValueKind::Typed)
      << "Unexpected kind for lookup result, " << lookup_result;
  return lookup_result_id;
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

auto HandleMemberAccessExpr(Context& context,
                            Parse::MemberAccessExprId parse_node) -> bool {
  SemIR::NameId name_id = context.node_stack().PopName();
  auto base_id = context.node_stack().PopExpr();

  // If the base is a name scope, such as a class or namespace, perform lookup
  // into that scope.
  if (auto name_scope_id = GetAsNameScope(context, base_id)) {
    auto inst_id =
        name_scope_id->is_valid()
            ? context.LookupQualifiedName(parse_node, name_id, *name_scope_id)
            : SemIR::InstId::BuiltinError;
    inst_id = GetExprValueForLookupResult(context, inst_id);
    auto inst = context.insts().Get(inst_id);
    // TODO: Track that this instruction was named within `base_id`.
    context.AddInstAndPush(
        {parse_node, SemIR::NameRef{inst.type_id(), name_id, inst_id}});
    return true;
  }

  // If the base isn't a scope, it must have a complete type.
  auto base_type_id = context.insts().Get(base_id).type_id();
  if (!context.TryToCompleteType(base_type_id, [&] {
        CARBON_DIAGNOSTIC(IncompleteTypeInMemberAccess, Error,
                          "Member access into object of incomplete type `{0}`.",
                          std::string);
        return context.emitter().Build(
            base_id, IncompleteTypeInMemberAccess,
            context.sem_ir().StringifyType(base_type_id));
      })) {
    context.node_stack().Push(parse_node, SemIR::InstId::BuiltinError);
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
          context.LookupQualifiedName(parse_node, name_id, class_scope_id);
      member_id = GetExprValueForLookupResult(context, member_id);

      // Perform instance binding if we found an instance member.
      auto member_type_id = context.insts().Get(member_id).type_id();
      if (auto unbound_element_type =
              context.types().TryGetAs<SemIR::UnboundElementType>(
                  member_type_id)) {
        // Convert the base to the type of the element if necessary.
        base_id = ConvertToValueOrRefOfType(
            context, parse_node, base_id, unbound_element_type->class_type_id);

        // Find the specified element, which could be either a field or a base
        // class, and build an element access expression.
        auto element_id = context.constant_values().Get(member_id);
        CARBON_CHECK(element_id.is_constant())
            << "Non-constant value " << context.insts().Get(member_id)
            << " of unbound element type";
        auto index = GetClassElementIndex(context, element_id.inst_id());
        auto access_id = context.AddInst(
            {parse_node,
             SemIR::ClassElementAccess{unbound_element_type->element_type_id,
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
        context.node_stack().Push(parse_node, access_id);
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
              {parse_node,
               SemIR::BoundMethod{
                   context.GetBuiltinType(SemIR::BuiltinKind::BoundMethodType),
                   base_id, member_id}});
          return true;
        }
      }

      // For a non-instance member, the result is that member.
      // TODO: Track that this was named within `base_id`.
      context.AddInstAndPush(
          {parse_node, SemIR::NameRef{member_type_id, name_id, member_id}});
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
              {parse_node, SemIR::StructAccess{field.field_type_id, base_id,
                                               SemIR::ElementIndex(i)}});
          return true;
        }
      }
      CARBON_DIAGNOSTIC(QualifiedExprNameNotFound, Error,
                        "Type `{0}` does not have a member `{1}`.", std::string,
                        std::string);
      context.emitter().Emit(parse_node, QualifiedExprNameNotFound,
                             context.sem_ir().StringifyType(base_type_id),
                             context.names().GetFormatted(name_id).str());
      break;
    }
    // TODO: `ConstType` should support member access just like the
    // corresponding non-const type, except that the result should have `const`
    // type if it creates a reference expression performing field access.
    default: {
      if (base_type_id != SemIR::TypeId::Error) {
        CARBON_DIAGNOSTIC(QualifiedExprUnsupported, Error,
                          "Type `{0}` does not support qualified expressions.",
                          std::string);
        context.emitter().Emit(parse_node, QualifiedExprUnsupported,
                               context.sem_ir().StringifyType(base_type_id));
      }
      break;
    }
  }

  // Should only be reached on error.
  context.node_stack().Push(parse_node, SemIR::InstId::BuiltinError);
  return true;
}

auto HandlePointerMemberAccessExpr(Context& context,
                                   Parse::PointerMemberAccessExprId parse_node)
    -> bool {
  return context.TODO(parse_node, "HandlePointerMemberAccessExpr");
}

static auto GetIdentifierAsName(Context& context, Parse::NodeId parse_node)
    -> std::optional<SemIR::NameId> {
  auto token = context.parse_tree().node_token(parse_node);
  if (context.tokens().GetKind(token) != Lex::TokenKind::Identifier) {
    CARBON_CHECK(context.parse_tree().node_has_error(parse_node));
    return std::nullopt;
  }
  return SemIR::NameId::ForIdentifier(context.tokens().GetIdentifier(token));
}

// Handle a name that is used as an expression by performing unqualified name
// lookup.
static auto HandleNameAsExpr(Context& context, Parse::NodeId parse_node,
                             SemIR::NameId name_id) -> bool {
  auto value_id = context.LookupUnqualifiedName(parse_node, name_id);
  value_id = GetExprValueForLookupResult(context, value_id);
  auto value = context.insts().Get(value_id);
  context.AddInstAndPush(
      {parse_node, SemIR::NameRef{value.type_id(), name_id, value_id}});
  return true;
}

auto HandleIdentifierName(Context& context, Parse::IdentifierNameId parse_node)
    -> bool {
  // The parent is responsible for binding the name.
  auto name_id = GetIdentifierAsName(context, parse_node);
  if (!name_id) {
    return context.TODO(parse_node, "Error recovery from keyword name.");
  }
  context.node_stack().Push(parse_node, *name_id);
  return true;
}

auto HandleIdentifierNameExpr(Context& context,
                              Parse::IdentifierNameExprId parse_node) -> bool {
  auto name_id = GetIdentifierAsName(context, parse_node);
  if (!name_id) {
    return context.TODO(parse_node, "Error recovery from keyword name.");
  }
  return HandleNameAsExpr(context, parse_node, *name_id);
}

auto HandleBaseName(Context& context, Parse::BaseNameId parse_node) -> bool {
  context.node_stack().Push(parse_node, SemIR::NameId::Base);
  return true;
}

auto HandleSelfTypeNameExpr(Context& context,
                            Parse::SelfTypeNameExprId parse_node) -> bool {
  return HandleNameAsExpr(context, parse_node, SemIR::NameId::SelfType);
}

auto HandleSelfValueName(Context& context, Parse::SelfValueNameId parse_node)
    -> bool {
  context.node_stack().Push(parse_node, SemIR::NameId::SelfValue);
  return true;
}

auto HandleSelfValueNameExpr(Context& context,
                             Parse::SelfValueNameExprId parse_node) -> bool {
  return HandleNameAsExpr(context, parse_node, SemIR::NameId::SelfValue);
}

auto HandleQualifiedName(Context& context, Parse::QualifiedNameId parse_node)
    -> bool {
  auto [parse_node2, name_id2] = context.node_stack().PopNameWithParseNode();

  Parse::NodeId parse_node1 = context.node_stack().PeekParseNode();
  switch (context.parse_tree().node_kind(parse_node1)) {
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
      context.decl_name_stack().ApplyNameQualifier(parse_node1, name_id);
      // Add the QualifiedName so that it can be used for bracketing.
      context.node_stack().Push(parse_node);
      break;
    }

    default:
      CARBON_FATAL() << "Unexpected node kind on left side of qualified "
                        "declaration name";
  }

  context.decl_name_stack().ApplyNameQualifier(parse_node2, name_id2);
  return true;
}

auto HandlePackageExpr(Context& context, Parse::PackageExprId parse_node)
    -> bool {
  context.AddInstAndPush(
      {parse_node,
       SemIR::NameRef{context.GetBuiltinType(SemIR::BuiltinKind::NamespaceType),
                      SemIR::NameId::PackageNamespace,
                      SemIR::InstId::PackageNamespace}});
  return true;
}

}  // namespace Carbon::Check
