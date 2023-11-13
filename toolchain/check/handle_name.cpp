// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/ADT/STLExtras.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

// Returns the name scope corresponding to base_id, or nullopt if not a scope.
// On invalid scopes, prints a diagnostic and still returns the scope.
static auto GetAsNameScope(Context& context, SemIR::InstId base_id)
    -> std::optional<SemIR::NameScopeId> {
  auto base = context.insts().Get(context.FollowNameReferences(base_id));
  if (auto base_as_namespace = base.TryAs<SemIR::Namespace>()) {
    return base_as_namespace->name_scope_id;
  }
  if (auto base_as_class = base.TryAs<SemIR::ClassType>()) {
    auto& class_info = context.classes().Get(base_as_class->class_id);
    if (!class_info.is_defined()) {
      CARBON_DIAGNOSTIC(QualifiedExprInIncompleteClassScope, Error,
                        "Member access into incomplete class `{0}`.",
                        std::string);
      auto builder = context.emitter().Build(
          context.insts().Get(base_id).parse_node(),
          QualifiedExprInIncompleteClassScope,
          context.sem_ir().StringifyTypeExpr(base_id, true));
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
    return context.sem_ir().GetTypeAllowBuiltinTypes(
        context.classes().Get(class_decl->class_id).self_type_id);
  }

  // Anything else should be a typed value already.
  CARBON_CHECK(lookup_result.kind().value_kind() == SemIR::InstValueKind::Typed)
      << "Unexpected kind for lookup result";
  return lookup_result_id;
}

auto HandleMemberAccessExpr(Context& context, Parse::Node parse_node) -> bool {
  SemIR::NameId name_id = context.node_stack().Pop<Parse::NodeKind::Name>();
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
        parse_node,
        SemIR::NameReference{parse_node, inst.type_id(), name_id, inst_id});
    return true;
  }

  // If the base isn't a scope, it must have a complete type.
  auto base_type_id = context.insts().Get(base_id).type_id();
  if (!context.TryToCompleteType(base_type_id, [&] {
        CARBON_DIAGNOSTIC(IncompleteTypeInMemberAccess, Error,
                          "Member access into object of incomplete type `{0}`.",
                          std::string);
        return context.emitter().Build(
            context.insts().Get(base_id).parse_node(),
            IncompleteTypeInMemberAccess,
            context.sem_ir().StringifyType(base_type_id, true));
      })) {
    context.node_stack().Push(parse_node, SemIR::InstId::BuiltinError);
    return true;
  }

  // Materialize a temporary for the base expression if necessary.
  base_id = ConvertToValueOrReferenceExpr(context, base_id);
  base_type_id = context.insts().Get(base_id).type_id();

  auto base_type = context.insts().Get(
      context.sem_ir().GetTypeAllowBuiltinTypes(base_type_id));

  switch (base_type.kind()) {
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
      auto member_type_inst = context.insts().Get(
          context.sem_ir().GetTypeAllowBuiltinTypes(member_type_id));
      if (auto unbound_field_type =
              member_type_inst.TryAs<SemIR::UnboundFieldType>()) {
        // TODO: Check that the unbound field type describes a member of this
        // class. Perform a conversion of the base if necessary.

        // Find the named field and build a field access expression.
        auto field_id = context.GetConstantValue(member_id);
        CARBON_CHECK(field_id.is_valid())
            << "Non-constant value " << context.insts().Get(member_id)
            << " of unbound field type";
        auto field = context.insts().Get(field_id).TryAs<SemIR::Field>();
        CARBON_CHECK(field)
            << "Unexpected value " << context.insts().Get(field_id)
            << " for field name expression";
        auto access_id = context.AddInst(SemIR::ClassFieldAccess{
            parse_node, unbound_field_type->field_type_id, base_id,
            field->index});
        if (SemIR::GetExprCategory(context.sem_ir(), base_id) ==
                SemIR::ExprCategory::Value &&
            SemIR::GetExprCategory(context.sem_ir(), access_id) !=
                SemIR::ExprCategory::Value) {
          // Class field access on a value expression produces an ephemeral
          // reference if the class's value representation is a pointer to the
          // object representation. Add a value binding in that case so that the
          // expression category of the result matches the expression category
          // of the base.
          access_id = ConvertToValueExpr(context, access_id);
        }
        context.node_stack().Push(parse_node, access_id);
        return true;
      }
      if (member_type_id ==
          context.GetBuiltinType(SemIR::BuiltinKind::FunctionType)) {
        // Find the named function and check whether it's an instance method.
        auto function_name_id = context.GetConstantValue(member_id);
        CARBON_CHECK(function_name_id.is_valid())
            << "Non-constant value " << context.insts().Get(member_id)
            << " of function type";
        auto function_decl =
            context.insts().Get(function_name_id).TryAs<SemIR::FunctionDecl>();
        CARBON_CHECK(function_decl)
            << "Unexpected value " << context.insts().Get(function_name_id)
            << " of function type";
        auto& function = context.functions().Get(function_decl->function_id);
        for (auto param_id :
             context.inst_blocks().Get(function.implicit_param_refs_id)) {
          if (context.insts().Get(param_id).Is<SemIR::SelfParam>()) {
            context.AddInstAndPush(
                parse_node,
                SemIR::BoundMethod{
                    parse_node,
                    context.GetBuiltinType(SemIR::BuiltinKind::BoundMethodType),
                    base_id, member_id});
            return true;
          }
        }
      }

      // For a non-instance member, the result is that member.
      // TODO: Track that this was named within `base_id`.
      context.AddInstAndPush(
          parse_node,
          SemIR::NameReference{parse_node, member_type_id, name_id, member_id});
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
              parse_node, SemIR::StructAccess{parse_node, field.field_type_id,
                                              base_id, SemIR::MemberIndex(i)});
          return true;
        }
      }
      CARBON_DIAGNOSTIC(QualifiedExprNameNotFound, Error,
                        "Type `{0}` does not have a member `{1}`.", std::string,
                        llvm::StringRef);
      context.emitter().Emit(parse_node, QualifiedExprNameNotFound,
                             context.sem_ir().StringifyType(base_type_id),
                             context.names().GetFormatted(name_id));
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

auto HandlePointerMemberAccessExpr(Context& context, Parse::Node parse_node)
    -> bool {
  return context.TODO(parse_node, "HandlePointerMemberAccessExpr");
}

auto HandleName(Context& context, Parse::Node parse_node) -> bool {
  auto name_id = SemIR::NameId::ForIdentifier(context.tokens().GetIdentifier(
      context.parse_tree().node_token(parse_node)));
  // The parent is responsible for binding the name.
  context.node_stack().Push(parse_node, name_id);
  return true;
}

auto HandleNameExpr(Context& context, Parse::Node parse_node) -> bool {
  auto name_id = SemIR::NameId::ForIdentifier(context.tokens().GetIdentifier(
      context.parse_tree().node_token(parse_node)));
  auto value_id = context.LookupUnqualifiedName(parse_node, name_id);
  value_id = GetExprValueForLookupResult(context, value_id);
  auto value = context.insts().Get(value_id);
  context.AddInstAndPush(
      parse_node,
      SemIR::NameReference{parse_node, value.type_id(), name_id, value_id});
  return true;
}

auto HandleQualifiedDecl(Context& context, Parse::Node parse_node) -> bool {
  auto [parse_node2, name_id2] =
      context.node_stack().PopWithParseNode<Parse::NodeKind::Name>();

  Parse::Node parse_node1 = context.node_stack().PeekParseNode();
  switch (context.parse_tree().node_kind(parse_node1)) {
    case Parse::NodeKind::QualifiedDecl:
      // This is the second or subsequent QualifiedDecl in a chain.
      // Nothing to do: the first QualifiedDecl remains as a
      // bracketing node for later QualifiedDecls.
      break;

    case Parse::NodeKind::Name: {
      // This is the first QualifiedDecl in a chain, and starts with a
      // name.
      auto name_id = context.node_stack().Pop<Parse::NodeKind::Name>();
      context.decl_name_stack().ApplyNameQualifier(parse_node1, name_id);
      // Add the QualifiedDecl so that it can be used for bracketing.
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

auto HandleSelfTypeNameExpr(Context& context, Parse::Node parse_node) -> bool {
  auto name_id = SemIR::NameId::SelfType;
  auto value_id = context.LookupUnqualifiedName(parse_node, name_id);
  auto value = context.insts().Get(value_id);
  context.AddInstAndPush(
      parse_node,
      SemIR::NameReference{parse_node, value.type_id(), name_id, value_id});
  return true;
}

auto HandleSelfValueName(Context& context, Parse::Node parse_node) -> bool {
  context.node_stack().Push(parse_node);
  return true;
}

auto HandleSelfValueNameExpr(Context& context, Parse::Node parse_node) -> bool {
  auto name_id = SemIR::NameId::SelfValue;
  auto value_id = context.LookupUnqualifiedName(parse_node, name_id);
  auto value = context.insts().Get(value_id);
  context.AddInstAndPush(
      parse_node,
      SemIR::NameReference{parse_node, value.type_id(), name_id, value_id});
  return true;
}

}  // namespace Carbon::Check
