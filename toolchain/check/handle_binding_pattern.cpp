// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/return.h"

namespace Carbon::Check {

auto HandleAnyBindingPattern(Context& context, Parse::NodeId node_id,
                             bool is_generic) -> bool {
  auto [type_node, parsed_type_id] = context.node_stack().PopExprWithNodeId();
  auto cast_type_id = ExprAsType(context, type_node, parsed_type_id);

  // TODO: Handle `_` bindings.

  // Every other kind of pattern binding has a name.
  auto [name_node, name_id] = context.node_stack().PopNameWithNodeId();

  // Create the appropriate kind of binding for this pattern.
  auto make_bind_name = [&](SemIR::TypeId type_id,
                            SemIR::InstId value_id) -> SemIR::LocIdAndInst {
    // TODO: Eventually the name will need to support associations with other
    // scopes, but right now we don't support qualified names here.
    auto bind_name_id = context.bind_names().Add(
        {.name_id = name_id,
         .enclosing_scope_id = context.scope_stack().PeekNameScopeId()});
    if (is_generic) {
      // TODO: Create a `BindTemplateName` instead inside a `template` pattern.
      return {name_node,
              SemIR::BindSymbolicName{type_id, bind_name_id, value_id}};
    } else {
      return {name_node, SemIR::BindName{type_id, bind_name_id, value_id}};
    }
  };

  // A `self` binding can only appear in an implicit parameter list.
  if (name_id == SemIR::NameId::SelfValue &&
      !context.node_stack().PeekIs<Parse::NodeKind::ImplicitParamListStart>()) {
    CARBON_DIAGNOSTIC(
        SelfOutsideImplicitParamList, Error,
        "`self` can only be declared in an implicit parameter list.");
    context.emitter().Emit(node_id, SelfOutsideImplicitParamList);
  }

  // Allocate an instruction of the appropriate kind, linked to the name for
  // error locations.
  // TODO: The node stack is a fragile way of getting context information.
  // Get this information from somewhere else.
  switch (auto context_node_kind = context.node_stack().PeekNodeKind()) {
    case Parse::NodeKind::ReturnedModifier:
    case Parse::NodeKind::VariableIntroducer: {
      if (is_generic) {
        CARBON_DIAGNOSTIC(
            CompileTimeBindingInVarDecl, Error,
            "`var` declaration cannot declare a compile-time binding.");
        context.emitter().Emit(type_node, CompileTimeBindingInVarDecl);
      }
      auto binding_id =
          is_generic
              ? Parse::NodeId::Invalid
              : context.parse_tree().As<Parse::BindingPatternId>(node_id);

      // A `var` declaration at class scope introduces a field.
      auto enclosing_class_decl = context.GetCurrentScopeAs<SemIR::ClassDecl>();
      cast_type_id = context.AsCompleteType(cast_type_id, [&] {
        CARBON_DIAGNOSTIC(IncompleteTypeInVarDecl, Error,
                          "{0} has incomplete type `{1}`.", llvm::StringLiteral,
                          SemIR::TypeId);
        return context.emitter().Build(type_node, IncompleteTypeInVarDecl,
                                       enclosing_class_decl
                                           ? llvm::StringLiteral("Field")
                                           : llvm::StringLiteral("Variable"),
                                       cast_type_id);
      });
      if (enclosing_class_decl) {
        CARBON_CHECK(context_node_kind == Parse::NodeKind::VariableIntroducer)
            << "`returned var` at class scope";
        auto& class_info =
            context.classes().Get(enclosing_class_decl->class_id);
        auto field_type_id = context.GetUnboundElementType(
            class_info.self_type_id, cast_type_id);
        auto field_id = context.AddInst(
            {binding_id, SemIR::FieldDecl{
                             field_type_id, name_id,
                             SemIR::ElementIndex(context.args_type_info_stack()
                                                     .PeekCurrentBlockContents()
                                                     .size())}});

        // Add a corresponding field to the object representation of the class.
        context.args_type_info_stack().AddInstId(context.AddInstInNoBlock(
            {binding_id, SemIR::StructTypeField{name_id, cast_type_id}}));
        context.node_stack().Push(node_id, field_id);
        break;
      }

      SemIR::InstId value_id = SemIR::InstId::Invalid;
      if (context_node_kind == Parse::NodeKind::ReturnedModifier) {
        // TODO: Should we check this for the `var` as a whole, rather than for
        // the name binding?
        value_id =
            CheckReturnedVar(context, context.node_stack().PeekNodeId(),
                             name_node, name_id, type_node, cast_type_id);
      } else {
        value_id = context.AddInst(
            {name_node, SemIR::VarStorage{cast_type_id, name_id}});
      }
      auto bind_id = context.AddInst(make_bind_name(cast_type_id, value_id));
      context.node_stack().Push(node_id, bind_id);

      if (context_node_kind == Parse::NodeKind::ReturnedModifier) {
        RegisterReturnedVar(context, bind_id);
      }
      break;
    }

    case Parse::NodeKind::ImplicitParamListStart:
    case Parse::NodeKind::TuplePatternStart: {
      // Parameters can have incomplete types in a function declaration, but not
      // in a function definition. We don't know which kind we have here.
      // TODO: A tuple pattern can appear in other places than function
      // parameters.
      auto param_id =
          context.AddInst({name_node, SemIR::Param{cast_type_id, name_id}});
      auto bind_id = context.AddInst(make_bind_name(cast_type_id, param_id));
      // TODO: Bindings should come into scope immediately in other contexts
      // too.
      context.AddNameToLookup(name_id, bind_id);
      context.node_stack().Push(node_id, bind_id);
      break;
    }

    case Parse::NodeKind::LetIntroducer:
      cast_type_id = context.AsCompleteType(cast_type_id, [&] {
        CARBON_DIAGNOSTIC(IncompleteTypeInLetDecl, Error,
                          "`let` binding has incomplete type `{0}`.",
                          SemIR::TypeId);
        return context.emitter().Build(type_node, IncompleteTypeInLetDecl,
                                       cast_type_id);
      });
      // Create the instruction, but don't add it to a block until after we've
      // formed its initializer.
      // TODO: For general pattern parsing, we'll need to create a block to hold
      // the `let` pattern before we see the initializer.
      context.node_stack().Push(
          node_id, context.AddPlaceholderInstInNoBlock(
                       make_bind_name(cast_type_id, SemIR::InstId::Invalid)));
      break;

    default:
      CARBON_FATAL() << "Found a pattern binding in unexpected context "
                     << context_node_kind;
  }
  return true;
}

auto HandleBindingPattern(Context& context, Parse::BindingPatternId node_id)
    -> bool {
  return HandleAnyBindingPattern(context, node_id, /*is_generic=*/false);
}

auto HandleCompileTimeBindingPattern(Context& context,
                                     Parse::CompileTimeBindingPatternId node_id)
    -> bool {
  return HandleAnyBindingPattern(context, node_id, /*is_generic=*/true);
}

auto HandleAddr(Context& context, Parse::AddrId node_id) -> bool {
  auto self_param_id = context.node_stack().PopPattern();
  if (auto self_param =
          context.insts().TryGetAs<SemIR::AnyBindName>(self_param_id);
      self_param &&
      context.bind_names().Get(self_param->bind_name_id).name_id ==
          SemIR::NameId::SelfValue) {
    // TODO: The type of an `addr_pattern` should probably be the non-pointer
    // type, because that's the type that the pattern matches.
    context.AddInstAndPush(
        {node_id, SemIR::AddrPattern{self_param->type_id, self_param_id}});
  } else {
    CARBON_DIAGNOSTIC(AddrOnNonSelfParam, Error,
                      "`addr` can only be applied to a `self` parameter.");
    context.emitter().Emit(TokenOnly(node_id), AddrOnNonSelfParam);
    context.node_stack().Push(node_id, self_param_id);
  }
  return true;
}

auto HandleTemplate(Context& context, Parse::TemplateId node_id) -> bool {
  return context.TODO(node_id, "HandleTemplate");
}

}  // namespace Carbon::Check
