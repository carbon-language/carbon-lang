// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/return.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

auto HandleAddress(Context& context, Parse::AddressId parse_node) -> bool {
  auto self_param_id = context.node_stack().PopPattern();
  if (auto self_param =
          context.insts().TryGetAs<SemIR::AnyBindName>(self_param_id);
      self_param && self_param->name_id == SemIR::NameId::SelfValue) {
    // TODO: The type of an `addr_pattern` should probably be the non-pointer
    // type, because that's the type that the pattern matches.
    context.AddInstAndPush(
        parse_node,
        SemIR::AddrPattern{parse_node, self_param->type_id, self_param_id});
  } else {
    CARBON_DIAGNOSTIC(AddrOnNonSelfParam, Error,
                      "`addr` can only be applied to a `self` parameter.");
    context.emitter().Emit(TokenOnly(parse_node), AddrOnNonSelfParam);
    context.node_stack().Push(parse_node, self_param_id);
  }
  return true;
}

auto HandleAnyBindingPattern(Context& context, Parse::NodeId parse_node,
                             bool is_generic) -> bool {
  auto [type_node, parsed_type_id] =
      context.node_stack().PopExprWithParseNode();
  auto type_node_copy = type_node;
  auto cast_type_id = ExprAsType(context, type_node, parsed_type_id);

  // TODO: Handle `_` bindings.

  // Every other kind of pattern binding has a name.
  auto [name_node, name_id] = context.node_stack().PopNameWithParseNode();

  // Create the appropriate kind of binding for this pattern.
  auto make_bind_name = [name_node = name_node, name_id = name_id, is_generic](
                            SemIR::TypeId type_id,
                            SemIR::InstId value_id) -> SemIR::Inst {
    if (is_generic) {
      // TODO: Create a `BindTemplateName` instead inside a `template` pattern.
      return SemIR::BindSymbolicName{name_node, type_id, name_id, value_id};
    } else {
      return SemIR::BindName{name_node, type_id, name_id, value_id};
    }
  };

  // A `self` binding can only appear in an implicit parameter list.
  if (name_id == SemIR::NameId::SelfValue &&
      !context.node_stack().PeekIs<Parse::NodeKind::ImplicitParamListStart>()) {
    CARBON_DIAGNOSTIC(
        SelfOutsideImplicitParamList, Error,
        "`self` can only be declared in an implicit parameter list.");
    context.emitter().Emit(parse_node, SelfOutsideImplicitParamList);
  }

  // Allocate an instruction of the appropriate kind, linked to the name for
  // error locations.
  // TODO: The node stack is a fragile way of getting context information.
  // Get this information from somewhere else.
  switch (auto context_parse_node_kind =
              context.node_stack().PeekParseNodeKind()) {
    case Parse::NodeKind::ReturnedModifier:
    case Parse::NodeKind::VariableIntroducer: {
      if (is_generic) {
        CARBON_DIAGNOSTIC(
            GenericBindingInVarDecl, Error,
            "`var` declaration cannot declare a generic binding.");
        context.emitter().Emit(type_node, GenericBindingInVarDecl);
      }
      auto binding_id = is_generic ? Parse::NodeId::Invalid
                                   : Parse::BindingPatternId(parse_node);

      // A `var` declaration at class scope introduces a field.
      auto enclosing_class_decl = context.GetCurrentScopeAs<SemIR::ClassDecl>();
      cast_type_id = context.AsCompleteType(cast_type_id, [&] {
        CARBON_DIAGNOSTIC(IncompleteTypeInVarDecl, Error,
                          "{0} has incomplete type `{1}`.", llvm::StringLiteral,
                          std::string);
        return context.emitter().Build(
            type_node_copy, IncompleteTypeInVarDecl,
            enclosing_class_decl ? llvm::StringLiteral("Field")
                                 : llvm::StringLiteral("Variable"),
            context.sem_ir().StringifyType(cast_type_id));
      });
      SemIR::InstId value_id = SemIR::InstId::Invalid;
      SemIR::TypeId value_type_id = cast_type_id;
      if (context_parse_node_kind == Parse::NodeKind::ReturnedModifier) {
        // TODO: Should we check this for the `var` as a whole, rather than for
        // the name binding?
        CARBON_CHECK(!enclosing_class_decl) << "`returned var` at class scope";
        value_id =
            CheckReturnedVar(context, context.node_stack().PeekParseNode(),
                             name_node, name_id, type_node, cast_type_id);
      } else if (enclosing_class_decl) {
        auto& class_info =
            context.classes().Get(enclosing_class_decl->class_id);
        auto field_type_inst_id = context.AddInst(SemIR::UnboundElementType{
            binding_id, context.GetBuiltinType(SemIR::BuiltinKind::TypeType),
            class_info.self_type_id, cast_type_id});
        value_type_id = context.CanonicalizeType(field_type_inst_id);
        value_id = context.AddInst(
            SemIR::FieldDecl{binding_id, value_type_id, name_id,
                             SemIR::ElementIndex(context.args_type_info_stack()
                                                     .PeekCurrentBlockContents()
                                                     .size())});

        // Add a corresponding field to the object representation of the class.
        context.args_type_info_stack().AddInst(
            SemIR::StructTypeField{binding_id, name_id, cast_type_id});
      } else {
        value_id = context.AddInst(
            SemIR::VarStorage{name_node, value_type_id, name_id});
      }
      auto bind_id = context.AddInst(make_bind_name(value_type_id, value_id));
      context.node_stack().Push(parse_node, bind_id);

      if (context_parse_node_kind == Parse::NodeKind::ReturnedModifier) {
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
          context.AddInst(SemIR::Param{name_node, cast_type_id, name_id});
      context.AddInstAndPush(parse_node,
                             make_bind_name(cast_type_id, param_id));
      break;
    }

    case Parse::NodeKind::LetIntroducer:
      cast_type_id = context.AsCompleteType(cast_type_id, [&] {
        CARBON_DIAGNOSTIC(IncompleteTypeInLetDecl, Error,
                          "`let` binding has incomplete type `{0}`.",
                          std::string);
        return context.emitter().Build(
            type_node_copy, IncompleteTypeInLetDecl,
            context.sem_ir().StringifyType(cast_type_id));
      });
      // Create the instruction, but don't add it to a block until after we've
      // formed its initializer.
      // TODO: For general pattern parsing, we'll need to create a block to hold
      // the `let` pattern before we see the initializer.
      context.node_stack().Push(parse_node,
                                context.insts().AddInNoBlock(make_bind_name(
                                    cast_type_id, SemIR::InstId::Invalid)));
      break;

    default:
      CARBON_FATAL() << "Found a pattern binding in unexpected context "
                     << context_parse_node_kind;
  }
  return true;
}

auto HandleBindingPattern(Context& context, Parse::BindingPatternId parse_node)
    -> bool {
  return HandleAnyBindingPattern(context, parse_node, /*is_generic=*/false);
}

auto HandleGenericBindingPattern(Context& context,
                                 Parse::GenericBindingPatternId parse_node)
    -> bool {
  return HandleAnyBindingPattern(context, parse_node, /*is_generic=*/true);
}

auto HandleTemplate(Context& context, Parse::TemplateId parse_node) -> bool {
  return context.TODO(parse_node, "HandleTemplate");
}

}  // namespace Carbon::Check
