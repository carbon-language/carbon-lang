// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/return.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

auto HandleAddress(Context& context, Parse::NodeId parse_node) -> bool {
  auto self_param_id =
      context.node_stack().Peek<Parse::NodeKind::BindingPattern>();
  if (auto self_param =
          context.insts().Get(self_param_id).TryAs<SemIR::SelfParam>()) {
    self_param->is_addr_self = SemIR::BoolValue::True;
    context.insts().Set(self_param_id, *self_param);
  } else {
    CARBON_DIAGNOSTIC(AddrOnNonSelfParam, Error,
                      "`addr` can only be applied to a `self` parameter.");
    context.emitter().Emit(parse_node, AddrOnNonSelfParam);
  }
  return true;
}

auto HandleGenericBindingPattern(Context& context, Parse::NodeId parse_node)
    -> bool {
  return context.TODO(parse_node, "GenericBindingPattern");
}

auto HandleBindingPattern(Context& context, Parse::NodeId parse_node) -> bool {
  auto [type_node, parsed_type_id] =
      context.node_stack().PopExprWithParseNode();
  auto type_node_copy = type_node;
  auto cast_type_id = ExprAsType(context, type_node, parsed_type_id);

  // A `self` binding doesn't have a name.
  if (auto self_node =
          context.node_stack()
              .PopForSoloParseNodeIf<Parse::NodeKind::SelfValueName>()) {
    if (context.parse_tree().node_kind(context.node_stack().PeekParseNode()) !=
        Parse::NodeKind::ImplicitParamListStart) {
      CARBON_DIAGNOSTIC(
          SelfOutsideImplicitParamList, Error,
          "`self` can only be declared in an implicit parameter list.");
      context.emitter().Emit(parse_node, SelfOutsideImplicitParamList);
    }
    context.AddInstAndPush(
        parse_node, SemIR::SelfParam{*self_node, cast_type_id,
                                     /*is_addr_self=*/SemIR::BoolValue::False});
    return true;
  }

  // TODO: Handle `_` bindings.

  // Every other kind of pattern binding has a name.
  auto [name_node, name_id] = context.node_stack().PopNameWithParseNode();

  // Allocate an instruction of the appropriate kind, linked to the name for
  // error locations.
  // TODO: The node stack is a fragile way of getting context information.
  // Get this information from somewhere else.
  switch (auto context_parse_node_kind = context.parse_tree().node_kind(
              context.node_stack().PeekParseNode())) {
    case Parse::NodeKind::ReturnedModifier:
    case Parse::NodeKind::VariableIntroducer: {
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
        CARBON_CHECK(!enclosing_class_decl) << "`returned var` at class scope";
        value_id =
            CheckReturnedVar(context, context.node_stack().PeekParseNode(),
                             name_node, name_id, type_node, cast_type_id);
      } else if (enclosing_class_decl) {
        auto& class_info =
            context.classes().Get(enclosing_class_decl->class_id);
        auto field_type_inst_id = context.AddInst(SemIR::UnboundElementType{
            parse_node, context.GetBuiltinType(SemIR::BuiltinKind::TypeType),
            class_info.self_type_id, cast_type_id});
        value_type_id = context.CanonicalizeType(field_type_inst_id);
        value_id = context.AddInst(
            SemIR::FieldDecl{parse_node, value_type_id, name_id,
                             SemIR::ElementIndex(context.args_type_info_stack()
                                                     .PeekCurrentBlockContents()
                                                     .size())});

        // Add a corresponding field to the object representation of the class.
        context.args_type_info_stack().AddInst(
            SemIR::StructTypeField{parse_node, name_id, cast_type_id});
      } else {
        value_id = context.AddInst(
            SemIR::VarStorage{name_node, value_type_id, name_id});
      }
      auto bind_id = context.AddInst(
          SemIR::BindName{name_node, value_type_id, name_id, value_id});
      context.node_stack().Push(parse_node, bind_id);

      if (context_parse_node_kind == Parse::NodeKind::ReturnedModifier) {
        RegisterReturnedVar(context, bind_id);
      }
      break;
    }

    case Parse::NodeKind::ImplicitParamListStart:
    case Parse::NodeKind::TuplePatternStart:
      // Parameters can have incomplete types in a function declaration, but not
      // in a function definition. We don't know which kind we have here.
      // TODO: A tuple pattern can appear in other places than function
      // parameters.
      context.AddInstAndPush(parse_node,
                             SemIR::Param{name_node, cast_type_id, name_id});
      // TODO: Create a `BindName` instruction.
      break;

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
      context.node_stack().Push(
          parse_node,
          context.insts().AddInNoBlock(SemIR::BindName{
              name_node, cast_type_id, name_id, SemIR::InstId::Invalid}));
      break;

    default:
      CARBON_FATAL() << "Found a pattern binding in unexpected context "
                     << context_parse_node_kind;
  }
  return true;
}

auto HandleTemplate(Context& context, Parse::NodeId parse_node) -> bool {
  // TODO: diagnose if this occurs in a `var` context.
  return context.TODO(parse_node, "HandleTemplate");
}

}  // namespace Carbon::Check
