// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/modifiers.h"

namespace Carbon::Check {

auto HandleClassIntroducer(Context& context, Parse::NodeId parse_node) -> bool {
  // Create an instruction block to hold the instructions created as part of the
  // class signature, such as generic parameters.
  context.inst_block_stack().Push();
  // Push the bracketing node.
  context.node_stack().Push(parse_node);
  // Optional modifiers and the name follow.
  context.decl_state_stack().Push(DeclState::Class, parse_node);
  context.decl_name_stack().PushScopeAndStartName();
  return true;
}

static auto BuildClassDecl(Context& context)
    -> std::tuple<SemIR::ClassId, SemIR::InstId> {
  auto name_context = context.decl_name_stack().FinishName();
  context.node_stack()
      .PopAndDiscardSoloParseNode<Parse::NodeKind::ClassIntroducer>();
  auto first_node = context.decl_state_stack().innermost().first_node;

  // Process modifiers.
  CheckAccessModifiersOnDecl(context, Lex::TokenKind::Class);
  LimitModifiersOnDecl(context,
                       KeywordModifierSet::Class | KeywordModifierSet::Access,
                       Lex::TokenKind::Class);

  auto modifiers = context.decl_state_stack().innermost().modifier_set;
  if (!!(modifiers & KeywordModifierSet::Access)) {
    context.TODO(context.decl_state_stack().innermost().saw_access_modifier,
                 "access modifier");
  }
  auto inheritance_kind =
      !!(modifiers & KeywordModifierSet::Abstract) ? SemIR::Class::Abstract
      : !!(modifiers & KeywordModifierSet::Base)   ? SemIR::Class::Base
                                                   : SemIR::Class::Final;

  context.decl_state_stack().Pop(DeclState::Class);
  auto decl_block_id = context.inst_block_stack().Pop();

  // Add the class declaration.
  auto class_decl =
      SemIR::ClassDecl{first_node, SemIR::ClassId::Invalid, decl_block_id};
  auto class_decl_id = context.AddInst(class_decl);

  // Check whether this is a redeclaration.
  auto existing_id =
      context.decl_name_stack().LookupOrAddName(name_context, class_decl_id);
  if (existing_id.is_valid()) {
    if (auto existing_class_decl =
            context.insts().Get(existing_id).TryAs<SemIR::ClassDecl>()) {
      // This is a redeclaration of an existing class.
      class_decl.class_id = existing_class_decl->class_id;
      auto& class_info = context.classes().Get(class_decl.class_id);

      // The introducer kind must match the previous declaration.
      // TODO: The rule here is not yet decided. See #3384.
      if (class_info.inheritance_kind != inheritance_kind) {
        CARBON_DIAGNOSTIC(ClassRedeclarationDifferentIntroducer, Error,
                          "Class redeclared with different inheritance kind.");
        CARBON_DIAGNOSTIC(ClassRedeclarationDifferentIntroducerPrevious, Note,
                          "Previously declared here.");
        context.emitter()
            .Build(first_node, ClassRedeclarationDifferentIntroducer)
            .Note(existing_class_decl->parse_node,
                  ClassRedeclarationDifferentIntroducerPrevious)
            .Emit();
      }

      // TODO: Check that the generic parameter list agrees with the prior
      // declaration.
    } else {
      // This is a redeclaration of something other than a class.
      context.DiagnoseDuplicateName(name_context.parse_node, existing_id);
    }
  }

  // Create a new class if this isn't a valid redeclaration.
  if (!class_decl.class_id.is_valid()) {
    // TODO: If this is an invalid redeclaration of a non-class entity or there
    // was an error in the qualifier, we will have lost track of the class name
    // here. We should keep track of it even if the name is invalid.
    class_decl.class_id = context.classes().Add(
        {.name_id =
             name_context.state == DeclNameStack::NameContext::State::Unresolved
                 ? name_context.unresolved_name_id
                 : SemIR::NameId::Invalid,
         // `.self_type_id` depends on `class_id`, so is set below.
         .self_type_id = SemIR::TypeId::Invalid,
         .decl_id = class_decl_id,
         .inheritance_kind = inheritance_kind});

    // Build the `Self` type.
    auto& class_info = context.classes().Get(class_decl.class_id);
    class_info.self_type_id =
        context.CanonicalizeType(context.AddInst(SemIR::ClassType{
            first_node, context.GetBuiltinType(SemIR::BuiltinKind::TypeType),
            class_decl.class_id}));
  }

  // Write the class ID into the ClassDecl.
  context.insts().Set(class_decl_id, class_decl);

  return {class_decl.class_id, class_decl_id};
}

auto HandleClassDecl(Context& context, Parse::NodeId /*parse_node*/) -> bool {
  BuildClassDecl(context);
  context.decl_name_stack().PopScope();
  return true;
}

auto HandleClassDefinitionStart(Context& context, Parse::NodeId parse_node)
    -> bool {
  auto [class_id, class_decl_id] = BuildClassDecl(context);
  auto& class_info = context.classes().Get(class_id);

  // Track that this declaration is the definition.
  if (class_info.definition_id.is_valid()) {
    CARBON_DIAGNOSTIC(ClassRedefinition, Error, "Redefinition of class {0}.",
                      std::string);
    CARBON_DIAGNOSTIC(ClassPreviousDefinition, Note,
                      "Previous definition was here.");
    context.emitter()
        .Build(parse_node, ClassRedefinition,
               context.names().GetFormatted(class_info.name_id).str())
        .Note(context.insts().Get(class_info.definition_id).parse_node(),
              ClassPreviousDefinition)
        .Emit();
  } else {
    class_info.definition_id = class_decl_id;
    class_info.scope_id = context.name_scopes().Add();
  }

  // Enter the class scope.
  context.PushScope(class_decl_id, class_info.scope_id);

  // Introduce `Self`.
  context.AddNameToLookup(parse_node, SemIR::NameId::SelfType,
                          context.types().GetInstId(class_info.self_type_id));

  context.inst_block_stack().Push();
  context.node_stack().Push(parse_node, class_id);
  context.args_type_info_stack().Push();

  // TODO: Handle the case where there's control flow in the class body. For
  // example:
  //
  //   class C {
  //     var v: if true then i32 else f64;
  //   }
  //
  // We may need to track a list of instruction blocks here, as we do for a
  // function.
  class_info.body_block_id = context.inst_block_stack().PeekOrAdd();
  return true;
}

auto HandleBaseIntroducer(Context& context, Parse::NodeId parse_node) -> bool {
  context.decl_state_stack().Push(DeclState::Base, parse_node);
  return true;
}

auto HandleBaseColon(Context& /*context*/, Parse::NodeId /*parse_node*/)
    -> bool {
  return true;
}

auto HandleBaseDecl(Context& context, Parse::NodeId parse_node) -> bool {
  auto base_type_expr_id = context.node_stack().PopExpr();

  // Process modifiers. `extend` is required, none others are allowed.
  LimitModifiersOnDecl(context, KeywordModifierSet::Extend,
                       Lex::TokenKind::Base);
  auto modifiers = context.decl_state_stack().innermost().modifier_set;
  if (!(modifiers & KeywordModifierSet::Extend)) {
    CARBON_DIAGNOSTIC(BaseMissingExtend, Error,
                      "Missing `extend` before `base` declaration in class.");
    context.emitter().Emit(context.decl_state_stack().innermost().first_node,
                           BaseMissingExtend);
  }
  context.decl_state_stack().Pop(DeclState::Base);

  auto enclosing_class_decl = context.GetCurrentScopeAs<SemIR::ClassDecl>();
  if (!enclosing_class_decl) {
    CARBON_DIAGNOSTIC(BaseOutsideClass, Error,
                      "`base` declaration can only be used in a class.");
    context.emitter().Emit(parse_node, BaseOutsideClass);
    return true;
  }

  auto& class_info = context.classes().Get(enclosing_class_decl->class_id);
  if (class_info.base_id.is_valid()) {
    CARBON_DIAGNOSTIC(BaseRepeated, Error,
                      "Multiple `base` declarations in class. Multiple "
                      "inheritance is not permitted.");
    CARBON_DIAGNOSTIC(BasePrevious, Note,
                      "Previous `base` declaration is here.");
    context.emitter()
        .Build(parse_node, BaseRepeated)
        .Note(context.insts().Get(class_info.base_id).parse_node(),
              BasePrevious)
        .Emit();
    return true;
  }

  auto base_type_id = ExprAsType(context, parse_node, base_type_expr_id);
  base_type_id = context.AsCompleteType(base_type_id, [&] {
    CARBON_DIAGNOSTIC(IncompleteTypeInBaseDecl, Error,
                      "Base `{0}` is an incomplete type.", std::string);
    return context.emitter().Build(
        parse_node, IncompleteTypeInBaseDecl,
        context.sem_ir().StringifyType(base_type_id));
  });

  if (base_type_id != SemIR::TypeId::Error) {
    // For now, we treat all types that aren't introduced by a `class`
    // declaration as being final classes.
    // TODO: Once we have a better idea of which types are considered to be
    // classes, produce a better diagnostic for deriving from a non-class type.
    auto base_class = context.types().TryGetAs<SemIR::ClassType>(base_type_id);
    if (!base_class ||
        context.classes().Get(base_class->class_id).inheritance_kind ==
            SemIR::Class::Final) {
      CARBON_DIAGNOSTIC(BaseIsFinal, Error,
                        "Deriving from final type `{0}`. Base type must be an "
                        "`abstract` or `base` class.",
                        std::string);
      context.emitter().Emit(parse_node, BaseIsFinal,
                             context.sem_ir().StringifyType(base_type_id));
    }
  }

  // The `base` value in the class scope has an unbound element type. Instance
  // binding will be performed when it's found by name lookup into an instance.
  auto field_type_inst_id = context.AddInst(SemIR::UnboundElementType{
      parse_node, context.GetBuiltinType(SemIR::BuiltinKind::TypeType),
      class_info.self_type_id, base_type_id});
  auto field_type_id = context.CanonicalizeType(field_type_inst_id);
  class_info.base_id = context.AddInst(SemIR::BaseDecl{
      parse_node, field_type_id, base_type_id,
      SemIR::ElementIndex(
          context.args_type_info_stack().PeekCurrentBlockContents().size())});

  // Add a corresponding field to the object representation of the class.
  // TODO: Consider whether we want to use `partial T` here.
  context.args_type_info_stack().AddInst(
      SemIR::StructTypeField{parse_node, SemIR::NameId::Base, base_type_id});

  // Bind the name `base` in the class to the base field.
  context.decl_name_stack().AddNameToLookup(
      context.decl_name_stack().MakeUnqualifiedName(parse_node,
                                                    SemIR::NameId::Base),
      class_info.base_id);
  return true;
}

auto HandleClassDefinition(Context& context, Parse::NodeId parse_node) -> bool {
  auto fields_id = context.args_type_info_stack().Pop();
  auto class_id =
      context.node_stack().Pop<Parse::NodeKind::ClassDefinitionStart>();
  context.inst_block_stack().Pop();
  context.PopScope();
  context.decl_name_stack().PopScope();

  // The class type is now fully defined.
  auto& class_info = context.classes().Get(class_id);
  class_info.object_repr_id =
      context.CanonicalizeStructType(parse_node, fields_id);
  return true;
}

}  // namespace Carbon::Check
