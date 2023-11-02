// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/lex/token_kind.h"

namespace Carbon::Check {

auto HandleClassIntroducer(Context& context, Parse::Node parse_node) -> bool {
  // Create an instruction block to hold the instructions created as part of the
  // class signature, such as generic parameters.
  context.inst_block_stack().Push();
  // Push the bracketing node.
  context.node_stack().Push(parse_node);
  // A name should always follow.
  context.declaration_name_stack().Push();
  return true;
}

static auto BuildClassDeclaration(Context& context)
    -> std::tuple<SemIR::ClassId, SemIR::InstId> {
  auto name_context = context.declaration_name_stack().Pop();
  auto class_keyword =
      context.node_stack()
          .PopForSoloParseNode<Parse::NodeKind::ClassIntroducer>();
  auto decl_block_id = context.inst_block_stack().Pop();

  // Add the class declaration.
  auto class_decl = SemIR::ClassDeclaration{
      class_keyword, SemIR::ClassId::Invalid, decl_block_id};
  auto class_decl_id = context.AddInst(class_decl);

  // Check whether this is a redeclaration.
  auto existing_id = context.declaration_name_stack().LookupOrAddName(
      name_context, class_decl_id);
  if (existing_id.is_valid()) {
    if (auto existing_class_decl =
            context.insts().Get(existing_id).TryAs<SemIR::ClassDeclaration>()) {
      // This is a redeclaration of an existing class.
      class_decl.class_id = existing_class_decl->class_id;
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
        {.name_id = name_context.state ==
                            DeclarationNameStack::NameContext::State::Unresolved
                        ? name_context.unresolved_name_id
                        : IdentifierId::Invalid,
         // `.self_type_id` depends on `class_id`, so is set below.
         .self_type_id = SemIR::TypeId::Invalid,
         .declaration_id = class_decl_id});

    // Build the `Self` type.
    auto& class_info = context.classes().Get(class_decl.class_id);
    class_info.self_type_id =
        context.CanonicalizeType(context.AddInst(SemIR::ClassType{
            class_keyword, context.GetBuiltinType(SemIR::BuiltinKind::TypeType),
            class_decl.class_id}));
  }

  // Write the class ID into the ClassDeclaration.
  context.insts().Set(class_decl_id, class_decl);

  return {class_decl.class_id, class_decl_id};
}

auto HandleClassDeclaration(Context& context, Parse::Node /*parse_node*/)
    -> bool {
  BuildClassDeclaration(context);
  return true;
}

auto HandleClassDefinitionStart(Context& context, Parse::Node parse_node)
    -> bool {
  auto [class_id, class_decl_id] = BuildClassDeclaration(context);
  auto& class_info = context.classes().Get(class_id);

  // Track that this declaration is the definition.
  if (class_info.definition_id.is_valid()) {
    CARBON_DIAGNOSTIC(ClassRedefinition, Error, "Redefinition of class {0}.",
                      llvm::StringRef);
    CARBON_DIAGNOSTIC(ClassPreviousDefinition, Note,
                      "Previous definition was here.");
    context.emitter()
        .Build(parse_node, ClassRedefinition,
               context.identifiers().Get(class_info.name_id))
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
  // TODO: This will shadow a local variable declared with name `r#Self`, but
  // should not. See #2984 and the corresponding code in
  // HandleSelfTypeNameExpression.
  context.AddNameToLookup(
      parse_node,
      context.identifiers().Add(
          Lex::TokenKind::SelfTypeIdentifier.fixed_spelling()),
      context.sem_ir().GetTypeAllowBuiltinTypes(class_info.self_type_id));

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

auto HandleClassDefinition(Context& context, Parse::Node parse_node) -> bool {
  auto fields_id = context.args_type_info_stack().Pop();
  auto class_id =
      context.node_stack().Pop<Parse::NodeKind::ClassDefinitionStart>();
  context.inst_block_stack().Pop();
  context.PopScope();

  // The class type is now fully defined.
  auto& class_info = context.classes().Get(class_id);
  class_info.object_representation_id =
      context.CanonicalizeStructType(parse_node, fields_id);
  return true;
}

}  // namespace Carbon::Check
