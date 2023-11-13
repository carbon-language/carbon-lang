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
  context.decl_name_stack().PushScopeAndStartName();
  return true;
}

auto HandleAbstractModifier(Context& context, Parse::Node parse_node) -> bool {
  context.node_stack().Push(parse_node);
  return true;
}

auto HandleBaseModifier(Context& context, Parse::Node parse_node) -> bool {
  context.node_stack().Push(parse_node);
  return true;
}

static auto BuildClassDecl(Context& context)
    -> std::tuple<SemIR::ClassId, SemIR::InstId> {
  auto name_context = context.decl_name_stack().FinishName();
  auto introducer = context.node_stack().PeekParseNode();
  bool abstract =
      context.node_stack()
          .PopAndDiscardSoloParseNodeIf<Parse::NodeKind::AbstractModifier>();
  bool base =
      context.node_stack()
          .PopAndDiscardSoloParseNodeIf<Parse::NodeKind::BaseModifier>();
  context.node_stack()
      .PopAndDiscardSoloParseNode<Parse::NodeKind::ClassIntroducer>();
  auto decl_block_id = context.inst_block_stack().Pop();

  CARBON_CHECK(!(abstract && base)) << "Cannot be both `abstract` and `base`";
  auto inheritance_kind = abstract ? SemIR::Class::Abstract
                          : base   ? SemIR::Class::Base
                                   : SemIR::Class::Final;

  // Add the class declaration.
  auto class_decl =
      SemIR::ClassDecl{introducer, SemIR::ClassId::Invalid, decl_block_id};
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
            .Build(introducer, ClassRedeclarationDifferentIntroducer)
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
            introducer, context.GetBuiltinType(SemIR::BuiltinKind::TypeType),
            class_decl.class_id}));
  }

  // Write the class ID into the ClassDecl.
  context.insts().Set(class_decl_id, class_decl);

  return {class_decl.class_id, class_decl_id};
}

auto HandleClassDecl(Context& context, Parse::Node /*parse_node*/) -> bool {
  BuildClassDecl(context);
  context.decl_name_stack().PopScope();
  return true;
}

auto HandleClassDefinitionStart(Context& context, Parse::Node parse_node)
    -> bool {
  auto [class_id, class_decl_id] = BuildClassDecl(context);
  auto& class_info = context.classes().Get(class_id);

  // Track that this declaration is the definition.
  if (class_info.definition_id.is_valid()) {
    CARBON_DIAGNOSTIC(ClassRedefinition, Error, "Redefinition of class {0}.",
                      llvm::StringRef);
    CARBON_DIAGNOSTIC(ClassPreviousDefinition, Note,
                      "Previous definition was here.");
    context.emitter()
        .Build(parse_node, ClassRedefinition,
               context.names().GetFormatted(class_info.name_id))
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
  context.AddNameToLookup(
      parse_node, SemIR::NameId::SelfType,
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
  context.decl_name_stack().PopScope();

  // The class type is now fully defined.
  auto& class_info = context.classes().Get(class_id);
  class_info.object_representation_id =
      context.CanonicalizeStructType(parse_node, fields_id);
  return true;
}

}  // namespace Carbon::Check
