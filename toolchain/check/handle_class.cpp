// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/class.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/merge.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// If `type_id` is a class type, get its corresponding `SemIR::Class` object.
// Otherwise returns `nullptr`.
static auto TryGetAsClass(Context& context, SemIR::TypeId type_id)
    -> SemIR::Class* {
  auto class_type = context.types().TryGetAs<SemIR::ClassType>(type_id);
  if (!class_type) {
    return nullptr;
  }
  return &context.classes().Get(class_type->class_id);
}

auto HandleClassIntroducer(Context& context, Parse::ClassIntroducerId node_id)
    -> bool {
  // Create an instruction block to hold the instructions created as part of the
  // class signature, such as generic parameters.
  context.inst_block_stack().Push();
  // Push the bracketing node.
  context.node_stack().Push(node_id);
  // Optional modifiers and the name follow.
  context.decl_state_stack().Push(DeclState::Class);
  context.decl_name_stack().PushScopeAndStartName();
  return true;
}

static auto BuildClassDecl(Context& context, Parse::AnyClassDeclId node_id,
                           bool is_definition)
    -> std::tuple<SemIR::ClassId, SemIR::InstId> {
  if (context.node_stack().PopIf<Parse::NodeKind::TuplePattern>()) {
    context.TODO(node_id, "generic class");
  }
  if (context.node_stack().PopIf<Parse::NodeKind::ImplicitParamList>()) {
    context.TODO(node_id, "generic class");
  }

  auto name_context = context.decl_name_stack().FinishName();
  context.node_stack()
      .PopAndDiscardSoloNodeId<Parse::NodeKind::ClassIntroducer>();

  // Process modifiers.
  CheckAccessModifiersOnDecl(context, Lex::TokenKind::Class,
                             name_context.target_scope_id);
  LimitModifiersOnDecl(context,
                       KeywordModifierSet::Class | KeywordModifierSet::Access |
                           KeywordModifierSet::Extern,
                       Lex::TokenKind::Class);
  RestrictExternModifierOnDecl(context, Lex::TokenKind::Class,
                               name_context.target_scope_id, is_definition);

  auto modifiers = context.decl_state_stack().innermost().modifier_set;
  if (!!(modifiers & KeywordModifierSet::Access)) {
    context.TODO(context.decl_state_stack().innermost().modifier_node_id(
                     ModifierOrder::Access),
                 "access modifier");
  }

  bool is_extern = !!(modifiers & KeywordModifierSet::Extern);
  auto inheritance_kind =
      !!(modifiers & KeywordModifierSet::Abstract) ? SemIR::Class::Abstract
      : !!(modifiers & KeywordModifierSet::Base)   ? SemIR::Class::Base
                                                   : SemIR::Class::Final;

  context.decl_state_stack().Pop(DeclState::Class);
  auto decl_block_id = context.inst_block_stack().Pop();

  // Add the class declaration.
  auto class_decl = SemIR::ClassDecl{SemIR::TypeId::TypeType,
                                     SemIR::ClassId::Invalid, decl_block_id};
  auto class_decl_id = context.AddPlaceholderInst({node_id, class_decl});

  SemIR::Class class_info = {
      .name_id = name_context.name_id_for_new_inst(),
      .enclosing_scope_id = name_context.enclosing_scope_id_for_new_inst(),
      // `.self_type_id` depends on the ClassType, so is set below.
      .self_type_id = SemIR::TypeId::Invalid,
      .decl_id = class_decl_id,
      .inheritance_kind = inheritance_kind};

  // Check whether this is a redeclaration.
  auto prev_id =
      context.decl_name_stack().LookupOrAddName(name_context, class_decl_id);
  if (prev_id.is_valid()) {
    auto prev_inst_for_merge =
        ResolvePrevInstForMerge(context, node_id, prev_id);

    if (auto prev_class_decl =
            prev_inst_for_merge.inst.TryAs<SemIR::ClassDecl>()) {
      // TODO: Fix prev_is_extern.
      if (MergeClassRedecl(context, node_id, class_info,
                           /*new_is_import=*/false, is_definition, is_extern,
                           prev_class_decl->class_id,
                           /*prev_is_extern=*/false,
                           prev_inst_for_merge.import_ir_inst_id)) {
        // When merging, use the existing entity rather than adding a new one.
        class_decl.class_id = prev_class_decl->class_id;
      }
    } else {
      // This is a redeclaration of something other than a class.
      context.DiagnoseDuplicateName(class_decl_id, prev_id);
    }
  }

  // Create a new class if this isn't a valid redeclaration.
  bool is_new_class = !class_decl.class_id.is_valid();
  if (is_new_class) {
    // TODO: If this is an invalid redeclaration of a non-class entity or there
    // was an error in the qualifier, we will have lost track of the class name
    // here. We should keep track of it even if the name is invalid.
    class_decl.class_id = context.classes().Add(class_info);
  }

  // Write the class ID into the ClassDecl.
  context.ReplaceInstBeforeConstantUse(class_decl_id, class_decl);

  if (is_new_class) {
    // Build the `Self` type using the resulting type constant.
    auto& class_info = context.classes().Get(class_decl.class_id);
    class_info.self_type_id = context.GetTypeIdForTypeInst(class_decl_id);
  }

  return {class_decl.class_id, class_decl_id};
}

auto HandleClassDecl(Context& context, Parse::ClassDeclId node_id) -> bool {
  BuildClassDecl(context, node_id, /*is_definition=*/false);
  context.decl_name_stack().PopScope();
  return true;
}

auto HandleClassDefinitionStart(Context& context,
                                Parse::ClassDefinitionStartId node_id) -> bool {
  auto [class_id, class_decl_id] =
      BuildClassDecl(context, node_id, /*is_definition=*/true);
  auto& class_info = context.classes().Get(class_id);

  // Track that this declaration is the definition.
  if (!class_info.is_defined()) {
    class_info.definition_id = class_decl_id;
    class_info.scope_id = context.name_scopes().Add(
        class_decl_id, SemIR::NameId::Invalid, class_info.enclosing_scope_id);
  }

  // Enter the class scope.
  context.scope_stack().Push(class_decl_id, class_info.scope_id);

  // Introduce `Self`.
  context.name_scopes()
      .Get(class_info.scope_id)
      .names.insert({SemIR::NameId::SelfType,
                     context.types().GetInstId(class_info.self_type_id)});

  context.inst_block_stack().Push();
  context.node_stack().Push(node_id, class_id);
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

// Diagnoses a class-specific declaration appearing outside a class.
static auto DiagnoseClassSpecificDeclOutsideClass(Context& context,
                                                  SemIRLoc loc,
                                                  Lex::TokenKind tok) -> void {
  CARBON_DIAGNOSTIC(ClassSpecificDeclOutsideClass, Error,
                    "`{0}` declaration can only be used in a class.",
                    Lex::TokenKind);
  context.emitter().Emit(loc, ClassSpecificDeclOutsideClass, tok);
}

// Returns the declaration of the immediately-enclosing class scope, or
// diagonses if there isn't one.
static auto GetEnclosingClassOrDiagnose(Context& context, SemIRLoc loc,
                                        Lex::TokenKind tok)
    -> std::optional<SemIR::ClassDecl> {
  auto class_scope = context.GetCurrentScopeAs<SemIR::ClassDecl>();
  if (!class_scope) {
    DiagnoseClassSpecificDeclOutsideClass(context, loc, tok);
  }
  return class_scope;
}

// Diagnoses a class-specific declaration that is repeated within a class, but
// is not permitted to be repeated.
static auto DiagnoseClassSpecificDeclRepeated(Context& context,
                                              SemIRLoc new_loc,
                                              SemIRLoc prev_loc,
                                              Lex::TokenKind tok) -> void {
  CARBON_DIAGNOSTIC(ClassSpecificDeclRepeated, Error,
                    "Multiple `{0}` declarations in class.{1}", Lex::TokenKind,
                    std::string);
  const llvm::StringRef extra = tok == Lex::TokenKind::Base
                                    ? " Multiple inheritance is not permitted."
                                    : "";
  CARBON_DIAGNOSTIC(ClassSpecificDeclPrevious, Note,
                    "Previous `{0}` declaration is here.", Lex::TokenKind);
  context.emitter()
      .Build(new_loc, ClassSpecificDeclRepeated, tok, extra.str())
      .Note(prev_loc, ClassSpecificDeclPrevious, tok)
      .Emit();
}

auto HandleAdaptIntroducer(Context& context,
                           Parse::AdaptIntroducerId /*node_id*/) -> bool {
  context.decl_state_stack().Push(DeclState::Adapt);
  return true;
}

auto HandleAdaptDecl(Context& context, Parse::AdaptDeclId node_id) -> bool {
  auto [adapted_type_node, adapted_type_expr_id] =
      context.node_stack().PopExprWithNodeId();

  // Process modifiers. `extend` is permitted, no others are allowed.
  LimitModifiersOnDecl(context, KeywordModifierSet::Extend,
                       Lex::TokenKind::Adapt);
  auto modifiers = context.decl_state_stack().innermost().modifier_set;
  context.decl_state_stack().Pop(DeclState::Adapt);

  auto enclosing_class_decl =
      GetEnclosingClassOrDiagnose(context, node_id, Lex::TokenKind::Adapt);
  if (!enclosing_class_decl) {
    return true;
  }

  auto& class_info = context.classes().Get(enclosing_class_decl->class_id);
  if (class_info.adapt_id.is_valid()) {
    DiagnoseClassSpecificDeclRepeated(context, node_id, class_info.adapt_id,
                                      Lex::TokenKind::Adapt);
    return true;
  }

  auto adapted_type_id = ExprAsType(context, node_id, adapted_type_expr_id);
  adapted_type_id = context.AsCompleteType(adapted_type_id, [&] {
    CARBON_DIAGNOSTIC(IncompleteTypeInAdaptDecl, Error,
                      "Adapted type `{0}` is an incomplete type.",
                      SemIR::TypeId);
    return context.emitter().Build(node_id, IncompleteTypeInAdaptDecl,
                                   adapted_type_id);
  });

  // Build a SemIR representation for the declaration.
  class_info.adapt_id =
      context.AddInst({node_id, SemIR::AdaptDecl{adapted_type_id}});

  // Extend the class scope with the adapted type's scope if requested.
  if (!!(modifiers & KeywordModifierSet::Extend)) {
    auto extended_scope_id = SemIR::NameScopeId::Invalid;
    if (adapted_type_id == SemIR::TypeId::Error) {
      // Recover by not extending any scope. We instead set has_error to true
      // below.
    } else if (auto* adapted_class_info =
                   TryGetAsClass(context, adapted_type_id)) {
      extended_scope_id = adapted_class_info->scope_id;
      CARBON_CHECK(adapted_class_info->scope_id.is_valid())
          << "Complete class should have a scope";
    } else {
      // TODO: Accept any type that has a scope.
      context.TODO(node_id, "extending non-class type");
    }

    auto& class_scope = context.name_scopes().Get(class_info.scope_id);
    if (extended_scope_id.is_valid()) {
      class_scope.extended_scopes.push_back(extended_scope_id);
    } else {
      class_scope.has_error = true;
    }
  }
  return true;
}

auto HandleBaseIntroducer(Context& context, Parse::BaseIntroducerId /*node_id*/)
    -> bool {
  context.decl_state_stack().Push(DeclState::Base);
  return true;
}

auto HandleBaseColon(Context& /*context*/, Parse::BaseColonId /*node_id*/)
    -> bool {
  return true;
}

namespace {
// Information gathered about a base type specified in a `base` declaration.
struct BaseInfo {
  // A `BaseInfo` representing an erroneous base.
  static const BaseInfo Error;

  SemIR::TypeId type_id;
  SemIR::NameScopeId scope_id;
};
constexpr BaseInfo BaseInfo::Error = {.type_id = SemIR::TypeId::Error,
                                      .scope_id = SemIR::NameScopeId::Invalid};
}  // namespace

// Diagnoses an attempt to derive from a final type.
static auto DiagnoseBaseIsFinal(Context& context, Parse::NodeId node_id,
                                SemIR::TypeId base_type_id) -> void {
  CARBON_DIAGNOSTIC(BaseIsFinal, Error,
                    "Deriving from final type `{0}`. Base type must be an "
                    "`abstract` or `base` class.",
                    SemIR::TypeId);
  context.emitter().Emit(node_id, BaseIsFinal, base_type_id);
}

// Checks that the specified base type is valid.
static auto CheckBaseType(Context& context, Parse::NodeId node_id,
                          SemIR::InstId base_expr_id) -> BaseInfo {
  auto base_type_id = ExprAsType(context, node_id, base_expr_id);
  base_type_id = context.AsCompleteType(base_type_id, [&] {
    CARBON_DIAGNOSTIC(IncompleteTypeInBaseDecl, Error,
                      "Base `{0}` is an incomplete type.", SemIR::TypeId);
    return context.emitter().Build(node_id, IncompleteTypeInBaseDecl,
                                   base_type_id);
  });

  if (base_type_id == SemIR::TypeId::Error) {
    return BaseInfo::Error;
  }

  auto* base_class_info = TryGetAsClass(context, base_type_id);

  // The base must not be a final class.
  if (!base_class_info) {
    // For now, we treat all types that aren't introduced by a `class`
    // declaration as being final classes.
    // TODO: Once we have a better idea of which types are considered to be
    // classes, produce a better diagnostic for deriving from a non-class type.
    DiagnoseBaseIsFinal(context, node_id, base_type_id);
    return BaseInfo::Error;
  }
  if (base_class_info->inheritance_kind == SemIR::Class::Final) {
    DiagnoseBaseIsFinal(context, node_id, base_type_id);
  }

  CARBON_CHECK(base_class_info->scope_id.is_valid())
      << "Complete class should have a scope";
  return {.type_id = base_type_id, .scope_id = base_class_info->scope_id};
}

auto HandleBaseDecl(Context& context, Parse::BaseDeclId node_id) -> bool {
  auto [base_type_node_id, base_type_expr_id] =
      context.node_stack().PopExprWithNodeId();

  // Process modifiers. `extend` is required, no others are allowed.
  LimitModifiersOnDecl(context, KeywordModifierSet::Extend,
                       Lex::TokenKind::Base);
  auto modifiers = context.decl_state_stack().innermost().modifier_set;
  if (!(modifiers & KeywordModifierSet::Extend)) {
    CARBON_DIAGNOSTIC(BaseMissingExtend, Error,
                      "Missing `extend` before `base` declaration in class.");
    context.emitter().Emit(node_id, BaseMissingExtend);
  }
  context.decl_state_stack().Pop(DeclState::Base);

  auto enclosing_class_decl =
      GetEnclosingClassOrDiagnose(context, node_id, Lex::TokenKind::Base);
  if (!enclosing_class_decl) {
    return true;
  }

  auto& class_info = context.classes().Get(enclosing_class_decl->class_id);
  if (class_info.base_id.is_valid()) {
    DiagnoseClassSpecificDeclRepeated(context, node_id, class_info.base_id,
                                      Lex::TokenKind::Base);
    return true;
  }

  auto base_info = CheckBaseType(context, base_type_node_id, base_type_expr_id);

  // The `base` value in the class scope has an unbound element type. Instance
  // binding will be performed when it's found by name lookup into an instance.
  auto field_type_id =
      context.GetUnboundElementType(class_info.self_type_id, base_info.type_id);
  class_info.base_id = context.AddInst(
      {node_id,
       SemIR::BaseDecl{field_type_id, base_info.type_id,
                       SemIR::ElementIndex(context.args_type_info_stack()
                                               .PeekCurrentBlockContents()
                                               .size())}});

  // Add a corresponding field to the object representation of the class.
  // TODO: Consider whether we want to use `partial T` here.
  // TODO: Should we diagnose if there are already any fields?
  context.args_type_info_stack().AddInstId(context.AddInstInNoBlock(
      {node_id,
       SemIR::StructTypeField{SemIR::NameId::Base, base_info.type_id}}));

  // Bind the name `base` in the class to the base field.
  context.decl_name_stack().AddNameToLookup(
      context.decl_name_stack().MakeUnqualifiedName(node_id,
                                                    SemIR::NameId::Base),
      class_info.base_id);

  // Extend the class scope with the base class.
  if (!!(modifiers & KeywordModifierSet::Extend)) {
    auto& class_scope = context.name_scopes().Get(class_info.scope_id);
    if (base_info.scope_id.is_valid()) {
      class_scope.extended_scopes.push_back(base_info.scope_id);
    } else {
      class_scope.has_error = true;
    }
  }
  return true;
}

auto HandleClassDefinition(Context& context,
                           Parse::ClassDefinitionId /*node_id*/) -> bool {
  auto fields_id = context.args_type_info_stack().Pop();
  auto class_id =
      context.node_stack().Pop<Parse::NodeKind::ClassDefinitionStart>();
  context.inst_block_stack().Pop();

  // The class type is now fully defined. Compute its object representation.
  auto& class_info = context.classes().Get(class_id);
  if (class_info.adapt_id.is_valid()) {
    class_info.object_repr_id = SemIR::TypeId::Error;
    if (class_info.base_id.is_valid()) {
      CARBON_DIAGNOSTIC(AdaptWithBase, Error,
                        "Adapter cannot have a base class.");
      CARBON_DIAGNOSTIC(AdaptBaseHere, Note, "`base` declaration is here.");
      context.emitter()
          .Build(class_info.adapt_id, AdaptWithBase)
          .Note(class_info.base_id, AdaptBaseHere)
          .Emit();
    } else if (!context.inst_blocks().Get(fields_id).empty()) {
      auto first_field_id = context.inst_blocks().Get(fields_id).front();
      CARBON_DIAGNOSTIC(AdaptWithFields, Error, "Adapter cannot have fields.");
      CARBON_DIAGNOSTIC(AdaptFieldHere, Note,
                        "First field declaration is here.");
      context.emitter()
          .Build(class_info.adapt_id, AdaptWithFields)
          .Note(first_field_id, AdaptFieldHere)
          .Emit();
    } else {
      // The object representation of the adapter is the object representation
      // of the adapted type.
      auto adapted_type_id = context.insts()
                                 .GetAs<SemIR::AdaptDecl>(class_info.adapt_id)
                                 .adapted_type_id;
      // If we adapt an adapter, directly track the non-adapter type we're
      // adapting so that we have constant-time access to it.
      if (auto adapted_class =
              context.types().TryGetAs<SemIR::ClassType>(adapted_type_id)) {
        auto& adapted_class_info =
            context.classes().Get(adapted_class->class_id);
        if (adapted_class_info.adapt_id.is_valid()) {
          adapted_type_id = adapted_class_info.object_repr_id;
        }
      }
      class_info.object_repr_id = adapted_type_id;
    }
  } else {
    class_info.object_repr_id = context.GetStructType(fields_id);
  }

  // The decl_name_stack and scopes are popped by `ProcessNodeIds`.
  return true;
}

}  // namespace Carbon::Check
