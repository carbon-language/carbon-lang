// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/decl_name_stack.h"
#include "toolchain/check/decl_state.h"
#include "toolchain/check/function.h"
#include "toolchain/check/interface.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/parse/tree_node_location_translator.h"
#include "toolchain/sem_ir/entry_point.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto HandleFunctionIntroducer(Context& context,
                              Parse::FunctionIntroducerId node_id) -> bool {
  // Create an instruction block to hold the instructions created as part of the
  // function signature, such as parameter and return types.
  context.inst_block_stack().Push();
  // Push the bracketing node.
  context.node_stack().Push(node_id);
  // Optional modifiers and the name follow.
  context.decl_state_stack().Push(DeclState::Fn);
  context.decl_name_stack().PushScopeAndStartName();
  return true;
}

auto HandleReturnType(Context& context, Parse::ReturnTypeId node_id) -> bool {
  // Propagate the type expression.
  auto [type_node_id, type_inst_id] = context.node_stack().PopExprWithNodeId();
  auto type_id = ExprAsType(context, type_node_id, type_inst_id);
  // TODO: Use a dedicated instruction rather than VarStorage here.
  context.AddInstAndPush(
      {node_id, SemIR::VarStorage{type_id, SemIR::NameId::ReturnSlot}});
  return true;
}

static auto DiagnoseModifiers(Context& context, bool is_definition,
                              SemIR::NameScopeId target_scope_id)
    -> KeywordModifierSet {
  const Lex::TokenKind decl_kind = Lex::TokenKind::Fn;
  CheckAccessModifiersOnDecl(context, decl_kind, target_scope_id);
  if (is_definition) {
    ForbidExternModifierOnDefinition(context, decl_kind);
  }
  if (target_scope_id.is_valid()) {
    auto target_id = context.name_scopes().Get(target_scope_id).inst_id;
    if (target_id.is_valid() &&
        !context.insts().Is<SemIR::Namespace>(target_id)) {
      ForbidModifiersOnDecl(context, KeywordModifierSet::Extern, decl_kind,
                            " that is a member");
    }
  }
  LimitModifiersOnDecl(context,
                       KeywordModifierSet::Access | KeywordModifierSet::Extern |
                           KeywordModifierSet::Method |
                           KeywordModifierSet::Interface,
                       decl_kind);
  CheckMethodModifiersOnFunction(context, target_scope_id);
  RequireDefaultFinalOnlyInInterfaces(context, decl_kind, target_scope_id);

  return context.decl_state_stack().innermost().modifier_set;
}

// Build a FunctionDecl describing the signature of a function. This
// handles the common logic shared by function declaration syntax and function
// definition syntax.
static auto BuildFunctionDecl(Context& context,
                              Parse::AnyFunctionDeclId node_id,
                              bool is_definition)
    -> std::pair<SemIR::FunctionId, SemIR::InstId> {
  auto decl_block_id = context.inst_block_stack().Pop();

  auto return_type_id = SemIR::TypeId::Invalid;
  auto return_slot_id = SemIR::InstId::Invalid;
  if (auto [return_node, return_storage_id] =
          context.node_stack().PopWithNodeIdIf<Parse::NodeKind::ReturnType>();
      return_storage_id) {
    return_type_id = context.insts().Get(*return_storage_id).type_id();

    return_type_id = context.AsCompleteType(return_type_id, [&] {
      CARBON_DIAGNOSTIC(IncompleteTypeInFunctionReturnType, Error,
                        "Function returns incomplete type `{0}`.",
                        SemIR::TypeId);
      return context.emitter().Build(
          return_node, IncompleteTypeInFunctionReturnType, return_type_id);
    });

    if (!SemIR::GetInitRepr(context.sem_ir(), return_type_id)
             .has_return_slot()) {
      // The function only has a return slot if it uses in-place initialization.
    } else {
      return_slot_id = *return_storage_id;
    }
  }

  SemIR::InstBlockId param_refs_id =
      context.node_stack().Pop<Parse::NodeKind::TuplePattern>();
  SemIR::InstBlockId implicit_param_refs_id =
      context.node_stack().PopIf<Parse::NodeKind::ImplicitParamList>().value_or(
          SemIR::InstBlockId::Empty);
  auto name_context = context.decl_name_stack().FinishName();
  context.node_stack()
      .PopAndDiscardSoloNodeId<Parse::NodeKind::FunctionIntroducer>();

  // Process modifiers.
  auto modifiers =
      DiagnoseModifiers(context, is_definition, name_context.target_scope_id);
  if (!!(modifiers & KeywordModifierSet::Access)) {
    context.TODO(context.decl_state_stack().innermost().modifier_node_id(
                     ModifierOrder::Access),
                 "access modifier");
  }
  bool is_extern = !!(modifiers & KeywordModifierSet::Extern);
  if (!!(modifiers & KeywordModifierSet::Method)) {
    context.TODO(context.decl_state_stack().innermost().modifier_node_id(
                     ModifierOrder::Decl),
                 "method modifier");
  }
  if (!!(modifiers & KeywordModifierSet::Interface)) {
    // TODO: Once we are saving the modifiers for a function, add check that
    // the function may only be defined if it is marked `default` or `final`.
    context.TODO(context.decl_state_stack().innermost().modifier_node_id(
                     ModifierOrder::Decl),
                 "interface modifier");
  }
  context.decl_state_stack().Pop(DeclState::Fn);

  // Add the function declaration.
  auto function_decl = SemIR::FunctionDecl{
      context.GetBuiltinType(SemIR::BuiltinKind::FunctionType),
      SemIR::FunctionId::Invalid, decl_block_id};
  auto function_info = SemIR::Function{
      .name_id = name_context.name_id_for_new_inst(),
      .enclosing_scope_id = name_context.enclosing_scope_id_for_new_inst(),
      .decl_id = context.AddPlaceholderInst({node_id, function_decl}),
      .implicit_param_refs_id = implicit_param_refs_id,
      .param_refs_id = param_refs_id,
      .return_type_id = return_type_id,
      .return_slot_id = return_slot_id,
      .is_extern = is_extern};
  if (is_definition) {
    function_info.definition_id = function_info.decl_id;
  }

  // At interface scope, a function declaration introduces an associated
  // function.
  auto lookup_result_id = function_info.decl_id;
  if (name_context.enclosing_scope_id_for_new_inst().is_valid() &&
      !name_context.has_qualifiers) {
    auto scope_inst_id = context.name_scopes().GetInstIdIfValid(
        name_context.enclosing_scope_id_for_new_inst());
    if (auto interface_scope =
            context.insts().TryGetAsIfValid<SemIR::InterfaceDecl>(
                scope_inst_id)) {
      lookup_result_id = BuildAssociatedEntity(
          context, interface_scope->interface_id, function_info.decl_id);
    }
  }

  // Check whether this is a redeclaration.
  auto existing_id =
      context.decl_name_stack().LookupOrAddName(name_context, lookup_result_id);
  if (existing_id.is_valid()) {
    if (auto existing_function_decl =
            context.insts().Get(existing_id).TryAs<SemIR::FunctionDecl>()) {
      if (MergeFunctionRedecl(context, node_id, function_info,
                              existing_function_decl->function_id,
                              is_definition)) {
        // When merging, use the existing function rather than adding a new one.
        function_decl.function_id = existing_function_decl->function_id;
      }
    } else {
      // This is a redeclaration of something other than a function. This
      // includes the case where an associated function redeclares another
      // associated function.
      context.DiagnoseDuplicateName(function_info.decl_id, existing_id);
    }
  }

  // Create a new function if this isn't a valid redeclaration.
  if (!function_decl.function_id.is_valid()) {
    function_decl.function_id = context.functions().Add(function_info);
  }

  // Write the function ID into the FunctionDecl.
  context.ReplaceInstBeforeConstantUse(function_info.decl_id,
                                       {node_id, function_decl});

  if (SemIR::IsEntryPoint(context.sem_ir(), function_decl.function_id)) {
    // TODO: Update this once valid signatures for the entry point are decided.
    if (!context.inst_blocks().Get(implicit_param_refs_id).empty() ||
        !context.inst_blocks().Get(param_refs_id).empty() ||
        (return_slot_id.is_valid() &&
         return_type_id !=
             context.GetBuiltinType(SemIR::BuiltinKind::BoolType) &&
         return_type_id != context.GetTupleType({}))) {
      CARBON_DIAGNOSTIC(InvalidMainRunSignature, Error,
                        "Invalid signature for `Main.Run` function. Expected "
                        "`fn ()` or `fn () -> i32`.");
      context.emitter().Emit(node_id, InvalidMainRunSignature);
    }
  }

  return {function_decl.function_id, function_info.decl_id};
}

auto HandleFunctionDecl(Context& context, Parse::FunctionDeclId node_id)
    -> bool {
  BuildFunctionDecl(context, node_id, /*is_definition=*/false);
  context.decl_name_stack().PopScope();
  return true;
}

auto HandleFunctionDefinitionStart(Context& context,
                                   Parse::FunctionDefinitionStartId node_id)
    -> bool {
  // Process the declaration portion of the function.
  auto [function_id, decl_id] =
      BuildFunctionDecl(context, node_id, /*is_definition=*/true);
  auto& function = context.functions().Get(function_id);

  // Create the function scope and the entry block.
  context.return_scope_stack().push_back({.decl_id = decl_id});
  context.inst_block_stack().Push();
  context.scope_stack().Push(decl_id);
  context.AddCurrentCodeBlockToFunction();

  // Bring the implicit and explicit parameters into scope.
  for (auto param_id : llvm::concat<SemIR::InstId>(
           context.inst_blocks().Get(function.implicit_param_refs_id),
           context.inst_blocks().Get(function.param_refs_id))) {
    auto param = context.insts().Get(param_id);

    // Find the parameter in the pattern.
    // TODO: More general pattern handling?
    if (auto addr_pattern = param.TryAs<SemIR::AddrPattern>()) {
      param_id = addr_pattern->inner_id;
      param = context.insts().Get(param_id);
    }

    // The parameter types need to be complete.
    context.TryToCompleteType(param.type_id(), [&] {
      CARBON_DIAGNOSTIC(
          IncompleteTypeInFunctionParam, Error,
          "Parameter has incomplete type `{0}` in function definition.",
          SemIR::TypeId);
      return context.emitter().Build(param_id, IncompleteTypeInFunctionParam,
                                     param.type_id());
    });
  }

  context.node_stack().Push(node_id, function_id);
  return true;
}

auto HandleFunctionDefinition(Context& context,
                              Parse::FunctionDefinitionId node_id) -> bool {
  SemIR::FunctionId function_id =
      context.node_stack().Pop<Parse::NodeKind::FunctionDefinitionStart>();

  // If the `}` of the function is reachable, reject if we need a return value
  // and otherwise add an implicit `return;`.
  if (context.is_current_position_reachable()) {
    if (context.functions().Get(function_id).return_type_id.is_valid()) {
      CARBON_DIAGNOSTIC(
          MissingReturnStatement, Error,
          "Missing `return` at end of function with declared return type.");
      context.emitter().Emit(TokenOnly(node_id), MissingReturnStatement);
    } else {
      context.AddInst({node_id, SemIR::Return{}});
    }
  }

  context.scope_stack().Pop();
  context.inst_block_stack().Pop();
  context.return_scope_stack().pop_back();
  context.decl_name_stack().PopScope();
  return true;
}

auto HandleBuiltinFunctionDefinitionStart(
    Context& context, Parse::BuiltinFunctionDefinitionStartId node_id) -> bool {
  // Process the declaration portion of the function.
  auto [function_id, _] =
      BuildFunctionDecl(context, node_id, /*is_definition=*/true);
  context.node_stack().Push(node_id, function_id);
  return true;
}

auto HandleBuiltinName(Context& context, Parse::BuiltinNameId node_id) -> bool {
  context.node_stack().Push(node_id);
  return true;
}

// Looks up a builtin function kind given its name as a string.
// TODO: Move this out to another file.
static auto LookupBuiltinFunctionKind(Context& context,
                                      Parse::BuiltinNameId name_id)
    -> SemIR::BuiltinFunctionKind {
  auto builtin_name = context.string_literal_values().Get(
      context.tokens().GetStringLiteralValue(
          context.parse_tree().node_token(name_id)));
  auto kind = llvm::StringSwitch<SemIR::BuiltinFunctionKind>(builtin_name)
                  .Case("int.add", SemIR::BuiltinFunctionKind::IntAdd)
                  .Default(SemIR::BuiltinFunctionKind::None);
  if (kind == SemIR::BuiltinFunctionKind::None) {
    CARBON_DIAGNOSTIC(UnknownBuiltinFunctionName, Error,
                      "Unknown builtin function name \"{0}\".", std::string);
    context.emitter().Emit(name_id, UnknownBuiltinFunctionName,
                           builtin_name.str());
  }
  return kind;
}

auto HandleBuiltinFunctionDefinition(
    Context& context, Parse::BuiltinFunctionDefinitionId /*node_id*/) -> bool {
  auto name_id =
      context.node_stack().PopForSoloNodeId<Parse::NodeKind::BuiltinName>();
  auto function_id =
      context.node_stack()
          .Pop<Parse::NodeKind::BuiltinFunctionDefinitionStart>();

  auto& function = context.functions().Get(function_id);
  function.builtin_kind = LookupBuiltinFunctionKind(context, name_id);

  context.decl_name_stack().PopScope();
  return true;
}

}  // namespace Carbon::Check
