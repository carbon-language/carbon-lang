// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/decl_name_stack.h"
#include "toolchain/check/decl_state.h"
#include "toolchain/check/function.h"
#include "toolchain/check/interface.h"
#include "toolchain/check/merge.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/parse/tree_node_diagnostic_converter.h"
#include "toolchain/sem_ir/builtin_function_kind.h"
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
  LimitModifiersOnDecl(context,
                       KeywordModifierSet::Access | KeywordModifierSet::Extern |
                           KeywordModifierSet::Method |
                           KeywordModifierSet::Interface,
                       decl_kind);
  RestrictExternModifierOnDecl(context, decl_kind, target_scope_id,
                               is_definition);
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
  auto return_storage_id = SemIR::InstId::Invalid;
  auto return_slot = SemIR::Function::ReturnSlot::NotComputed;
  if (auto [return_node, maybe_return_storage_id] =
          context.node_stack().PopWithNodeIdIf<Parse::NodeKind::ReturnType>();
      maybe_return_storage_id) {
    return_type_id = context.insts().Get(*maybe_return_storage_id).type_id();
    return_storage_id = *maybe_return_storage_id;
  } else {
    // If there's no return type, there's no return slot.
    return_slot = SemIR::Function::ReturnSlot::Absent;
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
      .return_storage_id = return_storage_id,
      .is_extern = is_extern,
      .return_slot = return_slot};
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
  auto prev_id =
      context.decl_name_stack().LookupOrAddName(name_context, lookup_result_id);
  if (prev_id.is_valid()) {
    auto prev_inst_for_merge =
        ResolvePrevInstForMerge(context, node_id, prev_id);

    if (auto prev_function_decl =
            prev_inst_for_merge.inst.TryAs<SemIR::FunctionDecl>()) {
      if (MergeFunctionRedecl(context, node_id, function_info,
                              /*new_is_import=*/false, is_definition,
                              prev_function_decl->function_id,
                              prev_inst_for_merge.import_ir_inst_id)) {
        // When merging, use the existing function rather than adding a new one.
        function_decl.function_id = prev_function_decl->function_id;
      }
    } else {
      // This is a redeclaration of something other than a function. This
      // includes the case where an associated function redeclares another
      // associated function.
      context.DiagnoseDuplicateName(function_info.decl_id, prev_id);
    }
  }

  // Create a new function if this isn't a valid redeclaration.
  if (!function_decl.function_id.is_valid()) {
    function_decl.function_id = context.functions().Add(function_info);
  }

  // Write the function ID into the FunctionDecl.
  context.ReplaceInstBeforeConstantUse(function_info.decl_id, function_decl);

  if (SemIR::IsEntryPoint(context.sem_ir(), function_decl.function_id)) {
    // TODO: Update this once valid signatures for the entry point are decided.
    if (!context.inst_blocks().Get(implicit_param_refs_id).empty() ||
        !context.inst_blocks().Get(param_refs_id).empty() ||
        (return_type_id.is_valid() &&
         return_type_id !=
             context.GetBuiltinType(SemIR::BuiltinKind::IntType) &&
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

// Processes a function definition after a signature for which we have already
// built a function ID. This logic is shared between processing regular function
// definitions and delayed parsing of inline method definitions.
static auto HandleFunctionDefinitionAfterSignature(
    Context& context, Parse::FunctionDefinitionStartId node_id,
    SemIR::FunctionId function_id, SemIR::InstId decl_id) -> void {
  auto& function = context.functions().Get(function_id);

  // Create the function scope and the entry block.
  context.return_scope_stack().push_back({.decl_id = decl_id});
  context.inst_block_stack().Push();
  context.scope_stack().Push(decl_id);
  context.AddCurrentCodeBlockToFunction();

  // Check the return type is complete.
  CheckFunctionReturnType(context, function.return_storage_id, function);

  // Check the parameter types are complete.
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
}

auto HandleFunctionDefinitionSuspend(Context& context,
                                     Parse::FunctionDefinitionStartId node_id)
    -> SuspendedFunction {
  // Process the declaration portion of the function.
  auto [function_id, decl_id] =
      BuildFunctionDecl(context, node_id, /*is_definition=*/true);
  return {function_id, decl_id, context.decl_name_stack().Suspend()};
}

auto HandleFunctionDefinitionResume(Context& context,
                                    Parse::FunctionDefinitionStartId node_id,
                                    SuspendedFunction sus_fn) -> void {
  context.decl_name_stack().Restore(sus_fn.saved_name_state);
  HandleFunctionDefinitionAfterSignature(context, node_id, sus_fn.function_id,
                                         sus_fn.decl_id);
}

auto HandleFunctionDefinitionStart(Context& context,
                                   Parse::FunctionDefinitionStartId node_id)
    -> bool {
  // Process the declaration portion of the function.
  auto [function_id, decl_id] =
      BuildFunctionDecl(context, node_id, /*is_definition=*/true);
  HandleFunctionDefinitionAfterSignature(context, node_id, function_id,
                                         decl_id);
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
  auto kind = SemIR::BuiltinFunctionKind::ForBuiltinName(builtin_name);
  if (kind == SemIR::BuiltinFunctionKind::None) {
    CARBON_DIAGNOSTIC(UnknownBuiltinFunctionName, Error,
                      "Unknown builtin function name \"{0}\".", std::string);
    context.emitter().Emit(name_id, UnknownBuiltinFunctionName,
                           builtin_name.str());
  }
  return kind;
}

// Returns whether `function` is a valid declaration of the builtin
// `builtin_kind`.
static auto IsValidBuiltinDeclaration(Context& context,
                                      const SemIR::Function& function,
                                      SemIR::BuiltinFunctionKind builtin_kind)
    -> bool {
  // Form the list of parameter types for the declaration.
  llvm::SmallVector<SemIR::TypeId> param_type_ids;
  auto implicit_param_refs =
      context.inst_blocks().Get(function.implicit_param_refs_id);
  auto param_refs = context.inst_blocks().Get(function.param_refs_id);
  param_type_ids.reserve(implicit_param_refs.size() + param_refs.size());
  for (auto param_id :
       llvm::concat<SemIR::InstId>(implicit_param_refs, param_refs)) {
    // TODO: We also need to track whether the parameter is declared with
    // `var`.
    param_type_ids.push_back(context.insts().Get(param_id).type_id());
  }

  // Get the return type. This is `()` if none was specified.
  auto return_type_id = function.return_type_id;
  if (!return_type_id.is_valid()) {
    return_type_id = context.GetTupleType({});
  }

  return builtin_kind.IsValidType(context.sem_ir(), param_type_ids,
                                  return_type_id);
}

auto HandleBuiltinFunctionDefinition(
    Context& context, Parse::BuiltinFunctionDefinitionId /*node_id*/) -> bool {
  auto name_id =
      context.node_stack().PopForSoloNodeId<Parse::NodeKind::BuiltinName>();
  auto [fn_node_id, function_id] =
      context.node_stack()
          .PopWithNodeId<Parse::NodeKind::BuiltinFunctionDefinitionStart>();

  auto builtin_kind = LookupBuiltinFunctionKind(context, name_id);
  if (builtin_kind != SemIR::BuiltinFunctionKind::None) {
    auto& function = context.functions().Get(function_id);
    if (IsValidBuiltinDeclaration(context, function, builtin_kind)) {
      function.builtin_kind = builtin_kind;
    } else {
      CARBON_DIAGNOSTIC(InvalidBuiltinSignature, Error,
                        "Invalid signature for builtin function \"{0}\".",
                        std::string);
      context.emitter().Emit(fn_node_id, InvalidBuiltinSignature,
                             builtin_kind.name().str());
    }
  }
  context.decl_name_stack().PopScope();
  return true;
}

}  // namespace Carbon::Check
