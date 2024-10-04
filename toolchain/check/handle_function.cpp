// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/decl_introducer_state.h"
#include "toolchain/check/decl_name_stack.h"
#include "toolchain/check/function.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/interface.h"
#include "toolchain/check/merge.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/check/name_component.h"
#include "toolchain/sem_ir/builtin_function_kind.h"
#include "toolchain/sem_ir/entry_point.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::FunctionIntroducerId node_id)
    -> bool {
  // Create an instruction block to hold the instructions created as part of the
  // function signature, such as parameter and return types.
  context.inst_block_stack().Push();
  // Push the bracketing node.
  context.node_stack().Push(node_id);
  // Optional modifiers and the name follow.
  context.decl_introducer_state_stack().Push<Lex::TokenKind::Fn>();
  context.decl_name_stack().PushScopeAndStartName();
  // The function is potentially generic.
  StartGenericDecl(context);
  // Start a new pattern block for the signature.
  context.pattern_block_stack().Push();
  return true;
}

auto HandleParseNode(Context& context, Parse::ReturnTypeId node_id) -> bool {
  // Propagate the type expression.
  auto [type_node_id, type_inst_id] = context.node_stack().PopExprWithNodeId();
  auto type_id = ExprAsType(context, type_node_id, type_inst_id).type_id;
  // TODO: Use a dedicated instruction rather than VarStorage here.
  context.AddInstAndPush<SemIR::VarStorage>(
      node_id, {.type_id = type_id, .name_id = SemIR::NameId::ReturnSlot});
  return true;
}

static auto DiagnoseModifiers(Context& context, DeclIntroducerState& introducer,
                              bool is_definition,
                              SemIR::InstId parent_scope_inst_id,
                              std::optional<SemIR::Inst> parent_scope_inst)
    -> void {
  CheckAccessModifiersOnDecl(context, introducer, parent_scope_inst);
  LimitModifiersOnDecl(context, introducer,
                       KeywordModifierSet::Access | KeywordModifierSet::Extern |
                           KeywordModifierSet::Method |
                           KeywordModifierSet::Interface);
  RestrictExternModifierOnDecl(context, introducer, parent_scope_inst,
                               is_definition);
  CheckMethodModifiersOnFunction(context, introducer, parent_scope_inst_id,
                                 parent_scope_inst);
  RequireDefaultFinalOnlyInInterfaces(context, introducer, parent_scope_inst);
}

// Checks that the parameter lists specified in a function declaration are
// valid for a function declaration, and numbers the parameters.
static auto CheckFunctionSignature(Context& context,
                                   const NameComponent& name_and_params)
    -> void {
  RequireGenericOrSelfImplicitFunctionParams(
      context, name_and_params.implicit_params_id);
  SemIR::RuntimeParamIndex next_index(0);
  for (auto param_id : llvm::concat<const SemIR::InstId>(
           context.inst_blocks().GetOrEmpty(name_and_params.implicit_params_id),
           context.inst_blocks().GetOrEmpty(name_and_params.params_id))) {
    // Find the parameter in the pattern.
    auto param_info =
        SemIR::Function::GetParamFromParamRefId(context.sem_ir(), param_id);

    // If this is a runtime parameter, number it.
    if (param_info.bind_name &&
        param_info.bind_name->kind == SemIR::BindName::Kind) {
      param_info.inst.runtime_index = next_index;
      context.ReplaceInstBeforeConstantUse(param_info.inst_id, param_info.inst);
      ++next_index.index;
    }
  }

  // TODO: Also assign a parameter index to the return storage, if present.
}

// Tries to merge new_function into prev_function_id. Since new_function won't
// have a definition even if one is upcoming, set is_definition to indicate the
// planned result.
//
// If merging is successful, returns true and may update the previous function.
// Otherwise, returns false. Prints a diagnostic when appropriate.
static auto MergeFunctionRedecl(Context& context, SemIRLoc new_loc,
                                SemIR::Function& new_function,
                                bool new_is_import, bool new_is_definition,
                                SemIR::FunctionId prev_function_id,
                                SemIR::ImportIRId prev_import_ir_id) -> bool {
  auto& prev_function = context.functions().Get(prev_function_id);

  if (!CheckFunctionTypeMatches(context, new_function, prev_function)) {
    return false;
  }

  CheckIsAllowedRedecl(context, Lex::TokenKind::Fn, prev_function.name_id,
                       RedeclInfo(new_function, new_loc, new_is_definition),
                       RedeclInfo(prev_function, prev_function.latest_decl_id(),
                                  prev_function.definition_id.is_valid()),
                       prev_import_ir_id);

  if (!prev_function.first_owning_decl_id.is_valid()) {
    prev_function.first_owning_decl_id = new_function.first_owning_decl_id;
  }
  if (new_is_definition) {
    // Track the signature from the definition, so that IDs in the body
    // match IDs in the signature.
    prev_function.MergeDefinition(new_function);
    prev_function.return_storage_id = new_function.return_storage_id;
  }
  if ((prev_import_ir_id.is_valid() && !new_is_import)) {
    ReplacePrevInstForMerge(context, new_function.parent_scope_id,
                            prev_function.name_id,
                            new_function.first_owning_decl_id);
  }
  return true;
}

// Check whether this is a redeclaration, merging if needed.
static auto TryMergeRedecl(Context& context, Parse::AnyFunctionDeclId node_id,
                           SemIR::InstId prev_id,
                           SemIR::FunctionDecl& function_decl,
                           SemIR::Function& function_info, bool is_definition)
    -> void {
  if (!prev_id.is_valid()) {
    return;
  }

  auto prev_function_id = SemIR::FunctionId::Invalid;
  auto prev_import_ir_id = SemIR::ImportIRId::Invalid;
  CARBON_KIND_SWITCH(context.insts().Get(prev_id)) {
    case CARBON_KIND(SemIR::FunctionDecl function_decl): {
      prev_function_id = function_decl.function_id;
      break;
    }
    case SemIR::ImportRefLoaded::Kind: {
      auto import_ir_inst =
          GetCanonicalImportIRInst(context, &context.sem_ir(), prev_id);

      // Verify the decl so that things like aliases are name conflicts.
      const auto* import_ir =
          context.import_irs().Get(import_ir_inst.ir_id).sem_ir;
      if (!import_ir->insts().Is<SemIR::FunctionDecl>(import_ir_inst.inst_id)) {
        break;
      }

      // Use the type to get the ID.
      if (auto struct_value = context.insts().TryGetAs<SemIR::StructValue>(
              context.constant_values().GetConstantInstId(prev_id))) {
        if (auto function_type = context.types().TryGetAs<SemIR::FunctionType>(
                struct_value->type_id)) {
          prev_function_id = function_type->function_id;
          prev_import_ir_id = import_ir_inst.ir_id;
        }
      }
      break;
    }
    default:
      break;
  }

  if (!prev_function_id.is_valid()) {
    context.DiagnoseDuplicateName(function_info.latest_decl_id(), prev_id);
    return;
  }

  if (MergeFunctionRedecl(context, node_id, function_info,
                          /*new_is_import=*/false, is_definition,
                          prev_function_id, prev_import_ir_id)) {
    // When merging, use the existing function rather than adding a new one.
    function_decl.function_id = prev_function_id;
  }
}

// Build a FunctionDecl describing the signature of a function. This
// handles the common logic shared by function declaration syntax and function
// definition syntax.
static auto BuildFunctionDecl(Context& context,
                              Parse::AnyFunctionDeclId node_id,
                              bool is_definition)
    -> std::pair<SemIR::FunctionId, SemIR::InstId> {
  auto return_storage_id = SemIR::InstId::Invalid;
  if (auto [return_node, maybe_return_storage_id] =
          context.node_stack().PopWithNodeIdIf<Parse::NodeKind::ReturnType>();
      maybe_return_storage_id) {
    return_storage_id = *maybe_return_storage_id;
  }

  auto name = PopNameComponent(context);
  if (!name.params_id.is_valid()) {
    context.TODO(node_id, "function with positional parameters");
    name.params_id = SemIR::InstBlockId::Empty;
  }

  // Check that the function signature is valid and number the parameters.
  CheckFunctionSignature(context, name);

  auto name_context = context.decl_name_stack().FinishName(name);
  context.node_stack()
      .PopAndDiscardSoloNodeId<Parse::NodeKind::FunctionIntroducer>();

  // Process modifiers.
  auto [parent_scope_inst_id, parent_scope_inst] =
      context.name_scopes().GetInstIfValid(name_context.parent_scope_id);
  auto introducer =
      context.decl_introducer_state_stack().Pop<Lex::TokenKind::Fn>();
  DiagnoseModifiers(context, introducer, is_definition, parent_scope_inst_id,
                    parent_scope_inst);
  bool is_extern = introducer.modifier_set.HasAnyOf(KeywordModifierSet::Extern);
  auto virtual_modifier =
      introducer.modifier_set.ToEnum<SemIR::Function::VirtualModifier>()
          .Case(KeywordModifierSet::Virtual,
                SemIR::Function::VirtualModifier::Virtual)
          .Case(KeywordModifierSet::Abstract,
                SemIR::Function::VirtualModifier::Abstract)
          .Case(KeywordModifierSet::Impl,
                SemIR::Function::VirtualModifier::Impl)
          .Default(SemIR::Function::VirtualModifier::None);
  if (introducer.modifier_set.HasAnyOf(KeywordModifierSet::Interface)) {
    // TODO: Once we are saving the modifiers for a function, add check that
    // the function may only be defined if it is marked `default` or `final`.
    context.TODO(introducer.modifier_node_id(ModifierOrder::Decl),
                 "interface modifier");
  }

  // Add the function declaration.
  auto decl_block_id = context.inst_block_stack().Pop();
  auto function_decl = SemIR::FunctionDecl{
      SemIR::TypeId::Invalid, SemIR::FunctionId::Invalid, decl_block_id};
  auto decl_id =
      context.AddPlaceholderInst(SemIR::LocIdAndInst(node_id, function_decl));

  // Build the function entity. This will be merged into an existing function if
  // there is one, or otherwise added to the function store.
  auto function_info =
      SemIR::Function{{name_context.MakeEntityWithParamsBase(
                          name, decl_id, is_extern, introducer.extern_library)},
                      {.return_storage_id = return_storage_id,
                       .virtual_modifier = virtual_modifier}};
  if (is_definition) {
    function_info.definition_id = decl_id;
  }

  TryMergeRedecl(context, node_id, name_context.prev_inst_id(), function_decl,
                 function_info, is_definition);

  // Create a new function if this isn't a valid redeclaration.
  if (!function_decl.function_id.is_valid()) {
    if (function_info.is_extern && context.IsImplFile()) {
      DiagnoseExternRequiresDeclInApiFile(context, node_id);
    }
    function_info.generic_id = FinishGenericDecl(context, decl_id);
    function_decl.function_id = context.functions().Add(function_info);
  } else {
    FinishGenericRedecl(context, decl_id, function_info.generic_id);
    // TODO: Validate that the redeclaration doesn't set an access modifier.
  }
  function_decl.type_id = context.GetFunctionType(
      function_decl.function_id, context.scope_stack().PeekSpecificId());

  // Write the function ID into the FunctionDecl.
  context.ReplaceInstBeforeConstantUse(decl_id, function_decl);

  // Diagnose 'definition of `abstract` function' using the canonical Function's
  // modifiers.
  if (is_definition &&
      context.functions().Get(function_decl.function_id).virtual_modifier ==
          SemIR::Function::VirtualModifier::Abstract) {
    CARBON_DIAGNOSTIC(DefinedAbstractFunction, Error,
                      "definition of `abstract` function");
    context.emitter().Emit(TokenOnly(node_id), DefinedAbstractFunction);
  }

  // Check if we need to add this to name lookup, now that the function decl is
  // done.
  if (!name_context.prev_inst_id().is_valid()) {
    // At interface scope, a function declaration introduces an associated
    // function.
    auto lookup_result_id = decl_id;
    if (parent_scope_inst && !name_context.has_qualifiers) {
      if (auto interface_scope =
              parent_scope_inst->TryAs<SemIR::InterfaceDecl>()) {
        lookup_result_id = BuildAssociatedEntity(
            context, interface_scope->interface_id, decl_id);
      }
    }

    context.decl_name_stack().AddName(name_context, lookup_result_id,
                                      introducer.modifier_set.GetAccessKind());
  }

  if (SemIR::IsEntryPoint(context.sem_ir(), function_decl.function_id)) {
    auto return_type_id = function_info.GetDeclaredReturnType(context.sem_ir());
    // TODO: Update this once valid signatures for the entry point are decided.
    if (function_info.implicit_param_refs_id.is_valid() ||
        !function_info.param_refs_id.is_valid() ||
        !context.inst_blocks().Get(function_info.param_refs_id).empty() ||
        (return_type_id.is_valid() &&
         return_type_id !=
             context.GetBuiltinType(SemIR::BuiltinInstKind::IntType) &&
         return_type_id != context.GetTupleType({}))) {
      CARBON_DIAGNOSTIC(InvalidMainRunSignature, Error,
                        "invalid signature for `Main.Run` function; expected "
                        "`fn ()` or `fn () -> i32`");
      context.emitter().Emit(node_id, InvalidMainRunSignature);
    }
  }

  if (!is_definition && context.IsImplFile() && !is_extern) {
    context.definitions_required().push_back(decl_id);
  }

  return {function_decl.function_id, decl_id};
}

auto HandleParseNode(Context& context, Parse::FunctionDeclId node_id) -> bool {
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
  StartGenericDefinition(context);
  context.AddCurrentCodeBlockToFunction();

  // Check the return type is complete.
  CheckFunctionReturnType(context, function.return_storage_id, function,
                          SemIR::SpecificId::Invalid);

  // Check the parameter types are complete.
  for (auto param_ref_id : llvm::concat<const SemIR::InstId>(
           context.inst_blocks().GetOrEmpty(function.implicit_param_refs_id),
           context.inst_blocks().GetOrEmpty(function.param_refs_id))) {
    auto param_info =
        SemIR::Function::GetParamFromParamRefId(context.sem_ir(), param_ref_id);

    // The parameter types need to be complete.
    context.TryToCompleteType(param_info.inst.type_id, [&] {
      CARBON_DIAGNOSTIC(
          IncompleteTypeInFunctionParam, Error,
          "parameter has incomplete type `{0}` in function definition",
          SemIR::TypeId);
      return context.emitter().Build(param_info.inst_id,
                                     IncompleteTypeInFunctionParam,
                                     param_info.inst.type_id);
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
  return {.function_id = function_id,
          .decl_id = decl_id,
          .saved_name_state = context.decl_name_stack().Suspend()};
}

auto HandleFunctionDefinitionResume(Context& context,
                                    Parse::FunctionDefinitionStartId node_id,
                                    SuspendedFunction suspended_fn) -> void {
  context.decl_name_stack().Restore(suspended_fn.saved_name_state);
  HandleFunctionDefinitionAfterSignature(
      context, node_id, suspended_fn.function_id, suspended_fn.decl_id);
}

auto HandleParseNode(Context& context, Parse::FunctionDefinitionStartId node_id)
    -> bool {
  // Process the declaration portion of the function.
  auto [function_id, decl_id] =
      BuildFunctionDecl(context, node_id, /*is_definition=*/true);
  HandleFunctionDefinitionAfterSignature(context, node_id, function_id,
                                         decl_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::FunctionDefinitionId node_id)
    -> bool {
  SemIR::FunctionId function_id =
      context.node_stack().Pop<Parse::NodeKind::FunctionDefinitionStart>();

  // If the `}` of the function is reachable, reject if we need a return value
  // and otherwise add an implicit `return;`.
  if (context.is_current_position_reachable()) {
    if (context.functions().Get(function_id).return_storage_id.is_valid()) {
      CARBON_DIAGNOSTIC(
          MissingReturnStatement, Error,
          "missing `return` at end of function with declared return type");
      context.emitter().Emit(TokenOnly(node_id), MissingReturnStatement);
    } else {
      context.AddInst<SemIR::Return>(node_id, {});
    }
  }

  context.scope_stack().Pop();
  context.inst_block_stack().Pop();
  context.return_scope_stack().pop_back();
  context.decl_name_stack().PopScope();

  // If this is a generic function, collect information about the definition.
  auto& function = context.functions().Get(function_id);
  FinishGenericDefinition(context, function.generic_id);

  return true;
}

auto HandleParseNode(Context& context,
                     Parse::BuiltinFunctionDefinitionStartId node_id) -> bool {
  // Process the declaration portion of the function.
  auto [function_id, _] =
      BuildFunctionDecl(context, node_id, /*is_definition=*/true);
  context.node_stack().Push(node_id, function_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::BuiltinNameId node_id) -> bool {
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
                      "unknown builtin function name \"{0}\"", std::string);
    context.emitter().Emit(name_id, UnknownBuiltinFunctionName,
                           builtin_name.str());
  }
  return kind;
}

// Returns whether `function` is a valid declaration of the builtin
// `builtin_inst_kind`.
static auto IsValidBuiltinDeclaration(Context& context,
                                      const SemIR::Function& function,
                                      SemIR::BuiltinFunctionKind builtin_kind)
    -> bool {
  // Form the list of parameter types for the declaration.
  llvm::SmallVector<SemIR::TypeId> param_type_ids;
  auto implicit_param_refs =
      context.inst_blocks().GetOrEmpty(function.implicit_param_refs_id);
  auto param_refs = context.inst_blocks().GetOrEmpty(function.param_refs_id);
  param_type_ids.reserve(implicit_param_refs.size() + param_refs.size());
  for (auto param_id :
       llvm::concat<const SemIR::InstId>(implicit_param_refs, param_refs)) {
    // TODO: We also need to track whether the parameter is declared with
    // `var`.
    param_type_ids.push_back(context.insts().Get(param_id).type_id());
  }

  // Get the return type. This is `()` if none was specified.
  auto return_type_id = function.GetDeclaredReturnType(context.sem_ir());
  if (!return_type_id.is_valid()) {
    return_type_id = context.GetTupleType({});
  }

  return builtin_kind.IsValidType(context.sem_ir(), param_type_ids,
                                  return_type_id);
}

auto HandleParseNode(Context& context,
                     Parse::BuiltinFunctionDefinitionId /*node_id*/) -> bool {
  auto name_id =
      context.node_stack().PopForSoloNodeId<Parse::NodeKind::BuiltinName>();
  auto [fn_node_id, function_id] =
      context.node_stack()
          .PopWithNodeId<Parse::NodeKind::BuiltinFunctionDefinitionStart>();

  auto builtin_kind = LookupBuiltinFunctionKind(context, name_id);
  if (builtin_kind != SemIR::BuiltinFunctionKind::None) {
    auto& function = context.functions().Get(function_id);
    if (IsValidBuiltinDeclaration(context, function, builtin_kind)) {
      function.builtin_function_kind = builtin_kind;
    } else {
      CARBON_DIAGNOSTIC(InvalidBuiltinSignature, Error,
                        "invalid signature for builtin function \"{0}\"",
                        std::string);
      context.emitter().Emit(fn_node_id, InvalidBuiltinSignature,
                             builtin_kind.name().str());
    }
  }
  context.decl_name_stack().PopScope();
  return true;
}

}  // namespace Carbon::Check
