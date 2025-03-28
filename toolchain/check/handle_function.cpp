// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/control_flow.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/decl_introducer_state.h"
#include "toolchain/check/decl_name_stack.h"
#include "toolchain/check/function.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/import.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/interface.h"
#include "toolchain/check/keyword_modifier_set.h"
#include "toolchain/check/literal.h"
#include "toolchain/check/merge.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/check/name_component.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/builtin_function_kind.h"
#include "toolchain/sem_ir/entry_point.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/pattern.h"
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
  return true;
}

auto HandleParseNode(Context& context, Parse::ReturnTypeId node_id) -> bool {
  // Propagate the type expression.
  auto [type_node_id, type_inst_id] = context.node_stack().PopExprWithNodeId();
  auto type_id = ExprAsType(context, type_node_id, type_inst_id).type_id;

  // If the previous node was `IdentifierNameBeforeParams`, then it would have
  // caused these entries to be pushed to the pattern stacks. But it's possible
  // to have a fn declaration without any parameters, in which case we find
  // `IdentifierNameNotBeforeParams` on the node stack. Then these entries are
  // not on the pattern stacks yet. They are only needed in that case if we have
  // a return type, which we now know that we do.
  if (context.node_stack().PeekNodeKind() ==
      Parse::NodeKind::IdentifierNameNotBeforeParams) {
    context.pattern_block_stack().Push();
    context.full_pattern_stack().PushFullPattern(
        FullPatternStack::Kind::ExplicitParamList);
  }

  auto return_slot_pattern_id = AddPatternInst<SemIR::ReturnSlotPattern>(
      context, node_id, {.type_id = type_id, .type_inst_id = type_inst_id});
  auto param_pattern_id = AddPatternInst<SemIR::OutParamPattern>(
      context, node_id,
      {.type_id = type_id,
       .subpattern_id = return_slot_pattern_id,
       .index = SemIR::CallParamIndex::None});
  context.node_stack().Push(node_id, param_pattern_id);
  return true;
}

// Returns the ID of the self parameter pattern, or None.
// TODO: Do this during initial traversal of implicit params.
static auto FindSelfPattern(Context& context,
                            SemIR::InstBlockId implicit_param_patterns_id)
    -> SemIR::InstId {
  auto implicit_param_patterns =
      context.inst_blocks().GetOrEmpty(implicit_param_patterns_id);
  if (const auto* i = llvm::find_if(implicit_param_patterns,
                                    [&](auto implicit_param_id) {
                                      return SemIR::IsSelfPattern(
                                          context.sem_ir(), implicit_param_id);
                                    });
      i != implicit_param_patterns.end()) {
    return *i;
  }
  return SemIR::InstId::None;
}

// Diagnoses issues with the modifiers, removing modifiers that shouldn't be
// present.
static auto DiagnoseModifiers(Context& context,
                              Parse::AnyFunctionDeclId node_id,
                              DeclIntroducerState& introducer,
                              bool is_definition,
                              SemIR::InstId parent_scope_inst_id,
                              std::optional<SemIR::Inst> parent_scope_inst,
                              SemIR::InstId self_param_id) -> void {
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

  if (introducer.modifier_set.HasAnyOf(KeywordModifierSet::Interface)) {
    // TODO: Once we are saving the modifiers for a function, add check that
    // the function may only be defined if it is marked `default` or `final`.
    context.TODO(introducer.modifier_node_id(ModifierOrder::Decl),
                 "interface modifier");
  }

  if (!self_param_id.has_value() &&
      introducer.modifier_set.HasAnyOf(KeywordModifierSet::Method)) {
    CARBON_DIAGNOSTIC(VirtualWithoutSelf, Error, "virtual class function");
    context.emitter().Emit(node_id, VirtualWithoutSelf);
    introducer.modifier_set.Remove(KeywordModifierSet::Method);
  }
}

// Returns the virtual-family modifier as an enum.
static auto GetVirtualModifier(const KeywordModifierSet& modifier_set)
    -> SemIR::Function::VirtualModifier {
  return modifier_set.ToEnum<SemIR::Function::VirtualModifier>()
      .Case(KeywordModifierSet::Virtual,
            SemIR::Function::VirtualModifier::Virtual)
      .Case(KeywordModifierSet::Abstract,
            SemIR::Function::VirtualModifier::Abstract)
      .Case(KeywordModifierSet::Impl, SemIR::Function::VirtualModifier::Impl)
      .Default(SemIR::Function::VirtualModifier::None);
}

// Tries to merge new_function into prev_function_id. Since new_function won't
// have a definition even if one is upcoming, set is_definition to indicate the
// planned result.
//
// If merging is successful, returns true and may update the previous function.
// Otherwise, returns false. Prints a diagnostic when appropriate.
static auto MergeFunctionRedecl(Context& context,
                                Parse::AnyFunctionDeclId node_id,
                                SemIR::Function& new_function,
                                bool new_is_definition,
                                SemIR::FunctionId prev_function_id,
                                SemIR::ImportIRId prev_import_ir_id) -> bool {
  auto& prev_function = context.functions().Get(prev_function_id);

  if (!CheckFunctionTypeMatches(context, new_function, prev_function)) {
    return false;
  }

  DiagnoseIfInvalidRedecl(
      context, Lex::TokenKind::Fn, prev_function.name_id,
      RedeclInfo(new_function, node_id, new_is_definition),
      RedeclInfo(prev_function, prev_function.latest_decl_id(),
                 prev_function.has_definition_started()),
      prev_import_ir_id);
  if (new_is_definition && prev_function.has_definition_started()) {
    return false;
  }

  if (!prev_function.first_owning_decl_id.has_value()) {
    prev_function.first_owning_decl_id = new_function.first_owning_decl_id;
  }
  if (new_is_definition) {
    // Track the signature from the definition, so that IDs in the body
    // match IDs in the signature.
    prev_function.MergeDefinition(new_function);
    prev_function.call_params_id = new_function.call_params_id;
    prev_function.return_slot_pattern_id = new_function.return_slot_pattern_id;
    prev_function.self_param_id = new_function.self_param_id;
  }
  if (prev_import_ir_id.has_value()) {
    ReplacePrevInstForMerge(context, new_function.parent_scope_id,
                            prev_function.name_id,
                            new_function.first_owning_decl_id);
  }
  return true;
}

// Check whether this is a redeclaration, merging if needed.
static auto TryMergeRedecl(Context& context, Parse::AnyFunctionDeclId node_id,
                           const DeclNameStack::NameContext& name_context,
                           SemIR::FunctionDecl& function_decl,
                           SemIR::Function& function_info, bool is_definition)
    -> void {
  if (name_context.state == DeclNameStack::NameContext::State::Poisoned) {
    DiagnosePoisonedName(context, name_context.name_id_for_new_inst(),
                         name_context.poisoning_loc_id, name_context.loc_id);
    return;
  }

  auto prev_id = name_context.prev_inst_id();
  if (!prev_id.has_value()) {
    return;
  }

  auto prev_function_id = SemIR::FunctionId::None;
  auto prev_type_id = SemIR::TypeId::None;
  auto prev_import_ir_id = SemIR::ImportIRId::None;
  CARBON_KIND_SWITCH(context.insts().Get(prev_id)) {
    case CARBON_KIND(SemIR::FunctionDecl function_decl): {
      prev_function_id = function_decl.function_id;
      prev_type_id = function_decl.type_id;
      break;
    }
    case SemIR::ImportRefLoaded::Kind: {
      auto import_ir_inst = GetCanonicalImportIRInst(context, prev_id);

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
          prev_type_id = struct_value->type_id;
          prev_import_ir_id = import_ir_inst.ir_id;
        }
      }
      break;
    }
    default:
      break;
  }

  if (!prev_function_id.has_value()) {
    DiagnoseDuplicateName(context, name_context.name_id, name_context.loc_id,
                          prev_id);
    return;
  }

  if (MergeFunctionRedecl(context, node_id, function_info, is_definition,
                          prev_function_id, prev_import_ir_id)) {
    // When merging, use the existing function rather than adding a new one.
    function_decl.function_id = prev_function_id;
    function_decl.type_id = prev_type_id;
  }
}

// Adds the declaration to name lookup when appropriate.
static auto MaybeAddToNameLookup(
    Context& context, const DeclNameStack::NameContext& name_context,
    const KeywordModifierSet& modifier_set,
    const std::optional<SemIR::Inst>& parent_scope_inst, SemIR::InstId decl_id)
    -> void {
  if (name_context.state == DeclNameStack::NameContext::State::Poisoned ||
      name_context.prev_inst_id().has_value()) {
    return;
  }

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
                                    modifier_set.GetAccessKind());
}

// If the function is the entry point, do corresponding validation.
static auto ValidateForEntryPoint(Context& context,
                                  Parse::AnyFunctionDeclId node_id,
                                  SemIR::FunctionId function_id,
                                  const SemIR::Function& function_info)
    -> void {
  if (!SemIR::IsEntryPoint(context.sem_ir(), function_id)) {
    return;
  }

  auto return_type_id = function_info.GetDeclaredReturnType(context.sem_ir());
  // TODO: Update this once valid signatures for the entry point are decided.
  if (function_info.implicit_param_patterns_id.has_value() ||
      !function_info.param_patterns_id.has_value() ||
      !context.inst_blocks().Get(function_info.param_patterns_id).empty() ||
      (return_type_id.has_value() &&
       return_type_id != GetTupleType(context, {}) &&
       // TODO: Decide on valid return types for `Main.Run`. Perhaps we should
       // have an interface for this.
       return_type_id != MakeIntType(context, node_id, SemIR::IntKind::Signed,
                                     context.ints().Add(32)))) {
    CARBON_DIAGNOSTIC(InvalidMainRunSignature, Error,
                      "invalid signature for `Main.Run` function; expected "
                      "`fn ()` or `fn () -> i32`");
    context.emitter().Emit(node_id, InvalidMainRunSignature);
  }
}

// Requests a vtable be created when processing a virtual function.
static auto RequestVtableIfVirtual(
    Context& context, Parse::AnyFunctionDeclId node_id,
    SemIR::Function::VirtualModifier virtual_modifier,
    const std::optional<SemIR::Inst>& parent_scope_inst, SemIR::InstId decl_id)
    -> void {
  // In order to request a vtable, the function must be virtual, and in a class
  // scope.
  if (virtual_modifier == SemIR::Function::VirtualModifier::None ||
      !parent_scope_inst) {
    return;
  }
  auto class_decl = parent_scope_inst->TryAs<SemIR::ClassDecl>();
  if (!class_decl) {
    return;
  }

  auto& class_info = context.classes().Get(class_decl->class_id);
  if (virtual_modifier == SemIR::Function::VirtualModifier::Impl &&
      !class_info.base_id.has_value()) {
    CARBON_DIAGNOSTIC(ImplWithoutBase, Error, "impl without base class");
    context.emitter().Emit(node_id, ImplWithoutBase);
  }
  // TODO: If this is an `impl` function, check there's a matching base
  // function that's impl or virtual.
  class_info.is_dynamic = true;
  context.vtable_stack().AddInstId(decl_id);
}

// Validates the `destroy` function's signature. May replace invalid values for
// recovery.
static auto ValidateIfDestroy(Context& context, bool is_redecl,
                              std::optional<SemIR::Inst> parent_scope_inst,
                              SemIR::Function& function_info) -> void {
  if (function_info.name_id != SemIR::NameId::Destroy) {
    return;
  }

  // For recovery, always force explicit parameters to be empty. We do this
  // before any of the returns for simplicity.
  auto orig_param_patterns_id = function_info.param_patterns_id;
  function_info.param_patterns_id = SemIR::InstBlockId::Empty;

  // Use differences on merge to diagnose remaining issues.
  if (is_redecl) {
    return;
  }

  if (!parent_scope_inst || !parent_scope_inst->Is<SemIR::ClassDecl>()) {
    CARBON_DIAGNOSTIC(DestroyFunctionOutsideClass, Error,
                      "declaring `fn destroy` in non-class scope");
    context.emitter().Emit(function_info.latest_decl_id(),
                           DestroyFunctionOutsideClass);
    return;
  }

  if (!function_info.self_param_id.has_value()) {
    CARBON_DIAGNOSTIC(DestroyFunctionMissingSelf, Error,
                      "missing implicit `self` parameter");
    context.emitter().Emit(function_info.latest_decl_id(),
                           DestroyFunctionMissingSelf);
    return;
  }

  // `self` must be the only implicit parameter.
  if (auto block =
          context.inst_blocks().Get(function_info.implicit_param_patterns_id);
      block.size() > 1) {
    // Point at the first non-`self` parameter.
    auto param_id = block[function_info.self_param_id == block[0] ? 1 : 0];
    CARBON_DIAGNOSTIC(DestroyFunctionUnexpectedImplicitParam, Error,
                      "unexpected implicit parameter");
    context.emitter().Emit(param_id, DestroyFunctionUnexpectedImplicitParam);
    return;
  }

  if (!orig_param_patterns_id.has_value()) {
    CARBON_DIAGNOSTIC(DestroyFunctionPositionalParams, Error,
                      "missing empty explicit parameter list");
    context.emitter().Emit(function_info.latest_decl_id(),
                           DestroyFunctionPositionalParams);
    return;
  }

  if (orig_param_patterns_id != SemIR::InstBlockId::Empty) {
    CARBON_DIAGNOSTIC(DestroyFunctionNonEmptyExplicitParams, Error,
                      "unexpected parameter");
    context.emitter().Emit(context.inst_blocks().Get(orig_param_patterns_id)[0],
                           DestroyFunctionNonEmptyExplicitParams);
    return;
  }

  if (auto return_type_id =
          function_info.GetDeclaredReturnType(context.sem_ir());
      return_type_id.has_value() &&
      return_type_id != GetTupleType(context, {})) {
    CARBON_DIAGNOSTIC(DestroyFunctionIncorrectReturnType, Error,
                      "incorrect return type; must be unspecified or `()`");
    context.emitter().Emit(function_info.return_slot_pattern_id,
                           DestroyFunctionIncorrectReturnType);
    return;
  }
}

// Diagnoses when positional params aren't supported. Reassigns the pattern
// block if needed.
static auto DiagnosePositionalParams(Context& context,
                                     SemIR::Function& function_info) -> void {
  if (function_info.param_patterns_id.has_value()) {
    return;
  }

  context.TODO(function_info.latest_decl_id(),
               "function with positional parameters");
  function_info.param_patterns_id = SemIR::InstBlockId::Empty;
}

// Build a FunctionDecl describing the signature of a function. This
// handles the common logic shared by function declaration syntax and function
// definition syntax.
static auto BuildFunctionDecl(Context& context,
                              Parse::AnyFunctionDeclId node_id,
                              bool is_definition)
    -> std::pair<SemIR::FunctionId, SemIR::InstId> {
  auto return_slot_pattern_id = SemIR::InstId::None;
  if (auto [return_node, maybe_return_slot_pattern_id] =
          context.node_stack().PopWithNodeIdIf<Parse::NodeKind::ReturnType>();
      maybe_return_slot_pattern_id) {
    return_slot_pattern_id = *maybe_return_slot_pattern_id;
  }

  auto name = PopNameComponent(context, return_slot_pattern_id);
  auto name_context = context.decl_name_stack().FinishName(name);

  context.node_stack()
      .PopAndDiscardSoloNodeId<Parse::NodeKind::FunctionIntroducer>();

  auto self_param_id =
      FindSelfPattern(context, name.implicit_param_patterns_id);

  // Process modifiers.
  auto [parent_scope_inst_id, parent_scope_inst] =
      context.name_scopes().GetInstIfValid(name_context.parent_scope_id);
  auto introducer =
      context.decl_introducer_state_stack().Pop<Lex::TokenKind::Fn>();
  DiagnoseModifiers(context, node_id, introducer, is_definition,
                    parent_scope_inst_id, parent_scope_inst, self_param_id);
  bool is_extern = introducer.modifier_set.HasAnyOf(KeywordModifierSet::Extern);
  auto virtual_modifier = GetVirtualModifier(introducer.modifier_set);

  // Add the function declaration.
  SemIR::FunctionDecl function_decl = {SemIR::TypeId::None,
                                       SemIR::FunctionId::None,
                                       context.inst_block_stack().Pop()};
  auto decl_id = AddPlaceholderInst(context, node_id, function_decl);
  RequestVtableIfVirtual(context, node_id, virtual_modifier, parent_scope_inst,
                         decl_id);

  // Build the function entity. This will be merged into an existing function if
  // there is one, or otherwise added to the function store.
  auto function_info =
      SemIR::Function{name_context.MakeEntityWithParamsBase(
                          name, decl_id, is_extern, introducer.extern_library),
                      {.call_params_id = name.call_params_id,
                       .return_slot_pattern_id = name.return_slot_pattern_id,
                       .virtual_modifier = virtual_modifier,
                       .self_param_id = self_param_id}};
  if (is_definition) {
    function_info.definition_id = decl_id;
  }

  // Analyze standard function signatures before positional parameters, so that
  // we can have more specific diagnostics and recovery.
  bool is_redecl =
      name_context.state == DeclNameStack::NameContext::State::Resolved;
  ValidateIfDestroy(context, is_redecl, parent_scope_inst, function_info);

  DiagnosePositionalParams(context, function_info);

  TryMergeRedecl(context, node_id, name_context, function_decl, function_info,
                 is_definition);

  // Create a new function if this isn't a valid redeclaration.
  if (!function_decl.function_id.has_value()) {
    if (function_info.is_extern && context.sem_ir().is_impl()) {
      DiagnoseExternRequiresDeclInApiFile(context, node_id);
    }
    function_info.generic_id = BuildGenericDecl(context, decl_id);
    function_decl.function_id = context.functions().Add(function_info);
    function_decl.type_id =
        GetFunctionType(context, function_decl.function_id,
                        context.scope_stack().PeekSpecificId());
  } else {
    auto prev_decl_generic_id =
        context.functions().Get(function_decl.function_id).generic_id;
    FinishGenericRedecl(context, prev_decl_generic_id);
    // TODO: Validate that the redeclaration doesn't set an access modifier.
  }

  // Write the function ID into the FunctionDecl.
  ReplaceInstBeforeConstantUse(context, decl_id, function_decl);

  // Diagnose 'definition of `abstract` function' using the canonical Function's
  // modifiers.
  if (is_definition &&
      context.functions().Get(function_decl.function_id).virtual_modifier ==
          SemIR::Function::VirtualModifier::Abstract) {
    CARBON_DIAGNOSTIC(DefinedAbstractFunction, Error,
                      "definition of `abstract` function");
    context.emitter().Emit(TokenOnly(node_id), DefinedAbstractFunction);
  }

  // Add to name lookup if needed, now that the decl is built.
  MaybeAddToNameLookup(context, name_context, introducer.modifier_set,
                       parent_scope_inst, decl_id);

  ValidateForEntryPoint(context, node_id, function_decl.function_id,
                        function_info);

  if (!is_definition && context.sem_ir().is_impl() && !is_extern) {
    context.definitions_required().push_back(decl_id);
  }

  return {function_decl.function_id, decl_id};
}

auto HandleParseNode(Context& context, Parse::FunctionDeclId node_id) -> bool {
  BuildFunctionDecl(context, node_id, /*is_definition=*/false);
  context.decl_name_stack().PopScope();
  return true;
}

static auto CheckFunctionDefinitionSignature(Context& context,
                                             SemIR::Function& function)
    -> void {
  auto params_to_complete =
      context.inst_blocks().GetOrEmpty(function.call_params_id);

  // Check the return type is complete.
  if (function.return_slot_pattern_id.has_value()) {
    CheckFunctionReturnType(
        context, context.insts().GetLocId(function.return_slot_pattern_id),
        function, SemIR::SpecificId::None);
    params_to_complete = params_to_complete.drop_back();
  }

  // Check the parameter types are complete.
  for (auto param_ref_id : params_to_complete) {
    if (param_ref_id == SemIR::ErrorInst::SingletonInstId) {
      continue;
    }

    // The parameter types need to be complete.
    RequireCompleteType(
        context, context.insts().GetAs<SemIR::AnyParam>(param_ref_id).type_id,
        context.insts().GetLocId(param_ref_id), [&] {
          CARBON_DIAGNOSTIC(
              IncompleteTypeInFunctionParam, Error,
              "parameter has incomplete type {0} in function definition",
              TypeOfInstId);
          return context.emitter().Build(
              param_ref_id, IncompleteTypeInFunctionParam, param_ref_id);
        });
  }
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
  context.region_stack().PushRegion(context.inst_block_stack().PeekOrAdd());
  context.scope_stack().Push(decl_id);
  StartGenericDefinition(context);

  CheckFunctionDefinitionSignature(context, function);

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
  if (IsCurrentPositionReachable(context)) {
    if (context.functions()
            .Get(function_id)
            .return_slot_pattern_id.has_value()) {
      CARBON_DIAGNOSTIC(
          MissingReturnStatement, Error,
          "missing `return` at end of function with declared return type");
      context.emitter().Emit(TokenOnly(node_id), MissingReturnStatement);
    } else {
      AddReturnCleanupBlock(context, node_id);
    }
  }

  context.scope_stack().Pop();
  context.inst_block_stack().Pop();
  context.return_scope_stack().pop_back();
  context.decl_name_stack().PopScope();

  auto& function = context.functions().Get(function_id);
  function.body_block_ids = context.region_stack().PopRegion();

  // If this is a generic function, collect information about the definition.
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

// Returns whether `function` is a valid declaration of `builtin_kind`.
static auto IsValidBuiltinDeclaration(Context& context,
                                      const SemIR::Function& function,
                                      SemIR::BuiltinFunctionKind builtin_kind)
    -> bool {
  // Form the list of parameter types for the declaration.
  llvm::SmallVector<SemIR::TypeId> param_type_ids;
  auto implicit_param_patterns =
      context.inst_blocks().GetOrEmpty(function.implicit_param_patterns_id);
  auto param_patterns =
      context.inst_blocks().GetOrEmpty(function.param_patterns_id);
  param_type_ids.reserve(implicit_param_patterns.size() +
                         param_patterns.size());
  for (auto param_id : llvm::concat<const SemIR::InstId>(
           implicit_param_patterns, param_patterns)) {
    // TODO: We also need to track whether the parameter is declared with
    // `var`.
    param_type_ids.push_back(context.insts().Get(param_id).type_id());
  }

  // Get the return type. This is `()` if none was specified.
  auto return_type_id = function.GetDeclaredReturnType(context.sem_ir());
  if (!return_type_id.has_value()) {
    return_type_id = GetTupleType(context, {});
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
    CheckFunctionDefinitionSignature(context, function);
    if (IsValidBuiltinDeclaration(context, function, builtin_kind)) {
      function.builtin_function_kind = builtin_kind;
      // Build an empty generic definition if this is a generic builtin.
      StartGenericDefinition(context);
      FinishGenericDefinition(context, function.generic_id);
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
