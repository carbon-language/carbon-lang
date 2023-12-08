// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/modifiers.h"
#include "toolchain/sem_ir/entry_point.h"

namespace Carbon::Check {

static auto DiagnoseModifiers(Context& context) -> KeywordModifierSet {
  Lex::TokenKind decl_kind = Lex::TokenKind::Fn;
  CheckAccessModifiersOnDecl(context, decl_kind);
  LimitModifiersOnDecl(context,
                       KeywordModifierSet::Access | KeywordModifierSet::Method |
                           KeywordModifierSet::Interface,
                       decl_kind);
  // Rules for abstract, virtual, and impl, which are only allowed in classes.
  if (auto class_decl = context.GetCurrentScopeAs<SemIR::ClassDecl>()) {
    auto inheritance_kind =
        context.classes().Get(class_decl->class_id).inheritance_kind;
    if (inheritance_kind == SemIR::Class::Final) {
      ForbidModifiersOnDecl(context, KeywordModifierSet::Virtual, decl_kind,
                            " in a non-abstract non-base `class` definition",
                            class_decl->parse_node);
    }
    if (inheritance_kind != SemIR::Class::Abstract) {
      ForbidModifiersOnDecl(context, KeywordModifierSet::Abstract, decl_kind,
                            " in a non-abstract `class` definition",
                            class_decl->parse_node);
    }
  } else {
    ForbidModifiersOnDecl(context, KeywordModifierSet::Method, decl_kind,
                          " outside of a class");
  }
  RequireDefaultFinalOnlyInInterfaces(context, decl_kind);

  return context.decl_state_stack().innermost().modifier_set;
}

// Build a FunctionDecl describing the signature of a function. This
// handles the common logic shared by function declaration syntax and function
// definition syntax.
static auto BuildFunctionDecl(Context& context, bool is_definition)
    -> std::pair<SemIR::FunctionId, SemIR::InstId> {
  // TODO: This contains the IR block for the parameters and return type. At
  // present, it's just loose, but it's not strictly required for parameter
  // refs; we should either stop constructing it completely or, if it turns out
  // to be needed, store it. Note, the underlying issue is that the LLVM IR has
  // nowhere clear to emit, so changing storage would require addressing that
  // problem. For comparison with function calls, the IR needs to be emitted
  // prior to the call.
  context.inst_block_stack().Pop();

  auto return_type_id = SemIR::TypeId::Invalid;
  auto return_slot_id = SemIR::InstId::Invalid;
  if (context.parse_tree().node_kind(context.node_stack().PeekParseNode()) ==
      Parse::NodeKind::ReturnType) {
    auto [return_node, return_storage_id] =
        context.node_stack().PopWithParseNode<Parse::NodeKind::ReturnType>();
    auto return_node_copy = return_node;
    return_type_id = context.insts().Get(return_storage_id).type_id();

    return_type_id = context.AsCompleteType(return_type_id, [&] {
      CARBON_DIAGNOSTIC(IncompleteTypeInFunctionReturnType, Error,
                        "Function returns incomplete type `{0}`.", std::string);
      return context.emitter().Build(
          return_node_copy, IncompleteTypeInFunctionReturnType,
          context.sem_ir().StringifyType(return_type_id));
    });

    if (!SemIR::GetInitRepr(context.sem_ir(), return_type_id)
             .has_return_slot()) {
      // The function only has a return slot if it uses in-place initialization.
    } else {
      return_slot_id = return_storage_id;
    }
  }

  SemIR::InstBlockId param_refs_id =
      context.node_stack().Pop<Parse::NodeKind::TuplePattern>();
  SemIR::InstBlockId implicit_param_refs_id =
      context.node_stack().PopIf<Parse::NodeKind::ImplicitParamList>().value_or(
          SemIR::InstBlockId::Empty);
  auto name_context = context.decl_name_stack().FinishName();
  context.node_stack()
      .PopAndDiscardSoloParseNode<Parse::NodeKind::FunctionIntroducer>();

  auto first_node = context.decl_state_stack().innermost().first_node;

  // Process modifiers.
  auto modifiers = DiagnoseModifiers(context);
  if (!!(modifiers & KeywordModifierSet::Access)) {
    context.TODO(context.decl_state_stack().innermost().saw_access_modifier,
                 "access modifier");
  }
  if (!!(modifiers & KeywordModifierSet::Method)) {
    context.TODO(context.decl_state_stack().innermost().saw_decl_modifier,
                 "method modifier");
  }
  if (!!(modifiers & KeywordModifierSet::Interface)) {
    // TODO: Once we are saving the modifiers for a function, add check that
    // the function may only be defined if it is marked `default` or `final`.
    context.TODO(context.decl_state_stack().innermost().saw_decl_modifier,
                 "interface modifier");
  }
  context.decl_state_stack().Pop(DeclState::Fn);

  // Add the function declaration.
  auto function_decl = SemIR::FunctionDecl{
      first_node, context.GetBuiltinType(SemIR::BuiltinKind::FunctionType),
      SemIR::FunctionId::Invalid};
  auto function_decl_id = context.AddInst(function_decl);

  // Check whether this is a redeclaration.
  auto existing_id =
      context.decl_name_stack().LookupOrAddName(name_context, function_decl_id);
  if (existing_id.is_valid()) {
    if (auto existing_function_decl =
            context.insts().Get(existing_id).TryAs<SemIR::FunctionDecl>()) {
      // This is a redeclaration of an existing function.
      function_decl.function_id = existing_function_decl->function_id;

      // TODO: Check that the signature matches!

      // Track the signature from the definition, so that IDs in the body match
      // IDs in the signature.
      if (is_definition) {
        auto& function_info =
            context.functions().Get(function_decl.function_id);
        function_info.implicit_param_refs_id = implicit_param_refs_id;
        function_info.param_refs_id = param_refs_id;
        function_info.return_type_id = return_type_id;
        function_info.return_slot_id = return_slot_id;
      }
    } else {
      // This is a redeclaration of something other than a function.
      context.DiagnoseDuplicateName(name_context.parse_node, existing_id);
    }
  }

  // Create a new function if this isn't a valid redeclaration.
  if (!function_decl.function_id.is_valid()) {
    function_decl.function_id = context.functions().Add(
        {.name_id =
             name_context.state == DeclNameStack::NameContext::State::Unresolved
                 ? name_context.unresolved_name_id
                 : SemIR::NameId::Invalid,
         .decl_id = function_decl_id,
         .implicit_param_refs_id = implicit_param_refs_id,
         .param_refs_id = param_refs_id,
         .return_type_id = return_type_id,
         .return_slot_id = return_slot_id});
  }

  // Write the function ID into the FunctionDecl.
  context.insts().Set(function_decl_id, function_decl);

  if (SemIR::IsEntryPoint(context.sem_ir(), function_decl.function_id)) {
    // TODO: Update this once valid signatures for the entry point are decided.
    if (!context.inst_blocks().Get(implicit_param_refs_id).empty() ||
        !context.inst_blocks().Get(param_refs_id).empty() ||
        (return_slot_id.is_valid() &&
         return_type_id !=
             context.GetBuiltinType(SemIR::BuiltinKind::BoolType) &&
         return_type_id != context.CanonicalizeTupleType(first_node, {}))) {
      CARBON_DIAGNOSTIC(InvalidMainRunSignature, Error,
                        "Invalid signature for `Main.Run` function. Expected "
                        "`fn ()` or `fn () -> i32`.");
      context.emitter().Emit(first_node, InvalidMainRunSignature);
    }
  }

  return {function_decl.function_id, function_decl_id};
}

auto HandleFunctionDecl(Context& context, Parse::NodeId /*parse_node*/)
    -> bool {
  BuildFunctionDecl(context, /*is_definition=*/false);
  context.decl_name_stack().PopScope();
  return true;
}

auto HandleFunctionDefinition(Context& context, Parse::NodeId parse_node)
    -> bool {
  SemIR::FunctionId function_id =
      context.node_stack().Pop<Parse::NodeKind::FunctionDefinitionStart>();

  // If the `}` of the function is reachable, reject if we need a return value
  // and otherwise add an implicit `return;`.
  if (context.is_current_position_reachable()) {
    if (context.functions().Get(function_id).return_type_id.is_valid()) {
      CARBON_DIAGNOSTIC(
          MissingReturnStatement, Error,
          "Missing `return` at end of function with declared return type.");
      context.emitter().Emit(parse_node, MissingReturnStatement);
    } else {
      context.AddInst(SemIR::Return{parse_node});
    }
  }

  context.PopScope();
  context.inst_block_stack().Pop();
  context.return_scope_stack().pop_back();
  context.decl_name_stack().PopScope();
  return true;
}

auto HandleFunctionDefinitionStart(Context& context, Parse::NodeId parse_node)
    -> bool {
  // Process the declaration portion of the function.
  auto [function_id, decl_id] =
      BuildFunctionDecl(context, /*is_definition=*/true);
  auto& function = context.functions().Get(function_id);

  // Track that this declaration is the definition.
  if (function.definition_id.is_valid()) {
    CARBON_DIAGNOSTIC(FunctionRedefinition, Error,
                      "Redefinition of function {0}.", std::string);
    CARBON_DIAGNOSTIC(FunctionPreviousDefinition, Note,
                      "Previous definition was here.");
    context.emitter()
        .Build(parse_node, FunctionRedefinition,
               context.names().GetFormatted(function.name_id).str())
        .Note(context.insts().Get(function.definition_id).parse_node(),
              FunctionPreviousDefinition)
        .Emit();
  } else {
    function.definition_id = decl_id;
  }

  // Create the function scope and the entry block.
  context.return_scope_stack().push_back({.decl_id = decl_id});
  context.inst_block_stack().Push();
  context.PushScope(decl_id);
  context.AddCurrentCodeBlockToFunction();

  // Bring the implicit and explicit parameters into scope.
  for (auto param_id : llvm::concat<SemIR::InstId>(
           context.inst_blocks().Get(function.implicit_param_refs_id),
           context.inst_blocks().Get(function.param_refs_id))) {
    auto param = context.insts().Get(param_id);

    // The parameter types need to be complete.
    context.TryToCompleteType(param.type_id(), [&] {
      CARBON_DIAGNOSTIC(
          IncompleteTypeInFunctionParam, Error,
          "Parameter has incomplete type `{0}` in function definition.",
          std::string);
      return context.emitter().Build(
          param.parse_node(), IncompleteTypeInFunctionParam,
          context.sem_ir().StringifyType(param.type_id()));
    });

    if (auto fn_param = param.TryAs<SemIR::Param>()) {
      context.AddNameToLookup(fn_param->parse_node, fn_param->name_id,
                              param_id);
    } else if (auto self_param = param.TryAs<SemIR::SelfParam>()) {
      context.AddNameToLookup(self_param->parse_node, SemIR::NameId::SelfValue,
                              param_id);
    } else {
      CARBON_FATAL() << "Unexpected kind of parameter in function definition "
                     << param;
    }
  }

  context.node_stack().Push(parse_node, function_id);
  return true;
}

auto HandleFunctionIntroducer(Context& context, Parse::NodeId parse_node)
    -> bool {
  // Create an instruction block to hold the instructions created as part of the
  // function signature, such as parameter and return types.
  context.inst_block_stack().Push();
  // Push the bracketing node.
  context.node_stack().Push(parse_node);
  // Optional modifiers and the name follow.
  context.decl_state_stack().Push(DeclState::Fn, parse_node);
  context.decl_name_stack().PushScopeAndStartName();
  return true;
}

auto HandleReturnType(Context& context, Parse::NodeId parse_node) -> bool {
  // Propagate the type expression.
  auto [type_parse_node, type_inst_id] =
      context.node_stack().PopExprWithParseNode();
  auto type_id = ExprAsType(context, type_parse_node, type_inst_id);
  // TODO: Use a dedicated instruction rather than VarStorage here.
  context.AddInstAndPush(
      parse_node,
      SemIR::VarStorage{parse_node, type_id, SemIR::NameId::ReturnSlot});
  return true;
}

}  // namespace Carbon::Check
