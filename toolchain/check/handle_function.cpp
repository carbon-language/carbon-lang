// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/sem_ir/entry_point.h"

namespace Carbon::Check {

// Build a FunctionDeclaration describing the signature of a function. This
// handles the common logic shared by function declaration syntax and function
// definition syntax.
static auto BuildFunctionDeclaration(Context& context, bool is_definition)
    -> std::pair<SemIR::FunctionId, SemIR::NodeId> {
  // TODO: This contains the IR block for the parameters and return type. At
  // present, it's just loose, but it's not strictly required for parameter
  // refs; we should either stop constructing it completely or, if it turns out
  // to be needed, store it. Note, the underlying issue is that the LLVM IR has
  // nowhere clear to emit, so changing storage would require addressing that
  // problem. For comparison with function calls, the IR needs to be emitted
  // prior to the call.
  context.node_block_stack().Pop();

  auto return_type_id = SemIR::TypeId::Invalid;
  auto return_slot_id = SemIR::NodeId::Invalid;
  if (context.parse_tree().node_kind(context.node_stack().PeekParseNode()) ==
      Parse::NodeKind::ReturnType) {
    auto [return_node, return_storage_id] =
        context.node_stack().PopWithParseNode<Parse::NodeKind::ReturnType>();
    auto return_node_copy = return_node;
    return_type_id =
        context.semantics_ir().GetNode(return_storage_id).type_id();

    if (!context.TryToCompleteType(return_type_id, [&] {
          CARBON_DIAGNOSTIC(IncompleteTypeInFunctionReturnType, Error,
                            "Function returns incomplete type `{0}`.",
                            std::string);
          return context.emitter().Build(
              return_node_copy, IncompleteTypeInFunctionReturnType,
              context.semantics_ir().StringifyType(return_type_id, true));
        })) {
      return_type_id = SemIR::TypeId::Error;
    } else if (!SemIR::GetInitializingRepresentation(context.semantics_ir(),
                                                     return_type_id)
                    .has_return_slot()) {
      // The function only has a return slot if it uses in-place initialization.
    } else {
      return_slot_id = return_storage_id;
    }
  }

  SemIR::NodeBlockId param_refs_id =
      context.node_stack().Pop<Parse::NodeKind::ParameterList>();
  auto name_context = context.declaration_name_stack().Pop();
  auto fn_node =
      context.node_stack()
          .PopForSoloParseNode<Parse::NodeKind::FunctionIntroducer>();

  // Add the function declaration.
  auto function_decl = SemIR::FunctionDeclaration(
      fn_node, context.GetBuiltinType(SemIR::BuiltinKind::FunctionType),
      SemIR::FunctionId::Invalid);
  auto function_decl_id = context.AddNode(function_decl);

  // Check whether this is a redeclaration.
  auto existing_id = context.declaration_name_stack().LookupOrAddName(
      name_context, function_decl_id);
  if (existing_id.is_valid()) {
    if (auto existing_function_decl =
            context.semantics_ir()
                .GetNode(existing_id)
                .TryAs<SemIR::FunctionDeclaration>()) {
      // This is a redeclaration of an existing function.
      function_decl.function_id = existing_function_decl->function_id;

      // TODO: Check that the signature matches!

      // Track the signature from the definition, so that IDs in the body match
      // IDs in the signature.
      if (is_definition) {
        auto& function_info =
            context.semantics_ir().GetFunction(function_decl.function_id);
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
    function_decl.function_id = context.semantics_ir().AddFunction(
        {.name_id = name_context.state ==
                            DeclarationNameStack::NameContext::State::Unresolved
                        ? name_context.unresolved_name_id
                        : SemIR::StringId(SemIR::StringId::InvalidIndex),
         .param_refs_id = param_refs_id,
         .return_type_id = return_type_id,
         .return_slot_id = return_slot_id});
  }

  // Write the function ID into the FunctionDeclaration.
  context.semantics_ir().ReplaceNode(function_decl_id, function_decl);

  if (SemIR::IsEntryPoint(context.semantics_ir(), function_decl.function_id)) {
    // TODO: Update this once valid signatures for the entry point are decided.
    if (!context.semantics_ir().GetNodeBlock(param_refs_id).empty() ||
        (return_slot_id.is_valid() &&
         return_type_id !=
             context.GetBuiltinType(SemIR::BuiltinKind::BoolType) &&
         return_type_id != context.CanonicalizeTupleType(fn_node, {}))) {
      CARBON_DIAGNOSTIC(InvalidMainRunSignature, Error,
                        "Invalid signature for `Main.Run` function. Expected "
                        "`fn ()` or `fn () -> i32`.");
      context.emitter().Emit(fn_node, InvalidMainRunSignature);
    }
  }

  return {function_decl.function_id, function_decl_id};
}

auto HandleFunctionDeclaration(Context& context, Parse::Node /*parse_node*/)
    -> bool {
  BuildFunctionDeclaration(context, /*is_definition=*/false);
  return true;
}

auto HandleFunctionDefinition(Context& context, Parse::Node parse_node)
    -> bool {
  SemIR::FunctionId function_id =
      context.node_stack().Pop<Parse::NodeKind::FunctionDefinitionStart>();

  // If the `}` of the function is reachable, reject if we need a return value
  // and otherwise add an implicit `return;`.
  if (context.is_current_position_reachable()) {
    if (context.semantics_ir()
            .GetFunction(function_id)
            .return_type_id.is_valid()) {
      CARBON_DIAGNOSTIC(
          MissingReturnStatement, Error,
          "Missing `return` at end of function with declared return type.");
      context.emitter().Emit(parse_node, MissingReturnStatement);
    } else {
      context.AddNode(SemIR::Return(parse_node));
    }
  }

  context.PopScope();
  context.node_block_stack().Pop();
  context.return_scope_stack().pop_back();
  return true;
}

auto HandleFunctionDefinitionStart(Context& context, Parse::Node parse_node)
    -> bool {
  // Process the declaration portion of the function.
  auto [function_id, decl_id] =
      BuildFunctionDeclaration(context, /*is_definition=*/true);
  auto& function = context.semantics_ir().GetFunction(function_id);

  // Track that this declaration is the definition.
  if (function.definition_id.is_valid()) {
    CARBON_DIAGNOSTIC(FunctionRedefinition, Error,
                      "Redefinition of function {0}.", llvm::StringRef);
    CARBON_DIAGNOSTIC(FunctionPreviousDefinition, Note,
                      "Previous definition was here.");
    context.emitter()
        .Build(parse_node, FunctionRedefinition,
               context.semantics_ir().GetString(function.name_id))
        .Note(
            context.semantics_ir().GetNode(function.definition_id).parse_node(),
            FunctionPreviousDefinition)
        .Emit();
  } else {
    function.definition_id = decl_id;
  }

  // Create the function scope and the entry block.
  context.return_scope_stack().push_back(decl_id);
  context.node_block_stack().Push();
  context.PushScope();
  context.AddCurrentCodeBlockToFunction();

  // Bring the parameters into scope.
  for (auto param_id :
       context.semantics_ir().GetNodeBlock(function.param_refs_id)) {
    auto param = context.semantics_ir().GetNodeAs<SemIR::Parameter>(param_id);

    // The parameter types need to be complete.
    context.TryToCompleteType(param.type_id, [&] {
      CARBON_DIAGNOSTIC(
          IncompleteTypeInFunctionParam, Error,
          "Parameter has incomplete type `{0}` in function definition.",
          std::string);
      return context.emitter().Build(
          param.parse_node, IncompleteTypeInFunctionParam,
          context.semantics_ir().StringifyType(param.type_id, true));
    });

    context.AddNameToLookup(param.parse_node, param.name_id, param_id);
  }

  context.node_stack().Push(parse_node, function_id);
  return true;
}

auto HandleFunctionIntroducer(Context& context, Parse::Node parse_node)
    -> bool {
  // Create a node block to hold the nodes created as part of the function
  // signature, such as parameter and return types.
  context.node_block_stack().Push();
  // Push the bracketing node.
  context.node_stack().Push(parse_node);
  // A name should always follow.
  context.declaration_name_stack().Push();
  return true;
}

auto HandleReturnType(Context& context, Parse::Node parse_node) -> bool {
  // Propagate the type expression.
  auto [type_parse_node, type_node_id] =
      context.node_stack().PopExpressionWithParseNode();
  auto type_id = ExpressionAsType(context, type_parse_node, type_node_id);
  // TODO: Use a dedicated node rather than VarStorage here.
  context.AddNodeAndPush(
      parse_node,
      SemIR::VarStorage(parse_node, type_id,
                        context.semantics_ir().AddString("return")));
  return true;
}

}  // namespace Carbon::Check
