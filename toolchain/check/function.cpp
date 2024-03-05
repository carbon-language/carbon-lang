// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/function.h"

namespace Carbon::Check {

// Returns true if the function signature has an error, which will have
// previously been diagnosed.
static auto FunctionSignatureHasError(Context& context,
                                      const SemIR::Function& fn) -> bool {
  if (fn.return_type_id == SemIR::TypeId::Error) {
    return true;
  }
  for (auto param_refs_id : {fn.implicit_param_refs_id, fn.param_refs_id}) {
    if (param_refs_id != SemIR::InstBlockId::Empty) {
      for (auto param_id : context.inst_blocks().Get(param_refs_id)) {
        if (context.insts().Get(param_id).type_id() == SemIR::TypeId::Error) {
          return true;
        }
      }
    }
  }
  return false;
}

// Returns true if a param agrees. The caller is expected to provide a
// diagnostic.
// TODO: Consider moving diagnostics here, particularly to differentiate
// between type and name mistakes. For now, taking the simpler approach because
// I also think we may want to refactor params.
static auto DoesParamAgree(Context& context, SemIR::InstId new_param_ref_id,
                           SemIR::InstId prev_param_ref_id) -> bool {
  auto new_param_ref = context.insts().Get(new_param_ref_id);
  auto prev_param_ref = context.insts().Get(prev_param_ref_id);
  if (new_param_ref.kind() != prev_param_ref.kind() ||
      new_param_ref.type_id() != prev_param_ref.type_id()) {
    return false;
  }

  if (new_param_ref.Is<SemIR::AddrPattern>()) {
    new_param_ref =
        context.insts().Get(new_param_ref.As<SemIR::AddrPattern>().inner_id);
    prev_param_ref =
        context.insts().Get(prev_param_ref.As<SemIR::AddrPattern>().inner_id);
  }

  if (new_param_ref.kind() != prev_param_ref.kind()) {
    return false;
  }

  if (new_param_ref.Is<SemIR::AnyBindName>()) {
    new_param_ref =
        context.insts().Get(new_param_ref.As<SemIR::AnyBindName>().value_id);
    prev_param_ref =
        context.insts().Get(prev_param_ref.As<SemIR::AnyBindName>().value_id);
  }

  CARBON_CHECK(new_param_ref.Is<SemIR::Param>()) << new_param_ref;
  CARBON_CHECK(prev_param_ref.Is<SemIR::Param>()) << prev_param_ref;

  auto new_param = new_param_ref.As<SemIR::Param>();
  auto prev_param = prev_param_ref.As<SemIR::Param>();
  return new_param.name_id == prev_param.name_id;
}

// Returns true if two param refs agree.
static auto DoParamAgrees(Context& context, SemIR::InstId new_decl_id,
                          SemIR::InstBlockId new_param_refs_id,
                          SemIR::InstId prev_decl_id,
                          SemIR::InstBlockId prev_param_refs_id,
                          llvm::StringLiteral param_diag_label) -> bool {
  // This will often occur for empty params.
  if (new_param_refs_id == prev_param_refs_id) {
    return true;
  }
  const auto new_param_ref_ids = context.inst_blocks().Get(new_param_refs_id);
  const auto prev_param_ref_ids = context.inst_blocks().Get(prev_param_refs_id);
  if (new_param_ref_ids.size() != prev_param_ref_ids.size()) {
    CARBON_DIAGNOSTIC(FunctionSignatureParamCountDisagree, Error,
                      "Function declared with {0}{1} parameter(s).", int32_t,
                      llvm::StringLiteral);
    CARBON_DIAGNOSTIC(FunctionSignatureParamCountPrevious, Note,
                      "Function previously declared with {0}{1} parameter(s).",
                      int32_t, llvm::StringLiteral);
    context.emitter()
        .Build(new_decl_id, FunctionSignatureParamCountDisagree,
               new_param_ref_ids.size(), param_diag_label)
        .Note(prev_decl_id, FunctionSignatureParamCountPrevious,
              prev_param_ref_ids.size(), param_diag_label)
        .Emit();
    return false;
  }
  for (auto [index, new_param_ref_id, prev_param_ref_id] :
       llvm::enumerate(new_param_ref_ids, prev_param_ref_ids)) {
    if (!DoesParamAgree(context, new_param_ref_id, prev_param_ref_id)) {
      CARBON_DIAGNOSTIC(FunctionSignatureParamDisagree, Error,
                        "Declaration of{1} parameter {0} disagrees.", int32_t,
                        llvm::StringLiteral);
      CARBON_DIAGNOSTIC(FunctionSignatureParamPrevious, Note,
                        "Previous declaration of{1} parameter {0}.", int32_t,
                        llvm::StringLiteral);
      context.emitter()
          .Build(new_param_ref_id, FunctionSignatureParamDisagree,
                 new_param_ref_ids.size(), param_diag_label)
          .Note(prev_param_ref_id, FunctionSignatureParamPrevious,
                prev_param_ref_ids.size(), param_diag_label)
          .Emit();
      return false;
    }
  }
  return true;
}

// Returns true if the provided function signatures agrees, in the sense that
// declarations can be merged.
static auto DoesFunctionSignatureAgree(Context& context,
                                       const SemIR::Function& new_function,
                                       const SemIR::Function& prev_function)
    -> bool {
  if (FunctionSignatureHasError(context, new_function) ||
      FunctionSignatureHasError(context, prev_function)) {
    return false;
  }
  if (!DoParamAgrees(context, new_function.decl_id,
                     new_function.implicit_param_refs_id, prev_function.decl_id,
                     prev_function.implicit_param_refs_id, " implicit") ||
      !DoParamAgrees(context, new_function.decl_id, new_function.param_refs_id,
                     prev_function.decl_id, prev_function.param_refs_id, "")) {
    return false;
  }
  if (new_function.return_type_id != prev_function.return_type_id) {
    CARBON_DIAGNOSTIC(FunctionSignatureReturnTypeDisagree, Error,
                      "Function declares a return type of `{0}`.",
                      SemIR::TypeId);
    CARBON_DIAGNOSTIC(FunctionSignatureReturnTypeDisagreeNoReturn, Error,
                      "Function declared with no return type.");
    auto diag =
        new_function.return_type_id.is_valid()
            ? context.emitter().Build(new_function.decl_id,
                                      FunctionSignatureReturnTypeDisagree,
                                      new_function.return_type_id)
            : context.emitter().Build(
                  new_function.decl_id,
                  FunctionSignatureReturnTypeDisagreeNoReturn);
    if (prev_function.return_type_id.is_valid()) {
      CARBON_DIAGNOSTIC(FunctionSignatureReturnTypePrevious, Note,
                        "Function previously declared with return type `{0}`.",
                        SemIR::TypeId);
      diag.Note(prev_function.decl_id, FunctionSignatureReturnTypePrevious,
                prev_function.return_type_id);
    } else {
      CARBON_DIAGNOSTIC(FunctionSignatureReturnTypePreviousNoReturn, Note,
                        "Function previously declared with no return type.");
      diag.Note(prev_function.decl_id,
                FunctionSignatureReturnTypePreviousNoReturn);
    }
    diag.Emit();
    return false;
  }

  return true;
}

auto MergeFunctionDecl(Context& context, Parse::NodeId parse_node,
                       SemIR::Function& new_function,
                       SemIR::FunctionId prev_function_id, bool is_definition)
    -> bool {
  auto& prev_function = context.functions().Get(prev_function_id);

  // TODO: Disallow redeclarations within classes?
  if (!DoesFunctionSignatureAgree(context, new_function, prev_function)) {
    return false;
  }

  if (!is_definition) {
    CARBON_DIAGNOSTIC(FunctionRedeclaration, Error,
                      "Redeclaration of function {0}.", SemIR::NameId);
    CARBON_DIAGNOSTIC(FunctionPreviousDeclaration, Note,
                      "Previous declaration was here.");
    context.emitter()
        .Build(parse_node, FunctionRedeclaration, prev_function.name_id)
        .Note(prev_function.decl_id, FunctionPreviousDeclaration)
        .Emit();
    // The diagnostic doesn't prevent a merge.
    return true;
  }

  if (prev_function.definition_id.is_valid()) {
    CARBON_DIAGNOSTIC(FunctionRedefinition, Error,
                      "Redefinition of function {0}.", SemIR::NameId);
    CARBON_DIAGNOSTIC(FunctionPreviousDefinition, Note,
                      "Previous definition was here.");
    context.emitter()
        .Build(parse_node, FunctionRedefinition, prev_function.name_id)
        .Note(prev_function.definition_id, FunctionPreviousDefinition)
        .Emit();
    // The second definition will be unused as a consequence of the error.
    return true;
  }

  // Track the signature from the definition, so that IDs in the body
  // match IDs in the signature.
  prev_function.definition_id = new_function.definition_id;
  prev_function.implicit_param_refs_id = new_function.implicit_param_refs_id;
  prev_function.param_refs_id = new_function.param_refs_id;
  prev_function.return_type_id = new_function.return_type_id;
  prev_function.return_slot_id = new_function.return_slot_id;
  return true;
}

}  // namespace Carbon::Check
