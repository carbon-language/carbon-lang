// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp_overload_resolution.h"

#include "clang/Sema/Overload.h"
#include "clang/Sema/Sema.h"
#include "toolchain/check/cpp_import.h"
#include "toolchain/check/cpp_type_mapping.h"

namespace Carbon::Check {

auto PerformCppOverloadResolution(Context& context, SemIR::LocId loc_id,
                                  SemIR::CppOverloadSetId overload_set_id,
                                  llvm::ArrayRef<SemIR::InstId> arg_ids)
    -> std::optional<SemIR::InstId> {
  Diagnostics::AnnotationScope annotate_diagnostics(
      &context.emitter(), [&](auto& builder) {
        CARBON_DIAGNOSTIC(InCallToCppFunction, Note,
                          "in call to Cpp function here");
        builder.Note(loc_id, InCallToCppFunction);
      });

  // Map Carbon call argument types to C++ types.
  llvm::SmallVector<clang::Expr*> arg_exprs;
  arg_exprs.reserve(arg_ids.size());
  for (SemIR::InstId arg_id : arg_ids) {
    clang::QualType arg_cpp_type = MapToCppType(context, arg_id);
    if (arg_cpp_type.isNull()) {
      CARBON_DIAGNOSTIC(CppCallArgTypeNotSupported, Error,
                        "call argument of type {0} is not supported",
                        TypeOfInstId);
      context.emitter().Emit(loc_id, CppCallArgTypeNotSupported, arg_id);
      return std::nullopt;
    }
    // TODO: Allocate these on the stack.
    arg_exprs.emplace_back(new (context.ast_context()) clang::OpaqueValueExpr(
        // TODO: Add location accordingly.
        clang::SourceLocation(), arg_cpp_type.getNonReferenceType(),
        clang::ExprValueKind::VK_LValue));
  }

  const SemIR::CppOverloadSet& overload_set =
      context.cpp_overload_sets().Get(overload_set_id);

  // Add candidate functions from the name lookup.
  clang::OverloadCandidateSet candidate_set(
      // TODO: Add location accordingly.
      clang::SourceLocation(),
      clang::OverloadCandidateSet::CandidateSetKind::CSK_Normal);

  clang::ASTUnit* ast = context.sem_ir().clang_ast_unit();
  CARBON_CHECK(ast);
  clang::Sema& sema = ast->getSema();

  // TODO: Add support for method calls.
  for (clang::NamedDecl* candidate : overload_set.candidate_functions) {
    if (auto* fn_decl = dyn_cast<clang::FunctionDecl>(candidate)) {
      sema.AddOverloadCandidate(
          fn_decl, clang::DeclAccessPair::make(fn_decl, candidate->getAccess()),
          arg_exprs, candidate_set);
    } else if (isa<clang::FunctionTemplateDecl>(candidate)) {
      CARBON_DIAGNOSTIC(CppTemplateFunctionNotSupported, Error,
                        "template function is not supported");
      context.emitter().Emit(loc_id, CppTemplateFunctionNotSupported);
      return std::nullopt;
    }
    // TODO: Diagnose if it's neither of these types.
  }

  // Find best viable function among the candidates.
  clang::OverloadCandidateSet::iterator best_viable_fn;
  clang::OverloadingResult overloading_result =
      // TODO: Add location accordingly.
      candidate_set.BestViableFunction(sema, clang::SourceLocation(),
                                       best_viable_fn);

  switch (overloading_result) {
    case clang::OverloadingResult::OR_Success: {
      // TODO: Handle the cases when Function is null.
      CARBON_CHECK(best_viable_fn->Function);
      SemIR::InstId result =
          ImportCppFunctionDecl(context, loc_id, best_viable_fn->Function);
      return result;
    }
    case clang::OverloadingResult::OR_No_Viable_Function: {
      // TODO: Add notes with the candidates.
      CARBON_DIAGNOSTIC(CppOverloadingNoViableFunctionFound, Error,
                        "no matching function for call to `{0}`",
                        SemIR::NameId);
      context.emitter().Emit(loc_id, CppOverloadingNoViableFunctionFound,
                             overload_set.name_id);
      return std::nullopt;
    }
    case clang::OverloadingResult::OR_Ambiguous: {
      // TODO: Add notes with the candidates.
      CARBON_DIAGNOSTIC(CppOverloadingAmbiguousCandidatesFound, Error,
                        "call to `{0}` is ambiguous", SemIR::NameId);
      context.emitter().Emit(loc_id, CppOverloadingAmbiguousCandidatesFound,
                             overload_set.name_id);
      return std::nullopt;
    }
    case clang::OverloadingResult::OR_Deleted: {
      // TODO: Add notes with the candidates.
      CARBON_DIAGNOSTIC(CppOverloadingDeletedFunctionFound, Error,
                        "call to deleted function `{0}`", SemIR::NameId);
      context.emitter().Emit(loc_id, CppOverloadingDeletedFunctionFound,
                             overload_set.name_id);
      return std::nullopt;
    }
  }
}

}  // namespace Carbon::Check
