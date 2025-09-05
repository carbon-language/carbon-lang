// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp_overload_resolution.h"

#include "clang/Sema/Overload.h"
#include "clang/Sema/Sema.h"
#include "toolchain/check/cpp_type_mapping.h"
#include "toolchain/check/import_cpp.h"

namespace Carbon::Check {

auto PerformCppOverloadResolution(Context& context, SemIR::LocId loc_id,
                                  SemIR::InstId callee_id,
                                  llvm::ArrayRef<SemIR::InstId> arg_ids)
    -> std::optional<SemIR::InstId> {
  Diagnostics::AnnotationScope annotate_diagnostics(
      &context.emitter(), [&](auto& builder) {
        CARBON_DIAGNOSTIC(InCallToCppFunction, Note,
                          "in call to a Cpp function here");
        builder.Note(loc_id, InCallToCppFunction);
      });

  // Map Carbon call argument types to C++ types.
  llvm::SmallVector<clang::Expr*> arg_exprs;
  arg_exprs.reserve(arg_ids.size());
  for (SemIR::InstId arg_id : arg_ids) {
    auto arg_cpp_type = MapToCppType(context, arg_id);
    if (!arg_cpp_type) {
      CARBON_DIAGNOSTIC(CallArgTypeNotSupported, Error,
                        "call arg of type {0} is not supported", TypeOfInstId);
      context.emitter().Emit(loc_id, CallArgTypeNotSupported, arg_id);
      return std::nullopt;
    }
    auto* arg_expr = new (context.ast_context()) clang::OpaqueValueExpr(
        // TODO: Add location accordingly.
        clang::SourceLocation(), arg_cpp_type->getNonReferenceType(),
        clang::ExprValueKind::VK_LValue);
    arg_exprs.emplace_back(arg_expr);
  }

  auto overload_set_type_inst =
      context.types().GetAsInst(context.insts().Get(callee_id).type_id());
  auto overload_set_type =
      overload_set_type_inst.TryAs<SemIR::CppOverloadSetType>();
  if (!overload_set_type) {
    return std::nullopt;
  }
  const SemIR::CppOverloadSet& overload_set =
      context.cpp_overload_sets().Get(overload_set_type->overload_set_id);

  // Add candidate functions from the name lookup.
  clang::OverloadCandidateSet candidate_set(
      // TODO: Add location accordingly.
      clang::SourceLocation(),
      clang::OverloadCandidateSet::CandidateSetKind::CSK_Normal);

  clang::ASTUnit* ast = context.sem_ir().clang_ast_unit();
  CARBON_CHECK(ast);
  clang::Sema& sema = ast->getSema();

  for (clang::NamedDecl* candidate : overload_set.candidate_functions) {
    if (auto* fn_decl = dyn_cast<clang::FunctionDecl>(candidate)) {
      sema.AddOverloadCandidate(
          fn_decl, clang::DeclAccessPair::make(fn_decl, candidate->getAccess()),
          arg_exprs, candidate_set);
    }
  }

  // Find best viable function among the candidates.
  // Note: In C++, a single non-templated function is also treated as an
  // overloaded set and goes through the overload resolution to ensure that the
  // function is viable for the call. Keeping the same behavior here for
  // consistency.
  clang::OverloadCandidateSet::iterator best_viable_fn;
  clang::OverloadingResult overloading_result =
      // TODO: Add location accordingly.
      candidate_set.BestViableFunction(sema, clang::SourceLocation(),
                                       best_viable_fn);

  switch (overloading_result) {
    case clang::OverloadingResult::OR_Success: {
      SemIR::InstId result =
          ImportFunctionDecl(context, loc_id, best_viable_fn->Function);
      return result;
    }
    case clang::OverloadingResult::OR_No_Viable_Function: {
      CARBON_DIAGNOSTIC(
          OverloadingNoViableFunctionFound, Error,
          "no viable function found during overloading resolution");
      context.emitter().Emit(loc_id, OverloadingNoViableFunctionFound);
      return std::nullopt;
    }
    case clang::OverloadingResult::OR_Ambiguous: {
      CARBON_DIAGNOSTIC(
          OverloadingAmbiguousCandidatesFound, Error,
          "ambiguous candidates found during overloading resolution");
      context.emitter().Emit(loc_id, OverloadingAmbiguousCandidatesFound);
      return std::nullopt;
    }
    case clang::OverloadingResult::OR_Deleted: {
      CARBON_DIAGNOSTIC(
          OverloadingDeletedFunctionFound, Error,
          "overloading resolution succeeded, but refers to a deleted function");
      context.emitter().Emit(loc_id, OverloadingDeletedFunctionFound);
      return std::nullopt;
    }
  }
}

}  // namespace Carbon::Check
