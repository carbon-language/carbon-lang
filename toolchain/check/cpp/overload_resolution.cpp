// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/overload_resolution.h"

#include "clang/Basic/DiagnosticSema.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Sema.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/cpp/access.h"
#include "toolchain/check/cpp/import.h"
#include "toolchain/check/cpp/location.h"
#include "toolchain/check/cpp/operators.h"
#include "toolchain/check/cpp/type_mapping.h"
#include "toolchain/check/member_access.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/name_scope.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Map a Carbon name into a C++ name.
static auto GetCppName(Context& context, SemIR::NameId name_id)
    -> clang::DeclarationName {
  // TODO: Some special names should probably use different formatting. In
  // particular, NameId::CppOperator should probably map back to a
  // CXXOperatorName.
  auto name_str = context.names().GetFormatted(name_id);
  return clang::DeclarationName(&context.ast_context().Idents.get(name_str));
}

// Adds the given overload candidates to the candidate set.
static auto AddOverloadCandidates(clang::Sema& sema,
                                  clang::OverloadCandidateSet& candidate_set,
                                  const clang::UnresolvedSet<4>& functions,
                                  clang::Expr* self_arg,
                                  llvm::ArrayRef<clang::Expr*> args) -> void {
  constexpr bool SuppressUserConversions = false;
  constexpr bool PartialOverloading = false;
  constexpr clang::TemplateArgumentListInfo* ExplicitTemplateArgs = nullptr;

  for (auto found_decl : functions.pairs()) {
    auto* decl = found_decl->getUnderlyingDecl();
    auto* template_decl = dyn_cast<clang::FunctionTemplateDecl>(decl);
    auto* fn_decl = template_decl ? template_decl->getTemplatedDecl()
                                  : cast<clang::FunctionDecl>(decl);
    auto* method_decl = dyn_cast<clang::CXXMethodDecl>(fn_decl);
    if (method_decl && !method_decl->isStatic() &&
        !isa<clang::CXXConstructorDecl>(fn_decl)) {
      clang::QualType self_type;
      clang::Expr::Classification self_classification;
      if (self_arg) {
        self_type = self_arg->getType();
        self_classification = self_arg->Classify(sema.Context);
      }
      if (template_decl) {
        sema.AddMethodTemplateCandidate(
            template_decl, found_decl,
            cast<clang::CXXRecordDecl>(template_decl->getDeclContext()),
            ExplicitTemplateArgs, self_type, self_classification, args,
            candidate_set, SuppressUserConversions, PartialOverloading);
      } else if (method_decl->isOverloadedOperator()) {
        sema.AddMemberOperatorCandidates(method_decl->getOverloadedOperator(),
                                         candidate_set.getLocation(), args,
                                         candidate_set);
      } else {
        sema.AddMethodCandidate(method_decl, found_decl,
                                method_decl->getParent(), self_type,
                                self_classification, args, candidate_set,
                                SuppressUserConversions, PartialOverloading);
      }
    } else if (template_decl) {
      sema.AddTemplateOverloadCandidate(
          template_decl, found_decl, ExplicitTemplateArgs, args, candidate_set,
          SuppressUserConversions, PartialOverloading);
    } else {
      sema.AddOverloadCandidate(fn_decl, found_decl, args, candidate_set,
                                SuppressUserConversions, PartialOverloading);
    }
  }
}

// Checks whether a selected overload is accessible and diagnoses if not.
static auto CheckOverloadAccess(Context& context, SemIR::LocId loc_id,
                                const SemIR::CppOverloadSet& overload_set,
                                clang::DeclAccessPair overload,
                                SemIR::InstId overload_inst_id) -> void {
  SemIR::AccessKind member_access_kind = MapCppAccess(overload);
  if (member_access_kind == SemIR::AccessKind::Public) {
    return;
  }

  auto name_scope_const_id = context.constant_values().Get(
      context.name_scopes().Get(overload_set.parent_scope_id).inst_id());
  SemIR::AccessKind allowed_access_kind =
      GetHighestAllowedAccess(context, loc_id, name_scope_const_id);
  CheckAccess(context, loc_id, SemIR::LocId(overload_inst_id),
              overload_set.name_id, member_access_kind,
              /*is_parent_access=*/false,
              {.constant_id = name_scope_const_id,
               .highest_allowed_access = allowed_access_kind});
}

auto PerformCppOverloadResolution(Context& context, SemIR::LocId loc_id,
                                  SemIR::CppOverloadSetId overload_set_id,
                                  SemIR::InstId self_id,
                                  llvm::ArrayRef<SemIR::InstId> arg_ids)
    -> SemIR::InstId {
  // Register an annotation scope to flush any Clang diagnostics when we return.
  // This is important to ensure that Clang diagnostics are properly interleaved
  // with Carbon diagnostics.
  Diagnostics::AnnotationScope annotate_diagnostics(&context.emitter(),
                                                    [](auto& /*builder*/) {});

  // Map Carbon call argument types to C++ types.
  clang::Expr* self_expr = nullptr;
  if (self_id.has_value()) {
    self_expr = InventClangArg(context, self_id);
    if (!self_expr) {
      return SemIR::ErrorInst::InstId;
    }
  }
  auto maybe_arg_exprs = InventClangArgs(context, arg_ids);
  if (!maybe_arg_exprs.has_value()) {
    return SemIR::ErrorInst::InstId;
  }
  auto& arg_exprs = *maybe_arg_exprs;

  const SemIR::CppOverloadSet& overload_set =
      context.cpp_overload_sets().Get(overload_set_id);

  clang::SourceLocation loc = GetCppLocation(context, loc_id);

  // Add candidate functions from the name lookup.
  clang::OverloadCandidateSet candidate_set(
      loc,
      overload_set.operator_rewrite_info.OriginalOperator
          ? clang::OverloadCandidateSet::CandidateSetKind::CSK_Operator
          : clang::OverloadCandidateSet::CandidateSetKind::CSK_Normal,
      overload_set.operator_rewrite_info);

  clang::Sema& sema = context.clang_sema();

  AddOverloadCandidates(sema, candidate_set, overload_set.candidate_functions,
                        self_expr, arg_exprs);

  // Find best viable function among the candidates.
  clang::OverloadCandidateSet::iterator best_viable_fn;
  clang::OverloadingResult overloading_result =
      candidate_set.BestViableFunction(sema, loc, best_viable_fn);

  switch (overloading_result) {
    case clang::OverloadingResult::OR_Success: {
      // TODO: Handle the cases when Function is null.
      CARBON_CHECK(best_viable_fn->Function);
      if (best_viable_fn->RewriteKind) {
        context.TODO(
            loc_id,
            llvm::formatv("Rewriting operator{0} using {1} is not supported",
                          clang::getOperatorSpelling(
                              candidate_set.getRewriteInfo().OriginalOperator),
                          best_viable_fn->Function->getNameAsString()));
        return SemIR::ErrorInst::InstId;
      }
      sema.MarkFunctionReferenced(loc, best_viable_fn->Function);
      SemIR::InstId result_id = ImportCppFunctionDecl(
          context, loc_id, best_viable_fn->Function,
          // If this is an operator method, the first arg will be used as self.
          arg_exprs.size() -
              (IsCppOperatorMethodDecl(best_viable_fn->Function) ? 1 : 0));
      CheckOverloadAccess(context, loc_id, overload_set,
                          best_viable_fn->FoundDecl, result_id);
      return result_id;
    }
    case clang::OverloadingResult::OR_No_Viable_Function: {
      candidate_set.NoteCandidates(
          clang::PartialDiagnosticAt(
              loc, sema.PDiag(clang::diag::err_ovl_no_viable_function_in_call)
                       << GetCppName(context, overload_set.name_id)),
          sema, clang::OCD_AllCandidates, arg_exprs);
      return SemIR::ErrorInst::InstId;
    }
    case clang::OverloadingResult::OR_Ambiguous: {
      candidate_set.NoteCandidates(
          clang::PartialDiagnosticAt(
              loc, sema.PDiag(clang::diag::err_ovl_ambiguous_call)
                       << GetCppName(context, overload_set.name_id)),
          sema, clang::OCD_AmbiguousCandidates, arg_exprs);
      return SemIR::ErrorInst::InstId;
    }
    case clang::OverloadingResult::OR_Deleted: {
      sema.DiagnoseUseOfDeletedFunction(
          loc, clang::SourceRange(loc, loc),
          GetCppName(context, overload_set.name_id), candidate_set,
          best_viable_fn->Function, arg_exprs);
      return SemIR::ErrorInst::InstId;
    }
  }
}

}  // namespace Carbon::Check
