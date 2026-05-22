// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/overload_resolution.h"

#include "clang/AST/DeclCXX.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Sema.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/cpp/access.h"
#include "toolchain/check/cpp/call.h"
#include "toolchain/check/cpp/import.h"
#include "toolchain/check/cpp/location.h"
#include "toolchain/check/cpp/operators.h"
#include "toolchain/check/cpp/type_mapping.h"
#include "toolchain/check/member_access.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/diagnostics/emitter.h"
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
static auto AddOverloadCandidates(
    Context& context, clang::OverloadCandidateSet& candidate_set,
    const clang::UnresolvedSet<4>& functions,
    llvm::ArrayRef<SemIR::InstId> template_arg_ids, clang::Expr* self_arg,
    llvm::ArrayRef<clang::Expr*> args) -> void {
  clang::Sema& sema = context.clang_sema();

  constexpr bool SuppressUserConversions = false;
  constexpr bool PartialOverloading = false;

  for (auto found_decl : functions.pairs()) {
    auto* decl = found_decl->getUnderlyingDecl();

    // Form an explicit template argument list if needed. Note that this is done
    // per-candidate, as the conversions performed on the template arguments
    // differ based on the corresponding template parameters.
    auto* template_decl = dyn_cast<clang::FunctionTemplateDecl>(decl);
    clang::TemplateArgumentListInfo explicit_template_arg_storage;
    clang::TemplateArgumentListInfo* explicit_template_args = nullptr;
    if (!template_arg_ids.empty()) {
      if (!template_decl) {
        continue;
      }
      if (!ConvertArgsToTemplateArgs(context, template_decl, template_arg_ids,
                                     explicit_template_arg_storage,
                                     /*diagnose=*/false)) {
        continue;
      }
      explicit_template_args = &explicit_template_arg_storage;
    }

    auto* fn_decl = template_decl ? template_decl->getTemplatedDecl()
                                  : cast<clang::FunctionDecl>(decl);
    if (IsObjectMemberFunction(*fn_decl)) {
      auto* method_decl = cast<clang::CXXMethodDecl>(fn_decl);
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
            explicit_template_args, self_type, self_classification, args,
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
          template_decl, found_decl, explicit_template_args, args,
          candidate_set, SuppressUserConversions, PartialOverloading);
    } else {
      sema.AddOverloadCandidate(fn_decl, found_decl, args, candidate_set,
                                SuppressUserConversions, PartialOverloading);
    }
  }
}

auto CheckCppOverloadAccess(
    Context& context, SemIR::LocId loc_id, clang::DeclAccessPair overload,
    SemIR::KnownInstId<SemIR::FunctionDecl> overload_inst_id,
    SemIR::NameScopeId parent_scope_id) -> void {
  SemIR::AccessKind member_access_kind = MapCppAccess(overload);
  if (member_access_kind == SemIR::AccessKind::Public) {
    return;
  }

  auto function_id = context.insts().Get(overload_inst_id).function_id;
  auto& function = context.functions().Get(function_id);
  if (!parent_scope_id.has_value()) {
    parent_scope_id = function.parent_scope_id;
  }

  auto name_scope_const_id = context.constant_values().Get(
      context.name_scopes().Get(parent_scope_id).inst_id());
  SemIR::AccessKind allowed_access_kind =
      GetHighestAllowedAccess(context, loc_id, name_scope_const_id);
  CheckAccess(context, loc_id, SemIR::LocId(overload_inst_id), function.name_id,
              member_access_kind,
              /*is_parent_access=*/false,
              {.constant_id = name_scope_const_id,
               .highest_allowed_access = allowed_access_kind});
}

// Computes the passing mode for a C++ function parameter that is a reference.
static auto ComputePassingModeForReferenceBinding(
    const clang::StandardConversionSequence& scs)
    -> SemIR::ClangDeclSignature::PassingMode {
  CARBON_CHECK(scs.ReferenceBinding);
  auto pointee_type = scs.getToType(2);
  if (pointee_type.isConstQualified() ||
      (scs.IsLvalueReference && scs.BindsToRvalue)) {
    // Reference to const is always mapped to Carbon pass by value. A non-const
    // lvalue reference bound to an rvalue only happens when initializing an
    // object parameter with no ref-qualifier from an rvalue, which we also
    // model as pass-by-value.
    return SemIR::ClangDeclSignature::PassingMode::ByValue;
  }
  // Rvalue reference to non-const is passed as a `var` to force a copy or move
  // in the caller. Lvalue reference to non-const is passed by reference.
  return scs.IsLvalueReference ? SemIR::ClangDeclSignature::PassingMode::ByRef
                               : SemIR::ClangDeclSignature::PassingMode::ByVar;
}

// Returns whether move-construction of type `type` is known to be equivalent to
// a copy. If so, it's safe to map C++ pass-by-value into Carbon pass-by-value
// instead of pass-by-var.
static auto IsMoveEquivalentToCopy(clang::QualType type) {
  // We can pass by copy instead of by move if:
  // - The type is not a class type.
  auto* record_decl = type->getAsCXXRecordDecl();
  if (!record_decl) {
    return true;
  }

  // - The move constructor is defaulted and deleted or non-existent, in
  //   which case overload resolution for a move will call the copy
  //   constructor.
  if (!record_decl->hasMoveConstructor() ||
      (!record_decl->hasUserDeclaredMoveConstructor() &&
       record_decl->defaultedMoveConstructorIsDeleted())) {
    return true;
  }

  // - Both move and copy are trivial and not deleted, in which case they
  //   are equivalent.
  if (record_decl->hasTrivialMoveConstructor() &&
      !record_decl->defaultedMoveConstructorIsDeleted() &&
      record_decl->hasTrivialCopyConstructor() &&
      !record_decl->defaultedCopyConstructorIsDeleted()) {
    return true;
  }

  // Otherwise we need a move, so we pass by var.
  return false;
}

auto GetPassingModeForCppParameter(const clang::ImplicitConversionSequence& ics,
                                   const clang::Expr* arg_expr)
    -> SemIR::ClangDeclSignature::PassingMode {
  if (ics.isStandard()) {
    const auto& scs = ics.Standard;
    if (scs.ReferenceBinding) {
      return ComputePassingModeForReferenceBinding(scs);
    }

    // Most standard conversions can be mapped to Carbon pass by value. The
    // exception is where the source is an initializing expression of record
    // type, which we map to pass by var, unless a copy would do the same thing.
    if (arg_expr->isXValue() && !IsMoveEquivalentToCopy(arg_expr->getType())) {
      return SemIR::ClangDeclSignature::PassingMode::ByVar;
    }

    return SemIR::ClangDeclSignature::PassingMode::ByValue;
  }

  if (ics.isUserDefined()) {
    const auto& ucs = ics.UserDefined;
    if (ucs.After.ReferenceBinding) {
      return ComputePassingModeForReferenceBinding(ucs.After);
    }

    const auto* ctor =
        dyn_cast_or_null<clang::CXXConstructorDecl>(ucs.ConversionFunction);
    if (ctor && ctor->isCopyConstructor()) {
      // Overload resolution wanted to call a copy constructor to initialize
      // this parameter. Pass by value instead; we'll copy in the thunk.
      return SemIR::ClangDeclSignature::PassingMode::ByValue;
    }

    // We're calling a user-defined conversion, so we're performing
    // initialization. Pass by move unless the type being initialized doesn't
    // distinguish moves and copies.
    return IsMoveEquivalentToCopy(ucs.After.getToType(2))
               ? SemIR::ClangDeclSignature::PassingMode::ByValue
               : SemIR::ClangDeclSignature::PassingMode::ByVar;
  }

  // TODO: Support ellipsis conversion sequences.
  CARBON_FATAL("Unexpected kind of implicit conversion sequence");
}

// Computes the signature for a C++ function candidate based on the conversions
// performed on the arguments.
auto ComputeClangDeclSignatureFromBestViableFunction(
    Context& context, clang::OverloadCandidateSet::iterator candidate,
    clang::Expr* self_expr, llvm::ArrayRef<clang::Expr*> arg_exprs,
    SemIR::ClangDeclSignature::Kind kind) -> SemIR::ClangDeclSignatureId {
  SemIR::ClangDeclSignature signature;
  signature.kind = kind;
  signature.num_params = static_cast<int32_t>(arg_exprs.size());
  signature.passing_modes.reserve(signature.num_params);

  for (auto [i, arg_expr] : llvm::enumerate(arg_exprs)) {
    // Compute which conversion sequence corresponds to this argument.
    // TODO: Clang should expose a way to compute this.
    int conversion_index = i;
    if (isa<clang::CXXMethodDecl>(candidate->Function) &&
        !isa<clang::CXXConstructorDecl>(candidate->Function)) {
      // Methods (both static and non-static, but not constructors) get an
      // object parameter conversion at index 0.
      ++conversion_index;
    }

    signature.passing_modes.push_back(GetPassingModeForCppParameter(
        candidate->Conversions[conversion_index], arg_expr));
  }

  if (IsObjectMemberFunction(*candidate->Function)) {
    signature.self_passing_mode =
        GetPassingModeForCppParameter(candidate->Conversions[0], self_expr);
  }

  return context.clang_decl_signatures().Add(std::move(signature));
}

auto PerformCppOverloadResolution(
    Context& context, SemIR::LocId loc_id,
    const SemIR::CppOverloadSet& overload_set,
    llvm::ArrayRef<SemIR::InstId> template_arg_ids, SemIR::InstId self_id,
    llvm::ArrayRef<SemIR::InstId> arg_ids) -> SemIR::InstId {
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

  clang::SourceLocation loc = GetCppLocation(context, loc_id);

  // Add candidate functions from the name lookup.
  clang::OverloadCandidateSet candidate_set(
      loc,
      overload_set.operator_rewrite_info.OriginalOperator
          ? clang::OverloadCandidateSet::CandidateSetKind::CSK_Operator
          : clang::OverloadCandidateSet::CandidateSetKind::CSK_Normal,
      overload_set.operator_rewrite_info);

  AddOverloadCandidates(context, candidate_set,
                        overload_set.candidate_functions, template_arg_ids,
                        self_expr, arg_exprs);

  // Find best viable function among the candidates.
  clang::Sema& sema = context.clang_sema();
  clang::OverloadCandidateSet::iterator best_viable_fn;
  clang::OverloadingResult overloading_result =
      candidate_set.BestViableFunction(sema, loc, best_viable_fn);

  switch (overloading_result) {
    case clang::OverloadingResult::OR_Success: {
      CARBON_CHECK(best_viable_fn->Function);
      CARBON_CHECK(!best_viable_fn->RewriteKind);
      SemIR::ClangDeclSignatureId signature_id =
          ComputeClangDeclSignatureFromBestViableFunction(
              context, best_viable_fn, self_expr, arg_exprs);

      SemIR::InstId result_id = ImportCppFunctionDecl(
          context, loc_id, best_viable_fn->Function, signature_id);
      if (result_id != SemIR::ErrorInst::InstId) {
        CheckCppOverloadAccess(
            context, loc_id, best_viable_fn->FoundDecl,
            context.insts().GetAsKnownInstId<SemIR::FunctionDecl>(result_id),
            overload_set.parent_scope_id);
      }
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
