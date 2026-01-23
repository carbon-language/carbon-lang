// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/operators.h"

#include "clang/Sema/Overload.h"
#include "clang/Sema/Sema.h"
#include "toolchain/check/core_identifier.h"
#include "toolchain/check/cpp/import.h"
#include "toolchain/check/cpp/location.h"
#include "toolchain/check/cpp/overload_resolution.h"
#include "toolchain/check/cpp/type_mapping.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Maps Carbon operator interface and operator names to Clang operator kinds.
static auto GetClangOperatorKind(Context& context, SemIR::LocId loc_id,
                                 CoreIdentifier interface_name,
                                 CoreIdentifier op_name)
    -> std::optional<clang::OverloadedOperatorKind> {
  switch (interface_name) {
      // Unary operators.
    case CoreIdentifier::Destroy:
    case CoreIdentifier::As:
    case CoreIdentifier::ImplicitAs:
    case CoreIdentifier::UnsafeAs:
    case CoreIdentifier::Copy: {
      // TODO: Support destructors and conversions.
      return std::nullopt;
    }

    // Increment and decrement.
    case CoreIdentifier::Inc: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_PlusPlus;
    }
    case CoreIdentifier::Dec: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_MinusMinus;
    }

    // Arithmetic.
    case CoreIdentifier::Negate: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_Minus;
    }

    // Bitwise.
    case CoreIdentifier::BitComplement: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_Tilde;
    }

    // Binary operators.

    // Arithmetic operators.
    case CoreIdentifier::AddWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_Plus;
    }
    case CoreIdentifier::SubWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_Minus;
    }
    case CoreIdentifier::MulWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_Star;
    }
    case CoreIdentifier::DivWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_Slash;
    }
    case CoreIdentifier::ModWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_Percent;
    }

    // Bitwise operators.
    case CoreIdentifier::BitAndWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_Amp;
    }
    case CoreIdentifier::BitOrWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_Pipe;
    }
    case CoreIdentifier::BitXorWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_Caret;
    }
    case CoreIdentifier::LeftShiftWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_LessLess;
    }
    case CoreIdentifier::RightShiftWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_GreaterGreater;
    }

    // Assignment.
    case CoreIdentifier::AssignWith: {
      // TODO: This is not yet reached because we don't use the `AssignWith`
      // interface for assignment yet.
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_Equal;
    }

    // Compound assignment arithmetic operators.
    case CoreIdentifier::AddAssignWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_PlusEqual;
    }
    case CoreIdentifier::SubAssignWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_MinusEqual;
    }
    case CoreIdentifier::MulAssignWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_StarEqual;
    }
    case CoreIdentifier::DivAssignWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_SlashEqual;
    }
    case CoreIdentifier::ModAssignWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_PercentEqual;
    }

    // Compound assignment bitwise operators.
    case CoreIdentifier::BitAndAssignWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_AmpEqual;
    }
    case CoreIdentifier::BitOrAssignWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_PipeEqual;
    }
    case CoreIdentifier::BitXorAssignWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_CaretEqual;
    }
    case CoreIdentifier::LeftShiftAssignWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_LessLessEqual;
    }
    case CoreIdentifier::RightShiftAssignWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_GreaterGreaterEqual;
    }

    // Relational operators.
    case CoreIdentifier::EqWith: {
      if (op_name == CoreIdentifier::Equal) {
        return clang::OO_EqualEqual;
      }
      CARBON_CHECK(op_name == CoreIdentifier::NotEqual);
      return clang::OO_ExclaimEqual;
    }
    case CoreIdentifier::OrderedWith: {
      switch (op_name) {
        case CoreIdentifier::Less:
          return clang::OO_Less;
        case CoreIdentifier::Greater:
          return clang::OO_Greater;
        case CoreIdentifier::LessOrEquivalent:
          return clang::OO_LessEqual;
        case CoreIdentifier::GreaterOrEquivalent:
          return clang::OO_GreaterEqual;
        default:
          CARBON_FATAL("Unexpected OrderedWith op `{0}`", op_name);
      }
    }

    // Array indexing.
    case CoreIdentifier::IndexWith: {
      CARBON_CHECK(op_name == CoreIdentifier::At);
      return clang::OO_Subscript;
    }

    default: {
      context.TODO(loc_id, llvm::formatv("Unsupported operator interface `{0}`",
                                         interface_name));
      return std::nullopt;
    }
  }
}

auto LookupCppOperator(Context& context, SemIR::LocId loc_id, Operator op,
                       llvm::ArrayRef<SemIR::InstId> arg_ids) -> SemIR::InstId {
  // Register an annotation scope to flush any Clang diagnostics when we return.
  // This is important to ensure that Clang diagnostics are properly interleaved
  // with Carbon diagnostics.
  Diagnostics::AnnotationScope annotate_diagnostics(&context.emitter(),
                                                    [](auto& /*builder*/) {});

  auto op_kind =
      GetClangOperatorKind(context, loc_id, op.interface_name, op.op_name);
  if (!op_kind) {
    return SemIR::InstId::None;
  }

  // Make sure all operands are complete before lookup.
  for (SemIR::InstId arg_id : arg_ids) {
    SemIR::TypeId arg_type_id = context.insts().Get(arg_id).type_id();
    if (!RequireCompleteType(context, arg_type_id, loc_id, [&] {
          CARBON_DIAGNOSTIC(
              IncompleteOperandTypeInCppOperatorLookup, Error,
              "looking up a C++ operator with incomplete operand type {0}",
              SemIR::TypeId);
          return context.emitter().Build(
              loc_id, IncompleteOperandTypeInCppOperatorLookup, arg_type_id);
        })) {
      return SemIR::ErrorInst::InstId;
    }
  }

  auto maybe_arg_exprs = InventClangArgs(context, arg_ids);
  if (!maybe_arg_exprs.has_value()) {
    return SemIR::ErrorInst::InstId;
  }
  auto& arg_exprs = *maybe_arg_exprs;

  clang::SourceLocation loc = GetCppLocation(context, loc_id);
  clang::OverloadCandidateSet::OperatorRewriteInfo operator_rewrite_info(
      *op_kind, loc, /*AllowRewritten=*/true);
  clang::OverloadCandidateSet candidate_set(
      loc, clang::OverloadCandidateSet::CSK_Operator, operator_rewrite_info);

  clang::Sema& sema = context.clang_sema();

  // This works for both unary and binary operators.
  sema.LookupOverloadedBinOp(candidate_set, *op_kind, clang::UnresolvedSet<0>{},
                             arg_exprs);

  clang::OverloadCandidateSet::iterator best_viable_fn;
  switch (candidate_set.BestViableFunction(sema, loc, best_viable_fn)) {
    case clang::OverloadingResult::OR_Success: {
      if (!best_viable_fn->Function) {
        // The best viable candidate was a builtin. Let the Carbon operator
        // machinery handle that.
        return SemIR::InstId::None;
      }
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
      auto result_id = ImportCppFunctionDecl(
          context, loc_id, best_viable_fn->Function,
          // If this is an operator method, the first arg will be used as self.
          arg_ids.size() -
              (isa<clang::CXXMethodDecl>(best_viable_fn->Function) ? 1 : 0));
      if (result_id != SemIR::ErrorInst::InstId) {
        CheckCppOverloadAccess(
            context, loc_id, best_viable_fn->FoundDecl,
            context.insts().GetAsKnownInstId<SemIR::FunctionDecl>(result_id));
      }
      return result_id;
    }
    case clang::OverloadingResult::OR_No_Viable_Function: {
      // OK, didn't find a viable C++ candidate, but this is not an error, as
      // there might be a Carbon candidate.
      return SemIR::InstId::None;
    }
    case clang::OverloadingResult::OR_Ambiguous: {
      const char* spelling = clang::getOperatorSpelling(*op_kind);
      candidate_set.NoteCandidates(
          clang::PartialDiagnosticAt(
              loc, sema.PDiag(clang::diag::err_ovl_ambiguous_oper_binary)
                       << spelling << arg_exprs[0]->getType()
                       << arg_exprs[1]->getType()),
          sema, clang::OCD_AmbiguousCandidates, arg_exprs, spelling, loc);
      return SemIR::ErrorInst::InstId;
    }
    case clang::OverloadingResult::OR_Deleted:
      const char* spelling = clang::getOperatorSpelling(*op_kind);
      auto* message = best_viable_fn->Function->getDeletedMessage();
      // The best viable function might be a different operator if the best
      // candidate is a rewritten candidate, so use the operator kind of the
      // candidate itself in the diagnostic.
      candidate_set.NoteCandidates(
          clang::PartialDiagnosticAt(
              loc, sema.PDiag(clang::diag::err_ovl_deleted_oper)
                       << clang::getOperatorSpelling(
                              best_viable_fn->Function->getOverloadedOperator())
                       << (message != nullptr)
                       << (message ? message->getString() : llvm::StringRef())),
          sema, clang::OCD_AllCandidates, arg_exprs, spelling, loc);
      return SemIR::ErrorInst::InstId;
  }
}

auto IsCppOperatorMethodDecl(clang::Decl* decl) -> bool {
  auto* clang_method_decl = dyn_cast<clang::CXXMethodDecl>(decl);
  return clang_method_decl && clang_method_decl->isOverloadedOperator();
}

static auto GetAsCppFunctionDecl(Context& context, SemIR::InstId inst_id)
    -> clang::FunctionDecl* {
  auto function_type = context.types().TryGetAs<SemIR::FunctionType>(
      context.insts().Get(inst_id).type_id());
  if (!function_type) {
    return nullptr;
  }
  SemIR::ClangDeclId clang_decl_id =
      context.functions().Get(function_type->function_id).clang_decl_id;
  return clang_decl_id.has_value()
             ? dyn_cast<clang::FunctionDecl>(
                   context.clang_decls().Get(clang_decl_id).key.decl)
             : nullptr;
}

auto IsCppOperatorMethod(Context& context, SemIR::InstId inst_id) -> bool {
  auto* function_decl = GetAsCppFunctionDecl(context, inst_id);
  return function_decl && IsCppOperatorMethodDecl(function_decl);
}

auto IsCppConstructorOrNonMethodOperator(Context& context,
                                         SemIR::InstId inst_id) -> bool {
  auto* function_decl = GetAsCppFunctionDecl(context, inst_id);
  if (!function_decl) {
    return false;
  }
  if (isa<clang::CXXConstructorDecl>(function_decl)) {
    return true;
  }
  return !isa<clang::CXXMethodDecl>(function_decl) &&
         function_decl->isOverloadedOperator();
}

}  // namespace Carbon::Check
