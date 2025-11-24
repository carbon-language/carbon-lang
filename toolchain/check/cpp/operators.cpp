// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/operators.h"

#include "clang/Sema/Overload.h"
#include "clang/Sema/Sema.h"
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
                                 llvm::StringLiteral interface_name,
                                 llvm::StringLiteral op_name)
    -> std::optional<clang::OverloadedOperatorKind> {
  // Unary operators.
  if (interface_name == "Destroy" || interface_name == "As" ||
      interface_name == "ImplicitAs") {
    // TODO: Support destructors and conversions.
    return std::nullopt;
  }

  // Increment and Decrement.
  if (interface_name == "Inc") {
    CARBON_CHECK(op_name == "Op");
    return clang::OO_PlusPlus;
  }
  if (interface_name == "Dec") {
    CARBON_CHECK(op_name == "Op");
    return clang::OO_MinusMinus;
  }

  // Arithmetic.
  if (interface_name == "Negate") {
    CARBON_CHECK(op_name == "Op");
    return clang::OO_Minus;
  }

  // Binary operators.

  // Arithmetic Operators.
  if (interface_name == "AddWith") {
    CARBON_CHECK(op_name == "Op");
    return clang::OO_Plus;
  }
  if (interface_name == "SubWith") {
    CARBON_CHECK(op_name == "Op");
    return clang::OO_Minus;
  }
  if (interface_name == "MulWith") {
    CARBON_CHECK(op_name == "Op");
    return clang::OO_Star;
  }
  if (interface_name == "DivWith") {
    CARBON_CHECK(op_name == "Op");
    return clang::OO_Slash;
  }
  if (interface_name == "ModWith") {
    CARBON_CHECK(op_name == "Op");
    return clang::OO_Percent;
  }

  // Bitwise Operators.
  if (interface_name == "BitAndWith") {
    CARBON_CHECK(op_name == "Op");
    return clang::OO_Amp;
  }
  if (interface_name == "BitOrWith") {
    CARBON_CHECK(op_name == "Op");
    return clang::OO_Pipe;
  }
  if (interface_name == "BitXorWith") {
    CARBON_CHECK(op_name == "Op");
    return clang::OO_Caret;
  }
  if (interface_name == "LeftShiftWith") {
    CARBON_CHECK(op_name == "Op");
    return clang::OO_LessLess;
  }
  if (interface_name == "RightShiftWith") {
    CARBON_CHECK(op_name == "Op");
    return clang::OO_GreaterGreater;
  }

  // Compound Assignment Arithmetic Operators.
  if (interface_name == "AddAssignWith") {
    CARBON_CHECK(op_name == "Op");
    return clang::OO_PlusEqual;
  }
  if (interface_name == "SubAssignWith") {
    CARBON_CHECK(op_name == "Op");
    return clang::OO_MinusEqual;
  }
  if (interface_name == "MulAssignWith") {
    CARBON_CHECK(op_name == "Op");
    return clang::OO_StarEqual;
  }
  if (interface_name == "DivAssignWith") {
    CARBON_CHECK(op_name == "Op");
    return clang::OO_SlashEqual;
  }
  if (interface_name == "ModAssignWith") {
    CARBON_CHECK(op_name == "Op");
    return clang::OO_PercentEqual;
  }

  // Compound Assignment Bitwise Operators.
  if (interface_name == "BitAndAssignWith") {
    CARBON_CHECK(op_name == "Op");
    return clang::OO_AmpEqual;
  }
  if (interface_name == "BitOrAssignWith") {
    CARBON_CHECK(op_name == "Op");
    return clang::OO_PipeEqual;
  }
  if (interface_name == "BitXorAssignWith") {
    CARBON_CHECK(op_name == "Op");
    return clang::OO_CaretEqual;
  }
  if (interface_name == "LeftShiftAssignWith") {
    CARBON_CHECK(op_name == "Op");
    return clang::OO_LessLessEqual;
  }
  if (interface_name == "RightShiftAssignWith") {
    CARBON_CHECK(op_name == "Op");
    return clang::OO_GreaterGreaterEqual;
  }

  // Relational Operators.
  if (interface_name == "EqWith") {
    if (op_name == "Equal") {
      return clang::OO_EqualEqual;
    }
    CARBON_CHECK(op_name == "NotEqual");
    return clang::OO_ExclaimEqual;
  }
  if (interface_name == "OrderedWith") {
    if (op_name == "Less") {
      return clang::OO_Less;
    }
    if (op_name == "Greater") {
      return clang::OO_Greater;
    }
    if (op_name == "LessOrEquivalent") {
      return clang::OO_LessEqual;
    }
    CARBON_CHECK(op_name == "GreaterOrEquivalent");
    return clang::OO_GreaterEqual;
  }

  context.TODO(loc_id, llvm::formatv("Unsupported operator interface `{0}`",
                                     interface_name));
  return std::nullopt;
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
      if (auto fn_decl =
              context.insts().TryGetAsWithId<SemIR::FunctionDecl>(result_id)) {
        CheckCppOverloadAccess(context, loc_id, best_viable_fn->FoundDecl,
                               fn_decl->inst_id);
      } else {
        CARBON_CHECK(result_id == SemIR::ErrorInst::InstId);
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

auto IsCppOperatorMethod(Context& context, SemIR::InstId inst_id) -> bool {
  auto function_type = context.types().TryGetAs<SemIR::FunctionType>(
      context.insts().Get(inst_id).type_id());
  if (!function_type) {
    return false;
  }
  SemIR::ClangDeclId clang_decl_id =
      context.functions().Get(function_type->function_id).clang_decl_id;
  return clang_decl_id.has_value() &&
         IsCppOperatorMethodDecl(
             context.clang_decls().Get(clang_decl_id).key.decl);
}

}  // namespace Carbon::Check
