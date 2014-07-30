//===--- SemaStmtAttr.cpp - Statement Attribute Handling ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements stmt-related attribute processing.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaInternal.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Sema/DelayedDiagnostic.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/LoopHint.h"
#include "clang/Sema/ScopeInfo.h"
#include "llvm/ADT/StringExtras.h"

using namespace clang;
using namespace sema;

static Attr *handleFallThroughAttr(Sema &S, Stmt *St, const AttributeList &A,
                                   SourceRange Range) {
  if (!isa<NullStmt>(St)) {
    S.Diag(A.getRange().getBegin(), diag::err_fallthrough_attr_wrong_target)
        << St->getLocStart();
    if (isa<SwitchCase>(St)) {
      SourceLocation L = S.getLocForEndOfToken(Range.getEnd());
      S.Diag(L, diag::note_fallthrough_insert_semi_fixit)
          << FixItHint::CreateInsertion(L, ";");
    }
    return nullptr;
  }
  if (S.getCurFunction()->SwitchStack.empty()) {
    S.Diag(A.getRange().getBegin(), diag::err_fallthrough_attr_outside_switch);
    return nullptr;
  }
  return ::new (S.Context) FallThroughAttr(A.getRange(), S.Context,
                                           A.getAttributeSpellingListIndex());
}

static Attr *handleLoopHintAttr(Sema &S, Stmt *St, const AttributeList &A,
                                SourceRange) {
  IdentifierLoc *PragmaNameLoc = A.getArgAsIdent(0);
  IdentifierLoc *OptionLoc = A.getArgAsIdent(1);
  IdentifierLoc *StateLoc = A.getArgAsIdent(2);
  Expr *ValueExpr = A.getArgAsExpr(3);

  bool PragmaUnroll = PragmaNameLoc->Ident->getName() == "unroll";
  bool PragmaNoUnroll = PragmaNameLoc->Ident->getName() == "nounroll";
  if (St->getStmtClass() != Stmt::DoStmtClass &&
      St->getStmtClass() != Stmt::ForStmtClass &&
      St->getStmtClass() != Stmt::CXXForRangeStmtClass &&
      St->getStmtClass() != Stmt::WhileStmtClass) {
    const char *Pragma =
        llvm::StringSwitch<const char *>(PragmaNameLoc->Ident->getName())
            .Case("unroll", "#pragma unroll")
            .Case("nounroll", "#pragma nounroll")
            .Default("#pragma clang loop");
    S.Diag(St->getLocStart(), diag::err_pragma_loop_precedes_nonloop) << Pragma;
    return nullptr;
  }

  LoopHintAttr::OptionType Option;
  LoopHintAttr::Spelling Spelling;
  if (PragmaUnroll) {
    Option = ValueExpr ? LoopHintAttr::UnrollCount : LoopHintAttr::Unroll;
    Spelling = LoopHintAttr::Pragma_unroll;
  } else if (PragmaNoUnroll) {
    Option = LoopHintAttr::Unroll;
    Spelling = LoopHintAttr::Pragma_nounroll;
  } else {
    assert(OptionLoc && OptionLoc->Ident &&
           "Attribute must have valid option info.");
    IdentifierInfo *OptionInfo = OptionLoc->Ident;
    Option = llvm::StringSwitch<LoopHintAttr::OptionType>(OptionInfo->getName())
                 .Case("vectorize", LoopHintAttr::Vectorize)
                 .Case("vectorize_width", LoopHintAttr::VectorizeWidth)
                 .Case("interleave", LoopHintAttr::Interleave)
                 .Case("interleave_count", LoopHintAttr::InterleaveCount)
                 .Case("unroll", LoopHintAttr::Unroll)
                 .Case("unroll_count", LoopHintAttr::UnrollCount)
                 .Default(LoopHintAttr::Vectorize);
    Spelling = LoopHintAttr::Pragma_clang_loop;
  }

  int ValueInt = 1;
  LoopHintAttr::LoopHintState State = LoopHintAttr::Default;
  if (PragmaNoUnroll) {
    State = LoopHintAttr::Disable;
  } else if (Option == LoopHintAttr::VectorizeWidth ||
             Option == LoopHintAttr::InterleaveCount ||
             Option == LoopHintAttr::UnrollCount) {
    // FIXME: We should support template parameters for the loop hint value.
    // See bug report #19610.
    llvm::APSInt ValueAPS;
    if (!ValueExpr || !ValueExpr->isIntegerConstantExpr(ValueAPS, S.Context) ||
        (ValueInt = ValueAPS.getSExtValue()) < 1) {
      S.Diag(A.getLoc(), diag::err_pragma_loop_invalid_value);
      return nullptr;
    }
  } else if (Option == LoopHintAttr::Vectorize ||
             Option == LoopHintAttr::Interleave ||
             Option == LoopHintAttr::Unroll) {
    // Default state is assumed if StateLoc is not specified, such as with
    // '#pragma unroll'.
    if (StateLoc && StateLoc->Ident) {
      if (StateLoc->Ident->isStr("disable"))
        State = LoopHintAttr::Disable;
      else
        State = LoopHintAttr::Enable;
    }
  }

  return LoopHintAttr::CreateImplicit(S.Context, Spelling, Option, State,
                                      ValueInt, A.getRange());
}

static void
CheckForIncompatibleAttributes(Sema &S,
                               const SmallVectorImpl<const Attr *> &Attrs) {
  // There are 3 categories of loop hints attributes: vectorize, interleave,
  // and unroll. Each comes in two variants: a state form and a numeric form.
  // The state form selectively defaults/enables/disables the transformation
  // for the loop (for unroll, default indicates full unrolling rather than
  // enabling the transformation).  The numeric form form provides an integer
  // hint (for example, unroll count) to the transformer. The following array
  // accumulates the hints encountered while iterating through the attributes
  // to check for compatibility.
  struct {
    const LoopHintAttr *StateAttr;
    const LoopHintAttr *NumericAttr;
  } HintAttrs[] = {{nullptr, nullptr}, {nullptr, nullptr}, {nullptr, nullptr}};

  for (const auto *I : Attrs) {
    const LoopHintAttr *LH = dyn_cast<LoopHintAttr>(I);

    // Skip non loop hint attributes
    if (!LH)
      continue;

    int Option = LH->getOption();
    int Category;
    enum { Vectorize, Interleave, Unroll };
    switch (Option) {
    case LoopHintAttr::Vectorize:
    case LoopHintAttr::VectorizeWidth:
      Category = Vectorize;
      break;
    case LoopHintAttr::Interleave:
    case LoopHintAttr::InterleaveCount:
      Category = Interleave;
      break;
    case LoopHintAttr::Unroll:
    case LoopHintAttr::UnrollCount:
      Category = Unroll;
      break;
    };

    auto &CategoryState = HintAttrs[Category];
    const LoopHintAttr *PrevAttr;
    if (Option == LoopHintAttr::Vectorize ||
        Option == LoopHintAttr::Interleave || Option == LoopHintAttr::Unroll) {
      // Enable|disable hint.  For example, vectorize(enable).
      PrevAttr = CategoryState.StateAttr;
      CategoryState.StateAttr = LH;
    } else {
      // Numeric hint.  For example, vectorize_width(8).
      PrevAttr = CategoryState.NumericAttr;
      CategoryState.NumericAttr = LH;
    }

    PrintingPolicy Policy(S.Context.getLangOpts());
    SourceLocation OptionLoc = LH->getRange().getBegin();
    if (PrevAttr)
      // Cannot specify same type of attribute twice.
      S.Diag(OptionLoc, diag::err_pragma_loop_compatibility)
          << /*Duplicate=*/true << PrevAttr->getDiagnosticName(Policy)
          << LH->getDiagnosticName(Policy);

    if (CategoryState.StateAttr && CategoryState.NumericAttr &&
        (Category == Unroll ||
         CategoryState.StateAttr->getState() == LoopHintAttr::Disable)) {
      // Disable hints are not compatible with numeric hints of the same
      // category.  As a special case, numeric unroll hints are also not
      // compatible with "enable" form of the unroll pragma, unroll(full).
      S.Diag(OptionLoc, diag::err_pragma_loop_compatibility)
          << /*Duplicate=*/false
          << CategoryState.StateAttr->getDiagnosticName(Policy)
          << CategoryState.NumericAttr->getDiagnosticName(Policy);
    }
  }
}

static Attr *ProcessStmtAttribute(Sema &S, Stmt *St, const AttributeList &A,
                                  SourceRange Range) {
  switch (A.getKind()) {
  case AttributeList::UnknownAttribute:
    S.Diag(A.getLoc(), A.isDeclspecAttribute() ?
           diag::warn_unhandled_ms_attribute_ignored :
           diag::warn_unknown_attribute_ignored) << A.getName();
    return nullptr;
  case AttributeList::AT_FallThrough:
    return handleFallThroughAttr(S, St, A, Range);
  case AttributeList::AT_LoopHint:
    return handleLoopHintAttr(S, St, A, Range);
  default:
    // if we're here, then we parsed a known attribute, but didn't recognize
    // it as a statement attribute => it is declaration attribute
    S.Diag(A.getRange().getBegin(), diag::err_attribute_invalid_on_stmt)
        << A.getName() << St->getLocStart();
    return nullptr;
  }
}

StmtResult Sema::ProcessStmtAttributes(Stmt *S, AttributeList *AttrList,
                                       SourceRange Range) {
  SmallVector<const Attr*, 8> Attrs;
  for (const AttributeList* l = AttrList; l; l = l->getNext()) {
    if (Attr *a = ProcessStmtAttribute(*this, S, *l, Range))
      Attrs.push_back(a);
  }

  CheckForIncompatibleAttributes(*this, Attrs);

  if (Attrs.empty())
    return S;

  return ActOnAttributedStmt(Range.getBegin(), Attrs, S);
}
