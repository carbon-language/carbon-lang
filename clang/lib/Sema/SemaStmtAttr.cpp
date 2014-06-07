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
  if (St->getStmtClass() != Stmt::DoStmtClass &&
      St->getStmtClass() != Stmt::ForStmtClass &&
      St->getStmtClass() != Stmt::CXXForRangeStmtClass &&
      St->getStmtClass() != Stmt::WhileStmtClass) {
    S.Diag(St->getLocStart(), diag::err_pragma_loop_precedes_nonloop);
    return nullptr;
  }

  IdentifierLoc *OptionLoc = A.getArgAsIdent(0);
  IdentifierInfo *OptionInfo = OptionLoc->Ident;
  IdentifierLoc *ValueLoc = A.getArgAsIdent(1);
  IdentifierInfo *ValueInfo = ValueLoc->Ident;
  Expr *ValueExpr = A.getArgAsExpr(2);

  assert(OptionInfo && "Attribute must have valid option info.");

  LoopHintAttr::OptionType Option =
      llvm::StringSwitch<LoopHintAttr::OptionType>(OptionInfo->getName())
          .Case("vectorize", LoopHintAttr::Vectorize)
          .Case("vectorize_width", LoopHintAttr::VectorizeWidth)
          .Case("interleave", LoopHintAttr::Interleave)
          .Case("interleave_count", LoopHintAttr::InterleaveCount)
          .Default(LoopHintAttr::Vectorize);

  int ValueInt;
  if (Option == LoopHintAttr::Vectorize || Option == LoopHintAttr::Interleave) {
    if (!ValueInfo) {
      S.Diag(ValueLoc->Loc, diag::err_pragma_loop_invalid_keyword)
          << /*MissingKeyword=*/true << "";
      return nullptr;
    }

    if (ValueInfo->isStr("disable"))
      ValueInt = 0;
    else if (ValueInfo->isStr("enable"))
      ValueInt = 1;
    else {
      S.Diag(ValueLoc->Loc, diag::err_pragma_loop_invalid_keyword)
          << /*MissingKeyword=*/false << ValueInfo;
      return nullptr;
    }
  } else if (Option == LoopHintAttr::VectorizeWidth ||
             Option == LoopHintAttr::InterleaveCount) {
    // FIXME: We should support template parameters for the loop hint value.
    // See bug report #19610.
    llvm::APSInt ValueAPS;
    if (!ValueExpr || !ValueExpr->isIntegerConstantExpr(ValueAPS, S.Context)) {
      S.Diag(ValueLoc->Loc, diag::err_pragma_loop_invalid_value)
          << /*MissingValue=*/true << "";
      return nullptr;
    }

    if ((ValueInt = ValueAPS.getSExtValue()) < 1) {
      S.Diag(ValueLoc->Loc, diag::err_pragma_loop_invalid_value)
          << /*MissingValue=*/false << ValueInt;
      return nullptr;
    }
  } else
    llvm_unreachable("Unknown loop hint option");

  return LoopHintAttr::CreateImplicit(S.Context, Option, ValueInt,
                                      A.getRange());
}

static void
CheckForIncompatibleAttributes(Sema &S, SmallVectorImpl<const Attr *> &Attrs) {
  int PrevOptionValue[4] = {-1, -1, -1, -1};
  int OptionId[4] = {LoopHintAttr::Vectorize, LoopHintAttr::VectorizeWidth,
                     LoopHintAttr::Interleave, LoopHintAttr::InterleaveCount};

  for (const auto *I : Attrs) {
    const LoopHintAttr *LH = dyn_cast<LoopHintAttr>(I);

    // Skip non loop hint attributes
    if (!LH)
      continue;

    int State, Value;
    int Option = LH->getOption();
    int ValueInt = LH->getValue();

    switch (Option) {
    case LoopHintAttr::Vectorize:
    case LoopHintAttr::VectorizeWidth:
      State = 0;
      Value = 1;
      break;
    case LoopHintAttr::Interleave:
    case LoopHintAttr::InterleaveCount:
      State = 2;
      Value = 3;
      break;
    }

    SourceLocation ValueLoc = LH->getRange().getEnd();

    // Compatibility testing is split into two cases.
    // 1. if the current loop hint sets state (enable/disable) - check against
    // previous state and value.
    // 2. if the current loop hint sets a value - check against previous state
    // and value.

    if (Option == State) {
      if (PrevOptionValue[State] != -1) {
        // Cannot specify state twice.
        int PrevValue = PrevOptionValue[State];
        S.Diag(ValueLoc, diag::err_pragma_loop_compatibility)
            << /*Duplicate=*/true << LoopHintAttr::getOptionName(Option)
            << LoopHintAttr::getValueName(PrevValue)
            << LoopHintAttr::getOptionName(Option)
            << LoopHintAttr::getValueName(Value);
      }

      if (PrevOptionValue[Value] != -1) {
        // Compare state with previous width/count.
        int PrevOption = OptionId[Value];
        int PrevValueInt = PrevOptionValue[Value];
        if ((ValueInt == 0 && PrevValueInt > 1) ||
            (ValueInt == 1 && PrevValueInt <= 1))
          S.Diag(ValueLoc, diag::err_pragma_loop_compatibility)
              << /*Duplicate=*/false << LoopHintAttr::getOptionName(PrevOption)
              << PrevValueInt << LoopHintAttr::getOptionName(Option)
              << LoopHintAttr::getValueName(ValueInt);
      }
    } else {
      if (PrevOptionValue[State] != -1) {
        // Compare width/count value with previous state.
        int PrevOption = OptionId[State];
        int PrevValueInt = PrevOptionValue[State];
        if ((ValueInt > 1 && PrevValueInt == 0) ||
            (ValueInt <= 1 && PrevValueInt == 1))
          S.Diag(ValueLoc, diag::err_pragma_loop_compatibility)
              << /*Duplicate=*/false << LoopHintAttr::getOptionName(PrevOption)
              << LoopHintAttr::getValueName(PrevValueInt)
              << LoopHintAttr::getOptionName(Option) << ValueInt;
      }

      if (PrevOptionValue[Value] != -1) {
        // Cannot specify a width/count twice.
        int PrevValueInt = PrevOptionValue[Value];
        S.Diag(ValueLoc, diag::err_pragma_loop_compatibility)
            << /*Duplicate=*/true << LoopHintAttr::getOptionName(Option)
            << PrevValueInt << LoopHintAttr::getOptionName(Option) << ValueInt;
      }
    }

    PrevOptionValue[Option] = ValueInt;
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
