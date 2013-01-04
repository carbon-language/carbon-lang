//===--- TransProtectedScope.cpp - Transformations to ARC mode ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Adds brackets in case statements that "contain" initialization of retaining
// variable, thus emitting the "switch case is in protected scope" error.
//
//===----------------------------------------------------------------------===//

#include "Transforms.h"
#include "Internals.h"
#include "clang/Sema/SemaDiagnostic.h"

using namespace clang;
using namespace arcmt;
using namespace trans;

namespace {

struct CaseInfo {
  SwitchCase *SC;
  SourceRange Range;
  bool FixedBypass;
  
  CaseInfo() : SC(0), FixedBypass(false) {}
  CaseInfo(SwitchCase *S, SourceRange Range)
    : SC(S), Range(Range), FixedBypass(false) {}
};

class CaseCollector : public RecursiveASTVisitor<CaseCollector> {
  llvm::SmallVectorImpl<CaseInfo> &Cases;

public:
  CaseCollector(llvm::SmallVectorImpl<CaseInfo> &Cases)
    : Cases(Cases) { }

  bool VisitSwitchStmt(SwitchStmt *S) {
    SourceLocation NextLoc = S->getLocEnd();
    SwitchCase *Curr = S->getSwitchCaseList();
    // We iterate over case statements in reverse source-order.
    while (Curr) {
      Cases.push_back(CaseInfo(Curr,SourceRange(Curr->getLocStart(), NextLoc)));
      NextLoc = Curr->getLocStart();
      Curr = Curr->getNextSwitchCase();
    }
    return true;
  }
};

} // anonymous namespace

static bool isInRange(FullSourceLoc Loc, SourceRange R) {
  return !Loc.isBeforeInTranslationUnitThan(R.getBegin()) &&
          Loc.isBeforeInTranslationUnitThan(R.getEnd());
}

static bool handleProtectedNote(const StoredDiagnostic &Diag,
                                llvm::SmallVectorImpl<CaseInfo> &Cases,
                                TransformActions &TA) {
  assert(Diag.getLevel() == DiagnosticsEngine::Note);

  for (unsigned i = 0; i != Cases.size(); i++) {
    CaseInfo &info = Cases[i];
    if (isInRange(Diag.getLocation(), info.Range)) {
      TA.clearDiagnostic(Diag.getID(), Diag.getLocation());
      if (!info.FixedBypass) {
        TA.insertAfterToken(info.SC->getColonLoc(), " {");
        TA.insert(info.Range.getEnd(), "}\n");
        info.FixedBypass = true;
      }
      return true;
    }
  }

  return false;
}

static void handleProtectedScopeError(CapturedDiagList::iterator &DiagI,
                                      CapturedDiagList::iterator DiagE,
                                      llvm::SmallVectorImpl<CaseInfo> &Cases,
                                      TransformActions &TA) {
  Transaction Trans(TA);
  assert(DiagI->getID() == diag::err_switch_into_protected_scope);
  SourceLocation ErrLoc = DiagI->getLocation();
  bool handledAllNotes = true;
  ++DiagI;
  for (; DiagI != DiagE && DiagI->getLevel() == DiagnosticsEngine::Note;
       ++DiagI) {
    if (!handleProtectedNote(*DiagI, Cases, TA))
      handledAllNotes = false;
  }

  if (handledAllNotes)
    TA.clearDiagnostic(diag::err_switch_into_protected_scope, ErrLoc);
}

void ProtectedScopeTraverser::traverseBody(BodyContext &BodyCtx) {
  MigrationPass &Pass = BodyCtx.getMigrationContext().Pass;
  SmallVector<CaseInfo, 16> Cases;
  CaseCollector(Cases).TraverseStmt(BodyCtx.getTopStmt());

  SourceRange BodyRange = BodyCtx.getTopStmt()->getSourceRange();
  const CapturedDiagList &DiagList = Pass.getDiags();
  CapturedDiagList::iterator I = DiagList.begin(), E = DiagList.end();
  while (I != E) {
    if (I->getID() == diag::err_switch_into_protected_scope &&
        isInRange(I->getLocation(), BodyRange)) {
      handleProtectedScopeError(I, E, Cases, Pass.TA);
      continue;
    }
    ++I;
  }
}
