//== ReturnStackAddressChecker.cpp ------------------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines ReturnStackAddressChecker, which is a path-sensitive
// check which looks for the addresses of stack variables being returned to
// callers.
//
//===----------------------------------------------------------------------===//

#include "GRExprEngineInternalChecks.h"
#include "clang/Analysis/PathSensitive/GRExprEngine.h"
#include "clang/Analysis/PathSensitive/BugReporter.h"
#include "clang/Analysis/PathSensitive/CheckerVisitor.h"
#include "llvm/ADT/SmallString.h"

using namespace clang;

namespace {
class VISIBILITY_HIDDEN ReturnStackAddressChecker : 
    public CheckerVisitor<ReturnStackAddressChecker> {      
  BuiltinBug *BT;
public:
    ReturnStackAddressChecker() : BT(0) {}
    static void *getTag();
    void PreVisitReturnStmt(CheckerContext &C, const ReturnStmt *RS);
};
}

void clang::RegisterReturnStackAddressChecker(GRExprEngine &Eng) {
  Eng.registerCheck(new ReturnStackAddressChecker());
}

void *ReturnStackAddressChecker::getTag() {
  static int x = 0; return &x;
}

void ReturnStackAddressChecker::PreVisitReturnStmt(CheckerContext &C,
                                                   const ReturnStmt *RS) {
  
  const Expr *RetE = RS->getRetValue();
  if (!RetE)
    return;
 
  SVal V = C.getState()->getSVal(RetE);
  const MemRegion *R = V.getAsRegion();

  if (!R || !R->hasStackStorage())
    return;  
  
  ExplodedNode *N = C.GenerateNode(RS, C.getState(), true);

  if (!N)
    return;
  
  if (!BT)
    BT = new BuiltinBug("Return of address to stack-allocated memory");
  
  // Generate a report for this bug.
  llvm::SmallString<100> buf;
  llvm::raw_svector_ostream os(buf);
  SourceRange range;
  
  // Check if the region is a compound literal.
  if (const CompoundLiteralRegion* CR = dyn_cast<CompoundLiteralRegion>(R)) {    
    const CompoundLiteralExpr* CL = CR->getLiteralExpr();
    os << "Address of stack memory associated with a compound literal "
          "declared on line "
       << C.getSourceManager().getInstantiationLineNumber(CL->getLocStart())
       << " returned to caller";    
    range = CL->getSourceRange();
  }
  else if (const AllocaRegion* AR = dyn_cast<AllocaRegion>(R)) {
    const Expr* ARE = AR->getExpr();
    SourceLocation L = ARE->getLocStart();
    range = ARE->getSourceRange();    
    os << "Address of stack memory allocated by call to alloca() on line "
       << C.getSourceManager().getInstantiationLineNumber(L)
       << " returned to caller";
  }
  else {
    os << "Address of stack memory associated with local variable '"
        << R->getString() << "' returned.";
  }

  RangedBugReport *report = new RangedBugReport(*BT, os.str().data(), N);
  report->addRange(RS->getSourceRange());
  if (range.isValid())
    report->addRange(range);
  
  C.EmitReport(report);
}
