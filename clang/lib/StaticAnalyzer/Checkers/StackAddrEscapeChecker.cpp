//=== StackAddrEscapeChecker.cpp ----------------------------------*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines stack address leak checker, which checks if an invalid 
// stack address is stored into a global or heap location. See CERT DCL30-C.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/SmallString.h"
using namespace clang;
using namespace ento;

namespace {
class StackAddrEscapeChecker : public Checker< check::PreStmt<ReturnStmt>,
                                               check::EndPath > {
  mutable llvm::OwningPtr<BuiltinBug> BT_stackleak;
  mutable llvm::OwningPtr<BuiltinBug> BT_returnstack;

public:
  void checkPreStmt(const ReturnStmt *RS, CheckerContext &C) const;
  void checkEndPath(EndOfFunctionNodeBuilder &B, ExprEngine &Eng) const;
private:
  void EmitStackError(CheckerContext &C, const MemRegion *R,
                      const Expr *RetE) const;
  static SourceRange GenName(raw_ostream &os, const MemRegion *R,
                             SourceManager &SM);
};
}

SourceRange StackAddrEscapeChecker::GenName(raw_ostream &os,
                                          const MemRegion *R,
                                          SourceManager &SM) {
    // Get the base region, stripping away fields and elements.
  R = R->getBaseRegion();
  SourceRange range;
  os << "Address of ";
  
  // Check if the region is a compound literal.
  if (const CompoundLiteralRegion* CR = dyn_cast<CompoundLiteralRegion>(R)) { 
    const CompoundLiteralExpr *CL = CR->getLiteralExpr();
    os << "stack memory associated with a compound literal "
          "declared on line "
        << SM.getExpansionLineNumber(CL->getLocStart())
        << " returned to caller";    
    range = CL->getSourceRange();
  }
  else if (const AllocaRegion* AR = dyn_cast<AllocaRegion>(R)) {
    const Expr *ARE = AR->getExpr();
    SourceLocation L = ARE->getLocStart();
    range = ARE->getSourceRange();    
    os << "stack memory allocated by call to alloca() on line "
       << SM.getExpansionLineNumber(L);
  }
  else if (const BlockDataRegion *BR = dyn_cast<BlockDataRegion>(R)) {
    const BlockDecl *BD = BR->getCodeRegion()->getDecl();
    SourceLocation L = BD->getLocStart();
    range = BD->getSourceRange();
    os << "stack-allocated block declared on line "
       << SM.getExpansionLineNumber(L);
  }
  else if (const VarRegion *VR = dyn_cast<VarRegion>(R)) {
    os << "stack memory associated with local variable '"
       << VR->getString() << '\'';
    range = VR->getDecl()->getSourceRange();
  }
  else if (const CXXTempObjectRegion *TOR = dyn_cast<CXXTempObjectRegion>(R)) {
    os << "stack memory associated with temporary object of type '"
       << TOR->getValueType().getAsString() << '\'';
    range = TOR->getExpr()->getSourceRange();
  }
  else {
    assert(false && "Invalid region in ReturnStackAddressChecker.");
  } 
  
  return range;
}

void StackAddrEscapeChecker::EmitStackError(CheckerContext &C, const MemRegion *R,
                                          const Expr *RetE) const {
  ExplodedNode *N = C.generateSink();

  if (!N)
    return;

  if (!BT_returnstack)
   BT_returnstack.reset(
                 new BuiltinBug("Return of address to stack-allocated memory"));

  // Generate a report for this bug.
  llvm::SmallString<512> buf;
  llvm::raw_svector_ostream os(buf);
  SourceRange range = GenName(os, R, C.getSourceManager());
  os << " returned to caller";
  BugReport *report = new BugReport(*BT_returnstack, os.str(), N);
  report->addRange(RetE->getSourceRange());
  if (range.isValid())
    report->addRange(range);

  C.EmitReport(report);
}

void StackAddrEscapeChecker::checkPreStmt(const ReturnStmt *RS,
                                        CheckerContext &C) const {
  
  const Expr *RetE = RS->getRetValue();
  if (!RetE)
    return;
 
  SVal V = C.getState()->getSVal(RetE);
  const MemRegion *R = V.getAsRegion();

  if (!R || !R->hasStackStorage())
    return;  
  
  if (R->hasStackStorage()) {
    // Automatic reference counting automatically copies blocks.
    if (C.getASTContext().getLangOptions().ObjCAutoRefCount &&
        isa<BlockDataRegion>(R))
      return;

    EmitStackError(C, R, RetE);
    return;
  }
}

void StackAddrEscapeChecker::checkEndPath(EndOfFunctionNodeBuilder &B,
                                        ExprEngine &Eng) const {

  const ProgramState *state = B.getState();

  // Iterate over all bindings to global variables and see if it contains
  // a memory region in the stack space.
  class CallBack : public StoreManager::BindingsHandler {
  private:
    ExprEngine &Eng;
    const StackFrameContext *CurSFC;
  public:
    SmallVector<std::pair<const MemRegion*, const MemRegion*>, 10> V;

    CallBack(ExprEngine &Eng, const LocationContext *LCtx)
      : Eng(Eng), CurSFC(LCtx->getCurrentStackFrame()) {}
    
    bool HandleBinding(StoreManager &SMgr, Store store,
                       const MemRegion *region, SVal val) {
      
      if (!isa<GlobalsSpaceRegion>(region->getMemorySpace()))
        return true;
      
      const MemRegion *vR = val.getAsRegion();
      if (!vR)
        return true;
        
      // Under automated retain release, it is okay to assign a block
      // directly to a global variable.
      if (Eng.getContext().getLangOptions().ObjCAutoRefCount &&
          isa<BlockDataRegion>(vR))
        return true;

      if (const StackSpaceRegion *SSR = 
          dyn_cast<StackSpaceRegion>(vR->getMemorySpace())) {
        // If the global variable holds a location in the current stack frame,
        // record the binding to emit a warning.
        if (SSR->getStackFrame() == CurSFC)
          V.push_back(std::make_pair(region, vR));
      }
      
      return true;
    }
  };
    
  CallBack cb(Eng, B.getPredecessor()->getLocationContext());
  state->getStateManager().getStoreManager().iterBindings(state->getStore(),cb);

  if (cb.V.empty())
    return;

  // Generate an error node.
  ExplodedNode *N = B.generateNode(state);
  if (!N)
    return;

  if (!BT_stackleak)
    BT_stackleak.reset(
      new BuiltinBug("Stack address stored into global variable",
                     "Stack address was saved into a global variable. "
                     "This is dangerous because the address will become "
                     "invalid after returning from the function"));
  
  for (unsigned i = 0, e = cb.V.size(); i != e; ++i) {
    // Generate a report for this bug.
    llvm::SmallString<512> buf;
    llvm::raw_svector_ostream os(buf);
    SourceRange range = GenName(os, cb.V[i].second,
                                Eng.getContext().getSourceManager());
    os << " is still referred to by the global variable '";
    const VarRegion *VR = cast<VarRegion>(cb.V[i].first->getBaseRegion());
    os << VR->getDecl()
       << "' upon returning to the caller.  This will be a dangling reference";
    BugReport *report = new BugReport(*BT_stackleak, os.str(), N);
    if (range.isValid())
      report->addRange(range);

    Eng.getBugReporter().EmitReport(report);
  }
}

void ento::registerStackAddrEscapeChecker(CheckerManager &mgr) {
  mgr.registerChecker<StackAddrEscapeChecker>();
}
