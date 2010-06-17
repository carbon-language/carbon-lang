//=== StackAddrLeakChecker.cpp ------------------------------------*- C++ -*--//
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

#include "GRExprEngineInternalChecks.h"
#include "clang/Checker/BugReporter/BugType.h"
#include "clang/Checker/PathSensitive/CheckerVisitor.h"
#include "clang/Checker/PathSensitive/GRState.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/SmallString.h"
using namespace clang;

namespace {
class StackAddrLeakChecker : public CheckerVisitor<StackAddrLeakChecker> {
  BuiltinBug *BT_stackleak;
  BuiltinBug *BT_returnstack;

public:
  StackAddrLeakChecker() : BT_stackleak(0), BT_returnstack(0) {}
  static void *getTag() {
    static int x;
    return &x;
  }
  void PreVisitReturnStmt(CheckerContext &C, const ReturnStmt *RS);
  void EvalEndPath(GREndPathNodeBuilder &B, void *tag, GRExprEngine &Eng);
private:
  void EmitStackError(CheckerContext &C, const MemRegion *R, const Expr *RetE);
};
}

void clang::RegisterStackAddrLeakChecker(GRExprEngine &Eng) {
  Eng.registerCheck(new StackAddrLeakChecker());
}
void StackAddrLeakChecker::EmitStackError(CheckerContext &C, const MemRegion *R,
                                          const Expr *RetE) {
  ExplodedNode *N = C.GenerateSink();

  if (!N)
    return;

  if (!BT_returnstack)
   BT_returnstack=new BuiltinBug("Return of address to stack-allocated memory");

  // Generate a report for this bug.
  llvm::SmallString<512> buf;
  llvm::raw_svector_ostream os(buf);
  SourceRange range;

  // Get the base region, stripping away fields and elements.
  R = R->getBaseRegion();

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
  else if (const BlockDataRegion *BR = dyn_cast<BlockDataRegion>(R)) {
    const BlockDecl *BD = BR->getCodeRegion()->getDecl();
    SourceLocation L = BD->getLocStart();
    range = BD->getSourceRange();
    os << "Address of stack-allocated block declared on line "
       << C.getSourceManager().getInstantiationLineNumber(L)
       << " returned to caller";
  }
  else if (const VarRegion *VR = dyn_cast<VarRegion>(R)) {
    os << "Address of stack memory associated with local variable '"
       << VR->getString() << "' returned";
    range = VR->getDecl()->getSourceRange();
  }
  else {
    assert(false && "Invalid region in ReturnStackAddressChecker.");
    return;
  }

  RangedBugReport *report = new RangedBugReport(*BT_returnstack, os.str(), N);
  report->addRange(RetE->getSourceRange());
  if (range.isValid())
    report->addRange(range);

  C.EmitReport(report);
}	

void StackAddrLeakChecker::PreVisitReturnStmt(CheckerContext &C,
                                              const ReturnStmt *RS) {
  
  const Expr *RetE = RS->getRetValue();
  if (!RetE)
    return;
 
  SVal V = C.getState()->getSVal(RetE);
  const MemRegion *R = V.getAsRegion();

  if (!R || !R->hasStackStorage())
    return;  
  
  if (R->hasStackStorage()) {
    EmitStackError(C, R, RetE);
    return;
  }
}

void StackAddrLeakChecker::EvalEndPath(GREndPathNodeBuilder &B, void *tag,
                                       GRExprEngine &Eng) {
  SaveAndRestore<bool> OldHasGen(B.HasGeneratedNode);
  const GRState *state = B.getState();

  // Iterate over all bindings to global variables and see if it contains
  // a memory region in the stack space.
  class CallBack : public StoreManager::BindingsHandler {
  private:
    const StackFrameContext *CurSFC;
  public:
    const MemRegion *src, *dst;    

    CallBack(const LocationContext *LCtx)
      : CurSFC(LCtx->getCurrentStackFrame()), src(0), dst(0) {}
    
    bool HandleBinding(StoreManager &SMgr, Store store,
                       const MemRegion *region, SVal val) {
      
      if (!isa<GlobalsSpaceRegion>(region->getMemorySpace()))
        return true;
      
      const MemRegion *vR = val.getAsRegion();
      if (!vR)
        return true;
      
      if (const StackSpaceRegion *SSR = 
          dyn_cast<StackSpaceRegion>(vR->getMemorySpace())) {
        // If the global variable holds a location in the current stack frame,
        // record the binding to emit a warning.
        if (SSR->getStackFrame() == CurSFC) {        
          src = region;
          dst = vR;
          return false;
        }
      }
      
      return true;
    }
  };
    
  CallBack cb(B.getPredecessor()->getLocationContext());
  state->getStateManager().getStoreManager().iterBindings(state->getStore(),cb);
  
  // Did we find any globals referencing stack memory?
  if (!cb.src)
    return;

  // Generate an error node.
  ExplodedNode *N = B.generateNode(state, tag, B.getPredecessor());
  if (!N)
    return;
  
  if (!BT_stackleak)
    BT_stackleak = new BuiltinBug("Stack address stored into global variable",
                      "Stack address was saved into a global variable. "
                      "is dangerous because the address will become invalid "
                      "after returning from the function");
  
  BugReport *R =
    new BugReport(*BT_stackleak,  BT_stackleak->getDescription(), N);

  Eng.getBugReporter().EmitReport(R);
}
