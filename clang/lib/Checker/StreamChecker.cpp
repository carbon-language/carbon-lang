//===-- StreamChecker.cpp -----------------------------------------*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines checkers that model and check stream handling functions.
//
//===----------------------------------------------------------------------===//

#include "GRExprEngineExperimentalChecks.h"
#include "clang/Checker/BugReporter/BugType.h"
#include "clang/Checker/PathSensitive/CheckerVisitor.h"
#include "clang/Checker/PathSensitive/GRState.h"
#include "clang/Checker/PathSensitive/GRStateTrait.h"
#include "clang/Checker/PathSensitive/SymbolManager.h"
#include "llvm/ADT/ImmutableMap.h"

using namespace clang;

namespace {

class StreamChecker : public CheckerVisitor<StreamChecker> {
  IdentifierInfo *II_fopen, *II_fread, *II_fwrite, 
                 *II_fseek, *II_ftell, *II_rewind, *II_fgetpos, *II_fsetpos,  
                 *II_clearerr, *II_feof, *II_ferror, *II_fileno;
  BuiltinBug *BT_nullfp, *BT_illegalwhence;

public:
  StreamChecker() 
    : II_fopen(0), II_fread(0), II_fwrite(0), 
      II_fseek(0), II_ftell(0), II_rewind(0), II_fgetpos(0), II_fsetpos(0), 
      II_clearerr(0), II_feof(0), II_ferror(0), II_fileno(0), 
      BT_nullfp(0), BT_illegalwhence(0) {}

  static void *getTag() {
    static int x;
    return &x;
  }

  virtual bool EvalCallExpr(CheckerContext &C, const CallExpr *CE);

private:
  void Fopen(CheckerContext &C, const CallExpr *CE);
  void Fread(CheckerContext &C, const CallExpr *CE);
  void Fwrite(CheckerContext &C, const CallExpr *CE);
  void Fseek(CheckerContext &C, const CallExpr *CE);
  void Ftell(CheckerContext &C, const CallExpr *CE);
  void Rewind(CheckerContext &C, const CallExpr *CE);
  void Fgetpos(CheckerContext &C, const CallExpr *CE);
  void Fsetpos(CheckerContext &C, const CallExpr *CE);
  void Clearerr(CheckerContext &C, const CallExpr *CE);
  void Feof(CheckerContext &C, const CallExpr *CE);
  void Ferror(CheckerContext &C, const CallExpr *CE);
  void Fileno(CheckerContext &C, const CallExpr *CE);
  
  // Return true indicates the stream pointer is NULL.
  const GRState *CheckNullStream(SVal SV, const GRState *state, 
                                 CheckerContext &C);
};

} // end anonymous namespace

void clang::RegisterStreamChecker(GRExprEngine &Eng) {
  Eng.registerCheck(new StreamChecker());
}

bool StreamChecker::EvalCallExpr(CheckerContext &C, const CallExpr *CE) {
  const GRState *state = C.getState();
  const Expr *Callee = CE->getCallee();
  SVal L = state->getSVal(Callee);
  const FunctionDecl *FD = L.getAsFunctionDecl();
  if (!FD)
    return false;

  ASTContext &Ctx = C.getASTContext();
  if (!II_fopen)
    II_fopen = &Ctx.Idents.get("fopen");
  if (!II_fread)
    II_fread = &Ctx.Idents.get("fread");
  if (!II_fwrite)
    II_fwrite = &Ctx.Idents.get("fwrite");
  if (!II_fseek)
    II_fseek = &Ctx.Idents.get("fseek");
  if (!II_ftell)
    II_ftell = &Ctx.Idents.get("ftell");
  if (!II_rewind)
    II_rewind = &Ctx.Idents.get("rewind");
  if (!II_fgetpos)
    II_fgetpos = &Ctx.Idents.get("fgetpos");
  if (!II_fsetpos)
    II_fsetpos = &Ctx.Idents.get("fsetpos");
  if (!II_clearerr)
    II_clearerr = &Ctx.Idents.get("clearerr");
  if (!II_feof)
    II_feof = &Ctx.Idents.get("feof");
  if (!II_ferror)
    II_ferror = &Ctx.Idents.get("ferror");
  if (!II_fileno)
    II_fileno = &Ctx.Idents.get("fileno");

  if (FD->getIdentifier() == II_fopen) {
    Fopen(C, CE);
    return true;
  }
  if (FD->getIdentifier() == II_fread) {
    Fread(C, CE);
    return true;
  }
  if (FD->getIdentifier() == II_fwrite) {
    Fwrite(C, CE);
    return true;
  }
  if (FD->getIdentifier() == II_fseek) {
    Fseek(C, CE);
    return true;
  }
  if (FD->getIdentifier() == II_ftell) {
    Ftell(C, CE);
    return true;
  }
  if (FD->getIdentifier() == II_rewind) {
    Rewind(C, CE);
    return true;
  }
  if (FD->getIdentifier() == II_fgetpos) {
    Fgetpos(C, CE);
    return true;
  }
  if (FD->getIdentifier() == II_fsetpos) {
    Fsetpos(C, CE);
    return true;
  }
  if (FD->getIdentifier() == II_clearerr) {
    Clearerr(C, CE);
    return true;
  }
  if (FD->getIdentifier() == II_feof) {
    Feof(C, CE);
    return true;
  }
  if (FD->getIdentifier() == II_ferror) {
    Ferror(C, CE);
    return true;
  }
  if (FD->getIdentifier() == II_fileno) {
    Fileno(C, CE);
    return true;
  }

  return false;
}

void StreamChecker::Fopen(CheckerContext &C, const CallExpr *CE) {
  const GRState *state = C.getState();
  unsigned Count = C.getNodeBuilder().getCurrentBlockCount();
  ValueManager &ValMgr = C.getValueManager();
  DefinedSVal RetVal = cast<DefinedSVal>(ValMgr.getConjuredSymbolVal(0, CE, 
                                                                     Count));
  state = state->BindExpr(CE, RetVal);

  ConstraintManager &CM = C.getConstraintManager();
  // Bifurcate the state into two: one with a valid FILE* pointer, the other
  // with a NULL.
  const GRState *stateNotNull, *stateNull;
  llvm::tie(stateNotNull, stateNull) = CM.AssumeDual(state, RetVal);

  C.addTransition(stateNotNull);
  C.addTransition(stateNull);
}

void StreamChecker::Fread(CheckerContext &C, const CallExpr *CE) {
  const GRState *state = C.getState();
  if (!CheckNullStream(state->getSVal(CE->getArg(3)), state, C))
    return;
}

void StreamChecker::Fwrite(CheckerContext &C, const CallExpr *CE) {
  const GRState *state = C.getState();
  if (!CheckNullStream(state->getSVal(CE->getArg(3)), state, C))
    return;
}

void StreamChecker::Fseek(CheckerContext &C, const CallExpr *CE) {
  const GRState *state = C.getState();
  if (!(state = CheckNullStream(state->getSVal(CE->getArg(0)), state, C)))
    return;
  // Check the legality of the 'whence' argument of 'fseek'.
  SVal Whence = state->getSVal(CE->getArg(2));
  bool WhenceIsLegal = true;
  const nonloc::ConcreteInt *CI = dyn_cast<nonloc::ConcreteInt>(&Whence);
  if (!CI)
    WhenceIsLegal = false;

  int64_t x = CI->getValue().getSExtValue();
  if (!(x == 0 || x == 1 || x == 2))
    WhenceIsLegal = false;

  if (!WhenceIsLegal) {
    if (ExplodedNode *N = C.GenerateSink(state)) {
      if (!BT_illegalwhence)
        BT_illegalwhence = new BuiltinBug("Illegal whence argument",
                                     "The whence argument to fseek() should be "
                                          "SEEK_SET, SEEK_END, or SEEK_CUR.");
      BugReport *R = new BugReport(*BT_illegalwhence, 
                                   BT_illegalwhence->getDescription(), N);
      C.EmitReport(R);
    }
    return;
  }

  C.addTransition(state);
}

void StreamChecker::Ftell(CheckerContext &C, const CallExpr *CE) {
  const GRState *state = C.getState();
  if (!CheckNullStream(state->getSVal(CE->getArg(0)), state, C))
    return;
}

void StreamChecker::Rewind(CheckerContext &C, const CallExpr *CE) {
  const GRState *state = C.getState();
  if (!CheckNullStream(state->getSVal(CE->getArg(0)), state, C))
    return;
}

void StreamChecker::Fgetpos(CheckerContext &C, const CallExpr *CE) {
  const GRState *state = C.getState();
  if (!CheckNullStream(state->getSVal(CE->getArg(0)), state, C))
    return;
}

void StreamChecker::Fsetpos(CheckerContext &C, const CallExpr *CE) {
  const GRState *state = C.getState();
  if (!CheckNullStream(state->getSVal(CE->getArg(0)), state, C))
    return;
}

void StreamChecker::Clearerr(CheckerContext &C, const CallExpr *CE) {
  const GRState *state = C.getState();
  if (!CheckNullStream(state->getSVal(CE->getArg(0)), state, C))
    return;
}

void StreamChecker::Feof(CheckerContext &C, const CallExpr *CE) {
  const GRState *state = C.getState();
  if (!CheckNullStream(state->getSVal(CE->getArg(0)), state, C))
    return;
}

void StreamChecker::Ferror(CheckerContext &C, const CallExpr *CE) {
  const GRState *state = C.getState();
  if (!CheckNullStream(state->getSVal(CE->getArg(0)), state, C))
    return;
}

void StreamChecker::Fileno(CheckerContext &C, const CallExpr *CE) {
  const GRState *state = C.getState();
  if (!CheckNullStream(state->getSVal(CE->getArg(0)), state, C))
    return;
}

const GRState *StreamChecker::CheckNullStream(SVal SV, const GRState *state,
                                    CheckerContext &C) {
  const DefinedSVal *DV = dyn_cast<DefinedSVal>(&SV);
  if (!DV)
    return 0;

  ConstraintManager &CM = C.getConstraintManager();
  const GRState *stateNotNull, *stateNull;
  llvm::tie(stateNotNull, stateNull) = CM.AssumeDual(state, *DV);

  if (!stateNotNull && stateNull) {
    if (ExplodedNode *N = C.GenerateSink(stateNull)) {
      if (!BT_nullfp)
        BT_nullfp = new BuiltinBug("NULL stream pointer",
                                     "Stream pointer might be NULL.");
      BugReport *R =new BugReport(*BT_nullfp, BT_nullfp->getDescription(), N);
      C.EmitReport(R);
    }
    return 0;
  }
  return stateNotNull;
}
