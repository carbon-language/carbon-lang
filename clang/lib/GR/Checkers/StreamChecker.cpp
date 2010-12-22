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
#include "clang/GR/BugReporter/BugType.h"
#include "clang/GR/PathSensitive/CheckerVisitor.h"
#include "clang/GR/PathSensitive/GRState.h"
#include "clang/GR/PathSensitive/GRStateTrait.h"
#include "clang/GR/PathSensitive/SymbolManager.h"
#include "llvm/ADT/ImmutableMap.h"

using namespace clang;
using namespace GR;

namespace {

struct StreamState {
  enum Kind { Opened, Closed, OpenFailed, Escaped } K;
  const Stmt *S;

  StreamState(Kind k, const Stmt *s) : K(k), S(s) {}

  bool isOpened() const { return K == Opened; }
  bool isClosed() const { return K == Closed; }
  //bool isOpenFailed() const { return K == OpenFailed; }
  //bool isEscaped() const { return K == Escaped; }

  bool operator==(const StreamState &X) const {
    return K == X.K && S == X.S;
  }

  static StreamState getOpened(const Stmt *s) { return StreamState(Opened, s); }
  static StreamState getClosed(const Stmt *s) { return StreamState(Closed, s); }
  static StreamState getOpenFailed(const Stmt *s) { 
    return StreamState(OpenFailed, s); 
  }
  static StreamState getEscaped(const Stmt *s) {
    return StreamState(Escaped, s);
  }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddInteger(K);
    ID.AddPointer(S);
  }
};

class StreamChecker : public CheckerVisitor<StreamChecker> {
  IdentifierInfo *II_fopen, *II_tmpfile, *II_fclose, *II_fread, *II_fwrite, 
                 *II_fseek, *II_ftell, *II_rewind, *II_fgetpos, *II_fsetpos,  
                 *II_clearerr, *II_feof, *II_ferror, *II_fileno;
  BuiltinBug *BT_nullfp, *BT_illegalwhence, *BT_doubleclose, *BT_ResourceLeak;

public:
  StreamChecker() 
    : II_fopen(0), II_tmpfile(0) ,II_fclose(0), II_fread(0), II_fwrite(0), 
      II_fseek(0), II_ftell(0), II_rewind(0), II_fgetpos(0), II_fsetpos(0), 
      II_clearerr(0), II_feof(0), II_ferror(0), II_fileno(0), 
      BT_nullfp(0), BT_illegalwhence(0), BT_doubleclose(0), 
      BT_ResourceLeak(0) {}

  static void *getTag() {
    static int x;
    return &x;
  }

  virtual bool evalCallExpr(CheckerContext &C, const CallExpr *CE);
  void evalDeadSymbols(CheckerContext &C, SymbolReaper &SymReaper);
  void evalEndPath(GREndPathNodeBuilder &B, void *tag, GRExprEngine &Eng);
  void PreVisitReturnStmt(CheckerContext &C, const ReturnStmt *S);

private:
  void Fopen(CheckerContext &C, const CallExpr *CE);
  void Tmpfile(CheckerContext &C, const CallExpr *CE);
  void Fclose(CheckerContext &C, const CallExpr *CE);
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

  void OpenFileAux(CheckerContext &C, const CallExpr *CE);
  
  const GRState *CheckNullStream(SVal SV, const GRState *state, 
                                 CheckerContext &C);
  const GRState *CheckDoubleClose(const CallExpr *CE, const GRState *state, 
                                 CheckerContext &C);
};

} // end anonymous namespace

namespace clang {
namespace GR {
  template <>
  struct GRStateTrait<StreamState> 
    : public GRStatePartialTrait<llvm::ImmutableMap<SymbolRef, StreamState> > {
    static void *GDMIndex() { return StreamChecker::getTag(); }
  };
}
}

void GR::RegisterStreamChecker(GRExprEngine &Eng) {
  Eng.registerCheck(new StreamChecker());
}

bool StreamChecker::evalCallExpr(CheckerContext &C, const CallExpr *CE) {
  const GRState *state = C.getState();
  const Expr *Callee = CE->getCallee();
  SVal L = state->getSVal(Callee);
  const FunctionDecl *FD = L.getAsFunctionDecl();
  if (!FD)
    return false;

  ASTContext &Ctx = C.getASTContext();
  if (!II_fopen)
    II_fopen = &Ctx.Idents.get("fopen");
  if (!II_tmpfile)
    II_tmpfile = &Ctx.Idents.get("tmpfile");
  if (!II_fclose)
    II_fclose = &Ctx.Idents.get("fclose");
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
  if (FD->getIdentifier() == II_tmpfile) {
    Tmpfile(C, CE);
    return true;
  }
  if (FD->getIdentifier() == II_fclose) {
    Fclose(C, CE);
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
  OpenFileAux(C, CE);
}

void StreamChecker::Tmpfile(CheckerContext &C, const CallExpr *CE) {
  OpenFileAux(C, CE);
}

void StreamChecker::OpenFileAux(CheckerContext &C, const CallExpr *CE) {
  const GRState *state = C.getState();
  unsigned Count = C.getNodeBuilder().getCurrentBlockCount();
  SValBuilder &svalBuilder = C.getSValBuilder();
  DefinedSVal RetVal =
    cast<DefinedSVal>(svalBuilder.getConjuredSymbolVal(0, CE, Count));
  state = state->BindExpr(CE, RetVal);
  
  ConstraintManager &CM = C.getConstraintManager();
  // Bifurcate the state into two: one with a valid FILE* pointer, the other
  // with a NULL.
  const GRState *stateNotNull, *stateNull;
  llvm::tie(stateNotNull, stateNull) = CM.assumeDual(state, RetVal);
  
  if (SymbolRef Sym = RetVal.getAsSymbol()) {
    // if RetVal is not NULL, set the symbol's state to Opened.
    stateNotNull =
      stateNotNull->set<StreamState>(Sym,StreamState::getOpened(CE));
    stateNull =
      stateNull->set<StreamState>(Sym, StreamState::getOpenFailed(CE));

    C.addTransition(stateNotNull);
    C.addTransition(stateNull);
  }
}

void StreamChecker::Fclose(CheckerContext &C, const CallExpr *CE) {
  const GRState *state = CheckDoubleClose(CE, C.getState(), C);
  if (state)
    C.addTransition(state);
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
  const nonloc::ConcreteInt *CI = dyn_cast<nonloc::ConcreteInt>(&Whence);

  if (!CI)
    return;

  int64_t x = CI->getValue().getSExtValue();
  if (x >= 0 && x <= 2)
    return;

  if (ExplodedNode *N = C.generateNode(state)) {
    if (!BT_illegalwhence)
      BT_illegalwhence = new BuiltinBug("Illegal whence argument",
					"The whence argument to fseek() should be "
					"SEEK_SET, SEEK_END, or SEEK_CUR.");
    BugReport *R = new BugReport(*BT_illegalwhence, 
				 BT_illegalwhence->getDescription(), N);
    C.EmitReport(R);
  }
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
  llvm::tie(stateNotNull, stateNull) = CM.assumeDual(state, *DV);

  if (!stateNotNull && stateNull) {
    if (ExplodedNode *N = C.generateSink(stateNull)) {
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

const GRState *StreamChecker::CheckDoubleClose(const CallExpr *CE,
                                               const GRState *state,
                                               CheckerContext &C) {
  SymbolRef Sym = state->getSVal(CE->getArg(0)).getAsSymbol();
  if (!Sym)
    return state;
  
  const StreamState *SS = state->get<StreamState>(Sym);

  // If the file stream is not tracked, return.
  if (!SS)
    return state;
  
  // Check: Double close a File Descriptor could cause undefined behaviour.
  // Conforming to man-pages
  if (SS->isClosed()) {
    ExplodedNode *N = C.generateSink();
    if (N) {
      if (!BT_doubleclose)
        BT_doubleclose = new BuiltinBug("Double fclose",
                                        "Try to close a file Descriptor already"
                                        " closed. Cause undefined behaviour.");
      BugReport *R = new BugReport(*BT_doubleclose,
                                   BT_doubleclose->getDescription(), N);
      C.EmitReport(R);
    }
    return NULL;
  }
  
  // Close the File Descriptor.
  return state->set<StreamState>(Sym, StreamState::getClosed(CE));
}

void StreamChecker::evalDeadSymbols(CheckerContext &C,SymbolReaper &SymReaper) {
  for (SymbolReaper::dead_iterator I = SymReaper.dead_begin(),
         E = SymReaper.dead_end(); I != E; ++I) {
    SymbolRef Sym = *I;
    const GRState *state = C.getState();
    const StreamState *SS = state->get<StreamState>(Sym);
    if (!SS)
      return;

    if (SS->isOpened()) {
      ExplodedNode *N = C.generateSink();
      if (N) {
        if (!BT_ResourceLeak)
          BT_ResourceLeak = new BuiltinBug("Resource Leak", 
                          "Opened File never closed. Potential Resource leak.");
        BugReport *R = new BugReport(*BT_ResourceLeak, 
                                     BT_ResourceLeak->getDescription(), N);
        C.EmitReport(R);
      }
    }
  }
}

void StreamChecker::evalEndPath(GREndPathNodeBuilder &B, void *tag,
                                GRExprEngine &Eng) {
  SaveAndRestore<bool> OldHasGen(B.HasGeneratedNode);
  const GRState *state = B.getState();
  typedef llvm::ImmutableMap<SymbolRef, StreamState> SymMap;
  SymMap M = state->get<StreamState>();
  
  for (SymMap::iterator I = M.begin(), E = M.end(); I != E; ++I) {
    StreamState SS = I->second;
    if (SS.isOpened()) {
      ExplodedNode *N = B.generateNode(state, tag, B.getPredecessor());
      if (N) {
        if (!BT_ResourceLeak)
          BT_ResourceLeak = new BuiltinBug("Resource Leak", 
                          "Opened File never closed. Potential Resource leak.");
        BugReport *R = new BugReport(*BT_ResourceLeak, 
                                     BT_ResourceLeak->getDescription(), N);
        Eng.getBugReporter().EmitReport(R);
      }
    }
  }
}

void StreamChecker::PreVisitReturnStmt(CheckerContext &C, const ReturnStmt *S) {
  const Expr *RetE = S->getRetValue();
  if (!RetE)
    return;
  
  const GRState *state = C.getState();
  SymbolRef Sym = state->getSVal(RetE).getAsSymbol();
  
  if (!Sym)
    return;
  
  const StreamState *SS = state->get<StreamState>(Sym);
  if(!SS)
    return;

  if (SS->isOpened())
    state = state->set<StreamState>(Sym, StreamState::getEscaped(S));

  C.addTransition(state);
}
