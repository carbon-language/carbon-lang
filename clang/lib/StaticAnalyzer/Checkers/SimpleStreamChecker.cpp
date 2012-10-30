//===-- SimpleStreamChecker.cpp -----------------------------------------*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Defines a checker for proper use of fopen/fclose APIs.
//   - If a file has been closed with fclose, it should not be accessed again.
//   Accessing a closed file results in undefined behavior.
//   - If a file was opened with fopen, it must be closed with fclose before
//   the execution ends. Failing to do so results in a resource leak.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

using namespace clang;
using namespace ento;

namespace {
typedef llvm::SmallVector<SymbolRef, 2> SymbolVector;

struct StreamState {
  enum Kind { Opened, Closed } K;

  StreamState(Kind InK) : K(InK) { }

  bool isOpened() const { return K == Opened; }
  bool isClosed() const { return K == Closed; }

  static StreamState getOpened() { return StreamState(Opened); }
  static StreamState getClosed() { return StreamState(Closed); }

  bool operator==(const StreamState &X) const {
    return K == X.K;
  }
  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddInteger(K);
  }
};

class SimpleStreamChecker: public Checker<check::PostStmt<CallExpr>,
                                          check::PreStmt<CallExpr>,
                                          check::DeadSymbols,
                                          eval::Assume > {

  mutable IdentifierInfo *IIfopen, *IIfclose;

  mutable OwningPtr<BugType> DoubleCloseBugType;
  mutable OwningPtr<BugType> LeakBugType;

  void initIdentifierInfo(ASTContext &Ctx) const;

  void reportDoubleClose(SymbolRef FileDescSym,
                         const CallExpr *Call,
                         CheckerContext &C) const;

  ExplodedNode *reportLeaks(SymbolVector LeakedStreams,
                            CheckerContext &C) const;

public:
  SimpleStreamChecker() : IIfopen(0), IIfclose(0) {}

  /// Process fopen.
  void checkPostStmt(const CallExpr *Call, CheckerContext &C) const;
  /// Process fclose.
  void checkPreStmt(const CallExpr *Call, CheckerContext &C) const;

  void checkDeadSymbols(SymbolReaper &SymReaper, CheckerContext &C) const;
  ProgramStateRef evalAssume(ProgramStateRef state, SVal Cond,
                             bool Assumption) const;

};

} // end anonymous namespace

/// The state of the checker is a map from tracked stream symbols to their
/// state. Let's store it in the ProgramState.
REGISTER_MAP_WITH_PROGRAMSTATE(StreamMap, SymbolRef, StreamState)

void SimpleStreamChecker::checkPostStmt(const CallExpr *Call,
                                        CheckerContext &C) const {
  initIdentifierInfo(C.getASTContext());

  if (C.getCalleeIdentifier(Call) != IIfopen)
    return;

  // Get the symbolic value corresponding to the file handle.
  SymbolRef FileDesc = C.getSVal(Call).getAsSymbol();
  if (!FileDesc)
    return;

  // Generate the next transition (an edge in the exploded graph).
  ProgramStateRef State = C.getState();
  State = State->set<StreamMap>(FileDesc, StreamState::getOpened());
  C.addTransition(State);
}

void SimpleStreamChecker::checkPreStmt(const CallExpr *Call,
                                       CheckerContext &C) const {
  initIdentifierInfo(C.getASTContext());

  if (C.getCalleeIdentifier(Call) != IIfclose || Call->getNumArgs() != 1)
    return;

  // Get the symbolic value corresponding to the file handle.
  SymbolRef FileDesc = C.getSVal(Call->getArg(0)).getAsSymbol();
  if (!FileDesc)
    return;

  // Check if the stream has already been closed.
  ProgramStateRef State = C.getState();
  const StreamState *SS = State->get<StreamMap>(FileDesc);
  if (SS && SS->isClosed())
    reportDoubleClose(FileDesc, Call, C);

  // Generate the next transition, in which the stream is closed.
  State = State->set<StreamMap>(FileDesc, StreamState::getClosed());
  C.addTransition(State);
}

void SimpleStreamChecker::checkDeadSymbols(SymbolReaper &SymReaper,
                                           CheckerContext &C) const {
  ProgramStateRef State = C.getState();
  StreamMapTy TrackedStreams = State->get<StreamMap>();
  SymbolVector LeakedStreams;
  for (StreamMapTy::iterator I = TrackedStreams.begin(),
                           E = TrackedStreams.end(); I != E; ++I) {
    SymbolRef Sym = I->first;
    if (SymReaper.isDead(Sym)) {
      const StreamState &SS = I->second;
      if (SS.isOpened())
        LeakedStreams.push_back(Sym);

      // Remove the dead symbol from the streams map.
      State = State->remove<StreamMap>(Sym);
    }
  }

  ExplodedNode *N = reportLeaks(LeakedStreams, C);
  C.addTransition(State, N);
}

// If a symbolic region is assumed to NULL (or another constant), stop tracking
// it - assuming that allocation failed on this path.
ProgramStateRef SimpleStreamChecker::evalAssume(ProgramStateRef State,
                                                SVal Cond,
                                                bool Assumption) const {
  StreamMapTy TrackedStreams = State->get<StreamMap>();
  SymbolVector LeakedStreams;
  for (StreamMapTy::iterator I = TrackedStreams.begin(),
                           E = TrackedStreams.end(); I != E; ++I) {
    SymbolRef Sym = I->first;
    if (State->getConstraintManager().isNull(State, Sym).isTrue())
      State = State->remove<StreamMap>(Sym);
  }
  return State;
}

void SimpleStreamChecker::reportDoubleClose(SymbolRef FileDescSym,
                                            const CallExpr *CallExpr,
                                            CheckerContext &C) const {
  // We reached a bug, stop exploring the path here by generating a sink.
  ExplodedNode *ErrNode = C.generateSink();
  // If this error node already exists, return.
  if (!ErrNode)
    return;

  // Initialize the bug type.
  if (!DoubleCloseBugType)
    DoubleCloseBugType.reset(new BugType("Double fclose",
                             "Unix Stream API Error"));
  // Generate the report.
  BugReport *R = new BugReport(*DoubleCloseBugType,
      "Closing a previously closed file stream", ErrNode);
  R->addRange(CallExpr->getSourceRange());
  R->markInteresting(FileDescSym);
  C.EmitReport(R);
}

ExplodedNode *SimpleStreamChecker::reportLeaks(SymbolVector LeakedStreams,
                                               CheckerContext &C) const {
  ExplodedNode *Pred = C.getPredecessor();
  if (LeakedStreams.empty())
    return Pred;

  // Generate an intermediate node representing the leak point.
  static SimpleProgramPointTag Tag("StreamChecker : Leak");
  ExplodedNode *ErrNode = C.addTransition(Pred->getState(), Pred, &Tag);
  if (!ErrNode)
    return Pred;

  // Initialize the bug type.
  if (!LeakBugType) {
    LeakBugType.reset(new BuiltinBug("Resource Leak",
                                     "Unix Stream API Error"));
    // Sinks are higher importance bugs as well as calls to assert() or exit(0).
    LeakBugType->setSuppressOnSink(true);
  }

  // Attach bug reports to the leak node.
  // TODO: Identify the leaked file descriptor.
  for (llvm::SmallVector<SymbolRef, 2>::iterator
      I = LeakedStreams.begin(), E = LeakedStreams.end(); I != E; ++I) {
    BugReport *R = new BugReport(*LeakBugType,
        "Opened file is never closed; potential resource leak", ErrNode);
    R->markInteresting(*I);
    C.EmitReport(R);
  }

  return ErrNode;
}

void SimpleStreamChecker::initIdentifierInfo(ASTContext &Ctx) const {
  if (IIfopen)
    return;
  IIfopen = &Ctx.Idents.get("fopen");
  IIfclose = &Ctx.Idents.get("fclose");
}

void ento::registerSimpleStreamChecker(CheckerManager &mgr) {
  mgr.registerChecker<SimpleStreamChecker>();
}
