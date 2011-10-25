//== CheckerContext.h - Context info for path-sensitive checkers--*- C++ -*--=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines CheckerContext that provides contextual info for
// path-sensitive checkers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SA_CORE_PATHSENSITIVE_CHECKERCONTEXT
#define LLVM_CLANG_SA_CORE_PATHSENSITIVE_CHECKERCONTEXT

#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"

namespace clang {
namespace ento {

class CheckerContext {
  ExprEngine &Eng;
  ExplodedNode *Pred;
  const ProgramPoint Location;
  NodeBuilder &NB;

public:
  CheckerContext(NodeBuilder &builder,
                 ExprEngine &eng,
                 ExplodedNode *pred,
                 const ProgramPoint &loc)
    : Eng(eng),
      Pred(pred),
      Location(loc),
      NB(builder) {}

  ~CheckerContext();

  AnalysisManager &getAnalysisManager() {
    return Eng.getAnalysisManager();
  }

  ConstraintManager &getConstraintManager() {
    return Eng.getConstraintManager();
  }

  StoreManager &getStoreManager() {
    return Eng.getStoreManager();
  }

  ExplodedNode *&getPredecessor() { return Pred; }
  const ProgramState *getState() { return Pred->getState(); }

  /// \brief Returns the number of times the current block has been visited
  /// along the analyzed path.
  unsigned getCurrentBlockCount() {
    return NB.getContext().getCurrentBlockCount();
  }

  ASTContext &getASTContext() {
    return Eng.getContext();
  }
  
  BugReporter &getBugReporter() {
    return Eng.getBugReporter();
  }
  
  SourceManager &getSourceManager() {
    return getBugReporter().getSourceManager();
  }

  SValBuilder &getSValBuilder() {
    return Eng.getSValBuilder();
  }

  SymbolManager &getSymbolManager() {
    return getSValBuilder().getSymbolManager();
  }

  bool isObjCGCEnabled() {
    return Eng.isObjCGCEnabled();
  }

  ProgramStateManager &getStateManager() {
    return Eng.getStateManager();
  }


  AnalysisDeclContext *getCurrentAnalysisDeclContext() const {
    return Pred->getLocationContext()->getAnalysisDeclContext();
  }

  /// \brief Generate a default checker node (containing checker tag but no
  /// checker state changes).
  ExplodedNode *generateNode() {
    return generateNode(getState());
  }
  
  /// \brief Generate a new checker node.
  ExplodedNode *generateNode(const ProgramState *state,
                             const ProgramPointTag *tag = 0) {
    return generateNodeImpl(state, false, 0, tag);
  }

  /// \brief Generate a new checker node with the given predecessor.
  /// Allows checkers to generate a chain of nodes.
  ExplodedNode *generateNode(const ProgramState *state,
                             ExplodedNode *pred,
                             const ProgramPointTag *tag = 0,
                             bool isSink = false) {
    return generateNodeImpl(state, isSink, pred, tag);
  }

  /// \brief Generate a sink node. Generating sink stops exploration of the
  /// given path.
  ExplodedNode *generateSink(const ProgramState *state = 0) {
    return generateNodeImpl(state ? state : getState(), true);
  }

  /// \brief Emit the diagnostics report.
  void EmitReport(BugReport *R) {
    Eng.getBugReporter().EmitReport(R);
  }

  void EmitBasicReport(StringRef Name,
                       StringRef Category,
                       StringRef Str, PathDiagnosticLocation Loc,
                       SourceRange* RBeg, unsigned NumRanges) {
    Eng.getBugReporter().EmitBasicReport(Name, Category, Str, Loc,
                                         RBeg, NumRanges);
  }

private:
  ExplodedNode *generateNodeImpl(const ProgramState *state,
                                 bool markAsSink,
                                 ExplodedNode *pred = 0,
                                 const ProgramPointTag *tag = 0) {
    assert(state);
    ExplodedNode *node = NB.generateNode(tag ? Location.withTag(tag) : Location,
                                        state,
                                        pred ? pred : Pred, markAsSink);
    return node;
  }
};

} // end GR namespace

} // end clang namespace

#endif
