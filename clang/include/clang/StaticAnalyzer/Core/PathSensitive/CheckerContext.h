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

  AnalysisManager &getAnalysisManager() {
    return Eng.getAnalysisManager();
  }

  ConstraintManager &getConstraintManager() {
    return Eng.getConstraintManager();
  }

  StoreManager &getStoreManager() {
    return Eng.getStoreManager();
  }

  /// \brief Returns the previous node in the exploded graph, which includes
  /// the state of the program before the checker ran. Note, checkers should
  /// not retain the node in their state since the nodes might get invalidated.
  ExplodedNode *getPredecessor() { return Pred; }
  const ProgramState *getState() { return Pred->getState(); }

  /// \brief Returns the number of times the current block has been visited
  /// along the analyzed path.
  unsigned getCurrentBlockCount() {
    return NB.getContext().getCurrentBlockCount();
  }

  ASTContext &getASTContext() {
    return Eng.getContext();
  }
  
  const LocationContext *getLocationContext() {
    return Pred->getLocationContext();
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

  /// \brief Generates a new transition in the program state graph
  /// (ExplodedGraph). Uses the default CheckerContext predecessor node.
  ///
  /// @param State The state of the generated node.
  /// @param Tag The tag is used to uniquely identify the creation site. If no
  ///        tag is specified, a default tag, unique to the given checker,
  ///        will be used. Tags are used to prevent states generated at
  ///        different sites from caching out.
  ExplodedNode *addTransition(const ProgramState *State,
                              const ProgramPointTag *Tag = 0) {
    return addTransitionImpl(State, false, 0, Tag);
  }

  /// \brief Generates a default transition (containing checker tag but no
  /// checker state changes).
  ExplodedNode *addTransition() {
    return addTransition(getState());
  }

  /// \brief Generates a new transition with the given predecessor.
  /// Allows checkers to generate a chain of nodes.
  ///
  /// @param State The state of the generated node.
  /// @param Pred The transition will be generated from the specified Pred node
  ///             to the newly generated node.
  /// @param Tag The tag to uniquely identify the creation site.
  /// @param IsSink Mark the new node as sink, which will stop exploration of
  ///               the given path.
  ExplodedNode *addTransition(const ProgramState *State,
                             ExplodedNode *Pred,
                             const ProgramPointTag *Tag = 0,
                             bool IsSink = false) {
    return addTransitionImpl(State, IsSink, Pred, Tag);
  }

  /// \brief Generate a sink node. Generating sink stops exploration of the
  /// given path.
  ExplodedNode *generateSink(const ProgramState *state = 0) {
    return addTransitionImpl(state ? state : getState(), true);
  }

  /// \brief Emit the diagnostics report.
  void EmitReport(BugReport *R) {
    Eng.getBugReporter().EmitReport(R);
  }

private:
  ExplodedNode *addTransitionImpl(const ProgramState *State,
                                 bool MarkAsSink,
                                 ExplodedNode *P = 0,
                                 const ProgramPointTag *Tag = 0) {
    assert(State);
    if (State == Pred->getState() && !Tag && !MarkAsSink)
      return Pred;

    ExplodedNode *node = NB.generateNode(Tag ? Location.withTag(Tag) : Location,
                                        State,
                                        P ? P : Pred, MarkAsSink);
    return node;
  }
};

} // end GR namespace

} // end clang namespace

#endif
