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

#include "clang/Analysis/Support/SaveAndRestore.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"

namespace clang {

namespace ento {

class CheckerContext {
  ExplodedNodeSet &Dst;
  ExprEngine &Eng;
  ExplodedNode *Pred;
  const ProgramPoint Location;
  const ProgramState *ST;
  const unsigned size;
  // TODO: Use global context.
  NodeBuilderContext Ctx;
  NodeBuilder &NB;
public:
  bool *respondsToCallback;
public:
  CheckerContext(ExplodedNodeSet &dst,
                 NodeBuilder &builder,
                 ExprEngine &eng,
                 ExplodedNode *pred,
                 const ProgramPoint &loc,
                 bool *respondsToCB = 0,
                 const ProgramState *st = 0)
    : Dst(dst),
      Eng(eng),
      Pred(pred),
      Location(loc),
      ST(st),
      size(Dst.size()),
      Ctx(builder.C.Eng, builder.getBlock(), pred),
      NB(builder),
      respondsToCallback(respondsToCB) {
    assert(!(ST && ST != Pred->getState()));
  }

  ~CheckerContext();

  ExprEngine &getEngine() {
    return Eng;
  }

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
  const ProgramState *getState() { return ST ? ST : Pred->getState(); }

  /// \brief Returns the number of times the current block has been visited
  /// along the analyzed path.
  unsigned getCurrentBlockCount() {return NB.getCurrentBlockCount();}

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

  /// \brief Generate a default checker node (containing checker tag but no
  /// checker state changes).
  ExplodedNode *generateNode(bool autoTransition = true) {
    return generateNode(getState(), autoTransition);
  }
  
  /// \brief Generate a new checker node with the given predecessor.
  /// Allows checkers to generate a chain of nodes.
  ExplodedNode *generateNode(const ProgramState *state,
                             ExplodedNode *pred,
                             const ProgramPointTag *tag = 0,
                             bool autoTransition = true,
                             bool isSink = false) {
    ExplodedNode *N = generateNodeImpl(state, isSink, pred, tag);
    return N;
  }

  /// \brief Generate a new checker node.
  ExplodedNode *generateNode(const ProgramState *state,
                             bool autoTransition = true,
                             const ProgramPointTag *tag = 0) {
    ExplodedNode *N = generateNodeImpl(state, false, 0, tag);
    return N;
  }

  /// \brief Generate a sink node. Generating sink stops exploration of the
  /// given path.
  ExplodedNode *generateSink(const ProgramState *state = 0) {
    return generateNodeImpl(state ? state : getState(), true);
  }

  void addTransition(const ProgramState *state,
                     const ProgramPointTag *tag = 0) {
    assert(state);
    // If the 'state' is not new, we need to check if the cached state 'ST'
    // is new.
    if (state != getState())
      generateNode(state, true, tag);
  }

  void EmitReport(BugReport *R) {
    Eng.getBugReporter().EmitReport(R);
  }

  AnalysisContext *getCurrentAnalysisContext() const {
    return Pred->getLocationContext()->getAnalysisContext();
  }

private:
  ExplodedNode *generateNodeImpl(const ProgramState *state,
                                 bool markAsSink,
                                 ExplodedNode *pred = 0,
                                 const ProgramPointTag *tag = 0) {

    ExplodedNode *node = NB.generateNode(tag ? Location.withTag(tag) : Location,
                                        state,
                                        pred ? pred : Pred, markAsSink);
    return node;
  }
};

} // end GR namespace

} // end clang namespace

#endif
