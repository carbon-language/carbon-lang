//===- LiveVariables.h - Live Variable Analysis for Source CFGs -*- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements Live Variables analysis for source-level CFGs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIVEVARIABLES_H
#define LLVM_CLANG_LIVEVARIABLES_H

#include "clang/AST/Decl.h"
#include "clang/Analysis/Support/ExprDeclBitVector.h"
#include "clang/Analysis/FlowSensitive/DataflowValues.h"

namespace clang {

class Stmt;
class DeclRefExpr;
class SourceManager;
  
struct LiveVariables_ValueTypes {

  struct ObserverTy;


  // We need to keep track of both declarations and CFGBlock-level expressions,
  // (so that we don't explore such expressions twice), but we only need
  // liveness information for declarations (hence 
  // ValTy = DeclBitVector_Types::ValTy instead of 
  // ValTy = ExprDeclBitVector_Types::ValTy).

  struct AnalysisDataTy : public ExprDeclBitVector_Types::AnalysisDataTy {
    ObserverTy* Observer;
    
    AnalysisDataTy() : Observer(NULL) {}
  };

    // We only keep actual dataflow state for declarations.
  typedef DeclBitVector_Types::ValTy ValTy;
  
  //===-----------------------------------------------------===//
  // ObserverTy - Observer for uninitialized values queries.
  //===-----------------------------------------------------===//

  struct ObserverTy {
    virtual ~ObserverTy() {}
    
    /// ObserveStmt - A callback invoked right before invoking the
    ///  liveness transfer function on the given statement.
    virtual void ObserveStmt(Stmt* S, const AnalysisDataTy& AD, 
                             const ValTy& V) {}
    
    virtual void ObserverKill(DeclRefExpr* DR) {}
  };
};

class LiveVariables : public DataflowValues<LiveVariables_ValueTypes,
                                            dataflow::backward_analysis_tag> {
public:
  typedef LiveVariables_ValueTypes::ObserverTy ObserverTy;
    
  LiveVariables(CFG& cfg) { getAnalysisData().setCFG(&cfg); }
  
  /// IsLive - Return true if a variable is live at beginning of a
  /// specified block.
  bool isLive(const CFGBlock* B, const VarDecl* D) const;
                                              
  /// IsLive - Returns true if a variable is live at the beginning of the
  ///  the statement.  This query only works if liveness information
  ///  has been recorded at the statement level (see runOnAllBlocks), and
  ///  only returns liveness information for block-level expressions.
  bool isLive(const Stmt* S, const VarDecl* D) const;
  
  /// IsLive - Return true if a variable is live according to the
  ///  provided livness bitvector.
  bool isLive(const ValTy& V, const VarDecl* D) const;
  
  /// dumpLiveness - Print to stderr the liveness information encoded
  ///  by a specified bitvector.
  void dumpLiveness(const ValTy& V, SourceManager& M) const;
  
  /// dumpBlockLiveness - Print to stderr the liveness information
  ///  associated with each basic block.
  void dumpBlockLiveness(SourceManager& M) const;
  
  /// getNumDecls - Return the number of variables (declarations) that
  ///  whose liveness status is being tracked by the dataflow
  ///  analysis.
  unsigned getNumDecls() const { return getAnalysisData().getNumDecls(); }
    
  /// IntializeValues - Create initial dataflow values and meta data for
  ///  a given CFG.  This is intended to be called by the dataflow solver.
  void InitializeValues(const CFG& cfg);
  
  void runOnCFG(CFG& cfg);
                                              
  /// runOnAllBlocks - Propagate the dataflow values once for each block,
  ///  starting from the current dataflow values.  'recordStmtValues' indicates
  ///  whether the method should store dataflow values per each individual
  ///  block-level expression.
  void runOnAllBlocks(const CFG& cfg, ObserverTy* Obs,
                      bool recordStmtValues=false);
};

} // end namespace clang

#endif
