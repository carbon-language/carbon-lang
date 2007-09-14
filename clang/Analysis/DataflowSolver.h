//===--- DataflowSolver.h - Skeleton Dataflow Analysis Code -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines skeleton code for implementing dataflow analyses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSES_DATAFLOW_SOLVER
#define LLVM_CLANG_ANALYSES_DATAFLOW_SOLVER

#include "clang/AST/CFG.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace clang {

//===----------------------------------------------------------------------===//
/// DataflowWorkListTy - Data structure representing the worklist used for
///  dataflow algorithms.

class DataflowWorkListTy {
  typedef llvm::SmallPtrSet<const CFGBlock*,20> BlockSet;
  BlockSet wlist;
public:
  /// enqueue - Add a block to the worklist.  Blocks already on the worklist
  ///  are not added a second time.  
  void enqueue(const CFGBlock* B) { wlist.insert(B); }
  
  /// dequeue - Remove a block from the worklist.
  const CFGBlock* dequeue() {
    assert (!wlist.empty());
    const CFGBlock* B = *wlist.begin();
    wlist.erase(B);
    return B;          
  }
  
  /// isEmpty - Return true if the worklist is empty.
  bool isEmpty() const { return wlist.empty(); }
};

//===----------------------------------------------------------------------===//
/// DataflowSolverTy - Generic dataflow solver.
template <typename _DFValuesTy,      // Usually a subclass of DataflowValues
          typename _TransferFuncsTy,
          typename _MergeOperatorTy >
class DataflowSolver {

  //===--------------------------------------------------------------------===//
  // Type declarations.
  //===--------------------------------------------------------------------===//

public:
  typedef _DFValuesTy                            DFValuesTy;
  typedef _TransferFuncsTy                       TransferFuncsTy;
  typedef _MergeOperatorTy                       MergeOperatorTy;

  typedef typename _DFValuesTy::AnalysisDirTag   AnalysisDirTag;
  typedef typename _DFValuesTy::ValTy            ValTy;
  typedef typename _DFValuesTy::BlockDataMapTy   BlockDataMapTy;
  typedef typename _DFValuesTy::ObserverTy       ObserverTy;

  //===--------------------------------------------------------------------===//
  // External interface: constructing and running the solver.
  //===--------------------------------------------------------------------===//
  
public:
  DataflowSolver(DFValuesTy& d, ObserverTy* o = NULL) : D(d), O(o) {}
  ~DataflowSolver() {}  
  
  /// runOnCFG - Computes dataflow values for all blocks in a CFG.
  void runOnCFG(const CFG& cfg) {
    // Set initial dataflow values and boundary conditions.
    D.InitializeValues(cfg);     
    // Tag dispatch to the kind of analysis we do: forward or backwards.
    SolveDataflowEquations(cfg,typename _DFValuesTy::AnalysisDirTag());    
  }
  
  /// runOnBlock - Computes dataflow values for a given block.
  ///  This should usually be invoked only after previously computing
  ///  dataflow values using runOnCFG, as runOnBlock is intended to
  ///  only be used for querying the dataflow values within a block with
  ///  and Observer object.
  void runOnBlock(const CFGBlock* B) {
    TransferFuncsTy TF (D.getMetaData(),O);
    ProcessBlock(B,TF,AnalysisDirTag());
  }
  
  //===--------------------------------------------------------------------===//
  // Internal solver logic.
  //===--------------------------------------------------------------------===//
  
private:
 
  /// SolveDataflowEquations (FORWARD ANALYSIS) - Perform the actual
  ///  worklist algorithm to compute dataflow values.
  void SolveDataflowEquations(const CFG& cfg, dataflow::forward_analysis_tag) {
    // Create the worklist.
    DataflowWorkListTy WorkList;
    
    // Enqueue the ENTRY block.
    WorkList.enqueue(&cfg.getEntry());
    
    // Create the state for transfer functions.
    TransferFuncsTy TF(D.getMetaData(),O);
    
    // Process the worklist until it is empty.    
    while (!WorkList.isEmpty()) {
      const CFGBlock* B = WorkList.dequeue();
      // If the dataflow values at the block's exit have changed,
      // enqueue all successor blocks onto the worklist to have
      // their values updated.
      if (ProcessBlock(B,TF,AnalysisDirTag()))
        for (CFGBlock::const_succ_iterator I=B->succ_begin(), E=B->succ_end();
             I != E; ++I)
          WorkList.enqueue(*I);    
    }                     
  }       
  
  /// SolveDataflowEquations (BACKWARD ANALYSIS) - Perform the actual
  ///  worklist algorithm to compute dataflow values.  
  void SolveDataflowEquations(const CFG& cfg, dataflow::backward_analysis_tag) {
    // Create the worklist.
    DataflowWorkListTy WorkList;
    
    // Enqueue the EXIT block.
    WorkList.enqueue(&cfg.getExit());
    
    // Create the state for transfer functions.
    TransferFuncsTy TF(D.getMetaData(),O);
    
    // Process the worklist until it is empty.    
    while (!WorkList.isEmpty()) {
      const CFGBlock* B = WorkList.dequeue();
      // If the dataflow values at the block's entry have changed,
      // enqueue all predecessor blocks onto the worklist to have
      // their values updated.
      if (ProcessBlock(B,TF,AnalysisDirTag()))
        for (CFGBlock::const_pred_iterator I=B->pred_begin(), E=B->pred_end();
             I != E; ++I)
          WorkList.enqueue(*I);    
    }
  }
  
  /// ProcessBlock (FORWARD ANALYSIS) - Process the transfer functions
  ///  for a given block based on a forward analysis.
  bool ProcessBlock(const CFGBlock* B, TransferFuncsTy& TF, 
                    dataflow::forward_analysis_tag) {
    
    ValTy& V = TF.getVal();

    // Merge dataflow values from all predecessors of this block.
    V.resetValues();
    MergeOperatorTy Merge;
  
    for (CFGBlock::const_pred_iterator I=B->pred_begin(), 
                                       E=B->pred_end(); I!=E; ++I)
      Merge(V,D.getBlockData(*I));

    // Process the statements in the block in the forward direction.
    for (CFGBlock::const_iterator I=B->begin(), E=B->end(); I!=E; ++I)
      TF.BlockStmt_Visit(const_cast<Stmt*>(*I));
      
    return UpdateBlockValue(B,V);
  }
  
  /// ProcessBlock (BACKWARD ANALYSIS) - Process the transfer functions
  ///  for a given block based on a forward analysis.
  bool ProcessBlock(const CFGBlock* B, TransferFuncsTy& TF,
                    dataflow::backward_analysis_tag) {
    
    ValTy& V = TF.getVal();
    
    // Merge dataflow values from all predecessors of this block.
    V.resetValues();
    MergeOperatorTy Merge;
    
    for (CFGBlock::const_succ_iterator I=B->succ_begin(), 
                                       E=B->succ_end(); I!=E; ++I)
      Merge(V,D.getBlockData(*I));
    
    // Process the statements in the block in the forward direction.
    for (CFGBlock::const_reverse_iterator I=B->begin(), E=B->end(); I!=E; ++I)
      TF.BlockStmt_Visit(const_cast<Stmt*>(*I));
    
    return UpdateBlockValue(B,V);                        
  }
  
  /// UpdateBlockValue - After processing the transfer functions for a block,
  ///  update the dataflow value associated with the block.  Return true
  ///  if the block's value has changed.  We do lazy instantiation of block
  ///  values, so if the block value has not been previously computed we
  ///  obviously return true.
  bool UpdateBlockValue(const CFGBlock* B, ValTy& V) {
    BlockDataMapTy& M = D.getBlockDataMap();
    typename BlockDataMapTy::iterator I = M.find(B);
    
    if (I == M.end()) {
      M[B].copyValues(V);
      return true;
    }
    else if (!V.equal(I->second)) {
      I->second.copyValues(V);
      return true;
    }
    
    return false;
  }

private:
  DFValuesTy& D;
  ObserverTy* O;
};  
  

} // end namespace clang
#endif
