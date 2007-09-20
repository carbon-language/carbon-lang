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
#include "functional" // STL

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
          typename _MergeOperatorTy,
          typename _Equal = std::equal_to<typename _DFValuesTy::ValTy> >
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
  typedef typename _DFValuesTy::EdgeDataMapTy    EdgeDataMapTy;

  //===--------------------------------------------------------------------===//
  // External interface: constructing and running the solver.
  //===--------------------------------------------------------------------===//
  
public:
  DataflowSolver(DFValuesTy& d) : D(d), TF(d.getAnalysisData()) {}
  ~DataflowSolver() {}  
  
  /// runOnCFG - Computes dataflow values for all blocks in a CFG.
  void runOnCFG(const CFG& cfg) {
    // Set initial dataflow values and boundary conditions.
    D.InitializeValues(cfg);     
    // Solve the dataflow equations.  This will populate D.EdgeDataMap
    // with dataflow values.
    SolveDataflowEquations(cfg);    
  }
  
  /// runOnBlock - Computes dataflow values for a given block.
  ///  This should usually be invoked only after previously computing
  ///  dataflow values using runOnCFG, as runOnBlock is intended to
  ///  only be used for querying the dataflow values within a block with
  ///  and Observer object.
  void runOnBlock(const CFGBlock* B) {
    if (hasData(B,AnalysisDirTag()))
      ProcessBlock(B,AnalysisDirTag());
  }
  
  void runOnBlock(const CFGBlock& B) { runOnBlock(&B); }
  void runOnBlock(CFG::iterator &I) { runOnBlock(*I); }
  void runOnBlock(CFG::const_iterator &I) { runOnBlock(*I); }

  void runOnAllBlocks(const CFG& cfg) {
    for (CFG::const_iterator I=cfg.begin(), E=cfg.end(); I!=E; ++I)
      runOnBlock(I);
  }
  
  //===--------------------------------------------------------------------===//
  // Internal solver logic.
  //===--------------------------------------------------------------------===//
  
private:
 
  /// SolveDataflowEquations - Perform the actual
  ///  worklist algorithm to compute dataflow values.  
  void SolveDataflowEquations(const CFG& cfg) {

    EnqueueFirstBlock(cfg,AnalysisDirTag());
    
    // Process the worklist until it is empty.    
    while (!WorkList.isEmpty()) {
      const CFGBlock* B = WorkList.dequeue();
      // If the dataflow values at the block's entry have changed,
      // enqueue all predecessor blocks onto the worklist to have
      // their values updated.
      ProcessBlock(B,AnalysisDirTag());
      UpdateEdges(B,TF.getVal(),AnalysisDirTag());
    }
  }
  
  void EnqueueFirstBlock(const CFG& cfg, dataflow::forward_analysis_tag) {
    WorkList.enqueue(&cfg.getEntry());
  }
  
  void EnqueueFirstBlock(const CFG& cfg, dataflow::backward_analysis_tag) {
    WorkList.enqueue(&cfg.getExit());
  }  
  
  /// ProcessBlock (FORWARD ANALYSIS) - Process the transfer functions
  ///  for a given block based on a forward analysis.
  void ProcessBlock(const CFGBlock* B, dataflow::forward_analysis_tag) {
      
    // Merge dataflow values from all predecessors of this block.
    ValTy& V = TF.getVal();
    V.resetValues(D.getAnalysisData());
    MergeOperatorTy Merge;
  
    EdgeDataMapTy& M = D.getEdgeDataMap();
    bool firstMerge = true;
  
    for (CFGBlock::const_pred_iterator I=B->pred_begin(), 
                                      E=B->pred_end(); I!=E; ++I) {
      typename EdgeDataMapTy::iterator BI = M.find(CFG::Edge(*I,B));
      if (BI != M.end()) {
        if (firstMerge) {
          firstMerge = false;
          V.copyValues(BI->second);
        }
        else
          Merge(V,BI->second);
      }
    }

    // Process the statements in the block in the forward direction.
    for (CFGBlock::const_iterator I=B->begin(), E=B->end(); I!=E; ++I)
      TF.BlockStmt_Visit(const_cast<Stmt*>(*I));      
  }
  
  /// ProcessBlock (BACKWARD ANALYSIS) - Process the transfer functions
  ///  for a given block based on a forward analysis.
  void ProcessBlock(const CFGBlock* B, TransferFuncsTy& TF,
                    dataflow::backward_analysis_tag) {
        
    // Merge dataflow values from all predecessors of this block.
    ValTy& V = TF.getVal();
    V.resetValues(D.getAnalysisData());
    MergeOperatorTy Merge;
    
    EdgeDataMapTy& M = D.getEdgeDataMap();
    bool firstMerge = true;

    for (CFGBlock::const_succ_iterator I=B->succ_begin(), 
                                       E=B->succ_end(); I!=E; ++I) {
      typename EdgeDataMapTy::iterator BI = M.find(CFG::Edge(B,*I));
      if (BI != M.end()) {
        if (firstMerge) {
          firstMerge = false;
          V.copyValues(BI->second);
        }
        else
          Merge(V,BI->second);
      }
    }
    
    // Process the statements in the block in the forward direction.
    for (CFGBlock::const_reverse_iterator I=B->begin(), E=B->end(); I!=E; ++I)
      TF.BlockStmt_Visit(const_cast<Stmt*>(*I));    
  }

  /// UpdateEdges (FORWARD ANALYSIS) - After processing the transfer
  ///   functions for a block, update the dataflow value associated with the
  ///   block's outgoing edges.  Enqueue any successor blocks for an
  ///   outgoing edge whose value has changed.  
  void UpdateEdges(const CFGBlock* B, ValTy& V,dataflow::forward_analysis_tag) {    
    for (CFGBlock::const_succ_iterator I=B->succ_begin(), E=B->succ_end();
          I!=E; ++I) {
                    
      CFG::Edge Edg(B,*I);
      UpdateEdgeValue(Edg,V,*I);
    }
  }
  
  /// UpdateEdges (BACKWARD ANALYSIS) - After processing the transfer
  ///   functions for a block, update the dataflow value associated with the
  ///   block's incoming edges.  Enqueue any predecessor blocks for an
  ///   outgoing edge whose value has changed.  
  void UpdateEdges(const CFGBlock* B, ValTy& V,dataflow::backward_analysis_tag){      
    for (CFGBlock::const_pred_iterator I=B->succ_begin(), E=B->succ_end();
         I!=E; ++I) {
      
      CFG::Edge Edg(*I,B);
      UpdateEdgeValue(Edg,V,*I);
    }
  }
  
  /// UpdateEdgeValue - Update the value associated with a given edge.
  void UpdateEdgeValue(CFG::Edge& E, ValTy& V, const CFGBlock* TargetBlock) {
  
    EdgeDataMapTy& M = D.getEdgeDataMap();
    typename EdgeDataMapTy::iterator I = M.find(E);
      
    if (I == M.end()) {
      // First value for this edge.
      M[E].copyValues(V);
      WorkList.enqueue(TargetBlock);
    }
    else if (!_Equal()(V,I->second)) {
      I->second.copyValues(V);
      WorkList.enqueue(TargetBlock);
    }
  }
  
  /// hasData (FORWARD ANALYSIS) - Is there any dataflow values associated
  ///  with the incoming edges of a block?
  bool hasData(const CFGBlock* B, dataflow::forward_analysis_tag) {  
    EdgeDataMapTy& M = D.getEdgeDataMap();

    for (CFGBlock::const_pred_iterator I=B->pred_begin(), E=B->pred_end();
         I!=E; ++I)
      if (M.find(CFG::Edge(*I,B)) != M.end())
        return true;
        
    return false;
  }
  
  /// hasData (BACKWARD ANALYSIS) - Is there any dataflow values associated
  ///  with the outgoing edges of a block?
  bool hasData(const CFGBlock* B, dataflow::backward_analysis_tag) {  
    EdgeDataMapTy& M = D.getEdgeDataMap();
    
    for (CFGBlock::const_succ_iterator I=B->succ_begin(), E=B->succ_end();
         I!=E; ++I)
      if (M.find(CFG::Edge(B,*I)) != M.end())
        return true;
    
    return false;
  }

private:
  DFValuesTy& D;
  DataflowWorkListTy WorkList;
  TransferFuncsTy TF;
};  
  

} // end namespace clang
#endif
