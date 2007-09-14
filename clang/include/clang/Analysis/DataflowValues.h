//===--- DataflowValues.h - Data structure for dataflow values --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a skeleton data structure for encapsulating the dataflow
// values for a CFG.  Typically this is subclassed to provide methods for
// computing these values from a CFG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSES_DATAFLOW_VALUES
#define LLVM_CLANG_ANALYSES_DATAFLOW_VALUES

#include "clang/AST/CFG.h"
#include "llvm/ADT/DenseMap.h"

namespace clang {

namespace dataflow {
  
  //===----------------------------------------------------------------------===//
  /// Dataflow Directional Tag Classes.  These are used for tag dispatching
  ///  within the dataflow solver/transfer functions to determine what direction
  ///  a dataflow analysis flows.
  //===----------------------------------------------------------------------===//   
  
  struct forward_analysis_tag {};
  struct backward_analysis_tag {};
  
} // end namespace dataflow
    

template <typename TypeClass,
          typename _AnalysisDirTag = dataflow::forward_analysis_tag >
class DataflowValues {

  //===--------------------------------------------------------------------===//
  // Type declarations.
  //===--------------------------------------------------------------------===//    

public:
  typedef typename TypeClass::ValTy                ValTy;
  typedef typename TypeClass::MetaDataTy           MetaDataTy;
  typedef typename TypeClass::ObserverTy           ObserverTy;
  
  typedef _AnalysisDirTag                          AnalysisDirTag;  
  typedef llvm::DenseMap<const CFGBlock*, ValTy>   BlockDataMapTy;

  //===--------------------------------------------------------------------===//
  // Predicates.
  //===--------------------------------------------------------------------===//

public:
  /// isForwardAnalysis - Returns true if the dataflow values are computed
  ///  from a forward analysis.  If this returns true, the value returned
  ///  from getBlockData() is the dataflow values associated with the END of
  ///  the block.
  bool isForwardAnalysis() { return isForwardAnalysis(AnalysisDirTag()); }
  
  /// isBackwardAnalysis - Returns true if the dataflow values are computed
  ///  from a backward analysis.  If this returns true, the value returned
  ///  from getBlockData() is the dataflow values associated with the ENTRY of
  ///  the block.
  bool isBackwardAnalysis() { return !isForwardAnalysis(); }
  
private:
  bool isForwardAnalysis(dataflow::forward_analysis_tag) { return true; }
  bool isForwardAnalysis(dataflow::backward_analysis_tag) { return false; }  
  
  //===--------------------------------------------------------------------===//
  // Initialization and accessors methods.
  //===--------------------------------------------------------------------===//

public:
  /// InitializeValues - Invoked by the solver to initialize state needed for
  ///  dataflow analysis.  This method is usually specialized by subclasses.
  void InitializeValues(const CFG& cfg) {};  

  /// getBlockData - Retrieves the dataflow values associated with a 
  ///  specified CFGBlock.  If the dataflow analysis is a forward analysis,
  ///  this data is associated with the END of the block.  If the analysis
  ///  is a backwards analysis, it is associated with the ENTRY of the block.
  ValTy& getBlockData(const CFGBlock* B) {
    typename BlockDataMapTy::iterator I = BlockDataMap.find(B);
    assert (I != BlockDataMap.end() && "No data associated with CFGBlock.");
    return I->second;
  }
  
  const ValTy& getBlockData(const CFGBlock*) const {
    return reinterpret_cast<DataflowValues*>(this)->getBlockData();
  }  
  
  /// getBlockDataMap - Retrieves the internal map between CFGBlocks and
  ///  dataflow values.  Usually used by a dataflow solver to compute
  ///  values for blocks.
  BlockDataMapTy& getBlockDataMap() { return BlockDataMap; }
  const BlockDataMapTy& getBlockDataMap() const { return BlockDataMap; }

  /// getMetaData - Retrieves the meta data associated with a dataflow analysis.
  ///  This is typically consumed by transfer function code (via the solver).
  ///  This can also be used by subclasses to interpret the dataflow values.
  MetaDataTy& getMetaData() { return Meta; }
  const MetaDataTy& getMetaData() const { return Meta; }
  
  //===--------------------------------------------------------------------===//
  // Internal data.
  //===--------------------------------------------------------------------===//
  
protected:
  BlockDataMapTy                     BlockDataMap;
  MetaDataTy     Meta;
};          

} // end namespace clang
#endif
