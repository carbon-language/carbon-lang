//===- UninitializedValues.h - unintialized values analysis ----*- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides the interface for the Unintialized Values analysis,
// a flow-sensitive analysis that detects when variable values are unintialized.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNITVALS_H
#define LLVM_CLANG_UNITVALS_H

#include "llvm/ADT/BitVector.h"
#include "clang/Analysis/DataflowValues.h"

namespace clang {

  class VarDecl;
  class Expr;

/// UninitializedValues_ValueTypes - Utility class to wrap type declarations
///   for dataflow values and dataflow analysis state for the
///   Unitialized Values analysis.
class UninitializedValues_ValueTypes {
public:
  class ValTy {
    llvm::BitVector BV;
  public:
    // Accessors to internal bitvector.    
    unsigned size() const { return BV.size(); }
    void resize(unsigned i) { BV.resize(i); }
    llvm::BitVector::reference operator[](unsigned i) { return BV[i]; }
    void operator|=(const ValTy& RHS) { BV |= RHS.BV; }
    
    // Used by the solver.
    void resetValues() { BV.reset(); }
    bool equal(ValTy& RHS) const { return BV == RHS.BV; }
    void copyValues(ValTy& RHS) { BV = RHS.BV; }    
  };
  
  struct AnalysisDataTy {
    llvm::DenseMap<const VarDecl*, unsigned > VMap;
    llvm::DenseMap<const Expr*, unsigned > EMap;
    unsigned Counter;
    
    AnalysisDataTy() : Counter(0) {}
  };
};

/// UninitializedValues - Objects of this class encapsulate dataflow analysis
///  information regarding what variable declarations in a function are
///  potentially unintialized.
class UninitializedValues : 
  public DataflowValues<UninitializedValues_ValueTypes> {
  
  //===--------------------------------------------------------------------===//
  // Public interface.
  //===--------------------------------------------------------------------===//      
    
  static void CheckUninitializedValues(const CFG& cfg);

  //===--------------------------------------------------------------------===//
  // Internal logic.
  //===--------------------------------------------------------------------===//      
  
private:
  UninitializedValues() {}
  
public:
  /// IntializeValues - Create initial dataflow values and meta data for
  ///  a given CFG.  This is intended to be called by the dataflow solver.
  void InitializeValues(const CFG& cfg);
};

} // end namespace clang
#endif
