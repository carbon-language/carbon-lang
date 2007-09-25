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
#include "clang/Analysis/FlowSensitive/DataflowValues.h"

namespace clang {

  class BlockVarDecl;
  class Expr;
  class DeclRefExpr;
  class VarDecl;
  
/// UninitializedValues_ValueTypes - Utility class to wrap type declarations
///   for dataflow values and dataflow analysis state for the
///   Unitialized Values analysis.
class UninitializedValues_ValueTypes {
public:

  //===--------------------------------------------------------------------===//
  // AnalysisDataTy - Whole-function meta data used by the transfer function
  //  logic.
  //===--------------------------------------------------------------------===//
  
  struct ObserverTy;
  
  struct AnalysisDataTy {
    llvm::DenseMap<const BlockVarDecl*, unsigned > VMap;
    llvm::DenseMap<const Expr*, unsigned > EMap;
    unsigned NumDecls;
    unsigned NumBlockExprs;
    ObserverTy* Observer;
    
    AnalysisDataTy() : NumDecls(0), NumBlockExprs(0), Observer(NULL) {}
    
    bool isTracked(const BlockVarDecl* VD) { 
      return VMap.find(VD) != VMap.end();
    }
    
    bool isTracked(const Expr* E) {
      return EMap.find(E) != EMap.end();
    }
    
    unsigned& operator[](const BlockVarDecl *VD) { return VMap[VD]; }
    unsigned& operator[](const Expr* E) { return EMap[E]; }
  };

  //===--------------------------------------------------------------------===//
  // ValTy - Dataflow value.
  //===--------------------------------------------------------------------===//
  
  struct ValTy {
    llvm::BitVector DeclBV;
    llvm::BitVector ExprBV;

    void resetValues(AnalysisDataTy& AD) {
      DeclBV.resize(AD.NumDecls);
      DeclBV.reset();
      ExprBV.resize(AD.NumBlockExprs);
      ExprBV.reset();
    }
    
    bool operator==(const ValTy& RHS) const { 
      return DeclBV == RHS.DeclBV && ExprBV == RHS.ExprBV; 
    }
    
    void copyValues(const ValTy& RHS) {
      DeclBV = RHS.DeclBV;
      ExprBV = RHS.ExprBV;
    }

    llvm::BitVector::reference getBitRef(const BlockVarDecl* VD,
                                         AnalysisDataTy& AD) {
      assert (AD.isTracked(VD) && "BlockVarDecl not tracked.");
      return DeclBV[AD.VMap[VD]];
    }
    
    llvm::BitVector::reference getBitRef(const Expr* E,
                                         AnalysisDataTy& AD) {
      assert (AD.isTracked(E) && "Expr not tracked.");                                                                                   
      return ExprBV[AD.EMap[E]];
    }
    
    bool sizesEqual(ValTy& RHS) {
      return DeclBV.size() == RHS.DeclBV.size() &&
             ExprBV.size() == RHS.ExprBV.size();
    }
  };  
  
  //===--------------------------------------------------------------------===//
  // ObserverTy - Observer for querying DeclRefExprs that use an uninitalized
  //   value.
  //===--------------------------------------------------------------------===//
  
  struct ObserverTy {
    virtual ~ObserverTy();
    virtual void ObserveDeclRefExpr(ValTy& Val, AnalysisDataTy& AD, 
                                    DeclRefExpr* DR, BlockVarDecl* VD) = 0;
  };  
};

/// UninitializedValues - Objects of this class encapsulate dataflow analysis
///  information regarding what variable declarations in a function are
///  potentially unintialized.
class UninitializedValues : 
  public DataflowValues<UninitializedValues_ValueTypes> {  
public:
  typedef UninitializedValues_ValueTypes::ObserverTy ObserverTy;

  UninitializedValues() {}
  
  /// IntializeValues - Create initial dataflow values and meta data for
  ///  a given CFG.  This is intended to be called by the dataflow solver.
  void InitializeValues(const CFG& cfg);
};

} // end namespace clang
#endif
