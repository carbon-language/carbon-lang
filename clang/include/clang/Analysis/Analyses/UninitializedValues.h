//===- UninitializedValues.h - unintialized values analysis ----*- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides the interface for the Unintialized Values analysis,
// a flow-sensitive analysis that detects when variable values are unintialized.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNITVALS_H
#define LLVM_CLANG_UNITVALS_H

#include "clang/Analysis/Support/BlkExprDeclBitVector.h"
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

  struct ObserverTy;

  struct AnalysisDataTy : public StmtDeclBitVector_Types::AnalysisDataTy {
    AnalysisDataTy() : Observer(NULL), FullUninitTaint(true) {}
    virtual ~AnalysisDataTy() {}

    ObserverTy* Observer;
    bool FullUninitTaint;
  };

  typedef StmtDeclBitVector_Types::ValTy ValTy;

  //===--------------------------------------------------------------------===//
  // ObserverTy - Observer for querying DeclRefExprs that use an uninitalized
  //   value.
  //===--------------------------------------------------------------------===//

  struct ObserverTy {
    virtual ~ObserverTy();
    virtual void ObserveDeclRefExpr(ValTy& Val, AnalysisDataTy& AD,
                                    DeclRefExpr* DR, VarDecl* VD) = 0;
  };
};

/// UninitializedValues - Objects of this class encapsulate dataflow analysis
///  information regarding what variable declarations in a function are
///  potentially unintialized.
class UninitializedValues :
  public DataflowValues<UninitializedValues_ValueTypes> {
public:
  typedef UninitializedValues_ValueTypes::ObserverTy ObserverTy;

  UninitializedValues(CFG &cfg) { getAnalysisData().setCFG(cfg); }

  /// IntializeValues - Create initial dataflow values and meta data for
  ///  a given CFG.  This is intended to be called by the dataflow solver.
  void InitializeValues(const CFG& cfg);
};

} // end namespace clang
#endif
