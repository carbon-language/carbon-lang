//== SimpleConstraintManager.h ----------------------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Code shared between BasicConstraintManager and RangeConstraintManager.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SIMPLE_CONSTRAINT_MANAGER_H
#define LLVM_CLANG_ANALYSIS_SIMPLE_CONSTRAINT_MANAGER_H

#include "clang/Analysis/PathSensitive/ConstraintManager.h"
#include "clang/Analysis/PathSensitive/GRState.h"

namespace clang {

class SimpleConstraintManager : public ConstraintManager {
protected:
  GRStateManager& StateMgr;
public:
  SimpleConstraintManager(GRStateManager& statemgr) 
    : StateMgr(statemgr) {}
  virtual ~SimpleConstraintManager();
  
  bool canReasonAbout(SVal X) const;
  
  virtual const GRState* Assume(const GRState* St, SVal Cond, bool Assumption,
                                bool& isFeasible);

  const GRState* Assume(const GRState* St, Loc Cond, bool Assumption,
                        bool& isFeasible);

  const GRState* AssumeAux(const GRState* St, Loc Cond,bool Assumption,
                           bool& isFeasible);

  const GRState* Assume(const GRState* St, NonLoc Cond, bool Assumption,
                        bool& isFeasible);

  const GRState* AssumeAux(const GRState* St, NonLoc Cond, bool Assumption,
                           bool& isFeasible);

  const GRState* AssumeSymInt(const GRState* St, bool Assumption,
                              const SymIntExpr *SE, bool& isFeasible);

  virtual const GRState* AssumeSymNE(const GRState* St, SymbolRef sym,
                                     const llvm::APSInt& V,
                                     bool& isFeasible) = 0;

  virtual const GRState* AssumeSymEQ(const GRState* St, SymbolRef sym,
                                     const llvm::APSInt& V,
                                     bool& isFeasible) = 0;

  virtual const GRState* AssumeSymLT(const GRState* St, SymbolRef sym,
                                     const llvm::APSInt& V,
                                     bool& isFeasible) = 0;

  virtual const GRState* AssumeSymGT(const GRState* St, SymbolRef sym,
                                     const llvm::APSInt& V,
                                     bool& isFeasible) = 0;

  virtual const GRState* AssumeSymLE(const GRState* St, SymbolRef sym,
				     const llvm::APSInt& V,
				     bool& isFeasible) = 0;

  virtual const GRState* AssumeSymGE(const GRState* St, SymbolRef sym,
				     const llvm::APSInt& V,
				     bool& isFeasible) = 0;

  const GRState* AssumeInBound(const GRState* St, SVal Idx, SVal UpperBound,
                               bool Assumption, bool& isFeasible);

private:
  BasicValueFactory& getBasicVals() { return StateMgr.getBasicVals(); }
  SymbolManager& getSymbolManager() const { return StateMgr.getSymbolManager(); }
};

}  // end clang namespace

#endif
