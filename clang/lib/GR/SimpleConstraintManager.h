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

#ifndef LLVM_CLANG_GR_SIMPLE_CONSTRAINT_MANAGER_H
#define LLVM_CLANG_GR_SIMPLE_CONSTRAINT_MANAGER_H

#include "clang/GR/PathSensitive/ConstraintManager.h"
#include "clang/GR/PathSensitive/GRState.h"

namespace clang {

namespace ento {

class SimpleConstraintManager : public ConstraintManager {
  SubEngine &SU;
public:
  SimpleConstraintManager(SubEngine &subengine) : SU(subengine) {}
  virtual ~SimpleConstraintManager();

  //===------------------------------------------------------------------===//
  // Common implementation for the interface provided by ConstraintManager.
  //===------------------------------------------------------------------===//

  bool canReasonAbout(SVal X) const;

  const GRState *assume(const GRState *state, DefinedSVal Cond,
                        bool Assumption);

  const GRState *assume(const GRState *state, Loc Cond, bool Assumption);

  const GRState *assume(const GRState *state, NonLoc Cond, bool Assumption);

  const GRState *assumeSymRel(const GRState *state,
                              const SymExpr *LHS,
                              BinaryOperator::Opcode op,
                              const llvm::APSInt& Int);

protected:

  //===------------------------------------------------------------------===//
  // Interface that subclasses must implement.
  //===------------------------------------------------------------------===//

  // Each of these is of the form "$sym+Adj <> V", where "<>" is the comparison
  // operation for the method being invoked.
  virtual const GRState *assumeSymNE(const GRState *state, SymbolRef sym,
                                     const llvm::APSInt& V,
                                     const llvm::APSInt& Adjustment) = 0;

  virtual const GRState *assumeSymEQ(const GRState *state, SymbolRef sym,
                                     const llvm::APSInt& V,
                                     const llvm::APSInt& Adjustment) = 0;

  virtual const GRState *assumeSymLT(const GRState *state, SymbolRef sym,
                                     const llvm::APSInt& V,
                                     const llvm::APSInt& Adjustment) = 0;

  virtual const GRState *assumeSymGT(const GRState *state, SymbolRef sym,
                                     const llvm::APSInt& V,
                                     const llvm::APSInt& Adjustment) = 0;

  virtual const GRState *assumeSymLE(const GRState *state, SymbolRef sym,
                                     const llvm::APSInt& V,
                                     const llvm::APSInt& Adjustment) = 0;

  virtual const GRState *assumeSymGE(const GRState *state, SymbolRef sym,
                                     const llvm::APSInt& V,
                                     const llvm::APSInt& Adjustment) = 0;

  //===------------------------------------------------------------------===//
  // Internal implementation.
  //===------------------------------------------------------------------===//

  const GRState *assumeAux(const GRState *state, Loc Cond,bool Assumption);

  const GRState *assumeAux(const GRState *state, NonLoc Cond, bool Assumption);
};

} // end GR namespace

} // end clang namespace

#endif
