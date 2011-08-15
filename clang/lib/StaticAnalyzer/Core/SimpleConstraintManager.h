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

#include "clang/StaticAnalyzer/Core/PathSensitive/ConstraintManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"

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

  const ProgramState *assume(const ProgramState *state, DefinedSVal Cond,
                        bool Assumption);

  const ProgramState *assume(const ProgramState *state, Loc Cond, bool Assumption);

  const ProgramState *assume(const ProgramState *state, NonLoc Cond, bool Assumption);

  const ProgramState *assumeSymRel(const ProgramState *state,
                              const SymExpr *LHS,
                              BinaryOperator::Opcode op,
                              const llvm::APSInt& Int);

protected:

  //===------------------------------------------------------------------===//
  // Interface that subclasses must implement.
  //===------------------------------------------------------------------===//

  // Each of these is of the form "$sym+Adj <> V", where "<>" is the comparison
  // operation for the method being invoked.
  virtual const ProgramState *assumeSymNE(const ProgramState *state, SymbolRef sym,
                                     const llvm::APSInt& V,
                                     const llvm::APSInt& Adjustment) = 0;

  virtual const ProgramState *assumeSymEQ(const ProgramState *state, SymbolRef sym,
                                     const llvm::APSInt& V,
                                     const llvm::APSInt& Adjustment) = 0;

  virtual const ProgramState *assumeSymLT(const ProgramState *state, SymbolRef sym,
                                     const llvm::APSInt& V,
                                     const llvm::APSInt& Adjustment) = 0;

  virtual const ProgramState *assumeSymGT(const ProgramState *state, SymbolRef sym,
                                     const llvm::APSInt& V,
                                     const llvm::APSInt& Adjustment) = 0;

  virtual const ProgramState *assumeSymLE(const ProgramState *state, SymbolRef sym,
                                     const llvm::APSInt& V,
                                     const llvm::APSInt& Adjustment) = 0;

  virtual const ProgramState *assumeSymGE(const ProgramState *state, SymbolRef sym,
                                     const llvm::APSInt& V,
                                     const llvm::APSInt& Adjustment) = 0;

  //===------------------------------------------------------------------===//
  // Internal implementation.
  //===------------------------------------------------------------------===//

  const ProgramState *assumeAux(const ProgramState *state, Loc Cond,bool Assumption);

  const ProgramState *assumeAux(const ProgramState *state, NonLoc Cond, bool Assumption);
};

} // end GR namespace

} // end clang namespace

#endif
