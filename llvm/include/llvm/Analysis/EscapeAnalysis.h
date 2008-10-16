//===------------- EscapeAnalysis.h - Pointer escape analysis -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface for the pointer escape analysis.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_ESCAPEANALYSIS_H
#define LLVM_ANALYSIS_ESCAPEANALYSIS_H

#include "llvm/Pass.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Target/TargetData.h"
#include <set>

namespace llvm {

/// EscapeAnalysis - This class determines whether an allocation (a MallocInst 
/// or an AllocaInst) can escape from the current function.  It performs some
/// precomputation, with the rest of the work happening on-demand.
class EscapeAnalysis : public FunctionPass {
private:
  std::set<Instruction*> EscapePoints;

public:
  static char ID; // Class identification, replacement for typeinfo

  EscapeAnalysis() : FunctionPass(intptr_t(&ID)) {}

  bool runOnFunction(Function &F);
  
  void releaseMemory() {
    EscapePoints.clear();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequiredTransitive<TargetData>();
    AU.addRequiredTransitive<AliasAnalysis>();
    AU.setPreservesAll();
  }

  //===---------------------------------------------------------------------
  // Client API

  /// escapes - returns true if the value, which must have a pointer type,
  /// can escape.
  bool escapes(Value* A);
};

} // end llvm namespace

#endif
