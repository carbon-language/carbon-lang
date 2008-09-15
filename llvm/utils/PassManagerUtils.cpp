//===-- PassManagerUtils.cpp - --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements pass manager utiliy routines.
//
//===----------------------------------------------------------------------===//

#include "llvm/PassManagerUtils.h"
#include "llvm/PassManagers.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Analysis/LoopPass.h"

/// AddOptimizationPasses - This routine adds optimization passes 
/// based on selected optimization level, OptLevel. This routine is
/// used by llvm-gcc and other tools.
///
/// OptLevel - Optimization Level
/// EnableIPO - Enables IPO passes. llvm-gcc enables this when
///             flag_unit_at_a_time is set.
/// InlinerSelection - 1 : Add function inliner.
///                  - 2 : Add AlwaysInliner.
/// OptLibCalls - Simplify lib calls, if set.
/// PruneEH - Add PruneEHPass, if set.
/// UnrollLoop - Unroll loops, if set.
void llvm::AddOptimizationPasses(FunctionPassManager &FPM, PassManager &MPM,
                                 unsigned OptLevel, bool EnableIPO,
                                 unsigned InlinerSelection, bool OptLibCalls,
                                 bool PruneEH, bool UnrollLoop) {
  if (OptLevel == 0) 
    return;

  FPM.add(createCFGSimplificationPass());
  if (OptLevel == 1)
    FPM.add(createPromoteMemoryToRegisterPass());
  else
    FPM.add(createScalarReplAggregatesPass());
  FPM.add(createInstructionCombiningPass());

  if (EnableIPO)
    MPM.add(createRaiseAllocationsPass());      // call %malloc -> malloc inst
  MPM.add(createCFGSimplificationPass());       // Clean up disgusting code
  MPM.add(createPromoteMemoryToRegisterPass()); // Kill useless allocas
  if (EnableIPO) {
    MPM.add(createGlobalOptimizerPass());       // OptLevel out global vars
    MPM.add(createGlobalDCEPass());             // Remove unused fns and globs
    MPM.add(createIPConstantPropagationPass()); // IP Constant Propagation
    MPM.add(createDeadArgEliminationPass());    // Dead argument elimination
  }
  MPM.add(createInstructionCombiningPass());    // Clean up after IPCP & DAE
  MPM.add(createCFGSimplificationPass());       // Clean up after IPCP & DAE
  if (EnableIPO && PruneEH)
    MPM.add(createPruneEHPass());               // Remove dead EH info
  if (InlinerSelection == 1)                    // respect -fno-inline-functions
    MPM.add(createFunctionInliningPass());      // Inline small functions
  else if (InlinerSelection == 2)
    MPM.add(createAlwaysInlinerPass());         // Inline always_inline functions
  if (OptLevel > 2)
    MPM.add(createArgumentPromotionPass());   // Scalarize uninlined fn args
  if (OptLibCalls)
    MPM.add(createSimplifyLibCallsPass());    // Library Call Optimizations
  MPM.add(createInstructionCombiningPass());  // Cleanup for scalarrepl.
  MPM.add(createJumpThreadingPass());         // Thread jumps.
  MPM.add(createCFGSimplificationPass());     // Merge & remove BBs
  MPM.add(createScalarReplAggregatesPass());  // Break up aggregate allocas
  MPM.add(createInstructionCombiningPass());  // Combine silly seq's
  MPM.add(createCondPropagationPass());       // Propagate conditionals
  MPM.add(createTailCallEliminationPass());   // Eliminate tail calls
  MPM.add(createCFGSimplificationPass());     // Merge & remove BBs
  MPM.add(createReassociatePass());           // Reassociate expressions
  MPM.add(createLoopRotatePass());            // Rotate Loop
  MPM.add(createLICMPass());                  // Hoist loop invariants
  MPM.add(createLoopUnswitchPass());
  MPM.add(createLoopIndexSplitPass());        // Split loop index
  MPM.add(createInstructionCombiningPass());  
  MPM.add(createIndVarSimplifyPass());        // Canonicalize indvars
  MPM.add(createLoopDeletionPass());          // Delete dead loops
  if (UnrollLoop)
    MPM.add(createLoopUnrollPass());          // Unroll small loops
  MPM.add(createInstructionCombiningPass());  // Clean up after the unroller
  MPM.add(createGVNPass());                   // Remove redundancies
  MPM.add(createMemCpyOptPass());             // Remove memcpy / form memset
  MPM.add(createSCCPPass());                  // Constant prop with SCCP
  
  // Run instcombine after redundancy elimination to exploit opportunities
  // opened up by them.
  MPM.add(createInstructionCombiningPass());
  MPM.add(createCondPropagationPass());       // Propagate conditionals
  MPM.add(createDeadStoreEliminationPass());  // Delete dead stores
  MPM.add(createAggressiveDCEPass());   // Delete dead instructions
  MPM.add(createCFGSimplificationPass());     // Merge & remove BBs
  
  if (EnableIPO) {
    MPM.add(createStripDeadPrototypesPass());   // Get rid of dead prototypes
    MPM.add(createDeadTypeEliminationPass());   // Eliminate dead types
  }
  
  if (OptLevel > 1 && EnableIPO)
    MPM.add(createConstantMergePass());       // Merge dup global constants 
  
  return;
}
