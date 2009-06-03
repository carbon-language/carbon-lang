//===-- llvm/Support/StandardPasses.h - Standard pass lists -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines utility functions for creating a "standard" set of
// optimization passes, so that compilers and tools which use optimization
// passes use the same set of standard passes.
//
// These are implemented as inline functions so that we do not have to worry
// about link issues.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_STANDARDPASSES_H
#define LLVM_SUPPORT_STANDARDPASSES_H

#include "llvm/PassManager.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/IPO.h"

namespace llvm {
  /// createStandardFunctionPasses - Add the standard list of function passes to
  /// the provided pass manager.
  ///
  /// \arg OptimizationLevel - The optimization level, corresponding to -O0,
  /// -O1, etc.
  static inline void createStandardFunctionPasses(FunctionPassManager *PM,
                                                  unsigned OptimizationLevel);

  /// createStandardModulePasses - Add the standard list of module passes to the
  /// provided pass manager.
  ///
  /// \arg OptimizationLevel - The optimization level, corresponding to -O0,
  /// -O1, etc.
  /// \arg OptimizeSize - Whether the transformations should optimize for size.
  /// \arg UnitAtATime - Allow passes which may make global module changes.
  /// \arg UnrollLoops - Allow loop unrolling.
  /// \arg SimplifyLibCalls - Allow library calls to be simplified.
  /// \arg HaveExceptions - Whether the module may have code using exceptions.
  /// \arg InliningPass - The inlining pass to use, if any, or null. This will
  /// always be added, even at -O0.a
  static inline void createStandardModulePasses(PassManager *PM,
                                                unsigned OptimizationLevel,
                                                bool OptimizeSize,
                                                bool UnitAtATime,
                                                bool UnrollLoops,
                                                bool SimplifyLibCalls,
                                                bool HaveExceptions,
                                                Pass *InliningPass);

  // Implementations

  static inline void createStandardFunctionPasses(FunctionPassManager *PM,
                                                  unsigned OptimizationLevel) {
    if (OptimizationLevel > 0) {
      PM->add(createCFGSimplificationPass());
      if (OptimizationLevel == 1)
        PM->add(createPromoteMemoryToRegisterPass());
      else
        PM->add(createScalarReplAggregatesPass());
      PM->add(createInstructionCombiningPass());
    }
  }

  static inline void createStandardModulePasses(PassManager *PM,
                                                unsigned OptimizationLevel,
                                                bool OptimizeSize,
                                                bool UnitAtATime,
                                                bool UnrollLoops,
                                                bool SimplifyLibCalls,
                                                bool HaveExceptions,
                                                Pass *InliningPass) {
    if (OptimizationLevel == 0) {
      if (InliningPass)
        PM->add(InliningPass);
    } else {
      if (UnitAtATime)
        PM->add(createRaiseAllocationsPass());    // call %malloc -> malloc inst
      PM->add(createCFGSimplificationPass());     // Clean up disgusting code
       // Kill useless allocas
      PM->add(createPromoteMemoryToRegisterPass());
      if (UnitAtATime) {
        PM->add(createGlobalOptimizerPass());     // Optimize out global vars
        PM->add(createGlobalDCEPass());           // Remove unused fns and globs
        // IP Constant Propagation
        PM->add(createIPConstantPropagationPass());
        PM->add(createDeadArgEliminationPass());  // Dead argument elimination
      }
      PM->add(createInstructionCombiningPass());  // Clean up after IPCP & DAE
      PM->add(createCFGSimplificationPass());     // Clean up after IPCP & DAE
      if (UnitAtATime) {
        if (HaveExceptions)
          PM->add(createPruneEHPass());           // Remove dead EH info
        PM->add(createFunctionAttrsPass());       // Set readonly/readnone attrs
      }
      if (InliningPass)
        PM->add(InliningPass);
      if (OptimizationLevel > 2)
        PM->add(createArgumentPromotionPass());   // Scalarize uninlined fn args
      if (SimplifyLibCalls)
        PM->add(createSimplifyLibCallsPass());    // Library Call Optimizations
      PM->add(createInstructionCombiningPass());  // Cleanup for scalarrepl.
      PM->add(createJumpThreadingPass());         // Thread jumps.
      PM->add(createCFGSimplificationPass());     // Merge & remove BBs
      PM->add(createScalarReplAggregatesPass());  // Break up aggregate allocas
      PM->add(createInstructionCombiningPass());  // Combine silly seq's
      PM->add(createCondPropagationPass());       // Propagate conditionals
      PM->add(createTailCallEliminationPass());   // Eliminate tail calls
      PM->add(createCFGSimplificationPass());     // Merge & remove BBs
      PM->add(createReassociatePass());           // Reassociate expressions
      PM->add(createLoopRotatePass());            // Rotate Loop
      PM->add(createLICMPass());                  // Hoist loop invariants
      PM->add(createLoopUnswitchPass(OptimizeSize ? true : false));
      PM->add(createLoopIndexSplitPass());        // Split loop index
      PM->add(createInstructionCombiningPass());  
      PM->add(createIndVarSimplifyPass());        // Canonicalize indvars
      PM->add(createLoopDeletionPass());          // Delete dead loops
      if (UnrollLoops)
        PM->add(createLoopUnrollPass());          // Unroll small loops
      PM->add(createInstructionCombiningPass());  // Clean up after the unroller
      PM->add(createGVNPass());                   // Remove redundancies
      PM->add(createMemCpyOptPass());             // Remove memcpy / form memset
      PM->add(createSCCPPass());                  // Constant prop with SCCP
    
      // Run instcombine after redundancy elimination to exploit opportunities
      // opened up by them.
      PM->add(createInstructionCombiningPass());
      PM->add(createCondPropagationPass());       // Propagate conditionals
      PM->add(createDeadStoreEliminationPass());  // Delete dead stores
      PM->add(createAggressiveDCEPass());         // Delete dead instructions
      PM->add(createCFGSimplificationPass());     // Merge & remove BBs

      if (UnitAtATime) {
        PM->add(createStripDeadPrototypesPass()); // Get rid of dead prototypes
        PM->add(createDeadTypeEliminationPass()); // Eliminate dead types
      }

      if (OptimizationLevel > 1 && UnitAtATime)
        PM->add(createConstantMergePass());       // Merge dup global constants
    }
  }
}

#endif
