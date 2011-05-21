//===-- llvm/Support/PasaMangerBuilder.h - Build Standard Pass --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the PassManagerBuilder class, which is used to set up a
// "standard" optimization sequence suitable for languages like C and C++.
//
// These are implemented as inline functions so that we do not have to worry
// about link issues.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_PASSMANAGERBUILDER_H
#define LLVM_SUPPORT_PASSMANAGERBUILDER_H

#include "llvm/PassManager.h"
#include "llvm/DefaultPasses.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/IPO.h"

namespace llvm {
  
/// PassManagerBuilder - This class is used to set up a standard optimization
/// sequence for languages like C and C++, allowing some APIs to customize the
/// pass sequence in various ways. A simple example of using it would be:
///
///  OptimizerBuilder Builder;
///  Builder.setOptimizationLevel(2);
///  Builder.populateFunctionPassManager(FPM);
///  Builder.populateModulePassManager(MPM);
///
class PassManagerBuilder {
  unsigned OptLevel;   // 0 = -O0, 1 = -O1, 2 = -O2, 3 = -O3
  unsigned SizeLevel;  // 0 = none, 1 = -Os, 2 = -Oz
  TargetLibraryInfo *TLI;
  Pass *InlinerPass;
  
  bool DisableSimplifyLibCalls;
  bool DisableUnitAtATime;
  bool DisableUnrollLoops;
public:
  PassManagerBuilder() {
    OptLevel = 2;
    SizeLevel = 0;
    TLI = 0;
    InlinerPass = 0;
    DisableSimplifyLibCalls = false;
    DisableUnitAtATime = false;
    DisableUnrollLoops = false;
  }
  
  ~PassManagerBuilder() {
    delete TLI;
    delete InlinerPass;
  }
  
  /// setOptimizationLevel - Specify the basic optimization level -O0 ... -O3.
  void setOptimizationLevel(unsigned L) { OptLevel = L; }

  /// setSizeLevel - Specify the size optimization level: none, -Os, -Oz.
  void setSizeLevel(unsigned L) { SizeLevel = L; }

  /// setLibraryInfo - Set information about the runtime library for the
  /// optimizer.  If this is specified, it is added to both the function and
  /// per-module pass pipeline.
  void setLibraryInfo(TargetLibraryInfo *LI) { TLI = LI; }
  
  /// setInliner - Specify the inliner to use.  If this is specified, it is
  /// added to the per-module passes.
  void setInliner(Pass *P) { InlinerPass = P; }
  
  
  void disableSimplifyLibCalls() { DisableSimplifyLibCalls = true; }
  void disableUnitAtATime() { DisableUnitAtATime = true; }
  void disableUnrollLoops() { DisableUnrollLoops = true; }
  
private:
  void addInitialAliasAnalysisPasses(PassManagerBase &PM) {
    // Add TypeBasedAliasAnalysis before BasicAliasAnalysis so that
    // BasicAliasAnalysis wins if they disagree. This is intended to help
    // support "obvious" type-punning idioms.
    PM.add(createTypeBasedAliasAnalysisPass());
    PM.add(createBasicAliasAnalysisPass());
  }
public:
  
  /// populateFunctionPassManager - This fills in the function pass manager,
  /// which is expected to be run on each function immediately as it is
  /// generated.  The idea is to reduce the size of the IR in memory.
  void populateFunctionPassManager(FunctionPassManager &FPM) {
    if (OptLevel == 0) return;
    
    // Add TLI if we have some.
    if (TLI) FPM.add(new TargetLibraryInfo(*TLI));
    
    addInitialAliasAnalysisPasses(FPM);
    
    FPM.add(createCFGSimplificationPass());
    FPM.add(createScalarReplAggregatesPass());
    FPM.add(createEarlyCSEPass());
  }
  
  /// populateModulePassManager - This sets up the primary pass manager.
  void populateModulePassManager(PassManagerBase &MPM) {
    // If all optimizations are disabled, just run the always-inline pass.
    if (OptLevel == 0) {
      if (InlinerPass) {
        MPM.add(InlinerPass);
        InlinerPass = 0;
      }
      return;
    }
      
    // Add TLI if we have some.
    if (TLI) MPM.add(new TargetLibraryInfo(*TLI));

    addInitialAliasAnalysisPasses(MPM);
    
    if (!DisableUnitAtATime) {
      MPM.add(createGlobalOptimizerPass());     // Optimize out global vars
      
      MPM.add(createIPSCCPPass());              // IP SCCP
      MPM.add(createDeadArgEliminationPass());  // Dead argument elimination
      
      MPM.add(createInstructionCombiningPass());// Clean up after IPCP & DAE
      MPM.add(createCFGSimplificationPass());   // Clean up after IPCP & DAE
    }
    
    // Start of CallGraph SCC passes.
    if (!DisableUnitAtATime)
      MPM.add(createPruneEHPass());             // Remove dead EH info
    if (InlinerPass) {
      MPM.add(InlinerPass);
      InlinerPass = 0;
    }
    if (!DisableUnitAtATime)
      MPM.add(createFunctionAttrsPass());       // Set readonly/readnone attrs
    if (OptLevel > 2)
      MPM.add(createArgumentPromotionPass());   // Scalarize uninlined fn args
    
    // Start of function pass.
    // Break up aggregate allocas, using SSAUpdater.
    MPM.add(createScalarReplAggregatesPass(-1, false));
    MPM.add(createEarlyCSEPass());              // Catch trivial redundancies
    if (!DisableSimplifyLibCalls)
      MPM.add(createSimplifyLibCallsPass());    // Library Call Optimizations
    MPM.add(createJumpThreadingPass());         // Thread jumps.
    MPM.add(createCorrelatedValuePropagationPass()); // Propagate conditionals
    MPM.add(createCFGSimplificationPass());     // Merge & remove BBs
    MPM.add(createInstructionCombiningPass());  // Combine silly seq's
    
    MPM.add(createTailCallEliminationPass());   // Eliminate tail calls
    MPM.add(createCFGSimplificationPass());     // Merge & remove BBs
    MPM.add(createReassociatePass());           // Reassociate expressions
    MPM.add(createLoopRotatePass());            // Rotate Loop
    MPM.add(createLICMPass());                  // Hoist loop invariants
    MPM.add(createLoopUnswitchPass(SizeLevel || OptLevel < 3));
    MPM.add(createInstructionCombiningPass());  
    MPM.add(createIndVarSimplifyPass());        // Canonicalize indvars
    MPM.add(createLoopIdiomPass());             // Recognize idioms like memset.
    MPM.add(createLoopDeletionPass());          // Delete dead loops
    if (!DisableUnrollLoops)
      MPM.add(createLoopUnrollPass());          // Unroll small loops
    if (OptLevel > 1)
      MPM.add(createGVNPass());                 // Remove redundancies
    MPM.add(createMemCpyOptPass());             // Remove memcpy / form memset
    MPM.add(createSCCPPass());                  // Constant prop with SCCP
    
    // Run instcombine after redundancy elimination to exploit opportunities
    // opened up by them.
    MPM.add(createInstructionCombiningPass());
    MPM.add(createJumpThreadingPass());         // Thread jumps
    MPM.add(createCorrelatedValuePropagationPass());
    MPM.add(createDeadStoreEliminationPass());  // Delete dead stores
    MPM.add(createAggressiveDCEPass());         // Delete dead instructions
    MPM.add(createCFGSimplificationPass());     // Merge & remove BBs
    MPM.add(createInstructionCombiningPass());  // Clean up after everything.
    
    if (!DisableUnitAtATime) {
      MPM.add(createStripDeadPrototypesPass()); // Get rid of dead prototypes
      MPM.add(createDeadTypeEliminationPass()); // Eliminate dead types
      
      // GlobalOpt already deletes dead functions and globals, at -O3 try a
      // late pass of GlobalDCE.  It is capable of deleting dead cycles.
      if (OptLevel > 2)
        MPM.add(createGlobalDCEPass());         // Remove dead fns and globals.
      
      if (OptLevel > 1)
        MPM.add(createConstantMergePass());     // Merge dup global constants
    }
  }
};

  
} // end namespace llvm
#endif
