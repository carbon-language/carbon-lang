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
/// In addition to setting up the basic passes, PassManagerBuilder allows
/// frontends to vend a plugin API, where plugins are allowed to add extensions
/// to the default pass manager.  They do this by specifying where in the pass
/// pipeline they want to be added, along with a callback function that adds
/// the pass(es).  For example, a plugin that wanted to add a loop optimization
/// could do something like this:
///
/// static void addMyLoopPass(const PMBuilder &Builder, PassManagerBase &PM) {
///   if (Builder.getOptLevel() > 2 && Builder.getOptSizeLevel() == 0)
///     PM.add(createMyAwesomePass());
/// }
///   ...
///   Builder.addExtension(PassManagerBuilder::EP_LoopOptimizerEnd,
///                        addMyLoopPass);
///   ...
class PassManagerBuilder {
public:
  
  /// Extensions are passed the builder itself (so they can see how it is
  /// configured) as well as the pass manager to add stuff to.
  typedef void (*ExtensionFn)(const PassManagerBuilder &Builder,
                              PassManagerBase &PM);
  enum ExtensionPointTy {
    /// EP_EarlyAsPossible - This extension point allows adding passes before
    /// any other transformations, allowing them to see the code as it is coming
    /// out of the frontend.
    EP_EarlyAsPossible,
    
    /// EP_LoopOptimizerEnd - This extension point allows adding loop passes to
    /// the end of the loop optimizer.
    EP_LoopOptimizerEnd
  };
  
  /// The Optimization Level - Specify the basic optimization level.
  ///    0 = -O0, 1 = -O1, 2 = -O2, 3 = -O3
  unsigned OptLevel;
  
  /// SizeLevel - How much we're optimizing for size.
  ///    0 = none, 1 = -Os, 2 = -Oz
  unsigned SizeLevel;
  
  /// LibraryInfo - Specifies information about the runtime library for the
  /// optimizer.  If this is non-null, it is added to both the function and
  /// per-module pass pipeline.
  TargetLibraryInfo *LibraryInfo;
  
  /// Inliner - Specifies the inliner to use.  If this is non-null, it is
  /// added to the per-module passes.
  Pass *Inliner;
  
  bool DisableSimplifyLibCalls;
  bool DisableUnitAtATime;
  bool DisableUnrollLoops;
  
private:
  /// ExtensionList - This is list of all of the extensions that are registered.
  std::vector<std::pair<ExtensionPointTy, ExtensionFn> > Extensions;
  
public:
  PassManagerBuilder() {
    OptLevel = 2;
    SizeLevel = 0;
    LibraryInfo = 0;
    Inliner = 0;
    DisableSimplifyLibCalls = false;
    DisableUnitAtATime = false;
    DisableUnrollLoops = false;
  }
  
  ~PassManagerBuilder() {
    delete LibraryInfo;
    delete Inliner;
  }
  
  void addExtension(ExtensionPointTy Ty, ExtensionFn Fn) {
    Extensions.push_back(std::make_pair(Ty, Fn));
  }
  
private:
  void addExtensionsToPM(ExtensionPointTy ETy, PassManagerBase &PM) const {
    for (unsigned i = 0, e = Extensions.size(); i != e; ++i)
      if (Extensions[i].first == ETy)
        Extensions[i].second(*this, PM);
  }
  
  void addInitialAliasAnalysisPasses(PassManagerBase &PM) const {
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
    addExtensionsToPM(EP_EarlyAsPossible, FPM);
    
    // Add LibraryInfo if we have some.
    if (LibraryInfo) FPM.add(new TargetLibraryInfo(*LibraryInfo));

    if (OptLevel == 0) return;

    addInitialAliasAnalysisPasses(FPM);
    
    FPM.add(createCFGSimplificationPass());
    FPM.add(createScalarReplAggregatesPass());
    FPM.add(createEarlyCSEPass());
  }
  
  /// populateModulePassManager - This sets up the primary pass manager.
  void populateModulePassManager(PassManagerBase &MPM) {
    // If all optimizations are disabled, just run the always-inline pass.
    if (OptLevel == 0) {
      if (Inliner) {
        MPM.add(Inliner);
        Inliner = 0;
      }
      return;
    }
      
    // Add LibraryInfo if we have some.
    if (LibraryInfo) MPM.add(new TargetLibraryInfo(*LibraryInfo));

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
    if (Inliner) {
      MPM.add(Inliner);
      Inliner = 0;
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
    addExtensionsToPM(EP_LoopOptimizerEnd, MPM);

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
