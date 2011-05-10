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
#include "llvm/PassSupport.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/IPO.h"

namespace llvm {

  /// RegisterStandardPassLists solves a circular dependency problem.  The
  /// default list of passes has to live somewhere.  It can't live in the core
  /// modules, because these don't link to the libraries that actually define
  /// the passes.  It's in this header, so that a copy is created in every
  /// library that requests the default set, while still allowing plugins to
  /// register new passes without requiring them to link anything more than
  /// VMCore.
  class RegisterStandardPassLists {
    public:
    RegisterStandardPassLists() {
      StandardPass::RegisterDefaultPasses = RegisterStandardPassList;
    }
    private:
    /// Passes must be registered with functions that take no arguments, so we
    /// have to wrap their existing constructors.  
    static Pass *createDefaultScalarReplAggregatesPass(void) {
      return createScalarReplAggregatesPass(-1, false);
    }
    static Pass *createDefaultLoopUnswitchPass(void) {
      return createLoopUnswitchPass(false);
    }
    static Pass *createSizeOptimizingLoopUnswitchPass(void) {
      return createLoopUnswitchPass(true);
    }
    static void RegisterStandardPassList(void) {
      // Standard alias analysis passes

      // Add TypeBasedAliasAnalysis before BasicAliasAnalysis so that
      // BasicAliasAnalysis wins if they disagree. This is intended to help
      // support "obvious" type-punning idioms.
#define DEFAULT_ALIAS_ANALYSIS_PASS(pass, flags)\
  StandardPass::RegisterDefaultPass(\
    PassInfo::NormalCtor_t(create ## pass ## Pass),\
    &DefaultStandardPasses::pass ## ID, 0, StandardPass::AliasAnalysis, flags)
      DEFAULT_ALIAS_ANALYSIS_PASS(BasicAliasAnalysis, 0);
      DEFAULT_ALIAS_ANALYSIS_PASS(TypeBasedAliasAnalysis, 0);
#undef DEFAULT_ALIAS_ANALYSIS_PASS

#define DEFAULT_FUNCTION_PASS(pass, flags)\
  StandardPass::RegisterDefaultPass(\
      PassInfo::NormalCtor_t(create ## pass ## Pass),\
      &DefaultStandardPasses::pass ## ID, 0, StandardPass::Function, flags)
      DEFAULT_FUNCTION_PASS(CFGSimplification,
          StandardPass::OptimizationFlags(1));
      DEFAULT_FUNCTION_PASS(ScalarReplAggregates,
          StandardPass::OptimizationFlags(1));
      DEFAULT_FUNCTION_PASS(EarlyCSE, StandardPass::OptimizationFlags(1));
#undef DEFAULT_FUNCTION_PASS

#define DEFAULT_MODULE_PASS(pass, flags)\
  StandardPass::RegisterDefaultPass(\
      PassInfo::NormalCtor_t(create ## pass ## Pass),\
      &DefaultStandardPasses::pass ## ID, 0, StandardPass::Module, flags)
      // Optimize out global vars
      DEFAULT_MODULE_PASS(GlobalOptimizer, StandardPass::UnitAtATime);
      // IP SCCP
      DEFAULT_MODULE_PASS(IPSCCP,
          StandardPass::OptimizationFlags(0, 0, StandardPass::UnitAtATime));
      // Dead argument elimination
      DEFAULT_MODULE_PASS(DeadArgElimination,
          StandardPass::OptimizationFlags(0, 0, StandardPass::UnitAtATime));
      // Clean up after IPCP & DAE
      DEFAULT_MODULE_PASS(InstructionCombining,
          StandardPass::OptimizationFlags(0, 0, StandardPass::UnitAtATime));
      // Clean up after IPCP & DAE
      DEFAULT_MODULE_PASS(CFGSimplification,
          StandardPass::OptimizationFlags(0, 0, StandardPass::UnitAtATime));

      // Placeholder that will be replaced by an inliner if one is specified
      StandardPass::RegisterDefaultPass(0,
        &DefaultStandardPasses::InlinerPlaceholderID , 0,
        StandardPass::Module);
      // Remove dead EH info
      DEFAULT_MODULE_PASS(PruneEH, StandardPass::OptimizationFlags(0, 0,
            StandardPass::UnitAtATime | StandardPass::HaveExceptions));
      // Set readonly/readnone attrs
      DEFAULT_MODULE_PASS(FunctionAttrs, StandardPass::OptimizationFlags(0, 0,
            StandardPass::UnitAtATime));
      // Scalarize uninlined fn args
      DEFAULT_MODULE_PASS(ArgumentPromotion, 2);
      // Start of function pass.
      // Break up aggregate allocas, using SSAUpdater.
      StandardPass::RegisterDefaultPass(
          PassInfo::NormalCtor_t(createDefaultScalarReplAggregatesPass),
          &DefaultStandardPasses::ScalarReplAggregatesID, 0,
          StandardPass::Module);
      // Catch trivial redundancies
      DEFAULT_MODULE_PASS(EarlyCSE, 0);
      // Library Call Optimizations
      DEFAULT_MODULE_PASS(SimplifyLibCalls, StandardPass::SimplifyLibCalls);
      // Thread jumps
      DEFAULT_MODULE_PASS(JumpThreading, 0);
      // Propagate conditionals
      DEFAULT_MODULE_PASS(CorrelatedValuePropagation, 0);
      // Merge & remove BBs
      DEFAULT_MODULE_PASS(CFGSimplification, 0);
      // Combine silly seq's
      DEFAULT_MODULE_PASS(InstructionCombining, 0);
      // Eliminate tail calls
      DEFAULT_MODULE_PASS(TailCallElimination, 0);
      // Merge & remove BBs
      DEFAULT_MODULE_PASS(CFGSimplification, 0);
      // Reassociate expressions
      DEFAULT_MODULE_PASS(Reassociate, 0);
      // Rotate Loop
      DEFAULT_MODULE_PASS(LoopRotate, 0);
      // Hoist loop invariants
      DEFAULT_MODULE_PASS(LICM, 0);
      // Optimize for size if the optimzation level is 0-2
      StandardPass::RegisterDefaultPass(
          PassInfo::NormalCtor_t(createSizeOptimizingLoopUnswitchPass),
          &DefaultStandardPasses::LoopUnswitchID, 0,
          StandardPass::Module,
          StandardPass::OptimizationFlags(0, 2));
      // Optimize for size if the optimzation level is >2, and OptimizeSize is
      // set
      StandardPass::RegisterDefaultPass(
          PassInfo::NormalCtor_t(createSizeOptimizingLoopUnswitchPass),
          &DefaultStandardPasses::LoopUnswitchID, 0,
          StandardPass::Module,
          StandardPass::OptimizationFlags(0, 3, StandardPass::OptimizeSize));
      // Don't optimize for size if optimisation level is >2 and OptimizeSize
      // is not set
      StandardPass::RegisterDefaultPass(
          PassInfo::NormalCtor_t(createSizeOptimizingLoopUnswitchPass),
          &DefaultStandardPasses::LoopUnswitchID, 0,
          StandardPass::Module,
          StandardPass::OptimizationFlags(0, 3, 0, StandardPass::OptimizeSize));
      DEFAULT_MODULE_PASS(InstructionCombining, 0);
      // Canonicalize indvars
      DEFAULT_MODULE_PASS(IndVarSimplify, 0);
      // Recognize idioms like memset.
      DEFAULT_MODULE_PASS(LoopIdiom, 0);
      // Delete dead loops
      DEFAULT_MODULE_PASS(LoopDeletion, 0);
      // Unroll small loops
      DEFAULT_MODULE_PASS(LoopUnroll, StandardPass::UnrollLoops);
      // Remove redundancies
      DEFAULT_MODULE_PASS(GVN, 2);
      // Remove memcpy / form memset
      DEFAULT_MODULE_PASS(MemCpyOpt, 0);
      // Constant prop with SCCP
      DEFAULT_MODULE_PASS(SCCP, 0);

      // Run instcombine after redundancy elimination to exploit opportunities
      // opened up by them.
      DEFAULT_MODULE_PASS(InstructionCombining, 0);
      // Thread jumps
      DEFAULT_MODULE_PASS(JumpThreading, 0);
      DEFAULT_MODULE_PASS(CorrelatedValuePropagation, 0);
      // Delete dead stores
      DEFAULT_MODULE_PASS(DeadStoreElimination, 0);
      // Delete dead instructions
      DEFAULT_MODULE_PASS(AggressiveDCE, 0);
      // Merge & remove BBs
      DEFAULT_MODULE_PASS(CFGSimplification, 0);
      // Clean up after everything.
      DEFAULT_MODULE_PASS(InstructionCombining, 0);

      // Get rid of dead prototypes
      DEFAULT_MODULE_PASS(StripDeadPrototypes, StandardPass::UnitAtATime);
      // Eliminate dead types
      DEFAULT_MODULE_PASS(DeadTypeElimination, StandardPass::UnitAtATime);

      // GlobalOpt already deletes dead functions and globals, at -O3 try a
      // late pass of GlobalDCE.  It is capable of deleting dead cycles.
      // Remove dead fns and globals.
      DEFAULT_MODULE_PASS(GlobalDCE, 3 | StandardPass::UnitAtATime);
      // Merge dup global constants
      DEFAULT_MODULE_PASS(ConstantMerge, 2 | StandardPass::UnitAtATime);
#undef DEFAULT_MODULE_PASS

#define DEFAULT_LTO_PASS(pass, flags)\
  StandardPass::RegisterDefaultPass(\
      PassInfo::NormalCtor_t(create ## pass ## Pass),\
      &DefaultStandardPasses::pass ## ID, 0, StandardPass::LTO, flags)

      // LTO passes

      // Propagate constants at call sites into the functions they call.  This
      // opens opportunities for globalopt (and inlining) by substituting
      // function pointers passed as arguments to direct uses of functions.  
      DEFAULT_LTO_PASS(IPSCCP, 0);

      // Now that we internalized some globals, see if we can hack on them!
      DEFAULT_LTO_PASS(GlobalOptimizer, 0);
      
      // Linking modules together can lead to duplicated global constants, only
      // keep one copy of each constant...
      DEFAULT_LTO_PASS(ConstantMerge, 0);
      
      // Remove unused arguments from functions...
      DEFAULT_LTO_PASS(DeadArgElimination, 0);

      // Reduce the code after globalopt and ipsccp.  Both can open up
      // significant simplification opportunities, and both can propagate
      // functions through function pointers.  When this happens, we often have
      // to resolve varargs calls, etc, so let instcombine do this.
      DEFAULT_LTO_PASS(InstructionCombining, 0);

      // Inline small functions
      DEFAULT_LTO_PASS(FunctionInlining,
          StandardPass::OptimizationFlags(0, 0xf, StandardPass::RunInliner));
      // Remove dead EH info.
      DEFAULT_LTO_PASS(PruneEH, 0);
      // Optimize globals again if we ran the inliner.
      DEFAULT_LTO_PASS(GlobalOptimizer,
          StandardPass::OptimizationFlags(0, 0xf, StandardPass::RunInliner));
      DEFAULT_LTO_PASS(GlobalDCE, 0);

      // If we didn't decide to inline a function, check to see if we can
      // transform it to pass arguments by value instead of by reference.
      DEFAULT_LTO_PASS(ArgumentPromotion, 0);

      // The IPO passes may leave cruft around.  Clean up after them.
      DEFAULT_LTO_PASS(InstructionCombining, 0);
      DEFAULT_LTO_PASS(JumpThreading, 0);
      // Break up allocas
      DEFAULT_LTO_PASS(ScalarReplAggregates, 0);

      // Run a few AA driven optimizations here and now, to cleanup the code.
      // Add nocapture.
      DEFAULT_LTO_PASS(FunctionAttrs, 0);
      // IP alias analysis.
      DEFAULT_LTO_PASS(GlobalsModRef, 0);

      // Hoist loop invariants.
      DEFAULT_LTO_PASS(LICM, 0);
      // Remove redundancies.
      DEFAULT_LTO_PASS(GVN, 0);
      // Remove dead memcpys.
      DEFAULT_LTO_PASS(MemCpyOpt, 0);
      // Nuke dead stores.
      DEFAULT_LTO_PASS(DeadStoreElimination, 0);

      // Cleanup and simplify the code after the scalar optimizations.
      DEFAULT_LTO_PASS(InstructionCombining, 0);

      DEFAULT_LTO_PASS(JumpThreading, 0);
      
      // Delete basic blocks, which optimization passes may have killed.
      DEFAULT_LTO_PASS(CFGSimplification, 0);

      // Now that we have optimized the program, discard unreachable functions.
      DEFAULT_LTO_PASS(GlobalDCE, 0);
#undef DEFAULT_LTO_PASS
    }
  };
  static RegisterStandardPassLists AutoRegister;


  static inline void createStandardAliasAnalysisPasses(PassManagerBase *PM) {
    StandardPass::AddPassesFromSet(PM, StandardPass::AliasAnalysis);
  }

  /// createStandardFunctionPasses - Add the standard list of function passes to
  /// the provided pass manager.
  ///
  /// \arg OptimizationLevel - The optimization level, corresponding to -O0,
  /// -O1, etc.
  static inline void createStandardFunctionPasses(PassManagerBase *PM,
                                                  unsigned OptimizationLevel) {
    StandardPass::AddPassesFromSet(PM, StandardPass::Function, OptimizationLevel);
  }

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
  /// always be added, even at -O0.
  static inline void createStandardModulePasses(PassManagerBase *PM,
                                                unsigned OptimizationLevel,
                                                bool OptimizeSize,
                                                bool UnitAtATime,
                                                bool UnrollLoops,
                                                bool SimplifyLibCalls,
                                                bool HaveExceptions,
                                                Pass *InliningPass) {
    createStandardAliasAnalysisPasses(PM);

    // If all optimizations are disabled, just run the always-inline pass.
    if (OptimizationLevel == 0) {
      if (InliningPass)
        PM->add(InliningPass);
      return;
    }

    StandardPass::AddPassesFromSet(PM, StandardPass::Module,
      StandardPass::OptimizationFlags(OptimizationLevel,
        (OptimizeSize ? StandardPass::OptimizeSize : 0) |
        (UnitAtATime ? StandardPass::UnitAtATime : 0) |
        (UnrollLoops ? StandardPass::UnrollLoops : 0) |
        (SimplifyLibCalls ? StandardPass::SimplifyLibCalls : 0) |
        (HaveExceptions ? StandardPass::HaveExceptions : 0)),
      InliningPass);
    
  }

  /// createStandardLTOPasses - Add the standard list of module passes suitable
  /// for link time optimization.
  ///
  /// Internalize - Run the internalize pass.
  /// RunInliner - Use a function inlining pass.
  /// VerifyEach - Run the verifier after each pass.
  static inline void createStandardLTOPasses(PassManagerBase *PM,
                                             bool Internalize,
                                             bool RunInliner,
                                             bool VerifyEach) {
    // Provide AliasAnalysis services for optimizations.
    createStandardAliasAnalysisPasses(PM);

    // Now that composite has been compiled, scan through the module, looking
    // for a main function.  If main is defined, mark all other functions
    // internal.
    if (Internalize) {
      PM->add(createInternalizePass(true));
      if (VerifyEach)
        PM->add(createVerifierPass());
    }

    StandardPass::AddPassesFromSet(PM, StandardPass::LTO,
      StandardPass::OptimizationFlags(0, 0, RunInliner ?
        StandardPass::RunInliner : 0), VerifyEach);
  }
}

#endif
