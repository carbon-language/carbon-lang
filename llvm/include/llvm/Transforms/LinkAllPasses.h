//===- llvm/Transforms/LinkAllPasses.h - Reference All Passes ---*- C++ -*-===//
// 
//                      The LLVM Compiler Infrastructure
//
// This file was developed by Jeff Cohen and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This header file is required for building with Microsoft's VC++, as it has
// no way of linking all registered passes into executables other than by
// explicit use.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_LINKALLPASSES_H
#define LLVM_TRANSFORMS_LINKALLPASSES_H

#ifdef _MSC_VER

#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/LoadValueNumbering.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/UnifyFunctionExitNodes.h"

// Trying not to include <windows.h>, though maybe we should... Problem is,
// it pollutes the global namespace in some really nasty ways.
extern "C" __declspec(dllimport) void* __stdcall GetCurrentProcess();

namespace {
  struct ForcePassLinking {
    ForcePassLinking() {
      // We must reference the passes in such a way that VC++ will not
      // delete it all as dead code, even with whole program optimization,
      // yet is effectively a NO-OP. As the compiler isn't smart enough
      // to know that GetCurrentProcess() never returns
      // INVALID_HANDLE_VALUE, this will do the job.
      if (GetCurrentProcess() != (void *) -1)
        return;

      (void) llvm::createAAEvalPass();
      (void) llvm::createAggressiveDCEPass();
      (void) llvm::createAliasAnalysisCounterPass();
      (void) llvm::createAndersensPass();
      (void) llvm::createArgumentPromotionPass();
      (void) llvm::createBasicAliasAnalysisPass();
      (void) llvm::createBasicVNPass();
      (void) llvm::createBlockPlacementPass();
      (void) llvm::createBlockProfilerPass();
      (void) llvm::createBreakCriticalEdgesPass();
      (void) llvm::createCFGSimplificationPass();
      (void) llvm::createCombineBranchesPass();
      (void) llvm::createConstantMergePass();
      (void) llvm::createConstantPropagationPass();
      (void) llvm::createCorrelatedExpressionEliminationPass();
      (void) llvm::createDSAAPass();
      (void) llvm::createDSOptPass();
      (void) llvm::createDeadArgEliminationPass();
      (void) llvm::createDeadCodeEliminationPass();
      (void) llvm::createDeadInstEliminationPass();
      (void) llvm::createDeadStoreEliminationPass();
      (void) llvm::createDeadTypeEliminationPass();
      (void) llvm::createEdgeProfilerPass();
      (void) llvm::createEmitFunctionTablePass();
      (void) llvm::createFunctionInliningPass();
      (void) llvm::createFunctionProfilerPass();
      (void) llvm::createFunctionResolvingPass();
      (void) llvm::createGCSEPass();
      (void) llvm::createGlobalDCEPass();
      (void) llvm::createGlobalOptimizerPass();
      (void) llvm::createGlobalsModRefPass();
      (void) llvm::createIPConstantPropagationPass();
      (void) llvm::createIPSCCPPass();
      (void) llvm::createIndVarSimplifyPass();
      (void) llvm::createInstructionCombiningPass();
      (void) llvm::createInternalizePass();
      (void) llvm::createLICMPass();
      (void) llvm::createLoadValueNumberingPass();
      (void) llvm::createLoopExtractorPass();
      (void) llvm::createLoopInstrumentationPass();
      (void) llvm::createLoopSimplifyPass();
      (void) llvm::createLoopStrengthReducePass();
      (void) llvm::createLoopUnrollPass();
      (void) llvm::createLoopUnswitchPass();
      (void) llvm::createLowerAllocationsPass();
      (void) llvm::createLowerConstantExpressionsPass();
      (void) llvm::createLowerGCPass();
      (void) llvm::createLowerInvokePass();
      (void) llvm::createLowerPackedPass();
      (void) llvm::createLowerSelectPass();
      (void) llvm::createLowerSetJmpPass();
      (void) llvm::createLowerSwitchPass();
      (void) llvm::createNoAAPass();
      (void) llvm::createNoProfileInfoPass();
      (void) llvm::createPREPass();
      (void) llvm::createProfileLoaderPass();
      (void) llvm::createProfilePathsPass();
      (void) llvm::createPromoteMemoryToRegisterPass();
      (void) llvm::createPruneEHPass();
      (void) llvm::createRaiseAllocationsPass();
      (void) llvm::createRaisePointerReferencesPass();
      (void) llvm::createReassociatePass();
      (void) llvm::createSCCPPass();
      (void) llvm::createScalarReplAggregatesPass();
      (void) llvm::createSingleLoopExtractorPass();
      (void) llvm::createSteensgaardPass();
      (void) llvm::createStripSymbolsPass();
      (void) llvm::createTailCallEliminationPass();
      (void) llvm::createTailDuplicationPass();
      (void) llvm::createTraceBasicBlockPass();
      (void) llvm::createTraceValuesPassForBasicBlocks();
      (void) llvm::createTraceValuesPassForFunction();
      (void) llvm::createUnifyFunctionExitNodesPass();
    }
  } _ForcePassLinking;
};

#endif // _MSC_VER

#endif
