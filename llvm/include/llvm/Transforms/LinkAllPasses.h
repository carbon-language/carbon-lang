//===- llvm/Transforms/IPO.h - Interprocedural Transformations --*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
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

#include "llvm/Config/config.h"

#ifdef LLVM_ON_WIN32

#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/UnifyFunctionExitNodes.h"

// Trying not to include <windows.h>, though maybe we should...
extern "C" __declspec(dllimport) void* __stdcall GetCurrentProcess();

namespace {
    struct ForceLinking {
        ForceLinking() {
            // We must reference the passes in such a way that VC++ will not
            // delete it all as dead code, even with whole program optimization,
            // yet is effectively a NO-OP.  As the compiler isn't smart enough
            // to know that GetCurrentProcess() never returns
            // INVALID_HANDLE_VALUE, this will do the job.
            if (GetCurrentProcess() != (void *) -1)
                return;

            std::vector<llvm::BasicBlock*> bbv;

            // The commented out calls below refer to non-existant creation
            // functions.  They will be uncommented as the functions are added.

            // (void) llvm::createADCEPass();
            // (void) llvm::createArgPromotionPass();
            // (void) llvm::createBasicBlockTracerPass();
            (void) llvm::createBlockExtractorPass(bbv);
            // (void) llvm::createBlockPlacementPass();
            // (void) llvm::createBlockProfilerPass();
            (void) llvm::createBreakCriticalEdgesPass();
            // (void) llvm::createCEEPass();
            // (void) llvm::createCFGSimplifyPass();
            (void) llvm::createCombineBranchesPass();
            // (void) llvm::createConstantExpressionsLowerPass();
            (void) llvm::createConstantMergePass();
            (void) llvm::createConstantPropagationPass();
            // (void) llvm::createDAEPass();
            // (void) llvm::createDCEPass();
            // (void) llvm::createDSEPass();
            // (void) llvm::createDTEPass();
            (void) llvm::createDeadInstEliminationPass();
            // (void) llvm::createEdgeProfilerPass();
            (void) llvm::createEmitFunctionTablePass();
            // (void) llvm::createFunctionProfilerPass();
            (void) llvm::createFunctionResolvingPass();
            // (void) llvm::createFunctionTracerPass();
            (void) llvm::createGCSEPass();
            (void) llvm::createGlobalDCEPass();
            (void) llvm::createGlobalOptimizerPass();
            // (void) llvm::createIPCPPass();
            (void) llvm::createIPSCCPPass();
            (void) llvm::createIndVarSimplifyPass();
            // (void) llvm::createInstCombinerPass();
            // (void) llvm::createInstLoopsPass();
            (void) llvm::createInternalizePass();
            (void) llvm::createLICMPass();
            // (void) llvm::createLoopExtractorPass();
            (void) llvm::createLoopSimplifyPass();
            (void) llvm::createLoopStrengthReducePass();
            (void) llvm::createLoopUnrollPass();
            (void) llvm::createLoopUnswitchPass();
            (void) llvm::createLowerAllocationsPass();
            (void) llvm::createLowerGCPass();
            (void) llvm::createLowerInvokePass();
            (void) llvm::createLowerPackedPass();
            (void) llvm::createLowerSelectPass();
            (void) llvm::createLowerSetJmpPass();
            (void) llvm::createLowerSwitchPass();
            // (void) llvm::createPREPass();
            // (void) llvm::createProfilePathsPass();
            // (void) llvm::createPromotePass();
            (void) llvm::createPruneEHPass();
            // (void) llvm::createRPRPass();
            (void) llvm::createRaiseAllocationsPass();
            (void) llvm::createReassociatePass();
            (void) llvm::createSCCPPass();
            // (void) llvm::createSROAPass();
            // (void) llvm::createSimpleInlinerPass();
            (void) llvm::createSingleLoopExtractorPass();
            (void) llvm::createStripSymbolsPass();
            (void) llvm::createTailCallEliminationPass();
            (void) llvm::createTailDuplicationPass();
            // (void) llvm::createTraceBasicBlocksPass();
            (void) llvm::createUnifyFunctionExitNodesPass();
        }
    } X;
};

#endif // LLVM_ON_WIN32

#endif
