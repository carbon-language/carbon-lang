//===- polly/LinkAllPasses.h ----------- Reference All Passes ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file pulls in all transformation and analysis passes for tools
// like opt and bugpoint that need this functionality.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_LINKALLPASSES_H
#define POLLY_LINKALLPASSES_H

#include "polly/CodeGen/PPCGCodeGeneration.h"
#include "polly/Config/config.h"
#include "polly/Support/DumpFunctionPass.h"
#include "polly/Support/DumpModulePass.h"
#include "llvm/ADT/StringRef.h"
#include <cstdlib>

namespace llvm {
class Pass;
class PassRegistry;
} // namespace llvm

namespace polly {
llvm::Pass *createCodePreparationPass();
llvm::Pass *createScopInlinerPass();
llvm::Pass *createDeadCodeElimWrapperPass();
llvm::Pass *createDependenceInfoPass();
llvm::Pass *createDependenceInfoWrapperPassPass();
llvm::Pass *createDOTOnlyPrinterPass();
llvm::Pass *createDOTOnlyViewerPass();
llvm::Pass *createDOTPrinterPass();
llvm::Pass *createDOTViewerPass();
llvm::Pass *createJSONExporterPass();
llvm::Pass *createJSONImporterPass();
llvm::Pass *createPollyCanonicalizePass();
llvm::Pass *createPolyhedralInfoPass();
llvm::Pass *createScopDetectionWrapperPassPass();
llvm::Pass *createScopInfoRegionPassPass();
llvm::Pass *createScopInfoWrapperPassPass();
llvm::Pass *createRewriteByrefParamsWrapperPass();
llvm::Pass *createIslAstInfoWrapperPassPass();
llvm::Pass *createCodeGenerationPass();
#ifdef GPU_CODEGEN
llvm::Pass *createPPCGCodeGenerationPass(GPUArch Arch = GPUArch::NVPTX64,
                                         GPURuntime Runtime = GPURuntime::CUDA);

llvm::Pass *
createManagedMemoryRewritePassPass(GPUArch Arch = GPUArch::NVPTX64,
                                   GPURuntime Runtime = GPURuntime::CUDA);
#endif
llvm::Pass *createIslScheduleOptimizerWrapperPass();
llvm::Pass *createFlattenSchedulePass();
llvm::Pass *createForwardOpTreeWrapperPass();
llvm::Pass *createDeLICMWrapperPass();
llvm::Pass *createMaximalStaticExpansionPass();
llvm::Pass *createSimplifyWrapperPass(int);
llvm::Pass *createPruneUnprofitableWrapperPass();

extern char &CodePreparationID;
} // namespace polly

namespace {
struct PollyForcePassLinking {
  PollyForcePassLinking() {
    // We must reference the passes in such a way that compilers will not
    // delete it all as dead code, even with whole program optimization,
    // yet is effectively a NO-OP. As the compiler isn't smart enough
    // to know that getenv() never returns -1, this will do the job.
    if (std::getenv("bar") != (char *)-1)
      return;

    polly::createCodePreparationPass();
    polly::createDeadCodeElimWrapperPass();
    polly::createDependenceInfoPass();
    polly::createDOTOnlyPrinterPass();
    polly::createDOTOnlyViewerPass();
    polly::createDOTPrinterPass();
    polly::createDOTViewerPass();
    polly::createJSONExporterPass();
    polly::createJSONImporterPass();
    polly::createScopDetectionWrapperPassPass();
    polly::createScopInfoRegionPassPass();
    polly::createPollyCanonicalizePass();
    polly::createPolyhedralInfoPass();
    polly::createRewriteByrefParamsWrapperPass();
    polly::createIslAstInfoWrapperPassPass();
    polly::createCodeGenerationPass();
#ifdef GPU_CODEGEN
    polly::createPPCGCodeGenerationPass();
    polly::createManagedMemoryRewritePassPass();
#endif
    polly::createIslScheduleOptimizerWrapperPass();
    polly::createMaximalStaticExpansionPass();
    polly::createFlattenSchedulePass();
    polly::createForwardOpTreeWrapperPass();
    polly::createDeLICMWrapperPass();
    polly::createDumpModuleWrapperPass("", true);
    polly::createDumpFunctionWrapperPass("");
    polly::createSimplifyWrapperPass(0);
    polly::createPruneUnprofitableWrapperPass();
  }
} PollyForcePassLinking; // Force link by creating a global definition.
} // namespace

namespace llvm {
class PassRegistry;
void initializeCodePreparationPass(llvm::PassRegistry &);
void initializeScopInlinerPass(llvm::PassRegistry &);
void initializeDeadCodeElimWrapperPassPass(llvm::PassRegistry &);
void initializeJSONExporterPass(llvm::PassRegistry &);
void initializeJSONImporterPass(llvm::PassRegistry &);
void initializeIslAstInfoWrapperPassPass(llvm::PassRegistry &);
void initializeCodeGenerationPass(llvm::PassRegistry &);
void initializeRewriteByrefParamsWrapperPassPass(llvm::PassRegistry &);
#ifdef GPU_CODEGEN
void initializePPCGCodeGenerationPass(llvm::PassRegistry &);
void initializeManagedMemoryRewritePassPass(llvm::PassRegistry &);
#endif
void initializeIslScheduleOptimizerWrapperPassPass(llvm::PassRegistry &);
void initializeMaximalStaticExpanderPass(llvm::PassRegistry &);
void initializePollyCanonicalizePass(llvm::PassRegistry &);
void initializeFlattenSchedulePass(llvm::PassRegistry &);
void initializeForwardOpTreeWrapperPassPass(llvm::PassRegistry &);
void initializeDeLICMWrapperPassPass(llvm::PassRegistry &);
void initializeSimplifyWrapperPassPass(llvm::PassRegistry &);
void initializePruneUnprofitableWrapperPassPass(llvm::PassRegistry &);
} // namespace llvm

#endif
