//===------ RegisterPasses.cpp - Add the Polly Passes to default passes  --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Add the Polly passes to the optimization passes executed at -O3.
//
//===----------------------------------------------------------------------===//
#include "llvm/Analysis/Passes.h"
#include "llvm/InitializePasses.h"
#include "llvm/PassManager.h"
#include "llvm/PassRegistry.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

#include "polly/LinkAllPasses.h"

static void registerPollyPasses(const llvm::PassManagerBuilder &Builder,
                                llvm::PassManagerBase &PM) {
  // Polly is only enabled at -O3
  if (Builder.OptLevel != 3)
    return;

  // We need to initialize the passes before we use them.
  //
  // This is not necessary for the opt tool, however clang crashes if passes
  // are not initialized. (FIXME?)
  PassRegistry &Registry = *PassRegistry::getPassRegistry();
  initializeDominatorTreePass(Registry);
  initializePostDominatorTreePass(Registry);
  initializeLoopInfoPass(Registry);
  initializeScalarEvolutionPass(Registry);
  initializeRegionInfoPass(Registry);
  initializeDominanceFrontierPass(Registry);
  initializeAliasAnalysisAnalysisGroup(Registry);

  // A standard set of optimization passes partially taken/copied from the
  // set of default optimization passes. It is used to bring the code into
  // a canonical form that can than be analyzed by Polly. This set of passes is
  // most probably not yet optimal. TODO: Investigate optimal set of passes.
  PM.add(llvm::createPromoteMemoryToRegisterPass());
  PM.add(llvm::createInstructionCombiningPass());  // Clean up after IPCP & DAE
  PM.add(llvm::createCFGSimplificationPass());     // Clean up after IPCP & DAE
  PM.add(llvm::createTailCallEliminationPass());   // Eliminate tail calls
  PM.add(llvm::createCFGSimplificationPass());     // Merge & remove BBs
  PM.add(llvm::createReassociatePass());           // Reassociate expressions
  PM.add(llvm::createLoopRotatePass());            // Rotate Loop
  PM.add(llvm::createInstructionCombiningPass());
  PM.add(llvm::createIndVarSimplifyPass());        // Canonicalize indvars
  PM.add(llvm::createRegionInfoPass());

  PM.add(polly::createCodePreperationPass());
  PM.add(polly::createRegionSimplifyPass());

  // FIXME: Needed as RegionSimplifyPass destroys the canonical form of
  //        induction variables (It changes the order of the operands in the
  //        PHI nodes).
  PM.add(llvm::createIndVarSimplifyPass());
  PM.add(polly::createScopDetectionPass());
  PM.add(polly::createIndependentBlocksPass());

  // FIXME: We should not need to schedule passes like the TempScopInfoPass
  //        explicitally, as it is alread required by the ScopInfo pass.
  //        However, without this clang crashes because of unitialized passes.
  PM.add(polly::createTempScopInfoPass());
  PM.add(polly::createScopInfoPass());
  PM.add(polly::createDependencesPass());
  PM.add(polly::createScheduleOptimizerPass());
  PM.add(polly::createCloogInfoPass());
  PM.add(polly::createCodeGenerationPass());
}

// Execute Polly together with a set of preparing passes.
//
// We run Polly that early to run before loop optimizer passes like LICM or
// the LoopIdomPass. Both transform the code in a way that Polly will recognize
// less scops.
static llvm::RegisterStandardPasses
PassRegister(llvm::PassManagerBuilder::EP_EarlyAsPossible,
             registerPollyPasses);
