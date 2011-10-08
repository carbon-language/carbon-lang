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
#include "llvm/Support/CommandLine.h"

#include "polly/LinkAllPasses.h"

#include "polly/Cloog.h"
#include "polly/Dependences.h"
#include "polly/ScopDetection.h"
#include "polly/ScopInfo.h"
#include "polly/TempScopInfo.h"

using namespace llvm;

static cl::opt<bool>
PollyViewer("enable-polly-viewer",
       cl::desc("Enable the Polly DOT viewer in -O3"), cl::Hidden,
       cl::value_desc("Run the Polly DOT viewer at -O3"),
       cl::init(false));
static cl::opt<bool>
PollyOnlyViewer("enable-polly-only-viewer",
       cl::desc("Enable the Polly DOT viewer in -O3 (no BB content)"),
       cl::Hidden,
       cl::value_desc("Run the Polly DOT viewer at -O3 (no BB content"),
       cl::init(false));
static cl::opt<bool>
PollyPrinter("enable-polly-printer",
       cl::desc("Enable the Polly DOT printer in -O3"), cl::Hidden,
       cl::value_desc("Run the Polly DOT printer at -O3"),
       cl::init(false));
static cl::opt<bool>
PollyOnlyPrinter("enable-polly-only-printer",
       cl::desc("Enable the Polly DOT printer in -O3 (no BB content)"),
       cl::Hidden,
       cl::value_desc("Run the Polly DOT printer at -O3 (no BB content"),
       cl::init(false));

void initializePollyPasses(PassRegistry &Registry) {
  initializeCloogInfoPass(Registry);
  initializeCodeGenerationPass(Registry);
  initializeCodePreparationPass(Registry);
  initializeDependencesPass(Registry);
  initializeIndependentBlocksPass(Registry);
  initializeIslScheduleOptimizerPass(Registry);
#ifdef SCOPLIB_FOUND
  initializePoccPass(Registry);
#endif
  initializeRegionSimplifyPass(Registry);
  initializeScopDetectionPass(Registry);
  initializeScopInfoPass(Registry);
  initializeTempScopInfoPass(Registry);
}

// Statically register all Polly passes such that they are available after
// loading Polly.
class StaticInitializer {

public:
    StaticInitializer() {
      PassRegistry &Registry = *PassRegistry::getPassRegistry();
      initializePollyPasses(Registry);
    }
};

static StaticInitializer InitializeEverything;

static void registerPollyPasses(const llvm::PassManagerBuilder &Builder,
                                llvm::PassManagerBase &PM) {
  // Polly is only enabled at -O3
  if (Builder.OptLevel != 3)
    return;

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

  PM.add(polly::createCodePreparationPass());
  PM.add(polly::createRegionSimplifyPass());
  // FIXME: Needed as RegionSimplifyPass destroys the canonical form of
  //        induction variables (It changes the order of the operands in the
  //        PHI nodes).
  PM.add(llvm::createIndVarSimplifyPass());

  if (PollyViewer)
    PM.add(polly::createDOTViewerPass());
  if (PollyOnlyViewer)
    PM.add(polly::createDOTOnlyViewerPass());
  if (PollyPrinter)
    PM.add(polly::createDOTPrinterPass());
  if (PollyOnlyPrinter)
    PM.add(polly::createDOTOnlyPrinterPass());

  PM.add(polly::createIslScheduleOptimizerPass());
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
