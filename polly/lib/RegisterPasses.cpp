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
DisableScheduler("polly-no-optimizer",
                 cl::desc("Disable Polly Scheduling Optimizer"), cl::Hidden,
                 cl::init(false));
static cl::opt<bool>
DisableCodegen("polly-no-codegen",
       cl::desc("Disable Polly Code Generation"), cl::Hidden,
       cl::init(false));
static cl::opt<bool>
UsePocc("polly-use-pocc",
       cl::desc("Use the PoCC optimizer instead of the one in isl"), cl::Hidden,
       cl::init(false));
static cl::opt<bool>
ImportJScop("polly-run-import-jscop",
            cl::desc("Export the JScop description of the detected Scops"),
            cl::Hidden, cl::init(false));
static cl::opt<bool>
ExportJScop("polly-run-export-jscop",
            cl::desc("Export the JScop description of the detected Scops"),
            cl::Hidden, cl::init(false));
static cl::opt<bool>
PollyViewer("polly-show",
       cl::desc("Enable the Polly DOT viewer in -O3"), cl::Hidden,
       cl::value_desc("Run the Polly DOT viewer at -O3"),
       cl::init(false));
static cl::opt<bool>
PollyOnlyViewer("polly-show-only",
       cl::desc("Enable the Polly DOT viewer in -O3 (no BB content)"),
       cl::Hidden,
       cl::value_desc("Run the Polly DOT viewer at -O3 (no BB content"),
       cl::init(false));
static cl::opt<bool>
PollyPrinter("polly-dot",
       cl::desc("Enable the Polly DOT printer in -O3"), cl::Hidden,
       cl::value_desc("Run the Polly DOT printer at -O3"),
       cl::init(false));
static cl::opt<bool>
PollyOnlyPrinter("polly-dot-only",
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
  initializeJSONExporterPass(Registry);
  initializeJSONImporterPass(Registry);
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
  bool RunScheduler = !DisableScheduler;
  bool RunCodegen = !DisableCodegen;

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
  // FIXME: The next two passes should not be necessary here. They are currently
  //        because of two problems:
  //
  //        1. The RegionSimplifyPass destroys the canonical form of induction
  //           variables,as it produces PHI nodes with incorrectly ordered
  //           operands. To fix this we run IndVarSimplify.
  //
  //        2. IndVarSimplify does not preserve the region information and
  //           the regioninfo pass does currently not recover simple regions.
  //           As a result we need to run the RegionSimplify pass again to
  //           recover them
  PM.add(llvm::createIndVarSimplifyPass());
  PM.add(polly::createRegionSimplifyPass());

  if (PollyViewer)
    PM.add(polly::createDOTViewerPass());
  if (PollyOnlyViewer)
    PM.add(polly::createDOTOnlyViewerPass());
  if (PollyPrinter)
    PM.add(polly::createDOTPrinterPass());
  if (PollyOnlyPrinter)
    PM.add(polly::createDOTOnlyPrinterPass());

  if (ImportJScop)
    PM.add(polly::createJSONImporterPass());

  if (RunScheduler) {
    if (UsePocc) {
#ifdef SCOPLIB_FOUND
      PM.add(polly::createPoccPass());
#else
      errs() << "Polly is compiled without scoplib support. As scoplib is "
             << "required to run PoCC, PoCC is also not available. Falling "
             << "back to the isl optimizer.\n";
      PM.add(polly::createIslScheduleOptimizerPass());
#endif
    } else {
      PM.add(polly::createIslScheduleOptimizerPass());
    }
  }

  if (ExportJScop)
    PM.add(polly::createJSONExporterPass());

  if (RunCodegen)
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
