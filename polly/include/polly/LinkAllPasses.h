//===- polly/LinkAllPasses.h ------------ Reference All Passes ---*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header file pulls in all transformation and analysis passes for tools
// like opt and bugpoint that need this functionality.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_LINKALLPASSES_H
#define POLLY_LINKALLPASSES_H

#include "polly/Config/config.h"
#include <cstdlib>

namespace llvm {
  class Pass;
  class PassInfo;
  class PassRegistry;
  class RegionPass;
}

using namespace llvm;

namespace polly {
  Pass *createAffSCEVItTesterPass();
#ifdef CLOOG_FOUND
  Pass *createCloogExporterPass();
  Pass *createCloogInfoPass();
  Pass *createCodeGenerationPass();
#endif
  Pass *createCodePreparationPass();
  Pass *createDeadCodeElimPass();
  Pass *createDependencesPass();
  Pass *createDOTOnlyPrinterPass();
  Pass *createDOTOnlyViewerPass();
  Pass *createDOTPrinterPass();
  Pass *createDOTViewerPass();
  Pass *createIndependentBlocksPass();
  Pass *createIndVarSimplifyPass();
  Pass *createJSONExporterPass();
  Pass *createJSONImporterPass();
  Pass *createRegionSimplifyPass();
  Pass *createScopDetectionPass();
  Pass *createScopInfoPass();
  Pass *createIslScheduleOptimizerPass();
  Pass *createTempScopInfoPass();

#ifdef OPENSCOP_FOUND
  Pass *createScopExporterPass();
  Pass *createScopImporterPass();
#endif

#ifdef SCOPLIB_FOUND
  Pass *createPoccPass();
  Pass *createScopLibExporterPass();
  Pass *createScopLibImporterPass();
#endif

  extern char &IndependentBlocksID;
  extern char &CodePreparationID;
}

using namespace polly;

namespace {
  struct PollyForcePassLinking {
    PollyForcePassLinking() {
      // We must reference the passes in such a way that compilers will not
      // delete it all as dead code, even with whole program optimization,
      // yet is effectively a NO-OP. As the compiler isn't smart enough
      // to know that getenv() never returns -1, this will do the job.
      if (std::getenv("bar") != (char*) -1)
        return;

       createAffSCEVItTesterPass();
#ifdef CLOOG_FOUND
       createCloogExporterPass();
       createCloogInfoPass();
       createCodeGenerationPass();
#endif
       createCodePreparationPass();
       createDeadCodeElimPass();
       createDependencesPass();
       createDOTOnlyPrinterPass();
       createDOTOnlyViewerPass();
       createDOTPrinterPass();
       createDOTViewerPass();
       createIndependentBlocksPass();
       createIndVarSimplifyPass();
       createJSONExporterPass();
       createJSONImporterPass();
       createRegionSimplifyPass();
       createScopDetectionPass();
       createScopInfoPass();
       createIslScheduleOptimizerPass();
       createTempScopInfoPass();

#ifdef OPENSCOP_FOUND
       createScopExporterPass();
       createScopImporterPass();
#endif
#ifdef SCOPLIB_FOUND
       createPoccPass();
       createScopLibExporterPass();
       createScopLibImporterPass();
#endif

    }
  } PollyForcePassLinking; // Force link by creating a global definition.
}

namespace llvm {
  class PassRegistry;
#ifdef CLOOG_FOUND
  void initializeCodeGenerationPass(llvm::PassRegistry&);
#endif
  void initializeCodePreparationPass(llvm::PassRegistry&);
  void initializeDeadCodeElimPass(llvm::PassRegistry&);
  void initializeIndependentBlocksPass(llvm::PassRegistry&);
  void initializeJSONExporterPass(llvm::PassRegistry&);
  void initializeJSONImporterPass(llvm::PassRegistry&);
  void initializeIslScheduleOptimizerPass(llvm::PassRegistry&);
#ifdef SCOPLIB_FOUND
  void initializePoccPass(llvm::PassRegistry&);
#endif
  void initializePollyIndVarSimplifyPass(llvm::PassRegistry&);
  void initializeRegionSimplifyPass(llvm::PassRegistry&);
}

#endif
