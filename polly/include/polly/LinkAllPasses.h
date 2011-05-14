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
  class RegionPass;
}

using namespace llvm;

namespace polly {
  Pass *createAffSCEVItTesterPass();
  Pass *createCloogExporterPass();
  Pass *createCloogInfoPass();
  Pass *createCodeGenerationPass();
  Pass *createCodePreperationPass();
  Pass *createDependencesPass();
  Pass *createDOTOnlyPrinterPass();
  Pass *createDOTOnlyViewerPass();
  Pass *createDOTPrinterPass();
  Pass *createDOTViewerPass();
  Pass *createIndependentBlocksPass();
  Pass *createInterchangePass();
  Pass *createJSONExporterPass();
  Pass *createJSONImporterPass();
  Pass *createRegionSimplifyPass();
  Pass *createScopInfoPass();
  Pass *createScheduleOptimizerPass();

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
  extern char &CodePreperationID;
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
       createCloogExporterPass();
       createCloogInfoPass();
       createCodeGenerationPass();
       createCodePreperationPass();
       createDependencesPass();
       createDOTOnlyPrinterPass();
       createDOTOnlyViewerPass();
       createDOTPrinterPass();
       createDOTViewerPass();
       createIndependentBlocksPass();
       createInterchangePass();
       createJSONExporterPass();
       createJSONImporterPass();
       createRegionSimplifyPass();
       createScopInfoPass();
       createScheduleOptimizerPass();

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

#endif
