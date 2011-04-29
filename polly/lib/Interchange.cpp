//===- Interchange.cpp - Interchange interface ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "polly/Cloog.h"
#include "polly/LinkAllPasses.h"

#include "polly/ScopInfo.h"
#include "polly/Dependences.h"

#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/CommandLine.h"

#include <isl/map.h>

#define DEBUG_TYPE "polly-interchange"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace polly;

namespace {

  class Interchange : public ScopPass {
  public:
    static char ID;
    explicit Interchange() : ScopPass(ID) {}

    virtual bool runOnScop(Scop &S);
    void getAnalysisUsage(AnalysisUsage &AU) const;
  };

}

char Interchange::ID = 0;
bool Interchange::runOnScop(Scop &S) {
  if (std::distance(S.begin(), S.end()) != 2) // One statement besides the final statement
    return false;

  for (Scop::iterator SI = S.begin(), SE = S.end(); SI != SE; ++SI) {
    ScopStmt *Stmt = *SI;
    if (!Stmt->isReduction())
      continue;

    isl_map *Scattering = isl_map_copy(Stmt->getScattering());

    const std::string MapString = "{scattering[i0, i1, i2, i3, i4] -> scattering[i0, i3, i2, i1, i4]}";
    isl_map *Map = isl_map_read_from_str(Stmt->getIslContext(), MapString.c_str(), -1);

    isl_map_add_dims(Map, isl_dim_param, Stmt->getNumParams());
    Scattering = isl_map_apply_range(Scattering, Map);
    Stmt->setScattering(Scattering);

    DEBUG(
      isl_printer *p = isl_printer_to_str(S.getCtx());
      isl_printer_print_map(p, Scattering);
      dbgs() << isl_printer_get_str(p) << '\n';
      isl_printer_flush(p);
      isl_printer_free(p);
    );
  }

  return false;
}

void Interchange::getAnalysisUsage(AnalysisUsage &AU) const {
  ScopPass::getAnalysisUsage(AU);
  AU.addRequired<Dependences>();
}

static RegisterPass<Interchange> A("polly-interchange",
                            "Polly - Perform loop interchange");

Pass* polly::createInterchangePass() {
  return new Interchange();
}
