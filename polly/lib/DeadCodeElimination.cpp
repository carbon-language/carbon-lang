//===- DeadCodeElimination.cpp - Eliminate dead iteration  ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a skeleton that is meant to contain a dead code elimination pass
// later on.
//
// The idea of this pass is to loop over all statements and to remove statement
// iterations that do not calculate any value that is read later on. We need to
// make sure to forward RAR and WAR dependences.
//
// A case where this pass might be useful is
// http://llvm.org/bugs/show_bug.cgi?id=5117
//
//===----------------------------------------------------------------------===//

#include "polly/Dependences.h"
#include "polly/LinkAllPasses.h"
#include "polly/ScopInfo.h"
#include "llvm/Support/CommandLine.h"
#include "isl/set.h"
#include "isl/map.h"
#include "isl/union_map.h"

using namespace llvm;
using namespace polly;

namespace {
enum DcePrecision {
  DCE_PRECISION_AUTO,
  DCE_PRECISION_HULL,
  DCE_PRECISION_FULL
};

cl::opt<DcePrecision> DcePrecision(
    "polly-dce-precision", cl::desc("Precision of Polyhedral DCE"),
    cl::values(
        clEnumValN(DCE_PRECISION_FULL, "full",
                   "Live set is not approximated at each iteration"),
        clEnumValN(
            DCE_PRECISION_HULL, "hull",
            "Live set is approximated with an affine hull at each iteration"),
        clEnumValN(DCE_PRECISION_AUTO, "auto", "Currently the same as hull"),
        clEnumValEnd),
    cl::init(DCE_PRECISION_AUTO));

class DeadCodeElim : public ScopPass {

public:
  static char ID;
  explicit DeadCodeElim() : ScopPass(ID) {}

  virtual bool runOnScop(Scop &S);

  void printScop(llvm::raw_ostream &OS) const;
  void getAnalysisUsage(AnalysisUsage &AU) const;

private:
  isl_union_set *getLastWrites(isl_union_map *Writes, isl_union_map *Schedule);
  bool eliminateDeadCode(Scop &S);
};
}

char DeadCodeElim::ID = 0;

/// Return the set of iterations that contains the last write for each location.
isl_union_set *DeadCodeElim::getLastWrites(__isl_take isl_union_map *Writes,
                                           __isl_take isl_union_map *Schedule) {
  isl_union_map *WriteIterations = isl_union_map_reverse(Writes);
  isl_union_map *WriteTimes =
      isl_union_map_apply_range(WriteIterations, isl_union_map_copy(Schedule));

  isl_union_map *LastWriteTimes = isl_union_map_lexmax(WriteTimes);
  isl_union_map *LastWriteIterations = isl_union_map_apply_range(
      LastWriteTimes, isl_union_map_reverse(Schedule));

  isl_union_set *Live = isl_union_map_range(LastWriteIterations);
  return isl_union_set_coalesce(Live);
}

/// Performs polyhedral dead iteration elimination by:
/// o Assuming that the last write to each location is live.
/// o Following each RAW dependency from a live iteration backwards and adding
///   that iteration to the live set.
bool DeadCodeElim::eliminateDeadCode(Scop &S) {
  isl_union_set *Live = this->getLastWrites(S.getWrites(), S.getSchedule());

  Dependences *D = &getAnalysis<Dependences>();
  isl_union_map *Dep = D->getDependences(Dependences::TYPE_RAW);
  Dep = isl_union_map_reverse(Dep);

  isl_union_set *OriginalDomain = S.getDomains();
  while (true) {
    isl_union_set *Extra;

    Extra =
        isl_union_set_apply(isl_union_set_copy(Live), isl_union_map_copy(Dep));

    if (isl_union_set_is_subset(Extra, Live)) {
      isl_union_set_free(Extra);
      break;
    }

    Live = isl_union_set_union(Live, Extra);
    if (DcePrecision != DCE_PRECISION_FULL)
      Live = isl_union_set_affine_hull(Live);
    Live = isl_union_set_intersect(Live, isl_union_set_copy(OriginalDomain));
  }
  isl_union_map_free(Dep);
  isl_union_set_free(OriginalDomain);

  return S.restrictDomains(isl_union_set_coalesce(Live));
}

bool DeadCodeElim::runOnScop(Scop &S) { return eliminateDeadCode(S); }

void DeadCodeElim::printScop(raw_ostream &OS) const {}

void DeadCodeElim::getAnalysisUsage(AnalysisUsage &AU) const {
  ScopPass::getAnalysisUsage(AU);
  AU.addRequired<Dependences>();
}

Pass *polly::createDeadCodeElimPass() { return new DeadCodeElim(); }

INITIALIZE_PASS_BEGIN(DeadCodeElim, "polly-dce",
                      "Polly - Remove dead iterations", false, false)
INITIALIZE_PASS_DEPENDENCY(Dependences)
INITIALIZE_PASS_DEPENDENCY(ScopInfo)
INITIALIZE_PASS_END(DeadCodeElim, "polly-dce", "Polly - Remove dead iterations",
                    false, false)
