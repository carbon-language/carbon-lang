//===- DeadCodeElimination.cpp - Eliminate dead iteration  ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The polyhedral dead code elimination pass analyses a SCoP to eliminate
// statement instances that can be proven dead.
// As a consequence, the code generated for this SCoP may execute a statement
// less often. This means, a statement may be executed only in certain loop
// iterations or it may not even be part of the generated code at all.
//
// This code:
//
//    for (i = 0; i < N; i++)
//        arr[i] = 0;
//    for (i = 0; i < N; i++)
//        arr[i] = 10;
//    for (i = 0; i < N; i++)
//        arr[i] = i;
//
// is e.g. simplified to:
//
//    for (i = 0; i < N; i++)
//        arr[i] = i;
//
// The idea and the algorithm used was first implemented by Sven Verdoolaege in
// the 'ppcg' tool.
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

cl::opt<int> DCEPreciseSteps(
    "polly-dce-precise-steps",
    cl::desc(
        "The number of precise steps between two approximating iterations"),
    cl::init(2));

class DeadCodeElim : public ScopPass {

public:
  static char ID;
  explicit DeadCodeElim() : ScopPass(ID) {}

  virtual bool runOnScop(Scop &S);

  void printScop(llvm::raw_ostream &OS) const;
  void getAnalysisUsage(AnalysisUsage &AU) const;

private:
  isl_union_set *getLastWrites(isl_union_map *Writes, isl_union_map *Schedule);
  bool eliminateDeadCode(Scop &S, int PreciseSteps);
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
///
/// To ensure the set of live iterations does not get too complex we always
/// combine a certain number of precise steps with one approximating step that
/// simplifies the life set with an affine hull.
bool DeadCodeElim::eliminateDeadCode(Scop &S, int PreciseSteps) {
  Dependences *D = &getAnalysis<Dependences>();

  if (!D->hasValidDependences())
    return false;

  isl_union_set *Live = this->getLastWrites(S.getWrites(), S.getSchedule());
  isl_union_map *Dep = D->getDependences(Dependences::TYPE_RAW);
  Dep = isl_union_map_reverse(Dep);

  isl_union_set *OriginalDomain = S.getDomains();
  int Steps = 0;
  while (true) {
    isl_union_set *Extra;
    Steps++;

    Extra =
        isl_union_set_apply(isl_union_set_copy(Live), isl_union_map_copy(Dep));

    if (isl_union_set_is_subset(Extra, Live)) {
      isl_union_set_free(Extra);
      break;
    }

    Live = isl_union_set_union(Live, Extra);

    if (Steps > PreciseSteps) {
      Steps = 0;
      Live = isl_union_set_affine_hull(Live);
    }

    Live = isl_union_set_intersect(Live, isl_union_set_copy(OriginalDomain));
  }
  isl_union_map_free(Dep);
  isl_union_set_free(OriginalDomain);

  return S.restrictDomains(isl_union_set_coalesce(Live));
}

bool DeadCodeElim::runOnScop(Scop &S) {
  return eliminateDeadCode(S, DCEPreciseSteps);
}

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
