//===--------- PolyhedralInfo.cpp  - Create Scops from LLVM IR-------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An interface to the Polyhedral analysis engine(Polly) of LLVM.
//
// This pass provides an interface to the polyhedral analysis performed by
// Polly.
//
// This interface provides basic interface like isParallel, isVectorizable
// that can be used in LLVM transformation passes.
//
// Work in progress, this file is subject to change.
//
//===----------------------------------------------------------------------===//

#include "polly/PolyhedralInfo.h"
#include "polly/DependenceInfo.h"
#include "polly/LinkAllPasses.h"
#include "polly/Options.h"
#include "polly/ScopInfo.h"
#include "polly/Support/GICHelper.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include "isl/union_map.h"

using namespace llvm;
using namespace polly;

#define DEBUG_TYPE "polyhedral-info"

static cl::opt<bool> CheckParallel("polly-check-parallel",
                                   cl::desc("Check for parallel loops"),
                                   cl::Hidden, cl::cat(PollyCategory));

static cl::opt<bool> CheckVectorizable("polly-check-vectorizable",
                                       cl::desc("Check for vectorizable loops"),
                                       cl::Hidden, cl::cat(PollyCategory));

void PolyhedralInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequiredTransitive<DependenceInfoWrapperPass>();
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addRequiredTransitive<ScopInfoWrapperPass>();
  AU.setPreservesAll();
}

bool PolyhedralInfo::runOnFunction(Function &F) {
  DI = &getAnalysis<DependenceInfoWrapperPass>();
  SI = getAnalysis<ScopInfoWrapperPass>().getSI();
  return false;
}

void PolyhedralInfo::print(raw_ostream &OS, const Module *) const {
  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  for (auto *TopLevelLoop : LI) {
    for (auto *L : depth_first(TopLevelLoop)) {
      OS.indent(2) << L->getHeader()->getName() << ":\t";
      if (CheckParallel && isParallel(L))
        OS << "Loop is parallel.\n";
      else if (CheckParallel)
        OS << "Loop is not parallel.\n";
    }
  }
}

bool PolyhedralInfo::checkParallel(Loop *L, isl_pw_aff **MinDepDistPtr) const {
  bool IsParallel;
  const Scop *S = getScopContainingLoop(L);
  if (!S)
    return false;
  const Dependences &D =
      DI->getDependences(const_cast<Scop *>(S), Dependences::AL_Access);
  if (!D.hasValidDependences())
    return false;
  LLVM_DEBUG(dbgs() << "Loop :\t" << L->getHeader()->getName() << ":\n");

  isl_union_map *Deps =
      D.getDependences(Dependences::TYPE_RAW | Dependences::TYPE_WAW |
                       Dependences::TYPE_WAR | Dependences::TYPE_RED)
          .release();

  LLVM_DEBUG(dbgs() << "Dependences :\t" << stringFromIslObj(Deps, "null")
                    << "\n");

  isl_union_map *Schedule = getScheduleForLoop(S, L);
  LLVM_DEBUG(dbgs() << "Schedule: \t" << stringFromIslObj(Schedule, "null")
                    << "\n");

  IsParallel = D.isParallel(Schedule, Deps, MinDepDistPtr);
  isl_union_map_free(Schedule);
  return IsParallel;
}

bool PolyhedralInfo::isParallel(Loop *L) const { return checkParallel(L); }

const Scop *PolyhedralInfo::getScopContainingLoop(Loop *L) const {
  assert((SI) && "ScopInfoWrapperPass is required by PolyhedralInfo pass!\n");
  for (auto &It : *SI) {
    Region *R = It.first;
    if (R->contains(L))
      return It.second.get();
  }
  return nullptr;
}

//  Given a Loop and the containing SCoP, we compute the partial schedule
//  by taking union of individual schedules of each ScopStmt within the loop
//  and projecting out the inner dimensions from the range of the schedule.
//   for (i = 0; i < n; i++)
//      for (j = 0; j < n; j++)
//        A[j] = 1;  //Stmt
//
//  The original schedule will be
//    Stmt[i0, i1] -> [i0, i1]
//  The schedule for the outer loop will be
//    Stmt[i0, i1] -> [i0]
//  The schedule for the inner loop will be
//    Stmt[i0, i1] -> [i0, i1]
__isl_give isl_union_map *PolyhedralInfo::getScheduleForLoop(const Scop *S,
                                                             Loop *L) const {
  isl_union_map *Schedule = isl_union_map_empty(S->getParamSpace().release());
  int CurrDim = S->getRelativeLoopDepth(L);
  LLVM_DEBUG(dbgs() << "Relative loop depth:\t" << CurrDim << "\n");
  assert(CurrDim >= 0 && "Loop in region should have at least depth one");

  for (auto &SS : *S) {
    if (L->contains(SS.getSurroundingLoop())) {

      unsigned int MaxDim = SS.getNumIterators();
      LLVM_DEBUG(dbgs() << "Maximum depth of Stmt:\t" << MaxDim << "\n");
      isl_map *ScheduleMap = SS.getSchedule().release();
      assert(
          ScheduleMap &&
          "Schedules that contain extension nodes require special handling.");

      ScheduleMap = isl_map_project_out(ScheduleMap, isl_dim_out, CurrDim + 1,
                                        MaxDim - CurrDim - 1);
      ScheduleMap = isl_map_set_tuple_id(ScheduleMap, isl_dim_in,
                                         SS.getDomainId().release());
      Schedule =
          isl_union_map_union(Schedule, isl_union_map_from_map(ScheduleMap));
    }
  }
  Schedule = isl_union_map_coalesce(Schedule);
  return Schedule;
}

char PolyhedralInfo::ID = 0;

Pass *polly::createPolyhedralInfoPass() { return new PolyhedralInfo(); }

INITIALIZE_PASS_BEGIN(PolyhedralInfo, "polyhedral-info",
                      "Polly - Interface to polyhedral analysis engine", false,
                      false);
INITIALIZE_PASS_DEPENDENCY(DependenceInfoWrapperPass);
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass);
INITIALIZE_PASS_DEPENDENCY(ScopInfoWrapperPass);
INITIALIZE_PASS_END(PolyhedralInfo, "polyhedral-info",
                    "Polly - Interface to polyhedral analysis engine", false,
                    false)

//===----------------------------------------------------------------------===//

namespace {
/// Print result from PolyhedralInfo.
class PolyhedralInfoPrinterLegacyPass final : public FunctionPass {
public:
  static char ID;

  PolyhedralInfoPrinterLegacyPass() : PolyhedralInfoPrinterLegacyPass(outs()) {}
  explicit PolyhedralInfoPrinterLegacyPass(llvm::raw_ostream &OS)
      : FunctionPass(ID), OS(OS) {}

  bool runOnFunction(Function &F) override {
    PolyhedralInfo &P = getAnalysis<PolyhedralInfo>();

    OS << "Printing analysis '" << P.getPassName() << "' for function '"
       << F.getName() << "':\n";
    P.print(OS);

    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    FunctionPass::getAnalysisUsage(AU);
    AU.addRequired<PolyhedralInfo>();
    AU.setPreservesAll();
  }

private:
  llvm::raw_ostream &OS;
};

char PolyhedralInfoPrinterLegacyPass::ID = 0;
} // namespace

Pass *polly::createPolyhedralInfoPrinterLegacyPass(raw_ostream &OS) {
  return new PolyhedralInfoPrinterLegacyPass(OS);
}

INITIALIZE_PASS_BEGIN(
    PolyhedralInfoPrinterLegacyPass, "print-polyhedral-info",
    "Polly - Print interface to polyhedral analysis engine analysis", false,
    false);
INITIALIZE_PASS_DEPENDENCY(PolyhedralInfo);
INITIALIZE_PASS_END(
    PolyhedralInfoPrinterLegacyPass, "print-polyhedral-info",
    "Polly - Print interface to polyhedral analysis engine analysis", false,
    false)
