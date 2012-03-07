//===- Dependency.cpp - Calculate dependency information for a Scop.  -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Calculate the data dependency relations for a Scop using ISL.
//
// The integer set library (ISL) from Sven, has a integrated dependency analysis
// to calculate data dependences. This pass takes advantage of this and
// calculate those dependences a Scop.
//
// The dependences in this pass are exact in terms that for a specific read
// statement instance only the last write statement instance is returned. In
// case of may writes a set of possible write instances is returned. This
// analysis will never produce redundant dependences.
//
//===----------------------------------------------------------------------===//
//
#include "polly/Dependences.h"

#include "polly/LinkAllPasses.h"
#include "polly/ScopInfo.h"
#include "polly/Support/GICHelper.h"

#define DEBUG_TYPE "polly-dependences"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"

#include <isl/flow.h>
#include <isl/aff.h>
#define CLOOG_INT_GMP 1
#include <cloog/cloog.h>
#include <cloog/isl/cloog.h>

using namespace polly;
using namespace llvm;

static cl::opt<bool>
  LegalityCheckDisabled("disable-polly-legality",
       cl::desc("Disable polly legality check"), cl::Hidden,
       cl::init(false));

//===----------------------------------------------------------------------===//
Dependences::Dependences() : ScopPass(ID) {
  RAW = WAR = WAW = NULL;
}

void Dependences::collectInfo(Scop &S,
                              isl_union_map **Read, isl_union_map **Write,
                              isl_union_map **MayWrite,
                              isl_union_map **Schedule) {
  isl_space *Space = S.getParamSpace();
  *Read = isl_union_map_empty(isl_space_copy(Space));
  *Write = isl_union_map_empty(isl_space_copy(Space));
  *MayWrite = isl_union_map_empty(isl_space_copy(Space));
  *Schedule = isl_union_map_empty(Space);

  for (Scop::iterator SI = S.begin(), SE = S.end(); SI != SE; ++SI) {
    ScopStmt *Stmt = *SI;

    for (ScopStmt::memacc_iterator MI = Stmt->memacc_begin(),
          ME = Stmt->memacc_end(); MI != ME; ++MI) {
      isl_set *domcp = Stmt->getDomain();
      isl_map *accdom = (*MI)->getAccessRelation();

      accdom = isl_map_intersect_domain(accdom, domcp);

      if ((*MI)->isRead())
        *Read = isl_union_map_add_map(*Read, accdom);
      else
        *Write = isl_union_map_add_map(*Write, accdom);
    }
    *Schedule = isl_union_map_add_map(*Schedule, Stmt->getScattering());
  }
}

void Dependences::calculateDependences(Scop &S) {
  isl_union_map *Read, *Write, *MayWrite, *Schedule;

  collectInfo(S, &Read, &Write, &MayWrite, &Schedule);

  isl_union_map_compute_flow(isl_union_map_copy(Read),
                              isl_union_map_copy(Write),
                              isl_union_map_copy(MayWrite),
                              isl_union_map_copy(Schedule),
                              &RAW, NULL, NULL, NULL);

  isl_union_map_compute_flow(isl_union_map_copy(Write),
                             isl_union_map_copy(Write),
                             Read, Schedule,
                             &WAW, &WAR, NULL, NULL);

  isl_union_map_free(MayWrite);
  isl_union_map_free(Write);

  RAW = isl_union_map_coalesce(RAW);
  WAW = isl_union_map_coalesce(WAW);
  WAR = isl_union_map_coalesce(WAR);
}

bool Dependences::runOnScop(Scop &S) {
  releaseMemory();
  calculateDependences(S);

  return false;
}

bool Dependences::isValidScattering(StatementToIslMapTy *NewScattering) {
  Scop &S = getCurScop();

  if (LegalityCheckDisabled)
    return true;

  isl_union_map *Dependences = getDependences(TYPE_ALL);
  isl_space *Space = S.getParamSpace();
  isl_union_map *Scattering = isl_union_map_empty(Space);

  isl_space *ScatteringSpace = 0;

  for (Scop::iterator SI = S.begin(), SE = S.end(); SI != SE; ++SI) {
    ScopStmt *Stmt = *SI;

    isl_map *StmtScat;

    if (NewScattering->find(*SI) == NewScattering->end())
      StmtScat = Stmt->getScattering();
    else
      StmtScat = isl_map_copy((*NewScattering)[Stmt]);

    if (!ScatteringSpace)
      ScatteringSpace = isl_space_range(isl_map_get_space(StmtScat));

    Scattering = isl_union_map_add_map(Scattering, StmtScat);
  }

  Dependences = isl_union_map_apply_domain(Dependences,
                                           isl_union_map_copy(Scattering));
  Dependences = isl_union_map_apply_range(Dependences, Scattering);

  isl_set *Zero = isl_set_universe(isl_space_copy(ScatteringSpace));
  for (unsigned i = 0; i < isl_set_dim(Zero, isl_dim_set); i++)
    Zero = isl_set_fix_si(Zero, isl_dim_set, i, 0);

  isl_union_set *UDeltas = isl_union_map_deltas(Dependences);
  isl_set *Deltas = isl_union_set_extract_set(UDeltas, ScatteringSpace);
  isl_union_set_free(UDeltas);

  isl_map *NonPositive = isl_set_lex_le_set(Deltas, Zero);
  bool IsValid = isl_map_is_empty(NonPositive);
  isl_map_free(NonPositive);

  return IsValid;
}

isl_union_map *getCombinedScheduleForSpace(Scop *scop, unsigned dimLevel) {
  isl_space *Space = scop->getParamSpace();
  isl_union_map *schedule = isl_union_map_empty(Space);

  for (Scop::iterator SI = scop->begin(), SE = scop->end(); SI != SE; ++SI) {
    ScopStmt *Stmt = *SI;
    unsigned remainingDimensions = Stmt->getNumScattering() - dimLevel;
    isl_map *Scattering = isl_map_project_out(Stmt->getScattering(),
                                              isl_dim_out, dimLevel,
                                              remainingDimensions);
    schedule = isl_union_map_add_map(schedule, Scattering);
  }

  return schedule;
}

bool Dependences::isParallelDimension(isl_set *loopDomain,
                                      unsigned parallelDimension) {
  Scop *S = &getCurScop();
  isl_union_map *schedule = getCombinedScheduleForSpace(S, parallelDimension);

  // Calculate distance vector.
  isl_union_set *scheduleSubset;
  isl_union_map *scheduleDeps, *restrictedDeps;
  isl_union_map *scheduleDeps_war, *restrictedDeps_war;
  isl_union_map *scheduleDeps_waw, *restrictedDeps_waw;

  scheduleSubset = isl_union_set_from_set(isl_set_copy(loopDomain));

  scheduleDeps = isl_union_map_apply_range(isl_union_map_copy(RAW),
                                           isl_union_map_copy(schedule));
  scheduleDeps = isl_union_map_apply_domain(scheduleDeps,
                                            isl_union_map_copy(schedule));

  scheduleDeps_war = isl_union_map_apply_range(isl_union_map_copy(WAR),
                                               isl_union_map_copy(schedule));
  scheduleDeps_war = isl_union_map_apply_domain(scheduleDeps_war,
                                                isl_union_map_copy(schedule));

  scheduleDeps_waw = isl_union_map_apply_range(isl_union_map_copy(WAW),
                                               isl_union_map_copy(schedule));
  scheduleDeps_waw = isl_union_map_apply_domain(scheduleDeps_waw, schedule);

  // Dependences need to originate and to terminate in the scheduling space
  // enumerated by this loop.
  restrictedDeps = isl_union_map_intersect_domain(scheduleDeps,
    isl_union_set_copy(scheduleSubset));
  restrictedDeps = isl_union_map_intersect_range(restrictedDeps,
    isl_union_set_copy(scheduleSubset));

  isl_union_set *distance = isl_union_map_deltas(restrictedDeps);

  restrictedDeps_war = isl_union_map_intersect_domain(scheduleDeps_war,
    isl_union_set_copy(scheduleSubset));
  restrictedDeps_war = isl_union_map_intersect_range(restrictedDeps_war,
    isl_union_set_copy(scheduleSubset));

  isl_union_set *distance_war = isl_union_map_deltas(restrictedDeps_war);

  restrictedDeps_waw = isl_union_map_intersect_domain(scheduleDeps_waw,
    isl_union_set_copy(scheduleSubset));
  restrictedDeps_waw = isl_union_map_intersect_range(restrictedDeps_waw,
    scheduleSubset);

  isl_union_set *distance_waw = isl_union_map_deltas(restrictedDeps_waw);

  isl_space *Space = isl_space_set_alloc(S->getIslCtx(), 0, parallelDimension);

  // [0, 0, 0, 0] - All zero
  isl_set *allZero = isl_set_universe(isl_space_copy(Space));
  unsigned dimensions = isl_space_dim(Space, isl_dim_set);

  for (unsigned i = 0; i < dimensions; i++)
    allZero = isl_set_fix_si(allZero, isl_dim_set, i, 0);

  allZero = isl_set_align_params(allZero, S->getParamSpace());

  // All zero, last unknown.
  // [0, 0, 0, ?]
  isl_set *lastUnknown = isl_set_universe(isl_space_copy(Space));

  for (unsigned i = 0; i < dimensions - 1; i++)
    lastUnknown = isl_set_fix_si(lastUnknown, isl_dim_set, i, 0);

  lastUnknown = isl_set_align_params(lastUnknown, S->getParamSpace());

  // Valid distance vectors
  isl_set *validDistances = isl_set_subtract(lastUnknown, allZero);
  validDistances = isl_set_complement(validDistances);
  isl_union_set *validDistancesUS = isl_union_set_from_set(validDistances);

  isl_union_set *nonValid = isl_union_set_subtract(distance,
    isl_union_set_copy(validDistancesUS));

  isl_union_set *nonValid_war = isl_union_set_subtract(distance_war,
    isl_union_set_copy(validDistancesUS));

  isl_union_set *nonValid_waw = isl_union_set_subtract(distance_waw,
                                                       validDistancesUS);
  bool is_parallel = isl_union_set_is_empty(nonValid)
    && isl_union_set_is_empty(nonValid_war)
    && isl_union_set_is_empty(nonValid_waw);

  isl_space_free(Space);
  isl_union_set_free(nonValid);
  isl_union_set_free(nonValid_war);
  isl_union_set_free(nonValid_waw);

  return is_parallel;
}

bool Dependences::isParallelFor(const clast_for *f) {
  isl_set *loopDomain = isl_set_from_cloog_domain(f->domain);
  assert(loopDomain && "Cannot access domain of loop");

  return isParallelDimension(loopDomain, isl_set_n_dim(loopDomain));
}

void Dependences::printScop(raw_ostream &OS) const {
}

void Dependences::releaseMemory() {
  isl_union_map_free(RAW);
  isl_union_map_free(WAR);
  isl_union_map_free(WAW);

  RAW = WAR = WAW  = NULL;
}

isl_union_map *Dependences::getDependences(int Kinds) {
  isl_space *Space = isl_union_map_get_space(RAW);
  isl_union_map *Deps = isl_union_map_empty(Space);

  if (Kinds & TYPE_RAW)
    Deps = isl_union_map_union(Deps, isl_union_map_copy(RAW));

  if (Kinds & TYPE_WAR)
    Deps = isl_union_map_union(Deps, isl_union_map_copy(WAR));

  if (Kinds & TYPE_WAW)
    Deps = isl_union_map_union(Deps, isl_union_map_copy(WAW));

  Deps = isl_union_map_coalesce(Deps);
  Deps = isl_union_map_detect_equalities(Deps);
  return Deps;
}

void Dependences::getAnalysisUsage(AnalysisUsage &AU) const {
  ScopPass::getAnalysisUsage(AU);
}

char Dependences::ID = 0;

INITIALIZE_PASS_BEGIN(Dependences, "polly-dependences",
                      "Polly - Calculate dependences", false, false)
INITIALIZE_PASS_DEPENDENCY(ScopInfo)
INITIALIZE_PASS_END(Dependences, "polly-dependences",
                    "Polly - Calculate dependences", false, false)

Pass *polly::createDependencesPass() {
  return new Dependences();
}
