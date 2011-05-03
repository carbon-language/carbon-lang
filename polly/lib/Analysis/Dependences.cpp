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
  must_dep = may_dep = NULL;
  must_no_source = may_no_source = NULL;
  sink = must_source = may_source = NULL;
  war_dep = waw_dep = NULL;
}

bool Dependences::runOnScop(Scop &S) {
  isl_dim *dim = isl_dim_alloc(S.getCtx(), S.getNumParams(), 0, 0);

  if (sink)
    isl_union_map_free(sink);

  if (must_source)
    isl_union_map_free(must_source);

  if (may_source)
    isl_union_map_free(may_source);

  sink = isl_union_map_empty(isl_dim_copy(dim));
  must_source = isl_union_map_empty(isl_dim_copy(dim));
  may_source = isl_union_map_empty(isl_dim_copy(dim));
  isl_union_map *schedule = isl_union_map_empty(dim);

  if (must_dep)
    isl_union_map_free(must_dep);

  if (may_dep)
    isl_union_map_free(may_dep);

  if (must_no_source)
    isl_union_map_free(must_no_source);

  if (may_no_source)
    isl_union_map_free(may_no_source);

  if (war_dep)
    isl_union_map_free(war_dep);

  if (waw_dep)
    isl_union_map_free(waw_dep);

  must_dep = may_dep = NULL;
  must_no_source = may_no_source = NULL;

  war_dep = waw_dep = NULL;

  for (Scop::iterator SI = S.begin(), SE = S.end(); SI != SE; ++SI) {
    ScopStmt *Stmt = *SI;

    for (ScopStmt::memacc_iterator MI = Stmt->memacc_begin(),
          ME = Stmt->memacc_end(); MI != ME; ++MI) {
      isl_set *domcp = isl_set_copy(Stmt->getDomain());
      isl_map *accdom = isl_map_copy((*MI)->getAccessFunction());

      accdom = isl_map_intersect_domain(accdom, domcp);

      if ((*MI)->isRead())
        isl_union_map_add_map(sink, accdom);
      else
        isl_union_map_add_map(must_source, accdom);
    }
    isl_map *scattering = isl_map_copy(Stmt->getScattering());
    isl_union_map_add_map(schedule, scattering);
  }

  DEBUG(
    dbgs().indent(4) << "Sink:\n";
    dbgs().indent(8) << stringFromIslObj(sink) << "\n";

    dbgs().indent(4) << "MustSource:\n";
    dbgs().indent(8) << stringFromIslObj(must_source) << "\n";

    dbgs().indent(4) << "MaySource:\n";
    dbgs().indent(8) << stringFromIslObj(may_source) << "\n";

    dbgs().indent(4) << "Schedule:\n";
    dbgs().indent(8) << stringFromIslObj(schedule) << "\n";
  );

  isl_union_map_compute_flow(isl_union_map_copy(sink),
                              isl_union_map_copy(must_source),
                              isl_union_map_copy(may_source),
                              isl_union_map_copy(schedule),
                              &must_dep, &may_dep, &must_no_source,
                              &may_no_source);

  isl_union_map_compute_flow(isl_union_map_copy(must_source),
                             isl_union_map_copy(must_source),
                             isl_union_map_copy(sink), schedule,
                             &waw_dep, &war_dep, NULL, NULL);

  // Remove redundant statements.
  must_dep = isl_union_map_coalesce(must_dep);
  may_dep = isl_union_map_coalesce(may_dep);
  must_no_source = isl_union_map_coalesce(must_no_source);
  may_no_source = isl_union_map_coalesce(may_no_source);
  waw_dep = isl_union_map_coalesce(waw_dep);
  war_dep = isl_union_map_coalesce(war_dep);

  return false;
}

bool Dependences::isValidScattering(StatementToIslMapTy *NewScattering) {
  Scop &S = getCurScop();

  if (LegalityCheckDisabled)
    return true;

  isl_dim *dim = isl_dim_alloc(S.getCtx(), S.getNumParams(), 0, 0);

  isl_union_map *schedule = isl_union_map_empty(dim);

  for (Scop::iterator SI = S.begin(), SE = S.end(); SI != SE; ++SI) {
    ScopStmt *Stmt = *SI;

    isl_map *scattering;

    if (NewScattering->find(*SI) == NewScattering->end())
      scattering = isl_map_copy(Stmt->getScattering());
    else
      scattering = isl_map_copy((*NewScattering)[Stmt]);

    isl_union_map_add_map(schedule, scattering);
  }

  isl_union_map *temp_must_dep, *temp_may_dep;
  isl_union_map *temp_must_no_source, *temp_may_no_source;

  DEBUG(
    dbgs().indent(4) << "Sink :=\n";
    dbgs().indent(8) << stringFromIslObj(sink) << ";\n";

    dbgs().indent(4) << "MustSource :=\n";
    dbgs().indent(8) << stringFromIslObj(must_source) << ";\n";

    dbgs().indent(4) << "MaySource :=\n";
    dbgs().indent(8) << stringFromIslObj(may_source) << ";\n";

    dbgs().indent(4) << "Schedule :=\n";
    dbgs().indent(8) << stringFromIslObj(schedule) << ";\n";
  );

  isl_union_map_compute_flow(isl_union_map_copy(sink),
                              isl_union_map_copy(must_source),
                              isl_union_map_copy(may_source), schedule,
                              &temp_must_dep, &temp_may_dep,
                              &temp_must_no_source, &temp_may_no_source);

  DEBUG(dbgs().indent(4) << "\nDependences calculated\n");
  DEBUG(
    dbgs().indent(4) << "TempMustDep:=\n";
    dbgs().indent(8) << stringFromIslObj(temp_must_dep) << ";\n";

    dbgs().indent(4) << "MustDep:=\n";
    dbgs().indent(8) << stringFromIslObj(must_dep) << ";\n";
  );

  // Remove redundant statements.
  temp_must_dep = isl_union_map_coalesce(temp_must_dep);
  temp_may_dep = isl_union_map_coalesce(temp_may_dep);
  temp_must_no_source = isl_union_map_coalesce(temp_must_no_source);
  temp_may_no_source = isl_union_map_coalesce(temp_may_no_source);

  if (!isl_union_map_is_equal(temp_must_dep, must_dep)) {
    DEBUG(dbgs().indent(4) << "\nEqual 1 calculated\n");
    return false;
  }

  DEBUG(dbgs().indent(4) << "\nEqual 1 calculated\n");

  if (!isl_union_map_is_equal(temp_may_dep, may_dep))
    return false;

  DEBUG(dbgs().indent(4) << "\nEqual 2 calculated\n");

  if (!isl_union_map_is_equal(temp_must_no_source, must_no_source))
    return false;

  if (!isl_union_map_is_equal(temp_may_no_source, may_no_source))
    return false;

  return true;
}

isl_union_map* getCombinedScheduleForDim(Scop *scop, unsigned dimLevel) {
  isl_dim *dim = isl_dim_alloc(scop->getCtx(), scop->getNumParams(), 0, 0);

  isl_union_map *schedule = isl_union_map_empty(dim);

  for (Scop::iterator SI = scop->begin(), SE = scop->end(); SI != SE; ++SI) {
    ScopStmt *Stmt = *SI;
    isl_map *scattering = isl_map_copy(Stmt->getScattering());
    unsigned remainingDimensions = isl_map_n_out(scattering) - dimLevel;
    scattering = isl_map_project_out(scattering, isl_dim_out, dimLevel,
                                     remainingDimensions);
    isl_union_map_add_map(schedule, scattering);
  }

  return schedule;
}

bool Dependences::isParallelDimension(isl_set *loopDomain,
                                      unsigned parallelDimension) {
  Scop *S = &getCurScop();
  isl_union_map *schedule = getCombinedScheduleForDim(S, parallelDimension);

  // Calculate distance vector.
  isl_union_set *scheduleSubset;
  isl_union_map *scheduleDeps, *restrictedDeps;
  isl_union_map *scheduleDeps_war, *restrictedDeps_war;
  isl_union_map *scheduleDeps_waw, *restrictedDeps_waw;

  scheduleSubset = isl_union_set_from_set(isl_set_copy(loopDomain));

  scheduleDeps = isl_union_map_apply_range(isl_union_map_copy(must_dep),
                                           isl_union_map_copy(schedule));
  scheduleDeps = isl_union_map_apply_domain(scheduleDeps,
                                            isl_union_map_copy(schedule));

  scheduleDeps_war = isl_union_map_apply_range(isl_union_map_copy(war_dep),
                                               isl_union_map_copy(schedule));
  scheduleDeps_war = isl_union_map_apply_domain(scheduleDeps_war,
                                                isl_union_map_copy(schedule));

  scheduleDeps_waw = isl_union_map_apply_range(isl_union_map_copy(waw_dep),
                                               isl_union_map_copy(schedule));
  scheduleDeps_waw = isl_union_map_apply_domain(scheduleDeps_waw,
                                                isl_union_map_copy(schedule));

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
    isl_union_set_copy(scheduleSubset));

  isl_union_set *distance_waw = isl_union_map_deltas(restrictedDeps_waw);

  isl_dim *dim = isl_dim_set_alloc(S->getCtx(), S->getNumParams(),
                                   parallelDimension);

  // [0, 0, 0, 0] - All zero
  isl_basic_set *allZeroBS = isl_basic_set_universe(isl_dim_copy(dim));
  unsigned dimensions = isl_dim_size(dim, isl_dim_set);

  for (unsigned i = 0; i < dimensions; i++) {
    isl_constraint *c = isl_equality_alloc(isl_dim_copy(dim));
    isl_int v;
    isl_int_init(v);
    isl_int_set_si(v, -1);
    isl_constraint_set_coefficient(c, isl_dim_set, i, v);
    allZeroBS = isl_basic_set_add_constraint(allZeroBS, c);
    isl_int_clear(v);
  }

  isl_set *allZero = isl_set_from_basic_set(allZeroBS);

  // All zero, last unknown.
  // [0, 0, 0, ?]
  isl_basic_set *lastUnknownBS = isl_basic_set_universe(isl_dim_copy(dim));
  dimensions = isl_dim_size(dim, isl_dim_set);

  for (unsigned i = 0; i < dimensions - 1; i++) {
    isl_constraint *c = isl_equality_alloc(isl_dim_copy(dim));
    isl_int v;
    isl_int_init(v);
    isl_int_set_si(v, -1);
    isl_constraint_set_coefficient(c, isl_dim_set, i, v);
    lastUnknownBS = isl_basic_set_add_constraint(lastUnknownBS, c);
    isl_int_clear(v);
  }

  isl_set *lastUnknown = isl_set_from_basic_set(lastUnknownBS);

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

  return isl_union_set_is_empty(nonValid)
    && isl_union_set_is_empty(nonValid_war)
    && isl_union_set_is_empty(nonValid_waw);
}

bool Dependences::isParallelFor(const clast_for *f) {
  isl_set *loopDomain = isl_set_from_cloog_domain(f->domain);
  assert(loopDomain && "Cannot access domain of loop");

  return isParallelDimension(loopDomain, isl_set_n_dim(loopDomain));
}

void Dependences::printScop(raw_ostream &OS) const {
  OS.indent(4) << "Must dependences:\n";
  OS.indent(8) << stringFromIslObj(must_dep) << "\n";

  OS.indent(4) << "May dependences:\n";
  OS.indent(8) << stringFromIslObj(may_dep) << "\n";

  OS.indent(4) << "Must no source:\n";
  OS.indent(8) << stringFromIslObj(must_no_source) << "\n";

  OS.indent(4) << "May no source:\n";
  OS.indent(8) << stringFromIslObj(may_no_source) << "\n";
}

void Dependences::releaseMemory() {
  if (must_dep)
    isl_union_map_free(must_dep);

  if (may_dep)
    isl_union_map_free(may_dep);

  if (must_no_source)
    isl_union_map_free(must_no_source);

  if (may_no_source)
    isl_union_map_free(may_no_source);

  if (war_dep)
    isl_union_map_free(war_dep);

  if (waw_dep)
    isl_union_map_free(waw_dep);

  must_dep = may_dep = NULL;
  must_no_source = may_no_source = NULL;
  war_dep = waw_dep = NULL;

  if (sink)
    isl_union_map_free(sink);

  if (must_source)
    isl_union_map_free(must_source);

  if (may_source)
    isl_union_map_free(may_source);

  sink = must_source = may_source = NULL;
}

void Dependences::getAnalysisUsage(AnalysisUsage &AU) const {
  ScopPass::getAnalysisUsage(AU);
}

char Dependences::ID = 0;

static RegisterPass<Dependences>
X("polly-dependences", "Polly - Calculate dependences for Scop");

Pass* polly::createDependencesPass() {
  return new Dependences();
}
