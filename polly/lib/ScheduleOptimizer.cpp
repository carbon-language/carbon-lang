//===- Schedule.cpp - Calculate an optimized schedule ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass the isl to calculate a schedule that is optimized for parallelism
// and tileablility. The algorithm used in isl is an optimized version of the
// algorithm described in following paper:
//
// U. Bondhugula, A. Hartono, J. Ramanujam, and P. Sadayappan.
// A Practical Automatic Polyhedral Parallelizer and Locality Optimizer.
// In Proceedings of the 2008 ACM SIGPLAN Conference On Programming Language
// Design and Implementation, PLDI ’08, pages 101–113. ACM, 2008.
//===----------------------------------------------------------------------===//

#include "polly/Cloog.h"
#include "polly/LinkAllPasses.h"

#include "polly/Dependences.h"
#include "polly/ScopInfo.h"

#include "isl/dim.h"
#include "isl/map.h"
#include "isl/constraint.h"
#include "isl/schedule.h"

#define DEBUG_TYPE "polly-optimize-isl"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace polly;

namespace {

  class ScheduleOptimizer : public ScopPass {

  public:
    static char ID;
    explicit ScheduleOptimizer() : ScopPass(ID) {}

    virtual bool runOnScop(Scop &S);
    void printScop(llvm::raw_ostream &OS) const;
    void getAnalysisUsage(AnalysisUsage &AU) const;
  };

}

char ScheduleOptimizer::ID = 0;

static int getSingleMap(__isl_take isl_map *map, void *user) {
  isl_map **singleMap = (isl_map **) user;
  *singleMap = map;

  return 0;
}

void extendScattering(Scop &S, unsigned scatDimensions) {
  for (Scop::iterator SI = S.begin(), SE = S.end(); SI != SE; ++SI) {
    ScopStmt *stmt = *SI;

    if (stmt->isFinalRead())
      continue;

    isl_map *scattering = stmt->getScattering();
    isl_dim *dim = isl_dim_alloc(isl_map_get_ctx(scattering),
                                 isl_map_n_param(scattering),
                                 isl_map_n_out(scattering),
                                 scatDimensions);
    isl_basic_map *changeScattering = isl_basic_map_universe(isl_dim_copy(dim));

    for (unsigned i = 0; i < isl_map_n_out(scattering); i++) {
      isl_constraint *c = isl_equality_alloc(isl_dim_copy(dim));
      isl_constraint_set_coefficient_si(c, isl_dim_in, i, 1);
      isl_constraint_set_coefficient_si(c, isl_dim_out, i, -1);
      changeScattering = isl_basic_map_add_constraint(changeScattering, c);
    }

    for (unsigned i = isl_map_n_out(scattering); i < scatDimensions; i++) {
      isl_constraint *c = isl_equality_alloc(isl_dim_copy(dim));
      isl_constraint_set_coefficient_si(c, isl_dim_out, i, 1);
      changeScattering = isl_basic_map_add_constraint(changeScattering, c);
    }

    isl_map *changeScatteringMap = isl_map_from_basic_map(changeScattering);

    stmt->setScattering(isl_map_apply_range(scattering, changeScatteringMap));
  }
}

// @brief Tile a band.
//
// This function recieves a map that assigns to the instances of a statement
// an execution time.
//
// [i_0, i_1, i_2] -> [o_0, o_1, o_2, i_0, i_1, i_2]:
//   o_0 % 32 = 0 and o_1 % 32 = 0 and o_2 % 32 = 0
//   and o0 <= i0 <= o0 + 32 and o1 <= i1 <= o1 + 32 and o2 <= i2 <= o2 + 32

isl_map *tileBand(isl_map *band) {
  int dimensions = isl_map_n_out(band);
  int tileSize = 32;

  isl_dim *dim = isl_dim_alloc(isl_map_get_ctx(band), isl_map_n_param(band),
                               dimensions, dimensions * 3);
  isl_basic_map *tiledBand = isl_basic_map_universe(isl_dim_copy(dim));

  for (int i = 0; i < dimensions; i++) {
    isl_constraint *c = isl_equality_alloc(isl_dim_copy(dim));
    isl_constraint_set_coefficient_si(c, isl_dim_out, i, 1);
    isl_constraint_set_coefficient_si(c, isl_dim_out, 2 * dimensions + i,
                                      -tileSize);
    tiledBand = isl_basic_map_add_constraint(tiledBand, c);


    c = isl_equality_alloc(isl_dim_copy(dim));
    isl_constraint_set_coefficient_si(c, isl_dim_in, i, -1);
    isl_constraint_set_coefficient_si(c, isl_dim_out, dimensions + i, 1);
    tiledBand = isl_basic_map_add_constraint(tiledBand, c);

    c = isl_inequality_alloc(isl_dim_copy(dim));
    isl_constraint_set_coefficient_si(c, isl_dim_out, i, -1);
    isl_constraint_set_coefficient_si(c, isl_dim_out, dimensions + i, 1);
    tiledBand = isl_basic_map_add_constraint(tiledBand, c);

    c = isl_inequality_alloc(isl_dim_copy(dim));
    isl_constraint_set_coefficient_si(c, isl_dim_out, i, 1);
    isl_constraint_set_coefficient_si(c, isl_dim_out, dimensions + i, -1);
    isl_constraint_set_constant_si(c, tileSize - 1);
    tiledBand = isl_basic_map_add_constraint(tiledBand, c);
  }

  // Project out auxilary dimensions (introduced to ensure 'ii % tileSize = 0')
  //
  // The real dimensions are transformed into existentially quantified ones.
  // This reduces the number of visible scattering dimensions.  Also, Cloog
  // produces better code, if auxilary dimensions are existentially quantified.
  tiledBand = isl_basic_map_project_out(tiledBand, isl_dim_out, 2 * dimensions,
                                        dimensions);

  return isl_map_apply_range(band, isl_map_from_basic_map(tiledBand));
}

bool ScheduleOptimizer::runOnScop(Scop &S) {
  Dependences *D = &getAnalysis<Dependences>();

  // Build input data.
  int dependencyKinds = Dependences::TYPE_RAW
                          | Dependences::TYPE_WAR
                          | Dependences::TYPE_WAW;

  isl_union_map *validity = D->getDependences(dependencyKinds);
  isl_union_map *proximity = D->getDependences(dependencyKinds);
  isl_union_set *domain = NULL;

  for (Scop::iterator SI = S.begin(), SE = S.end(); SI != SE; ++SI)
    if ((*SI)->isFinalRead())
      continue;
    else if (!domain)
      domain = isl_union_set_from_set((*SI)->getDomain());
    else
      domain = isl_union_set_union(domain,
        isl_union_set_from_set((*SI)->getDomain()));

  if (!domain)
    return false;

  DEBUG(dbgs() << "\n\nCompute schedule from: ");
  DEBUG(dbgs() << "Domain := "; isl_union_set_dump(domain); dbgs() << ";\n");
  DEBUG(dbgs() << "Proximity := "; isl_union_map_dump(proximity);
        dbgs() << ";\n");
  DEBUG(dbgs() << "Validity := "; isl_union_map_dump(validity);
        dbgs() << ";\n");

  isl_schedule *schedule;

  schedule  = isl_union_set_compute_schedule(domain, validity, proximity);

  // Get the complete schedule.
  isl_union_map *scheduleMap = isl_schedule_get_map(schedule);

  DEBUG(dbgs() << "Computed schedule: ");
  DEBUG(isl_union_map_dump(scheduleMap));
  DEBUG(dbgs() << "Individual bands: ");

  // Get individual tileable bands.
  for (int i = 0; i <  isl_schedule_n_band(schedule); i++) {
    isl_union_map *band = isl_schedule_get_band(schedule, i);

    DEBUG(dbgs() << "Band " << i << ": ");
    DEBUG(isl_union_map_dump(band));

    for (Scop::iterator SI = S.begin(), SE = S.end(); SI != SE; ++SI) {
      ScopStmt *stmt = *SI;

      if (stmt->isFinalRead())
        continue;

      isl_set *domain = stmt->getDomain();
      isl_union_map *stmtBand;
      stmtBand = isl_union_map_intersect_domain(isl_union_map_copy(band),
                                                isl_union_set_from_set(domain));

      isl_map *sband;
      isl_union_map_foreach_map(stmtBand, getSingleMap, &sband);

      sband = tileBand(sband);
      DEBUG(dbgs() << "tiled band: ");
      DEBUG(isl_map_dump(sband));

      if (i == 0)
        stmt->setScattering(sband);
      else {
        isl_map *scattering = stmt->getScattering();
        scattering = isl_map_range_product(scattering, sband);
        scattering = isl_map_flatten(scattering);
        stmt->setScattering(scattering);
      }
    }

  }

  unsigned maxScatDims = 0;

  for (Scop::iterator SI = S.begin(), SE = S.end(); SI != SE; ++SI)
    maxScatDims = std::max(isl_map_n_out((*SI)->getScattering()), maxScatDims);

  extendScattering(S, maxScatDims);
  isl_schedule_free(schedule);
  return false;
}

void ScheduleOptimizer::printScop(raw_ostream &OS) const {
}

void ScheduleOptimizer::getAnalysisUsage(AnalysisUsage &AU) const {
  ScopPass::getAnalysisUsage(AU);
  AU.addRequired<Dependences>();
}

static RegisterPass<ScheduleOptimizer> A("polly-optimize-isl",
                                         "Polly - Calculate optimized "
                                         "schedules using the isl schedule "
                                         "calculator");

Pass* polly::createScheduleOptimizerPass() {
  return new ScheduleOptimizer();
}
