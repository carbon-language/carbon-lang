//===- Schedule.cpp - Calculate an optimized schedule ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass generates an entirey new schedule tree from the data dependences
// and iteration domains. The new schedule tree is computed in two steps:
//
// 1) The isl scheduling optimizer is run
//
// The isl scheduling optimizer creates a new schedule tree that maximizes
// parallelism and tileability and minimizes data-dependence distances. The
// algorithm used is a modified version of the ``Pluto'' algorithm:
//
//   U. Bondhugula, A. Hartono, J. Ramanujam, and P. Sadayappan.
//   A Practical Automatic Polyhedral Parallelizer and Locality Optimizer.
//   In Proceedings of the 2008 ACM SIGPLAN Conference On Programming Language
//   Design and Implementation, PLDI ’08, pages 101–113. ACM, 2008.
//
// 2) A set of post-scheduling transformations is applied on the schedule tree.
//
// These optimizations include:
//
//  - Tiling of the innermost tilable bands
//  - Prevectorization - The coice of a possible outer loop that is strip-mined
//                       to the innermost level to enable inner-loop
//                       vectorization.
//  - Some optimizations for spatial locality are also planned.
//
// For a detailed description of the schedule tree itself please see section 6
// of:
//
// Polyhedral AST generation is more than scanning polyhedra
// Tobias Grosser, Sven Verdoolaege, Albert Cohen
// ACM Transations on Programming Languages and Systems (TOPLAS),
// 37(4), July 2015
// http://www.grosser.es/#pub-polyhedral-AST-generation
//
// This publication also contains a detailed discussion of the different options
// for polyhedral loop unrolling, full/partial tile separation and other uses
// of the schedule tree.
//
//===----------------------------------------------------------------------===//

#include "polly/ScheduleOptimizer.h"
#include "polly/CodeGen/CodeGeneration.h"
#include "polly/DependenceInfo.h"
#include "polly/LinkAllPasses.h"
#include "polly/Options.h"
#include "polly/ScopInfo.h"
#include "polly/Support/GICHelper.h"
#include "llvm/Support/Debug.h"
#include "isl/aff.h"
#include "isl/band.h"
#include "isl/constraint.h"
#include "isl/map.h"
#include "isl/options.h"
#include "isl/printer.h"
#include "isl/schedule.h"
#include "isl/schedule_node.h"
#include "isl/space.h"
#include "isl/union_map.h"
#include "isl/union_set.h"

using namespace llvm;
using namespace polly;

#define DEBUG_TYPE "polly-opt-isl"

static cl::opt<std::string>
    OptimizeDeps("polly-opt-optimize-only",
                 cl::desc("Only a certain kind of dependences (all/raw)"),
                 cl::Hidden, cl::init("all"), cl::ZeroOrMore,
                 cl::cat(PollyCategory));

static cl::opt<std::string>
    SimplifyDeps("polly-opt-simplify-deps",
                 cl::desc("Dependences should be simplified (yes/no)"),
                 cl::Hidden, cl::init("yes"), cl::ZeroOrMore,
                 cl::cat(PollyCategory));

static cl::opt<int> MaxConstantTerm(
    "polly-opt-max-constant-term",
    cl::desc("The maximal constant term allowed (-1 is unlimited)"), cl::Hidden,
    cl::init(20), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> MaxCoefficient(
    "polly-opt-max-coefficient",
    cl::desc("The maximal coefficient allowed (-1 is unlimited)"), cl::Hidden,
    cl::init(20), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<std::string> FusionStrategy(
    "polly-opt-fusion", cl::desc("The fusion strategy to choose (min/max)"),
    cl::Hidden, cl::init("min"), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<std::string>
    MaximizeBandDepth("polly-opt-maximize-bands",
                      cl::desc("Maximize the band depth (yes/no)"), cl::Hidden,
                      cl::init("yes"), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> PrevectorWidth(
    "polly-prevect-width",
    cl::desc(
        "The number of loop iterations to strip-mine for pre-vectorization"),
    cl::Hidden, cl::init(4), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool> FirstLevelTiling("polly-tiling",
                                      cl::desc("Enable loop tiling"),
                                      cl::init(true), cl::ZeroOrMore,
                                      cl::cat(PollyCategory));

static cl::opt<int> FirstLevelDefaultTileSize(
    "polly-default-tile-size",
    cl::desc("The default tile size (if not enough were provided by"
             " --polly-tile-sizes)"),
    cl::Hidden, cl::init(32), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::list<int> FirstLevelTileSizes(
    "polly-tile-sizes", cl::desc("A tile size for each loop dimension, filled "
                                 "with --polly-default-tile-size"),
    cl::Hidden, cl::ZeroOrMore, cl::CommaSeparated, cl::cat(PollyCategory));

static cl::opt<bool>
    SecondLevelTiling("polly-2nd-level-tiling",
                      cl::desc("Enable a 2nd level loop of loop tiling"),
                      cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> SecondLevelDefaultTileSize(
    "polly-2nd-level-default-tile-size",
    cl::desc("The default 2nd-level tile size (if not enough were provided by"
             " --polly-2nd-level-tile-sizes)"),
    cl::Hidden, cl::init(16), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::list<int>
    SecondLevelTileSizes("polly-2nd-level-tile-sizes",
                         cl::desc("A tile size for each loop dimension, filled "
                                  "with --polly-default-tile-size"),
                         cl::Hidden, cl::ZeroOrMore, cl::CommaSeparated,
                         cl::cat(PollyCategory));

static cl::opt<bool> RegisterTiling("polly-register-tiling",
                                    cl::desc("Enable register tiling"),
                                    cl::init(false), cl::ZeroOrMore,
                                    cl::cat(PollyCategory));

static cl::opt<int> RegisterDefaultTileSize(
    "polly-register-tiling-default-tile-size",
    cl::desc("The default register tile size (if not enough were provided by"
             " --polly-register-tile-sizes)"),
    cl::Hidden, cl::init(2), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::list<int>
    RegisterTileSizes("polly-register-tile-sizes",
                      cl::desc("A tile size for each loop dimension, filled "
                               "with --polly-register-tile-size"),
                      cl::Hidden, cl::ZeroOrMore, cl::CommaSeparated,
                      cl::cat(PollyCategory));

__isl_give isl_schedule_node *
ScheduleTreeOptimizer::prevectSchedBand(__isl_take isl_schedule_node *Node,
                                        unsigned DimToVectorize,
                                        int VectorWidth) {
  assert(isl_schedule_node_get_type(Node) == isl_schedule_node_band);

  auto Space = isl_schedule_node_band_get_space(Node);
  auto ScheduleDimensions = isl_space_dim(Space, isl_dim_set);
  isl_space_free(Space);
  assert(DimToVectorize < ScheduleDimensions);

  if (DimToVectorize > 0) {
    Node = isl_schedule_node_band_split(Node, DimToVectorize);
    Node = isl_schedule_node_child(Node, 0);
  }
  if (DimToVectorize < ScheduleDimensions - 1)
    Node = isl_schedule_node_band_split(Node, 1);
  Space = isl_schedule_node_band_get_space(Node);
  auto Sizes = isl_multi_val_zero(Space);
  auto Ctx = isl_schedule_node_get_ctx(Node);
  Sizes =
      isl_multi_val_set_val(Sizes, 0, isl_val_int_from_si(Ctx, VectorWidth));
  Node = isl_schedule_node_band_tile(Node, Sizes);
  Node = isl_schedule_node_child(Node, 0);
  // Make sure the "trivially vectorizable loop" is not unrolled. Otherwise,
  // we will have troubles to match it in the backend.
  Node = isl_schedule_node_band_set_ast_build_options(
      Node, isl_union_set_read_from_str(Ctx, "{ unroll[x]: 1 = 0 }"));
  Node = isl_schedule_node_band_sink(Node);
  Node = isl_schedule_node_child(Node, 0);
  return Node;
}

__isl_give isl_schedule_node *
ScheduleTreeOptimizer::tileNode(__isl_take isl_schedule_node *Node,
                                const char *Identifier, ArrayRef<int> TileSizes,
                                int DefaultTileSize) {
  auto Ctx = isl_schedule_node_get_ctx(Node);
  auto Space = isl_schedule_node_band_get_space(Node);
  auto Dims = isl_space_dim(Space, isl_dim_set);
  auto Sizes = isl_multi_val_zero(Space);
  std::string IdentifierString(Identifier);
  for (unsigned i = 0; i < Dims; i++) {
    auto tileSize = i < TileSizes.size() ? TileSizes[i] : DefaultTileSize;
    Sizes = isl_multi_val_set_val(Sizes, i, isl_val_int_from_si(Ctx, tileSize));
  }
  auto TileLoopMarkerStr = IdentifierString + " - Tiles";
  isl_id *TileLoopMarker =
      isl_id_alloc(Ctx, TileLoopMarkerStr.c_str(), nullptr);
  Node = isl_schedule_node_insert_mark(Node, TileLoopMarker);
  Node = isl_schedule_node_child(Node, 0);
  Node = isl_schedule_node_band_tile(Node, Sizes);
  Node = isl_schedule_node_child(Node, 0);
  auto PointLoopMarkerStr = IdentifierString + " - Points";
  isl_id *PointLoopMarker =
      isl_id_alloc(Ctx, PointLoopMarkerStr.c_str(), nullptr);
  Node = isl_schedule_node_insert_mark(Node, PointLoopMarker);
  Node = isl_schedule_node_child(Node, 0);
  return Node;
}

bool ScheduleTreeOptimizer::isTileableBandNode(
    __isl_keep isl_schedule_node *Node) {
  if (isl_schedule_node_get_type(Node) != isl_schedule_node_band)
    return false;

  if (isl_schedule_node_n_children(Node) != 1)
    return false;

  if (!isl_schedule_node_band_get_permutable(Node))
    return false;

  auto Space = isl_schedule_node_band_get_space(Node);
  auto Dims = isl_space_dim(Space, isl_dim_set);
  isl_space_free(Space);

  if (Dims <= 1)
    return false;

  auto Child = isl_schedule_node_get_child(Node, 0);
  auto Type = isl_schedule_node_get_type(Child);
  isl_schedule_node_free(Child);

  if (Type != isl_schedule_node_leaf)
    return false;

  return true;
}

__isl_give isl_schedule_node *
ScheduleTreeOptimizer::optimizeBand(__isl_take isl_schedule_node *Node,
                                    void *User) {
  if (!isTileableBandNode(Node))
    return Node;

  if (FirstLevelTiling)
    Node = tileNode(Node, "1st level tiling", FirstLevelTileSizes,
                    FirstLevelDefaultTileSize);

  if (SecondLevelTiling)
    Node = tileNode(Node, "2nd level tiling", SecondLevelTileSizes,
                    SecondLevelDefaultTileSize);

  if (RegisterTiling) {
    auto *Ctx = isl_schedule_node_get_ctx(Node);
    Node = tileNode(Node, "Register tiling", RegisterTileSizes,
                    RegisterDefaultTileSize);
    Node = isl_schedule_node_band_set_ast_build_options(
        Node, isl_union_set_read_from_str(Ctx, "{unroll[x]}"));
  }

  if (PollyVectorizerChoice == VECTORIZER_NONE)
    return Node;

  auto Space = isl_schedule_node_band_get_space(Node);
  auto Dims = isl_space_dim(Space, isl_dim_set);
  isl_space_free(Space);

  for (int i = Dims - 1; i >= 0; i--)
    if (isl_schedule_node_band_member_get_coincident(Node, i)) {
      Node = prevectSchedBand(Node, i, PrevectorWidth);
      break;
    }

  return Node;
}

__isl_give isl_schedule *
ScheduleTreeOptimizer::optimizeSchedule(__isl_take isl_schedule *Schedule) {
  isl_schedule_node *Root = isl_schedule_get_root(Schedule);
  Root = optimizeScheduleNode(Root);
  isl_schedule_free(Schedule);
  auto S = isl_schedule_node_get_schedule(Root);
  isl_schedule_node_free(Root);
  return S;
}

__isl_give isl_schedule_node *ScheduleTreeOptimizer::optimizeScheduleNode(
    __isl_take isl_schedule_node *Node) {
  Node = isl_schedule_node_map_descendant_bottom_up(Node, optimizeBand, NULL);
  return Node;
}

bool ScheduleTreeOptimizer::isProfitableSchedule(
    Scop &S, __isl_keep isl_union_map *NewSchedule) {
  // To understand if the schedule has been optimized we check if the schedule
  // has changed at all.
  // TODO: We can improve this by tracking if any necessarily beneficial
  // transformations have been performed. This can e.g. be tiling, loop
  // interchange, or ...) We can track this either at the place where the
  // transformation has been performed or, in case of automatic ILP based
  // optimizations, by comparing (yet to be defined) performance metrics
  // before/after the scheduling optimizer
  // (e.g., #stride-one accesses)
  isl_union_map *OldSchedule = S.getSchedule();
  bool changed = !isl_union_map_is_equal(OldSchedule, NewSchedule);
  isl_union_map_free(OldSchedule);
  return changed;
}

namespace {
class IslScheduleOptimizer : public ScopPass {
public:
  static char ID;
  explicit IslScheduleOptimizer() : ScopPass(ID) { LastSchedule = nullptr; }

  ~IslScheduleOptimizer() { isl_schedule_free(LastSchedule); }

  bool runOnScop(Scop &S) override;
  void printScop(raw_ostream &OS, Scop &S) const override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  isl_schedule *LastSchedule;

  using llvm::Pass::doFinalization;

  virtual bool doFinalization() override {
    isl_schedule_free(LastSchedule);
    LastSchedule = nullptr;
    return true;
  }
};
}

char IslScheduleOptimizer::ID = 0;

bool IslScheduleOptimizer::runOnScop(Scop &S) {

  // Skip empty SCoPs but still allow code generation as it will delete the
  // loops present but not needed.
  if (S.getSize() == 0) {
    S.markAsOptimized();
    return false;
  }

  const Dependences &D = getAnalysis<DependenceInfo>().getDependences();

  if (!D.hasValidDependences())
    return false;

  isl_schedule_free(LastSchedule);
  LastSchedule = nullptr;

  // Build input data.
  int ValidityKinds =
      Dependences::TYPE_RAW | Dependences::TYPE_WAR | Dependences::TYPE_WAW;
  int ProximityKinds;

  if (OptimizeDeps == "all")
    ProximityKinds =
        Dependences::TYPE_RAW | Dependences::TYPE_WAR | Dependences::TYPE_WAW;
  else if (OptimizeDeps == "raw")
    ProximityKinds = Dependences::TYPE_RAW;
  else {
    errs() << "Do not know how to optimize for '" << OptimizeDeps << "'"
           << " Falling back to optimizing all dependences.\n";
    ProximityKinds =
        Dependences::TYPE_RAW | Dependences::TYPE_WAR | Dependences::TYPE_WAW;
  }

  isl_union_set *Domain = S.getDomains();

  if (!Domain)
    return false;

  isl_union_map *Validity = D.getDependences(ValidityKinds);
  isl_union_map *Proximity = D.getDependences(ProximityKinds);

  // Simplify the dependences by removing the constraints introduced by the
  // domains. This can speed up the scheduling time significantly, as large
  // constant coefficients will be removed from the dependences. The
  // introduction of some additional dependences reduces the possible
  // transformations, but in most cases, such transformation do not seem to be
  // interesting anyway. In some cases this option may stop the scheduler to
  // find any schedule.
  if (SimplifyDeps == "yes") {
    Validity = isl_union_map_gist_domain(Validity, isl_union_set_copy(Domain));
    Validity = isl_union_map_gist_range(Validity, isl_union_set_copy(Domain));
    Proximity =
        isl_union_map_gist_domain(Proximity, isl_union_set_copy(Domain));
    Proximity = isl_union_map_gist_range(Proximity, isl_union_set_copy(Domain));
  } else if (SimplifyDeps != "no") {
    errs() << "warning: Option -polly-opt-simplify-deps should either be 'yes' "
              "or 'no'. Falling back to default: 'yes'\n";
  }

  DEBUG(dbgs() << "\n\nCompute schedule from: ");
  DEBUG(dbgs() << "Domain := " << stringFromIslObj(Domain) << ";\n");
  DEBUG(dbgs() << "Proximity := " << stringFromIslObj(Proximity) << ";\n");
  DEBUG(dbgs() << "Validity := " << stringFromIslObj(Validity) << ";\n");

  unsigned IslSerializeSCCs;

  if (FusionStrategy == "max") {
    IslSerializeSCCs = 0;
  } else if (FusionStrategy == "min") {
    IslSerializeSCCs = 1;
  } else {
    errs() << "warning: Unknown fusion strategy. Falling back to maximal "
              "fusion.\n";
    IslSerializeSCCs = 0;
  }

  int IslMaximizeBands;

  if (MaximizeBandDepth == "yes") {
    IslMaximizeBands = 1;
  } else if (MaximizeBandDepth == "no") {
    IslMaximizeBands = 0;
  } else {
    errs() << "warning: Option -polly-opt-maximize-bands should either be 'yes'"
              " or 'no'. Falling back to default: 'yes'\n";
    IslMaximizeBands = 1;
  }

  isl_options_set_schedule_serialize_sccs(S.getIslCtx(), IslSerializeSCCs);
  isl_options_set_schedule_maximize_band_depth(S.getIslCtx(), IslMaximizeBands);
  isl_options_set_schedule_max_constant_term(S.getIslCtx(), MaxConstantTerm);
  isl_options_set_schedule_max_coefficient(S.getIslCtx(), MaxCoefficient);
  isl_options_set_tile_scale_tile_loops(S.getIslCtx(), 0);

  isl_options_set_on_error(S.getIslCtx(), ISL_ON_ERROR_CONTINUE);

  isl_schedule_constraints *ScheduleConstraints;
  ScheduleConstraints = isl_schedule_constraints_on_domain(Domain);
  ScheduleConstraints =
      isl_schedule_constraints_set_proximity(ScheduleConstraints, Proximity);
  ScheduleConstraints = isl_schedule_constraints_set_validity(
      ScheduleConstraints, isl_union_map_copy(Validity));
  ScheduleConstraints =
      isl_schedule_constraints_set_coincidence(ScheduleConstraints, Validity);
  isl_schedule *Schedule;
  Schedule = isl_schedule_constraints_compute_schedule(ScheduleConstraints);
  isl_options_set_on_error(S.getIslCtx(), ISL_ON_ERROR_ABORT);

  // In cases the scheduler is not able to optimize the code, we just do not
  // touch the schedule.
  if (!Schedule)
    return false;

  DEBUG({
    auto *P = isl_printer_to_str(S.getIslCtx());
    P = isl_printer_set_yaml_style(P, ISL_YAML_STYLE_BLOCK);
    P = isl_printer_print_schedule(P, Schedule);
    dbgs() << "NewScheduleTree: \n" << isl_printer_get_str(P) << "\n";
    isl_printer_free(P);
  });

  isl_schedule *NewSchedule = ScheduleTreeOptimizer::optimizeSchedule(Schedule);
  isl_union_map *NewScheduleMap = isl_schedule_get_map(NewSchedule);

  if (!ScheduleTreeOptimizer::isProfitableSchedule(S, NewScheduleMap)) {
    isl_union_map_free(NewScheduleMap);
    isl_schedule_free(NewSchedule);
    return false;
  }

  S.setScheduleTree(NewSchedule);
  S.markAsOptimized();

  isl_union_map_free(NewScheduleMap);
  return false;
}

void IslScheduleOptimizer::printScop(raw_ostream &OS, Scop &) const {
  isl_printer *p;
  char *ScheduleStr;

  OS << "Calculated schedule:\n";

  if (!LastSchedule) {
    OS << "n/a\n";
    return;
  }

  p = isl_printer_to_str(isl_schedule_get_ctx(LastSchedule));
  p = isl_printer_print_schedule(p, LastSchedule);
  ScheduleStr = isl_printer_get_str(p);
  isl_printer_free(p);

  OS << ScheduleStr << "\n";
}

void IslScheduleOptimizer::getAnalysisUsage(AnalysisUsage &AU) const {
  ScopPass::getAnalysisUsage(AU);
  AU.addRequired<DependenceInfo>();
}

Pass *polly::createIslScheduleOptimizerPass() {
  return new IslScheduleOptimizer();
}

INITIALIZE_PASS_BEGIN(IslScheduleOptimizer, "polly-opt-isl",
                      "Polly - Optimize schedule of SCoP", false, false);
INITIALIZE_PASS_DEPENDENCY(DependenceInfo);
INITIALIZE_PASS_DEPENDENCY(ScopInfo);
INITIALIZE_PASS_END(IslScheduleOptimizer, "polly-opt-isl",
                    "Polly - Optimize schedule of SCoP", false, false)
