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

namespace polly {
bool DisablePollyTiling;
}
static cl::opt<bool, true>
    DisableTiling("polly-no-tiling",
                  cl::desc("Disable tiling in the scheduler"),
                  cl::location(polly::DisablePollyTiling), cl::init(false),
                  cl::ZeroOrMore, cl::cat(PollyCategory));

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

static cl::opt<int> DefaultTileSize(
    "polly-default-tile-size",
    cl::desc("The default tile size (if not enough were provided by"
             " --polly-tile-sizes)"),
    cl::Hidden, cl::init(32), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::list<int> TileSizes("polly-tile-sizes",
                               cl::desc("A tile size"
                                        " for each loop dimension, filled with"
                                        " --polly-default-tile-size"),
                               cl::Hidden, cl::ZeroOrMore, cl::CommaSeparated,
                               cl::cat(PollyCategory));
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

  /// @brief Decide if the @p NewSchedule is profitable for @p S.
  ///
  /// @param S           The SCoP we optimize.
  /// @param NewSchedule The new schedule we computed.
  ///
  /// @return True, if we believe @p NewSchedule is an improvement for @p S.
  bool isProfitableSchedule(Scop &S, __isl_keep isl_union_map *NewSchedule);

  /// @brief Pre-vectorizes one scheduling dimension of a schedule band.
  ///
  /// prevectSchedBand splits out the dimension DimToVectorize, tiles it and
  /// sinks the resulting point loop.
  ///
  /// Example (DimToVectorize=0, VectorWidth=4):
  ///
  /// | Before transformation:
  /// |
  /// | A[i,j] -> [i,j]
  /// |
  /// | for (i = 0; i < 128; i++)
  /// |    for (j = 0; j < 128; j++)
  /// |      A(i,j);
  ///
  /// | After transformation:
  /// |
  /// | for (it = 0; it < 32; it+=1)
  /// |    for (j = 0; j < 128; j++)
  /// |      for (ip = 0; ip <= 3; ip++)
  /// |        A(4 * it + ip,j);
  ///
  /// The goal of this transformation is to create a trivially vectorizable
  /// loop.  This means a parallel loop at the innermost level that has a
  /// constant number of iterations corresponding to the target vector width.
  ///
  /// This transformation creates a loop at the innermost level. The loop has
  /// a constant number of iterations, if the number of loop iterations at
  /// DimToVectorize can be divided by VectorWidth. The default VectorWidth is
  /// currently constant and not yet target specific. This function does not
  /// reason about parallelism.
  static __isl_give isl_schedule_node *
  prevectSchedBand(__isl_take isl_schedule_node *Node, unsigned DimToVectorize,
                   int VectorWidth = 4);

  /// @brief Apply additional optimizations on the bands in the schedule tree.
  ///
  /// We are looking for an innermost band node and apply the following
  /// transformations:
  ///
  ///  - Tile the band
  ///      - if the band is tileable
  ///      - if the band has more than one loop dimension
  ///
  ///  - Prevectorize the schedule of the band (or the point loop in case of
  ///    tiling).
  ///      - if vectorization is enabled
  ///
  /// @param Node The schedule node to (possibly) optimize.
  /// @param User A pointer to forward some use information (currently unused).
  static isl_schedule_node *optimizeBand(isl_schedule_node *Node, void *User);

  /// @brief Apply post-scheduling transformations.
  ///
  /// This function applies a set of additional local transformations on the
  /// schedule tree as it computed by the isl scheduler. Local transformations
  /// applied include:
  ///
  ///   - Tiling
  ///   - Prevectorization
  ///
  /// @param Schedule The schedule object post-transformations will be applied
  ///                 on.
  /// @returns        The transformed schedule.
  static __isl_give isl_schedule *
  addPostTransforms(__isl_take isl_schedule *Schedule);

  using llvm::Pass::doFinalization;

  virtual bool doFinalization() override {
    isl_schedule_free(LastSchedule);
    LastSchedule = nullptr;
    return true;
  }
};
}

char IslScheduleOptimizer::ID = 0;

__isl_give isl_schedule_node *
IslScheduleOptimizer::prevectSchedBand(__isl_take isl_schedule_node *Node,
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
  Node = isl_schedule_node_band_sink(Node);
  Node = isl_schedule_node_child(Node, 0);
  return Node;
}

isl_schedule_node *IslScheduleOptimizer::optimizeBand(isl_schedule_node *Node,
                                                      void *User) {
  if (isl_schedule_node_get_type(Node) != isl_schedule_node_band)
    return Node;

  if (isl_schedule_node_n_children(Node) != 1)
    return Node;

  if (!isl_schedule_node_band_get_permutable(Node))
    return Node;

  auto Space = isl_schedule_node_band_get_space(Node);
  auto Dims = isl_space_dim(Space, isl_dim_set);

  if (Dims <= 1) {
    isl_space_free(Space);
    return Node;
  }

  auto Child = isl_schedule_node_get_child(Node, 0);
  auto Type = isl_schedule_node_get_type(Child);
  isl_schedule_node_free(Child);

  if (Type != isl_schedule_node_leaf) {
    isl_space_free(Space);
    return Node;
  }

  auto Sizes = isl_multi_val_zero(Space);
  auto Ctx = isl_schedule_node_get_ctx(Node);

  for (unsigned i = 0; i < Dims; i++) {
    auto tileSize = TileSizes.size() > i ? TileSizes[i] : DefaultTileSize;
    Sizes = isl_multi_val_set_val(Sizes, i, isl_val_int_from_si(Ctx, tileSize));
  }

  isl_schedule_node *Res;

  if (DisableTiling) {
    isl_multi_val_free(Sizes);
    Res = Node;
  } else {
    Res = isl_schedule_node_band_tile(Node, Sizes);
    Res = isl_schedule_node_child(Res, 0);
  }

  if (PollyVectorizerChoice == VECTORIZER_NONE)
    return Res;

  for (int i = Dims - 1; i >= 0; i--)
    if (isl_schedule_node_band_member_get_coincident(Res, i)) {
      Res = IslScheduleOptimizer::prevectSchedBand(Res, i);
      break;
    }

  return Res;
}

__isl_give isl_schedule *
IslScheduleOptimizer::addPostTransforms(__isl_take isl_schedule *Schedule) {
  isl_schedule_node *Root = isl_schedule_get_root(Schedule);
  isl_schedule_free(Schedule);
  Root = isl_schedule_node_map_descendant_bottom_up(
      Root, IslScheduleOptimizer::optimizeBand, NULL);
  auto S = isl_schedule_node_get_schedule(Root);
  isl_schedule_node_free(Root);
  return S;
}

bool IslScheduleOptimizer::isProfitableSchedule(
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

  isl_schedule *NewSchedule = addPostTransforms(Schedule);
  isl_union_map *NewScheduleMap = isl_schedule_get_map(NewSchedule);

  if (!isProfitableSchedule(S, NewScheduleMap)) {
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
