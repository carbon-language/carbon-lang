//===- IslAst.cpp - isl code generator interface --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The isl code generator interface takes a Scop and generates a isl_ast. This
// ist_ast can either be returned directly or it can be pretty printed to
// stdout.
//
// A typical isl_ast output looks like this:
//
// for (c2 = max(0, ceild(n + m, 2); c2 <= min(511, floord(5 * n, 3)); c2++) {
//   bb2(c2);
// }
//
//===----------------------------------------------------------------------===//

#include "polly/CodeGen/CodeGeneration.h"
#include "polly/CodeGen/IslAst.h"
#include "polly/Dependences.h"
#include "polly/LinkAllPasses.h"
#include "polly/Options.h"
#include "polly/ScopInfo.h"
#include "llvm/Support/Debug.h"

#include "isl/union_map.h"
#include "isl/list.h"
#include "isl/ast_build.h"
#include "isl/set.h"
#include "isl/map.h"
#include "isl/aff.h"

#define DEBUG_TYPE "polly-ast"

using namespace llvm;
using namespace polly;

using IslAstUserPayload = IslAstInfo::IslAstUserPayload;

static cl::opt<bool> UseContext("polly-ast-use-context",
                                cl::desc("Use context"), cl::Hidden,
                                cl::init(false), cl::ZeroOrMore,
                                cl::cat(PollyCategory));

static cl::opt<bool> DetectParallel("polly-ast-detect-parallel",
                                    cl::desc("Detect parallelism"), cl::Hidden,
                                    cl::init(false), cl::ZeroOrMore,
                                    cl::cat(PollyCategory));

namespace polly {
class IslAst {
public:
  IslAst(Scop *Scop, Dependences &D);

  ~IslAst();

  /// Print a source code representation of the program.
  void pprint(llvm::raw_ostream &OS);

  __isl_give isl_ast_node *getAst();

  /// @brief Get the run-time conditions for the Scop.
  __isl_give isl_ast_expr *getRunCondition();

private:
  Scop *S;
  isl_ast_node *Root;
  isl_ast_expr *RunCondition;

  void buildRunCondition(__isl_keep isl_ast_build *Build);
};
} // End namespace polly.

/// @brief Free an IslAstUserPayload object pointed to by @p Ptr
static void freeIslAstUserPayload(void *Ptr) {
  delete ((IslAstInfo::IslAstUserPayload *)Ptr);
}

IslAstInfo::IslAstUserPayload::~IslAstUserPayload() {
  isl_ast_build_free(Build);
  isl_pw_aff_free(MinimalDependenceDistance);
}

/// @brief Temporary information used when building the ast.
struct AstBuildUserInfo {
  /// @brief Construct and initialize the helper struct for AST creation.
  AstBuildUserInfo()
      : Deps(nullptr), InParallelFor(false), LastForNodeId(nullptr) {}

  /// @brief The dependence information used for the parallelism check.
  Dependences *Deps;

  /// @brief Flag to indicate that we are inside a parallel for node.
  bool InParallelFor;

  /// @brief The last iterator id created for the current SCoP.
  isl_id *LastForNodeId;
};

/// @brief Print a string @p str in a single line using @p Printer.
static isl_printer *printLine(__isl_take isl_printer *Printer,
                              const std::string &str,
                              __isl_keep isl_pw_aff *PWA = nullptr) {
  Printer = isl_printer_start_line(Printer);
  Printer = isl_printer_print_str(Printer, str.c_str());
  if (PWA)
    Printer = isl_printer_print_pw_aff(Printer, PWA);
  return isl_printer_end_line(Printer);
}

/// @brief Return all broken reductions as a string of clauses (OpenMP style).
static const std::string getBrokenReductionsStr(__isl_keep isl_ast_node *Node) {
  IslAstInfo::MemoryAccessSet *BrokenReductions;
  std::string str;

  BrokenReductions = IslAstInfo::getBrokenReductions(Node);
  if (!BrokenReductions || BrokenReductions->empty())
    return "";

  // Map each type of reduction to a comma separated list of the base addresses.
  std::map<MemoryAccess::ReductionType, std::string> Clauses;
  for (MemoryAccess *MA : *BrokenReductions)
    if (MA->isWrite())
      Clauses[MA->getReductionType()] +=
          ", " + MA->getBaseAddr()->getName().str();

  // Now print the reductions sorted by type. Each type will cause a clause
  // like:  reduction (+ : sum0, sum1, sum2)
  for (const auto &ReductionClause : Clauses) {
    str += " reduction (";
    str += MemoryAccess::getReductionOperatorStr(ReductionClause.first);
    // Remove the first two symbols (", ") to make the output look pretty.
    str += " : " + ReductionClause.second.substr(2) + ")";
  }

  return str;
}

/// @brief Callback executed for each for node in the ast in order to print it.
static isl_printer *cbPrintFor(__isl_take isl_printer *Printer,
                               __isl_take isl_ast_print_options *Options,
                               __isl_keep isl_ast_node *Node, void *) {

  isl_pw_aff *DD = IslAstInfo::getMinimalDependenceDistance(Node);
  const std::string BrokenReductionsStr = getBrokenReductionsStr(Node);
  const std::string DepDisPragmaStr = "#pragma minimal dependence distance: ";
  const std::string SimdPragmaStr = "#pragma simd";
  const std::string OmpPragmaStr = "#pragma omp parallel for";

  if (DD)
    Printer = printLine(Printer, DepDisPragmaStr, DD);

  if (IslAstInfo::isInnermostParallel(Node))
    Printer = printLine(Printer, SimdPragmaStr + BrokenReductionsStr);

  if (IslAstInfo::isOutermostParallel(Node))
    Printer = printLine(Printer, OmpPragmaStr + BrokenReductionsStr);

  isl_pw_aff_free(DD);
  return isl_ast_node_for_print(Node, Printer, Options);
}

/// @brief Check if the current scheduling dimension is parallel
///
/// In case the dimension is parallel we also check if any reduction
/// dependences is broken when we exploit this parallelism. If so,
/// @p IsReductionParallel will be set to true. The reduction dependences we use
/// to check are actually the union of the transitive closure of the initial
/// reduction dependences together with their reveresal. Even though these
/// dependences connect all iterations with each other (thus they are cyclic)
/// we can perform the parallelism check as we are only interested in a zero
/// (or non-zero) dependence distance on the dimension in question.
static bool astScheduleDimIsParallel(__isl_keep isl_ast_build *Build,
                                     Dependences *D,
                                     IslAstUserPayload *NodeInfo) {
  if (!D->hasValidDependences())
    return false;

  isl_union_map *Schedule = isl_ast_build_get_schedule(Build);
  isl_union_map *Deps = D->getDependences(
      Dependences::TYPE_RAW | Dependences::TYPE_WAW | Dependences::TYPE_WAR);

  if (!D->isParallel(Schedule, Deps, &NodeInfo->MinimalDependenceDistance) &&
      !isl_union_map_free(Schedule))
    return false;

  isl_union_map *RedDeps = D->getDependences(Dependences::TYPE_TC_RED);
  if (!D->isParallel(Schedule, RedDeps))
    NodeInfo->IsReductionParallel = true;

  if (!NodeInfo->IsReductionParallel && !isl_union_map_free(Schedule))
    return true;

  // Annotate reduction parallel nodes with the memory accesses which caused the
  // reduction dependences parallel execution of the node conflicts with.
  for (const auto &MaRedPair : D->getReductionDependences()) {
    if (!MaRedPair.second)
      continue;
    RedDeps = isl_union_map_from_map(isl_map_copy(MaRedPair.second));
    if (!D->isParallel(Schedule, RedDeps))
      NodeInfo->BrokenReductions.insert(MaRedPair.first);
  }

  isl_union_map_free(Schedule);
  return true;
}

// This method is executed before the construction of a for node. It creates
// an isl_id that is used to annotate the subsequently generated ast for nodes.
//
// In this function we also run the following analyses:
//
// - Detection of openmp parallel loops
//
static __isl_give isl_id *astBuildBeforeFor(__isl_keep isl_ast_build *Build,
                                            void *User) {
  AstBuildUserInfo *BuildInfo = (AstBuildUserInfo *)User;
  IslAstUserPayload *Payload = new IslAstUserPayload();
  isl_id *Id = isl_id_alloc(isl_ast_build_get_ctx(Build), "", Payload);
  Id = isl_id_set_free_user(Id, freeIslAstUserPayload);
  BuildInfo->LastForNodeId = Id;

  // Test for parallelism only if we are not already inside a parallel loop
  if (!BuildInfo->InParallelFor)
    BuildInfo->InParallelFor = Payload->IsOutermostParallel =
        astScheduleDimIsParallel(Build, BuildInfo->Deps, Payload);

  return Id;
}

// This method is executed after the construction of a for node.
//
// It performs the following actions:
//
// - Reset the 'InParallelFor' flag, as soon as we leave a for node,
//   that is marked as openmp parallel.
//
static __isl_give isl_ast_node *
astBuildAfterFor(__isl_take isl_ast_node *Node, __isl_keep isl_ast_build *Build,
                 void *User) {
  isl_id *Id = isl_ast_node_get_annotation(Node);
  assert(Id && "Post order visit assumes annotated for nodes");
  IslAstUserPayload *Payload = (IslAstUserPayload *)isl_id_get_user(Id);
  assert(Payload && "Post order visit assumes annotated for nodes");

  AstBuildUserInfo *BuildInfo = (AstBuildUserInfo *)User;
  assert(!Payload->Build && "Build environment already set");
  Payload->Build = isl_ast_build_copy(Build);
  Payload->IsInnermost = (Id == BuildInfo->LastForNodeId);

  // Innermost loops that are surrounded by parallel loops have not yet been
  // tested for parallelism. Test them here to ensure we check all innermost
  // loops for parallelism.
  if (Payload->IsInnermost && BuildInfo->InParallelFor) {
    if (Payload->IsOutermostParallel)
      Payload->IsInnermostParallel = true;
    else
      Payload->IsInnermostParallel =
          astScheduleDimIsParallel(Build, BuildInfo->Deps, Payload);
  }
  if (Payload->IsOutermostParallel)
    BuildInfo->InParallelFor = false;

  isl_id_free(Id);
  return Node;
}

static __isl_give isl_ast_node *AtEachDomain(__isl_take isl_ast_node *Node,
                                             __isl_keep isl_ast_build *Build,
                                             void *User) {
  assert(!isl_ast_node_get_annotation(Node) && "Node already annotated");

  IslAstUserPayload *Payload = new IslAstUserPayload();
  isl_id *Id = isl_id_alloc(isl_ast_build_get_ctx(Build), "", Payload);
  Id = isl_id_set_free_user(Id, freeIslAstUserPayload);

  Payload->Build = isl_ast_build_copy(Build);

  return isl_ast_node_set_annotation(Node, Id);
}

void IslAst::buildRunCondition(__isl_keep isl_ast_build *Build) {
  // The conditions that need to be checked at run-time for this scop are
  // available as an isl_set in the AssumedContext. We generate code for this
  // check as follows. First, we generate an isl_pw_aff that is 1, if a certain
  // combination of parameter values fulfills the conditions in the assumed
  // context, and that is 0 otherwise. We then translate this isl_pw_aff into
  // an isl_ast_expr. At run-time this expression can be evaluated and the
  // optimized scop can be executed conditionally according to the result of the
  // run-time check.

  isl_aff *Zero =
      isl_aff_zero_on_domain(isl_local_space_from_space(S->getParamSpace()));
  isl_aff *One =
      isl_aff_zero_on_domain(isl_local_space_from_space(S->getParamSpace()));

  One = isl_aff_add_constant_si(One, 1);

  isl_pw_aff *PwZero = isl_pw_aff_from_aff(Zero);
  isl_pw_aff *PwOne = isl_pw_aff_from_aff(One);

  PwOne = isl_pw_aff_intersect_domain(PwOne, S->getAssumedContext());
  PwZero = isl_pw_aff_intersect_domain(
      PwZero, isl_set_complement(S->getAssumedContext()));

  isl_pw_aff *Cond = isl_pw_aff_union_max(PwOne, PwZero);

  RunCondition = isl_ast_build_expr_from_pw_aff(Build, Cond);

  // Create the alias checks from the minimal/maximal accesses in each alias
  // group. This operation is by construction quadratic in the number of
  // elements in each alias group.
  isl_ast_expr *NonAliasGroup, *MinExpr, *MaxExpr;
  for (const Scop::MinMaxVectorTy *MinMaxAccesses : S->getAliasGroups()) {
    auto AccEnd = MinMaxAccesses->end();
    for (auto AccIt0 = MinMaxAccesses->begin(); AccIt0 != AccEnd; ++AccIt0) {
      for (auto AccIt1 = AccIt0 + 1; AccIt1 != AccEnd; ++AccIt1) {
        MinExpr =
            isl_ast_expr_address_of(isl_ast_build_access_from_pw_multi_aff(
                Build, isl_pw_multi_aff_copy(AccIt0->first)));
        MaxExpr =
            isl_ast_expr_address_of(isl_ast_build_access_from_pw_multi_aff(
                Build, isl_pw_multi_aff_copy(AccIt1->second)));
        NonAliasGroup = isl_ast_expr_le(MaxExpr, MinExpr);
        MinExpr =
            isl_ast_expr_address_of(isl_ast_build_access_from_pw_multi_aff(
                Build, isl_pw_multi_aff_copy(AccIt1->first)));
        MaxExpr =
            isl_ast_expr_address_of(isl_ast_build_access_from_pw_multi_aff(
                Build, isl_pw_multi_aff_copy(AccIt0->second)));
        NonAliasGroup =
            isl_ast_expr_or(NonAliasGroup, isl_ast_expr_le(MaxExpr, MinExpr));
        RunCondition = isl_ast_expr_and(RunCondition, NonAliasGroup);
      }
    }
  }
}

IslAst::IslAst(Scop *Scop, Dependences &D) : S(Scop) {
  isl_ctx *Ctx = S->getIslCtx();
  isl_options_set_ast_build_atomic_upper_bound(Ctx, true);
  isl_ast_build *Build;
  AstBuildUserInfo BuildInfo;

  if (UseContext)
    Build = isl_ast_build_from_context(S->getContext());
  else
    Build = isl_ast_build_from_context(isl_set_universe(S->getParamSpace()));

  Build = isl_ast_build_set_at_each_domain(Build, AtEachDomain, nullptr);

  isl_union_map *Schedule =
      isl_union_map_intersect_domain(S->getSchedule(), S->getDomains());

  if (DetectParallel || PollyVectorizerChoice != VECTORIZER_NONE) {
    BuildInfo.Deps = &D;
    BuildInfo.InParallelFor = 0;

    Build = isl_ast_build_set_before_each_for(Build, &astBuildBeforeFor,
                                              &BuildInfo);
    Build =
        isl_ast_build_set_after_each_for(Build, &astBuildAfterFor, &BuildInfo);
  }

  buildRunCondition(Build);

  Root = isl_ast_build_ast_from_schedule(Build, Schedule);

  isl_ast_build_free(Build);
}

IslAst::~IslAst() {
  isl_ast_node_free(Root);
  isl_ast_expr_free(RunCondition);
}

__isl_give isl_ast_node *IslAst::getAst() { return isl_ast_node_copy(Root); }
__isl_give isl_ast_expr *IslAst::getRunCondition() {
  return isl_ast_expr_copy(RunCondition);
}

void IslAstInfo::releaseMemory() {
  if (Ast) {
    delete Ast;
    Ast = 0;
  }
}

bool IslAstInfo::runOnScop(Scop &Scop) {
  if (Ast)
    delete Ast;

  S = &Scop;

  Dependences &D = getAnalysis<Dependences>();

  Ast = new IslAst(&Scop, D);

  DEBUG(printScop(dbgs()));
  return false;
}

__isl_give isl_ast_node *IslAstInfo::getAst() const { return Ast->getAst(); }
__isl_give isl_ast_expr *IslAstInfo::getRunCondition() const {
  return Ast->getRunCondition();
}

IslAstUserPayload *IslAstInfo::getNodePayload(__isl_keep isl_ast_node *Node) {
  isl_id *Id = isl_ast_node_get_annotation(Node);
  if (!Id)
    return nullptr;
  IslAstUserPayload *Payload = (IslAstUserPayload *)isl_id_get_user(Id);
  isl_id_free(Id);
  return Payload;
}

bool IslAstInfo::isInnermost(__isl_keep isl_ast_node *Node) {
  IslAstUserPayload *Payload = getNodePayload(Node);
  return Payload && Payload->IsInnermost;
}

bool IslAstInfo::isParallel(__isl_keep isl_ast_node *Node) {
  return IslAstInfo::isInnermostParallel(Node) ||
         IslAstInfo::isOutermostParallel(Node);
}

bool IslAstInfo::isInnermostParallel(__isl_keep isl_ast_node *Node) {
  IslAstUserPayload *Payload = getNodePayload(Node);
  return Payload && Payload->IsInnermostParallel;
}

bool IslAstInfo::isOutermostParallel(__isl_keep isl_ast_node *Node) {
  IslAstUserPayload *Payload = getNodePayload(Node);
  return Payload && Payload->IsOutermostParallel;
}

bool IslAstInfo::isReductionParallel(__isl_keep isl_ast_node *Node) {
  IslAstUserPayload *Payload = getNodePayload(Node);
  return Payload && Payload->IsReductionParallel;
}

isl_union_map *IslAstInfo::getSchedule(__isl_keep isl_ast_node *Node) {
  IslAstUserPayload *Payload = getNodePayload(Node);
  return Payload ? isl_ast_build_get_schedule(Payload->Build) : nullptr;
}

isl_pw_aff *
IslAstInfo::getMinimalDependenceDistance(__isl_keep isl_ast_node *Node) {
  IslAstUserPayload *Payload = getNodePayload(Node);
  return Payload ? isl_pw_aff_copy(Payload->MinimalDependenceDistance)
                 : nullptr;
}

IslAstInfo::MemoryAccessSet *
IslAstInfo::getBrokenReductions(__isl_keep isl_ast_node *Node) {
  IslAstUserPayload *Payload = getNodePayload(Node);
  return Payload ? &Payload->BrokenReductions : nullptr;
}

isl_ast_build *IslAstInfo::getBuild(__isl_keep isl_ast_node *Node) {
  IslAstUserPayload *Payload = getNodePayload(Node);
  return Payload ? Payload->Build : nullptr;
}

void IslAstInfo::printScop(raw_ostream &OS) const {
  isl_ast_print_options *Options;
  isl_ast_node *RootNode = getAst();
  isl_ast_expr *RunCondition = getRunCondition();
  char *RtCStr, *AstStr;

  Scop &S = getCurScop();
  Options = isl_ast_print_options_alloc(S.getIslCtx());
  Options = isl_ast_print_options_set_print_for(Options, cbPrintFor, nullptr);

  isl_printer *P = isl_printer_to_str(S.getIslCtx());
  P = isl_printer_print_ast_expr(P, RunCondition);
  RtCStr = isl_printer_get_str(P);
  P = isl_printer_flush(P);
  P = isl_printer_indent(P, 4);
  P = isl_printer_set_output_format(P, ISL_FORMAT_C);
  P = isl_ast_node_print(RootNode, P, Options);
  AstStr = isl_printer_get_str(P);

  Function *F = S.getRegion().getEntry()->getParent();
  isl_union_map *Schedule =
      isl_union_map_intersect_domain(S.getSchedule(), S.getDomains());

  OS << ":: isl ast :: " << F->getName() << " :: " << S.getRegion().getNameStr()
     << "\n";
  DEBUG(dbgs() << S.getContextStr() << "\n"; isl_union_map_dump(Schedule));
  OS << "\nif (" << RtCStr << ")\n\n";
  OS << AstStr << "\n";
  OS << "else\n";
  OS << "    {  /* original code */ }\n\n";

  isl_ast_expr_free(RunCondition);
  isl_union_map_free(Schedule);
  isl_ast_node_free(RootNode);
  isl_printer_free(P);
}

void IslAstInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  // Get the Common analysis usage of ScopPasses.
  ScopPass::getAnalysisUsage(AU);
  AU.addRequired<ScopInfo>();
  AU.addRequired<Dependences>();
}

char IslAstInfo::ID = 0;

Pass *polly::createIslAstInfoPass() { return new IslAstInfo(); }

INITIALIZE_PASS_BEGIN(IslAstInfo, "polly-ast",
                      "Polly - Generate an AST of the SCoP (isl)", false,
                      false);
INITIALIZE_PASS_DEPENDENCY(ScopInfo);
INITIALIZE_PASS_DEPENDENCY(Dependences);
INITIALIZE_PASS_END(IslAstInfo, "polly-ast",
                    "Polly - Generate an AST from the SCoP (isl)", false, false)
