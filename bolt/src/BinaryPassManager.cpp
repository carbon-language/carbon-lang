//===--- BinaryPassManager.cpp - Binary-level analysis/optimization passes ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "BinaryPassManager.h"
#include "Passes/Aligner.h"
#include "Passes/AllocCombiner.h"
#include "Passes/FrameOptimizer.h"
#include "Passes/IdenticalCodeFolding.h"
#include "Passes/IndirectCallPromotion.h"
#include "Passes/Inliner.h"
#include "Passes/Instrumentation.h"
#include "Passes/JTFootprintReduction.h"
#include "Passes/LongJmp.h"
#include "Passes/PLTCall.h"
#include "Passes/PatchEntries.h"
#include "Passes/RegReAssign.h"
#include "Passes/ReorderData.h"
#include "Passes/ReorderFunctions.h"
#include "Passes/RetpolineInsertion.h"
#include "Passes/SplitFunctions.h"
#include "Passes/StokeInfo.h"
#include "Passes/ValidateInternalCalls.h"
#include "Passes/VeneerElimination.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>

using namespace llvm;

namespace opts {

extern cl::OptionCategory BoltOptCategory;
extern cl::OptionCategory BoltCategory;

extern cl::opt<bool> Instrument;

extern cl::opt<unsigned> Verbosity;
extern cl::opt<bool> PrintAll;
extern cl::opt<bool> PrintDynoStats;
extern cl::opt<bool> DumpDotAll;
extern cl::opt<bolt::PLTCall::OptType> PLT;

static cl::opt<bool>
DynoStatsAll("dyno-stats-all",
  cl::desc("print dyno stats after each stage"),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltCategory));

static cl::opt<bool>
EliminateUnreachable("eliminate-unreachable",
  cl::desc("eliminate unreachable code"),
  cl::init(true),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

cl::opt<bool>
ICF("icf",
  cl::desc("fold functions with identical code"),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
JTFootprintReductionFlag("jt-footprint-reduction",
  cl::desc("make jump tables size smaller at the cost of using more "
           "instructions at jump sites"),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
PrintJTFootprintReduction("print-after-jt-footprint-reduction",
  cl::desc("print function after jt-footprint-reduction pass"),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

cl::opt<bool>
NeverPrint("never-print",
  cl::desc("never print"),
  cl::init(false),
  cl::ZeroOrMore,
  cl::ReallyHidden,
  cl::cat(BoltOptCategory));

cl::opt<bool>
PrintAfterBranchFixup("print-after-branch-fixup",
  cl::desc("print function after fixing local branches"),
  cl::Hidden,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
PrintAfterLowering("print-after-lowering",
  cl::desc("print function after instruction lowering"),
  cl::Hidden,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
PrintFOP("print-fop",
  cl::desc("print functions after frame optimizer pass"),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

cl::opt<bool>
PrintFinalized("print-finalized",
  cl::desc("print function after CFG is finalized"),
  cl::Hidden,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
PrintLongJmp("print-longjmp",
  cl::desc("print functions after longjmp pass"),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
PrintICF("print-icf",
  cl::desc("print functions after ICF optimization"),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
PrintICP("print-icp",
  cl::desc("print functions after indirect call promotion"),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
PrintRegReAssign("print-regreassign",
  cl::desc("print functions after regreassign pass"),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
PrintInline("print-inline",
  cl::desc("print functions after inlining optimization"),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
PrintOptimizeBodyless("print-optimize-bodyless",
  cl::desc("print functions after bodyless optimization"),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
PrintPLT("print-plt",
  cl::desc("print functions after PLT optimization"),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
PrintPeepholes("print-peepholes",
  cl::desc("print functions after peephole optimization"),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

cl::opt<bool>
PrintReordered("print-reordered",
  cl::desc("print functions after layout optimization"),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
PrintReorderedFunctions("print-reordered-functions",
  cl::desc("print functions after clustering"),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
PrintSCTC("print-sctc",
  cl::desc("print functions after conditional tail call simplification"),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
PrintSimplifyROLoads("print-simplify-rodata-loads",
  cl::desc("print functions after simplification of RO data loads"),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
PrintSplit("print-split",
  cl::desc("print functions after code splitting"),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
PrintUCE("print-uce",
  cl::desc("print functions after unreachable code elimination"),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
PrintProfileStats("print-profile-stats",
  cl::desc("print profile quality/bias analysis"),
  cl::ZeroOrMore,
  cl::init(false),
  cl::cat(BoltCategory));

static cl::opt<bool>
SimplifyConditionalTailCalls("simplify-conditional-tail-calls",
  cl::desc("simplify conditional tail calls by removing unnecessary jumps"),
  cl::init(true),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
SimplifyRODataLoads("simplify-rodata-loads",
  cl::desc("simplify loads from read-only sections by replacing the memory "
           "operand with the constant found in the corresponding section"),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
RegReAssign("reg-reassign",
  cl::desc("reassign registers so as to avoid using REX prefixes in hot code"),
  cl::init(false),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
StringOps("inline-memcpy",
  cl::desc("inline memcpy using 'rep movsb' instruction (X86-only)"),
  cl::init(false),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::list<std::string>
SpecializeMemcpy1("memcpy1-spec",
  cl::desc("list of functions with call sites for which to specialize memcpy() "
           "for size 1"),
  cl::value_desc("func1,func2:cs1:cs2,func3:cs1,..."),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::opt<bool>
StripRepRet("strip-rep-ret",
  cl::desc("strip 'repz' prefix from 'repz retq' sequence (on by default)"),
  cl::init(true),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

llvm::cl::opt<bool>
TimeOpts("time-opts",
  cl::desc("print time spent in each optimization"),
  cl::init(false),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static llvm::cl::opt<bool>
VerifyCFG("verify-cfg",
  cl::desc("verify the CFG after every pass"),
  cl::init(false),
  cl::Hidden,
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static llvm::cl::opt<bool>
Stoke("stoke",
  cl::desc("turn on the stoke analysis"),
  cl::init(false),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static llvm::cl::opt<bool>
PrintStoke("print-stoke",
  cl::desc("print functions after stoke analysis"),
  cl::init(false),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static llvm::cl::opt<bool>
  PrintVeneerElimination("print-veneer-elimination",
    cl::desc("print functions after veneer elimination pass"),
    cl::init(false),
    cl::ZeroOrMore,
    cl::cat(BoltOptCategory));

static llvm::cl::opt<bool>
  PrintRetpolineInsertion("print-retpoline-insertion",
    cl::desc("print functions after retpoline insertion pass"),
    cl::init(false),
    cl::ZeroOrMore,
    cl::cat(BoltCategory));

} // namespace opts

namespace llvm {
namespace bolt {

using namespace opts;

const char BinaryFunctionPassManager::TimerGroupName[] =
    "passman";
const char BinaryFunctionPassManager::TimerGroupDesc[] =
    "Binary Function Pass Manager";

void BinaryFunctionPassManager::runPasses() {
  auto &BFs = BC.getBinaryFunctions();
  for (const auto &OptPassPair : Passes) {
    if (!OptPassPair.first)
      continue;

    auto &Pass = OptPassPair.second;

    if (opts::Verbosity > 0) {
      outs() << "BOLT-INFO: Starting pass: " << Pass->getName() << "\n";
    }

    NamedRegionTimer T(Pass->getName(), Pass->getName(), TimerGroupName,
                       TimerGroupDesc, TimeOpts);

    callWithDynoStats(
      [this,&Pass] {
        Pass->runOnFunctions(BC);
      },
      BFs,
      Pass->getName(),
      opts::DynoStatsAll
    );

    if (opts::VerifyCFG &&
        !std::accumulate(
           BFs.begin(), BFs.end(),
           true,
           [](const bool Valid,
              const std::pair<const uint64_t, BinaryFunction> &It) {
             return Valid && It.second.validateCFG();
           })) {
      errs() << "BOLT-ERROR: Invalid CFG detected after pass "
             << Pass->getName() << "\n";
      exit(1);
    }

    if (opts::Verbosity > 0) {
      outs() << "BOLT-INFO: Finished pass: " << Pass->getName() << "\n";
    }

    if (!opts::PrintAll && !opts::DumpDotAll && !Pass->printPass())
      continue;

    const std::string Message = std::string("after ") + Pass->getName();

    for (auto &It : BFs) {
      auto &Function = It.second;

      if (!Pass->shouldPrint(Function))
        continue;

      Function.print(outs(), Message, true);

      if (opts::DumpDotAll)
        Function.dumpGraphForPass(Pass->getName());
    }
  }
}

void BinaryFunctionPassManager::runAllPasses(BinaryContext &BC) {
  BinaryFunctionPassManager Manager(BC);

  const auto InitialDynoStats = getDynoStats(BC.getBinaryFunctions());

  if (opts::Instrument) {
    Manager.registerPass(llvm::make_unique<Instrumentation>(NeverPrint));
  }

  // Here we manage dependencies/order manually, since passes are run in the
  // order they're registered.

  // Run this pass first to use stats for the original functions.
  Manager.registerPass(llvm::make_unique<PrintProgramStats>(NeverPrint));

  if (opts::PrintProfileStats)
    Manager.registerPass(llvm::make_unique<PrintProfileStats>(NeverPrint));

  Manager.registerPass(llvm::make_unique<ValidateInternalCalls>(NeverPrint));

  Manager.registerPass(llvm::make_unique<StripRepRet>(NeverPrint),
                       opts::StripRepRet);

  Manager.registerPass(llvm::make_unique<IdenticalCodeFolding>(PrintICF),
                       opts::ICF);

  if (BC.isAArch64())
      Manager.registerPass(
          llvm::make_unique<VeneerElimination>(PrintVeneerElimination));

  Manager.registerPass(
      llvm::make_unique<SpecializeMemcpy1>(NeverPrint, opts::SpecializeMemcpy1),
      !opts::SpecializeMemcpy1.empty());

  Manager.registerPass(llvm::make_unique<InlineMemcpy>(NeverPrint),
                       opts::StringOps);

  Manager.registerPass(llvm::make_unique<IndirectCallPromotion>(PrintICP));

  Manager.registerPass(
      llvm::make_unique<JTFootprintReduction>(PrintJTFootprintReduction),
      opts::JTFootprintReductionFlag);

  Manager.registerPass(
    llvm::make_unique<SimplifyRODataLoads>(PrintSimplifyROLoads),
    opts::SimplifyRODataLoads);

  Manager.registerPass(llvm::make_unique<RegReAssign>(PrintRegReAssign),
                       opts::RegReAssign);

  Manager.registerPass(llvm::make_unique<Inliner>(PrintInline));

  Manager.registerPass(llvm::make_unique<IdenticalCodeFolding>(PrintICF),
                       opts::ICF);

  Manager.registerPass(llvm::make_unique<PLTCall>(PrintPLT));

  Manager.registerPass(llvm::make_unique<ReorderBasicBlocks>(PrintReordered));

  Manager.registerPass(
    llvm::make_unique<EliminateUnreachableBlocks>(PrintUCE),
    opts::EliminateUnreachable);

  Manager.registerPass(llvm::make_unique<SplitFunctions>(PrintSplit));

  // This pass syncs local branches with CFG. If any of the following
  // passes breaks the sync - they either need to re-run the pass or
  // fix branches consistency internally.
  Manager.registerPass(llvm::make_unique<FixupBranches>(PrintAfterBranchFixup));

  // This pass should come close to last since it uses the estimated hot
  // size of a function to determine the order.  It should definitely
  // also happen after any changes to the call graph are made, e.g. inlining.
  Manager.registerPass(
    llvm::make_unique<ReorderFunctions>(PrintReorderedFunctions));

  // Print final dyno stats right while CFG and instruction analysis are intact.
  Manager.registerPass(
    llvm::make_unique<DynoStatsPrintPass>(
      InitialDynoStats, "after all optimizations before SCTC and FOP"),
    opts::PrintDynoStats | opts::DynoStatsAll);

  // Add the StokeInfo pass, which extract functions for stoke optimization and
  // get the liveness information for them
  Manager.registerPass(llvm::make_unique<StokeInfo>(PrintStoke), opts::Stoke);

  // This pass introduces conditional jumps into external functions.
  // Between extending CFG to support this and isolating this pass we chose
  // the latter. Thus this pass will do double jump removal and unreachable
  // code elimination if necessary and won't rely on peepholes/UCE for these
  // optimizations.
  // More generally this pass should be the last optimization pass that
  // modifies branches/control flow.  This pass is run after function
  // reordering so that it can tell whether calls are forward/backward
  // accurately.
  Manager.registerPass(
      llvm::make_unique<SimplifyConditionalTailCalls>(PrintSCTC),
      opts::SimplifyConditionalTailCalls);

  Manager.registerPass(llvm::make_unique<Peepholes>(PrintPeepholes));

  Manager.registerPass(llvm::make_unique<AlignerPass>());

  // Perform reordering on data contained in one or more sections using
  // memory profiling data.
  Manager.registerPass(llvm::make_unique<ReorderData>());

  // This pass should always run last.*
  Manager.registerPass(llvm::make_unique<FinalizeFunctions>(PrintFinalized));

  // FrameOptimizer has an implicit dependency on FinalizeFunctions.
  // FrameOptimizer move values around and needs to update CFIs. To do this, it
  // must read CFI, interpret it and rewrite it, so CFIs need to be correctly
  // placed according to the final layout.
  Manager.registerPass(llvm::make_unique<FrameOptimizerPass>(PrintFOP));

  Manager.registerPass(llvm::make_unique<AllocCombinerPass>(PrintFOP));

  Manager.registerPass(
      llvm::make_unique<RetpolineInsertion>(PrintRetpolineInsertion));

  // Assign each function an output section.
  Manager.registerPass(llvm::make_unique<AssignSections>());

  // Patch original function entries
  Manager.registerPass(llvm::make_unique<PatchEntries>());

  // Tighten branches according to offset differences between branch and
  // targets. No extra instructions after this pass, otherwise we may have
  // relocations out of range and crash during linking.
  if (BC.isAArch64())
    Manager.registerPass(llvm::make_unique<LongJmpPass>(PrintLongJmp));

  // This pass turns tail calls into jumps which makes them invisible to
  // function reordering. It's unsafe to use any CFG or instruction analysis
  // after this point.
  Manager.registerPass(
    llvm::make_unique<InstructionLowering>(PrintAfterLowering));

  // In non-relocation mode, mark functions that do not fit into their original
  // space as non-simple if we have to (e.g. for correct debug info update).
  // NOTE: this pass depends on finalized code.
  if (!BC.HasRelocations)
    Manager.registerPass(llvm::make_unique<CheckLargeFunctions>(NeverPrint));

  Manager.registerPass(llvm::make_unique<LowerAnnotations>(NeverPrint));

  Manager.runPasses();
}

} // namespace bolt
} // namespace llvm
