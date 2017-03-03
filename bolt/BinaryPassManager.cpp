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
#include "Passes/FrameOptimizer.h"
#include "Passes/Inliner.h"
#include "llvm/Support/Timer.h"

using namespace llvm;

namespace opts {

extern cl::opt<bool> PrintAll;
extern cl::opt<bool> DumpDotAll;
extern cl::opt<bool> DynoStatsAll;

llvm::cl::opt<bool> TimeOpts("time-opts",
                             cl::desc("print time spent in each optimization"),
                             cl::init(false), cl::ZeroOrMore);

static cl::opt<bool>
EliminateUnreachable("eliminate-unreachable",
                     cl::desc("eliminate unreachable code"),
                     cl::init(true),
                     cl::ZeroOrMore);

static cl::opt<bool>
OptimizeBodylessFunctions(
    "optimize-bodyless-functions",
    cl::desc("optimize functions that just do a tail call"),
    cl::ZeroOrMore);

static cl::opt<bool>
IndirectCallPromotion("indirect-call-promotion",
                      cl::desc("indirect call promotion"),
                      cl::ZeroOrMore);

static cl::opt<bool>
InlineSmallFunctions(
    "inline-small-functions",
    cl::desc("inline functions with a single basic block"),
    cl::ZeroOrMore);

static cl::opt<bool>
SimplifyConditionalTailCalls("simplify-conditional-tail-calls",
                             cl::desc("simplify conditional tail calls "
                                      "by removing unnecessary jumps"),
                             cl::ZeroOrMore);

static cl::opt<bool>
Peepholes("peepholes",
          cl::desc("run peephole optimizations"),
          cl::ZeroOrMore);

static cl::opt<bool>
SimplifyRODataLoads("simplify-rodata-loads",
                    cl::desc("simplify loads from read-only sections by "
                             "replacing the memory operand with the "
                             "constant found in the corresponding "
                             "section"),
                    cl::ZeroOrMore);

static cl::opt<bool>
StripRepRet("strip-rep-ret",
    cl::desc("strip 'repz' prefix from 'repz retq' sequence (on by default)"),
    cl::init(true),
    cl::ZeroOrMore);

static cl::opt<bool> OptimizeFrameAccesses(
    "frame-opt", cl::desc("optimize stack frame accesses"), cl::ZeroOrMore);

static cl::opt<bool>
PrintReordered("print-reordered",
               cl::desc("print functions after layout optimization"),
               cl::ZeroOrMore,
               cl::Hidden);

static cl::opt<bool>
PrintReorderedFunctions("print-reordered-functions",
               cl::desc("print functions after clustering"),
               cl::ZeroOrMore,
               cl::Hidden);

static cl::opt<bool>
PrintOptimizeBodyless("print-optimize-bodyless",
            cl::desc("print functions after bodyless optimization"),
            cl::ZeroOrMore,
            cl::Hidden);

static cl::opt<bool>
PrintAfterBranchFixup("print-after-branch-fixup",
                      cl::desc("print function after fixing local branches"),
                      cl::Hidden);

static cl::opt<bool>
PrintFinalized("print-finalized",
                cl::desc("print function after CFG is finalized"),
                cl::Hidden);

static cl::opt<bool>
PrintAfterLowering("print-after-lowering",
    cl::desc("print function after instruction lowering"),
    cl::Hidden);

static cl::opt<bool>
PrintUCE("print-uce",
         cl::desc("print functions after unreachable code elimination"),
         cl::ZeroOrMore,
         cl::Hidden);

static cl::opt<bool>
PrintSCTC("print-sctc",
         cl::desc("print functions after conditional tail call simplification"),
         cl::ZeroOrMore,
         cl::Hidden);

static cl::opt<bool>
PrintPeepholes("print-peepholes",
               cl::desc("print functions after peephole optimization"),
               cl::ZeroOrMore,
               cl::Hidden);

static cl::opt<bool>
PrintSimplifyROLoads("print-simplify-rodata-loads",
                     cl::desc("print functions after simplification of RO data"
                              " loads"),
                     cl::ZeroOrMore,
                     cl::Hidden);

static cl::opt<bool>
PrintICF("print-icf",
         cl::desc("print functions after ICF optimization"),
         cl::ZeroOrMore,
         cl::Hidden);

static cl::opt<bool>
PrintICP("print-icp",
         cl::desc("print functions after indirect call promotion"),
         cl::ZeroOrMore,
         cl::Hidden);

static cl::opt<bool>
PrintInline("print-inline",
            cl::desc("print functions after inlining optimization"),
            cl::ZeroOrMore,
            cl::Hidden);

static cl::opt<bool>
PrintFOP("print-fop",
         cl::desc("print functions after frame optimizer pass"),
         cl::ZeroOrMore,
         cl::Hidden);

static cl::opt<bool>
NeverPrint("never-print",
           cl::desc("never print"),
           cl::init(false),
           cl::ZeroOrMore,
           cl::ReallyHidden);

} // namespace opts

namespace llvm {
namespace bolt {

using namespace opts;

const char BinaryFunctionPassManager::TimerGroupName[] =
    "Binary Function Pass Manager";

cl::opt<bool> BinaryFunctionPassManager::AlwaysOn(
  "always-run-pass",
  cl::desc("Used for passes that are always enabled"),
  cl::init(true),
  cl::ReallyHidden);

void BinaryFunctionPassManager::runPasses() {
  for (const auto &OptPassPair : Passes) {
    if (!OptPassPair.first)
      continue;

    auto &Pass = OptPassPair.second;

    NamedRegionTimer T(Pass->getName(), TimerGroupName, TimeOpts);

    callWithDynoStats(
      [this,&Pass] {
        Pass->runOnFunctions(BC, BFs, LargeFunctions);
      },
      BFs,
      Pass->getName(),
      opts::DynoStatsAll
    );

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

void BinaryFunctionPassManager::runAllPasses(
  BinaryContext &BC,
  std::map<uint64_t, BinaryFunction> &Functions,
  std::set<uint64_t> &LargeFunctions
) {
  BinaryFunctionPassManager Manager(BC, Functions, LargeFunctions);

  // Here we manage dependencies/order manually, since passes are run in the
  // order they're registered.

  // Run this pass first to use stats for the original functions.
  Manager.registerPass(llvm::make_unique<PrintSortedBy>(NeverPrint));

  Manager.registerPass(llvm::make_unique<StripRepRet>(NeverPrint),
                       opts::StripRepRet);

  Manager.registerPass(llvm::make_unique<IdenticalCodeFolding>(PrintICF));

  Manager.registerPass(llvm::make_unique<IndirectCallPromotion>(PrintICP),
                       opts::IndirectCallPromotion);

  Manager.registerPass(llvm::make_unique<Peepholes>(PrintPeepholes),
                       opts::Peepholes);

  Manager.registerPass(llvm::make_unique<InlineSmallFunctions>(PrintInline),
                       opts::InlineSmallFunctions);

  Manager.registerPass(
    llvm::make_unique<OptimizeBodylessFunctions>(PrintOptimizeBodyless),
    opts::OptimizeBodylessFunctions);

  Manager.registerPass(
    llvm::make_unique<SimplifyRODataLoads>(PrintSimplifyROLoads),
    opts::SimplifyRODataLoads);

  Manager.registerPass(llvm::make_unique<IdenticalCodeFolding>(PrintICF));

  Manager.registerPass(llvm::make_unique<ReorderBasicBlocks>(PrintReordered));

  Manager.registerPass(llvm::make_unique<Peepholes>(PrintPeepholes),
                       opts::Peepholes);

  Manager.registerPass(
    llvm::make_unique<EliminateUnreachableBlocks>(PrintUCE),
    opts::EliminateUnreachable);

  // This pass syncs local branches with CFG. If any of the following
  // passes breaks the sync - they either need to re-run the pass or
  // fix branches consistency internally.
  Manager.registerPass(llvm::make_unique<FixupBranches>(PrintAfterBranchFixup));

  Manager.registerPass(llvm::make_unique<FrameOptimizerPass>(PrintFOP),
                       OptimizeFrameAccesses);

  // This pass introduces conditional jumps into external functions.
  // Between extending CFG to support this and isolating this pass we chose
  // the latter. Thus this pass will do unreachable code elimination
  // if necessary and wouldn't rely on UCE for this.
  // More generally this pass should be the last optimization pass.
  Manager.registerPass(
    llvm::make_unique<SimplifyConditionalTailCalls>(PrintSCTC),
    opts::SimplifyConditionalTailCalls);

  Manager.registerPass(llvm::make_unique<Peepholes>(PrintPeepholes),
                       opts::Peepholes);

  Manager.registerPass(
    llvm::make_unique<EliminateUnreachableBlocks>(PrintUCE),
    opts::EliminateUnreachable);

  Manager.registerPass(
    llvm::make_unique<ReorderFunctions>(PrintReorderedFunctions));

  Manager.registerPass(llvm::make_unique<FinalizeFunctions>(PrintFinalized));

  Manager.registerPass(
    llvm::make_unique<InstructionLowering>(PrintAfterLowering));

  Manager.runPasses();
}

} // namespace bolt
} // namespace llvm
