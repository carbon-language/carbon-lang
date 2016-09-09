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

using namespace llvm;

namespace opts {

extern llvm::cl::opt<bool> PrintAll;
extern llvm::cl::opt<bool> DumpDotAll;
extern llvm::cl::opt<bool> DynoStatsAll;

static cl::opt<bool>
EliminateUnreachable("eliminate-unreachable",
                     cl::desc("eliminate unreachable code"),
                     cl::ZeroOrMore);

static cl::opt<bool>
OptimizeBodylessFunctions(
    "optimize-bodyless-functions",
    cl::desc("optimize functions that just do a tail call"),
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
IdenticalCodeFolding(
    "icf",
    cl::desc("fold functions with identical code"),
    cl::ZeroOrMore);

static cl::opt<bool>
PrintReordered("print-reordered",
               cl::desc("print functions after layout optimization"),
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
PrintAfterFixup("print-after-fixup",
                cl::desc("print function after fixup"),
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
PrintInline("print-inline",
            cl::desc("print functions after inlining optimization"),
            cl::ZeroOrMore,
            cl::Hidden);

} // namespace opts

namespace llvm {
namespace bolt {

using namespace opts;

cl::opt<bool> BinaryFunctionPassManager::AlwaysOn(
  "always-run-pass",
  cl::desc("Used for passes that are always enabled"),
  cl::init(true),
  cl::ReallyHidden);

bool BinaryFunctionPassManager::NagUser = false;

void BinaryFunctionPassManager::runPasses() {
  for (const auto &OptPassPair : Passes) {
    if (!OptPassPair.first)
      continue;

    auto &Pass = OptPassPair.second;

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

  // Here we manage dependencies/order manually, since passes are ran in the
  // order they're registered.

  Manager.registerPass(llvm::make_unique<IdenticalCodeFolding>(PrintICF),
                       opts::IdenticalCodeFolding);

  Manager.registerPass(llvm::make_unique<InlineSmallFunctions>(PrintInline),
                       opts::InlineSmallFunctions);

  Manager.registerPass(
    llvm::make_unique<EliminateUnreachableBlocks>(PrintUCE, Manager.NagUser),
    opts::EliminateUnreachable);

  Manager.registerPass(
    llvm::make_unique<OptimizeBodylessFunctions>(PrintOptimizeBodyless),
    opts::OptimizeBodylessFunctions);

  Manager.registerPass(
    llvm::make_unique<SimplifyRODataLoads>(PrintSimplifyROLoads),
    opts::SimplifyRODataLoads);

  Manager.registerPass(
    llvm::make_unique<EliminateUnreachableBlocks>(PrintUCE, Manager.NagUser),
    opts::EliminateUnreachable);

  Manager.registerPass(llvm::make_unique<ReorderBasicBlocks>(PrintReordered));

  Manager.registerPass(llvm::make_unique<Peepholes>(PrintPeepholes),
                       opts::Peepholes);

  // This pass syncs local branches with CFG. If any of the following
  // passes breaks the sync - they either need to re-run the pass or
  // fix branches consistency internally.
  Manager.registerPass(llvm::make_unique<FixupBranches>(PrintAfterBranchFixup));

  // This pass introduces conditional jumps into external functions.
  // Between extending CFG to support this and isolating this pass we chose
  // the latter. Thus this pass will do unreachable code elimination
  // if necessary and wouldn't rely on UCE for this.
  // More generally this pass should be the last optimization pass.
  Manager.registerPass(
    llvm::make_unique<SimplifyConditionalTailCalls>(PrintSCTC),
    opts::SimplifyConditionalTailCalls);

  Manager.registerPass(llvm::make_unique<FixupFunctions>(PrintAfterFixup));

  Manager.runPasses();
}

} // namespace bolt
} // namespace llvm
