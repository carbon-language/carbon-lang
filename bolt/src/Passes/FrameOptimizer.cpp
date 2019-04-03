//===--- Passes/FrameOptimizer.cpp ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "FrameOptimizer.h"
#include "ShrinkWrapping.h"
#include "StackAvailableExpressions.h"
#include "StackReachingUses.h"
#include "llvm/Support/Timer.h"
#include <queue>
#include <unordered_map>

#define DEBUG_TYPE "fop"

using namespace llvm;

namespace opts {
extern cl::opt<unsigned> Verbosity;
extern cl::opt<bool> TimeOpts;
extern cl::OptionCategory BoltOptCategory;

using namespace bolt;

cl::opt<FrameOptimizationType>
FrameOptimization("frame-opt",
  cl::init(FOP_NONE),
  cl::desc("optimize stack frame accesses"),
  cl::values(
    clEnumValN(FOP_NONE, "none", "do not perform frame optimization"),
    clEnumValN(FOP_HOT, "hot", "perform FOP on hot functions"),
    clEnumValN(FOP_ALL, "all", "perform FOP on all functions")),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

cl::opt<bool>
RemoveStores("frame-opt-rm-stores",
  cl::init(FOP_NONE),
  cl::desc("apply additional analysis to remove stores (experimental)"),
  cl::init(false),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));


} // namespace opts

namespace llvm {
namespace bolt {

void FrameOptimizerPass::removeUnnecessaryLoads(const RegAnalysis &RA,
                                                const FrameAnalysis &FA,
                                                const BinaryContext &BC,
                                                BinaryFunction &BF) {
  StackAvailableExpressions SAE(RA, FA, BC, BF);
  SAE.run();

  DEBUG(dbgs() << "Performing unnecessary loads removal\n");
  std::deque<std::pair<BinaryBasicBlock *, MCInst *>> ToErase;
  bool Changed = false;
  const auto ExprEnd = SAE.expr_end();
  for (auto &BB : BF) {
    DEBUG(dbgs() <<"\tNow at BB " << BB.getName() << "\n");
    const MCInst *Prev = nullptr;
    for (auto &Inst : BB) {
      DEBUG({
        dbgs() << "\t\tNow at ";
        Inst.dump();
        for (auto I = Prev ? SAE.expr_begin(*Prev) : SAE.expr_begin(BB);
             I != ExprEnd; ++I) {
          dbgs() << "\t\t\tReached by: ";
          (*I)->dump();
        }
      });
      // if Inst is a load from stack and the current available expressions show
      // this value is available in a register or immediate, replace this load
      // with move from register or from immediate.
      auto FIEX = FA.getFIEFor(Inst);
      if (!FIEX) {
        Prev = &Inst;
        continue;
      }
      // FIXME: Change to remove IsSimple == 0. We're being conservative here,
      // but once replaceMemOperandWithReg is ready, we should feed it with all
      // sorts of complex instructions.
      if (FIEX->IsLoad == false || FIEX->IsSimple == false ||
          FIEX->StackOffset >= 0) {
        Prev = &Inst;
        continue;
      }

      for (auto I = Prev ? SAE.expr_begin(*Prev) : SAE.expr_begin(BB);
           I != ExprEnd; ++I) {
        const MCInst *AvailableInst = *I;
        auto FIEY = FA.getFIEFor(*AvailableInst);
        if (!FIEY)
          continue;
        assert(FIEY->IsStore && FIEY->IsSimple);
        if (FIEX->StackOffset != FIEY->StackOffset || FIEX->Size != FIEY->Size)
          continue;
        // TODO: Change push/pops to stack adjustment instruction
        if (BC.MIB->isPop(Inst))
          continue;

        ++NumRedundantLoads;
        Changed = true;
        DEBUG(dbgs() << "Redundant load instruction: ");
        DEBUG(Inst.dump());
        DEBUG(dbgs() << "Related store instruction: ");
        DEBUG(AvailableInst->dump());
        DEBUG(dbgs() << "@BB: " << BB.getName() << "\n");
        // Replace load
        if (FIEY->IsStoreFromReg) {
          if (!BC.MIB->replaceMemOperandWithReg(Inst, FIEY->RegOrImm)) {
            DEBUG(dbgs() << "FAILED to change operand to a reg\n");
            break;
          }
          ++NumLoadsChangedToReg;
          BC.MIB->removeAnnotation(Inst, "FrameAccessEntry");
          DEBUG(dbgs() << "Changed operand to a reg\n");
          if (BC.MIB->isRedundantMove(Inst)) {
            ++NumLoadsDeleted;
            DEBUG(dbgs() << "Created a redundant move\n");
            // Delete it!
            ToErase.push_front(std::make_pair(&BB, &Inst));
          }
        } else {
          char Buf[8] = {0, 0, 0, 0, 0, 0, 0, 0};
          support::ulittle64_t::ref(Buf + 0) = FIEY->RegOrImm;
          DEBUG(dbgs() << "Changing operand to an imm... ");
          if (!BC.MIB->replaceMemOperandWithImm(Inst, StringRef(Buf, 8), 0)) {
            DEBUG(dbgs() << "FAILED\n");
          } else {
            ++NumLoadsChangedToImm;
            BC.MIB->removeAnnotation(Inst, "FrameAccessEntry");
            DEBUG(dbgs() << "Ok\n");
          }
        }
        DEBUG(dbgs() << "Changed to: ");
        DEBUG(Inst.dump());
        break;
      }
      Prev = &Inst;
    }
  }
  if (Changed) {
    DEBUG(dbgs() << "FOP modified \"" << BF.getPrintName() << "\"\n");
  }
  // TODO: Implement an interface of eraseInstruction that works out the
  // complete list of elements to remove.
  for (auto I : ToErase) {
    I.first->eraseInstruction(I.first->findInstruction(I.second));
  }
}

void FrameOptimizerPass::removeUnusedStores(const FrameAnalysis &FA,
                                            const BinaryContext &BC,
                                            BinaryFunction &BF) {
  StackReachingUses SRU(FA, BC, BF);
  SRU.run();

  DEBUG(dbgs() << "Performing unused stores removal\n");
  std::vector<std::pair<BinaryBasicBlock *, MCInst *>> ToErase;
  bool Changed = false;
  for (auto &BB : BF) {
    DEBUG(dbgs() <<"\tNow at BB " << BB.getName() << "\n");
    const MCInst *Prev = nullptr;
    for (auto I = BB.rbegin(), E = BB.rend(); I != E; ++I) {
      auto &Inst = *I;
      DEBUG({
        dbgs() << "\t\tNow at ";
        Inst.dump();
        for (auto I = Prev ? SRU.expr_begin(*Prev) : SRU.expr_begin(BB);
             I != SRU.expr_end(); ++I) {
          dbgs() << "\t\t\tReached by: ";
          (*I)->dump();
        }
      });
      auto FIEX = FA.getFIEFor(Inst);
      if (!FIEX) {
        Prev = &Inst;
        continue;
      }
      if (FIEX->IsLoad || !FIEX->IsSimple || FIEX->StackOffset >= 0) {
        Prev = &Inst;
        continue;
      }

      if (SRU.isStoreUsed(*FIEX,
                          Prev ? SRU.expr_begin(*Prev) : SRU.expr_begin(BB))) {
        Prev = &Inst;
        continue;
      }
      // TODO: Change push/pops to stack adjustment instruction
      if (BC.MIB->isPush(Inst))
        continue;

      ++NumRedundantStores;
      Changed = true;
      DEBUG(dbgs() << "Unused store instruction: ");
      DEBUG(Inst.dump());
      DEBUG(dbgs() << "@BB: " << BB.getName() << "\n");
      DEBUG(dbgs() << "FIE offset = " << FIEX->StackOffset
                   << " size = " << (int)FIEX->Size << "\n");
      // Delete it!
      ToErase.push_back(std::make_pair(&BB, &Inst));
      Prev = &Inst;
    }
  }

  for (auto I : ToErase) {
    I.first->eraseInstruction(I.first->findInstruction(I.second));
  }
  if (Changed) {
    DEBUG(dbgs() << "FOP modified \"" << BF.getPrintName() << "\"\n");
  }
}

void FrameOptimizerPass::runOnFunctions(BinaryContext &BC) {
  if (opts::FrameOptimization == FOP_NONE)
    return;

  // Run FrameAnalysis pass
  BinaryFunctionCallGraph CG = buildCallGraph(BC);
  FrameAnalysis FA(BC, CG);
  RegAnalysis RA(BC, &BC.getBinaryFunctions(), &CG);

  // Our main loop: perform caller-saved register optimizations, then
  // callee-saved register optimizations (shrink wrapping).
  for (auto &I : BC.getBinaryFunctions()) {
    if (!FA.hasFrameInfo(I.second))
      continue;
    // Restrict pass execution if user asked to only run on hot functions
    if (opts::FrameOptimization == FOP_HOT) {
      if (I.second.getKnownExecutionCount() < BC.getHotThreshold())
        continue;
      DEBUG(dbgs() << "Considering " << I.second.getPrintName()
                   << " for frame optimizations because its execution count ( "
                   << I.second.getKnownExecutionCount()
                   << " ) exceeds our hotness threshold ( "
                   << BC.getHotThreshold() << " )\n");
    }
    {
      NamedRegionTimer T1("removeloads", "remove loads", "FOP", "FOP breakdown",
                          opts::TimeOpts);
      removeUnnecessaryLoads(RA, FA, BC, I.second);
    }
    if (opts::RemoveStores) {
      NamedRegionTimer T1("removestores", "remove stores", "FOP",
                          "FOP breakdown", opts::TimeOpts);
      removeUnusedStores(FA, BC, I.second);
    }
    // Don't even start shrink wrapping if no profiling info is available
    if (I.second.getKnownExecutionCount() == 0)
      continue;
    {
      NamedRegionTimer T1("movespills", "move spills", "FOP", "FOP breakdown",
                          opts::TimeOpts);
      DataflowInfoManager Info(BC, I.second, &RA, &FA);
      ShrinkWrapping SW(FA, BC, I.second, Info);
      if (SW.perform())
        FuncsChanged.insert(&I.second);
    }
  }

  outs() << "BOLT-INFO: FOP optimized " << NumRedundantLoads
         << " redundant load(s) and " << NumRedundantStores
         << " unused store(s)\n";
  outs() << "BOLT-INFO: FOP changed " << NumLoadsChangedToReg
         << " load(s) to use a register instead of a stack access, and "
         << NumLoadsChangedToImm << " to use an immediate.\n"
         << "BOLT-INFO: FOP deleted " << NumLoadsDeleted << " load(s) and "
         << NumRedundantStores << " store(s).\n";
  FA.printStats();
  ShrinkWrapping::printStats();
}

} // namespace bolt
} // namespace llvm
