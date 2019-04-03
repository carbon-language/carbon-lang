#include "AllocCombiner.h"

#define DEBUG_TYPE "alloccombiner"

using namespace llvm;

namespace opts {
extern bool shouldProcess(const bolt::BinaryFunction &Function);

extern cl::opt<bolt::FrameOptimizationType> FrameOptimization;

} // end namespace opts

namespace llvm {
namespace bolt {

namespace {

bool getStackAdjustmentSize(const BinaryContext &BC, const MCInst &Inst,
                            int64_t &Adjustment) {
  return BC.MIB->evaluateSimple(Inst, Adjustment,
                                std::make_pair(BC.MIB->getStackPointer(), 0LL),
                                std::make_pair(0, 0LL));
}

bool isIndifferentToSP(const MCInst &Inst, const BinaryContext &BC) {
  if (BC.MIB->isCFI(Inst))
    return true;

  const auto II = BC.MII->get(Inst.getOpcode());
  if (BC.MIB->isTerminator(Inst) ||
      II.hasImplicitDefOfPhysReg(BC.MIB->getStackPointer(), BC.MRI.get()) ||
      II.hasImplicitUseOfPhysReg(BC.MIB->getStackPointer()))
    return false;

  for (int I = 0, E = MCPlus::getNumPrimeOperands(Inst); I != E; ++I) {
    const auto &Operand = Inst.getOperand(I);
    if (Operand.isReg() && Operand.getReg() == BC.MIB->getStackPointer()) {
      return false;
    }
  }
  return true;
}

bool shouldProc(BinaryFunction &Function) {
  return Function.isSimple() && Function.hasCFG() &&
         opts::shouldProcess(Function) && (Function.getSize() > 0);
}

void runForAllWeCare(std::map<uint64_t, BinaryFunction> &BFs,
                     std::function<void(BinaryFunction &)> Task) {
  for (auto &It : BFs) {
    auto &Function = It.second;
    if (shouldProc(Function))
      Task(Function);
  }
}

} // end anonymous namespace

void AllocCombinerPass::combineAdjustments(BinaryContext &BC,
                                           BinaryFunction &BF) {
  for (auto &BB : BF) {
    MCInst *Prev = nullptr;
    for (auto I = BB.rbegin(), E = BB.rend(); I != E; ++I) {
      auto &Inst = *I;
      if (isIndifferentToSP(Inst, BC))
        continue; // Skip updating Prev

      int64_t Adjustment{0LL};
      if (!Prev || !BC.MIB->isStackAdjustment(Inst) ||
          !BC.MIB->isStackAdjustment(*Prev) ||
          !getStackAdjustmentSize(BC, *Prev, Adjustment)) {
        Prev = &Inst;
        continue;
      }

      DEBUG({
        dbgs() << "At \"" << BF.getPrintName() << "\", combining: \n";
        Inst.dump();
        Prev->dump();
        dbgs() << "Adjustment: " << Adjustment << "\n";
      });

      if (BC.MIB->isSUB(Inst))
        Adjustment = -Adjustment;

      BC.MIB->addToImm(Inst, Adjustment, BC.Ctx.get());

      DEBUG({
        dbgs() << "After adjustment:\n";
        Inst.dump();
      });

      BB.eraseInstruction(BB.findInstruction(Prev));
      ++NumCombined;
      FuncsChanged.insert(&BF);
      Prev = &Inst;
    }
  }
}

void AllocCombinerPass::runOnFunctions(BinaryContext &BC) {
  if (opts::FrameOptimization == FOP_NONE)
    return;

  runForAllWeCare(
      BC.getBinaryFunctions(),
      [&](BinaryFunction &Function) { combineAdjustments(BC, Function); });

  outs() << "BOLT-INFO: Allocation combiner: " << NumCombined
         << " empty spaces coalesced.\n";
}

} // end namespace bolt
} // end namespace llvm
