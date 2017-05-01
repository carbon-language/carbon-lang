#include "DataflowAnalysis.h"

namespace llvm {
namespace bolt {

void doForAllPreds(const BinaryContext &BC, const BinaryBasicBlock &BB,
                   std::function<void(ProgramPoint)> Task) {
  for (auto Pred : BB.predecessors()) {
    if (Pred->isValid())
      Task(ProgramPoint::getLastPointAt(*Pred));
  }
  if (!BB.isLandingPad())
    return;
  for (auto Thrower : BB.throwers()) {
    for (auto &Inst : *Thrower) {
      if (!BC.MIA->isInvoke(Inst) ||
          BC.MIA->getEHInfo(Inst).first != BB.getLabel())
        continue;
      Task(ProgramPoint(&Inst));
    }
  }
}

/// Operates on all successors of a basic block.
void doForAllSuccs(const BinaryBasicBlock &BB,
                   std::function<void(ProgramPoint)> Task) {
  for (auto Succ : BB.successors()) {
    if (Succ->isValid())
      Task(ProgramPoint::getFirstPointAt(*Succ));
  }
}

} // namespace bolt
} // namespace llvm

llvm::raw_ostream &llvm::operator<<(llvm::raw_ostream &OS,
                                    const BitVector &Val) {
  OS << "BitVector";
  return OS;
}
