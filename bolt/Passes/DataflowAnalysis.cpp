#include "DataflowAnalysis.h"

namespace llvm {

raw_ostream &operator<<(raw_ostream &OS, const BitVector &Val) {
  OS << "BitVector";
  return OS;
}

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

void RegStatePrinter::print(raw_ostream &OS, const BitVector &State) const {
  for (auto I = State.find_first(); I != -1; I = State.find_next(I)) {
    OS << BC.MRI->getName(I) << " ";
  }
}

} // namespace bolt
} // namespace llvm
