//===-- llvm/CodeGen/AsmPrinter/DbgValueHistoryCalculator.cpp -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DbgValueHistoryCalculator.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetRegisterInfo.h"

#define DEBUG_TYPE "dwarfdebug"

namespace llvm {

// Return true if debug value, encoded by DBG_VALUE instruction, is in a
// defined reg.
static bool isDbgValueInDefinedReg(const MachineInstr *MI) {
  assert(MI->isDebugValue() && "Invalid DBG_VALUE machine instruction!");
  return MI->getNumOperands() == 3 && MI->getOperand(0).isReg() &&
         MI->getOperand(0).getReg() &&
         (MI->getOperand(1).isImm() ||
          (MI->getOperand(1).isReg() && MI->getOperand(1).getReg() == 0U));
}

void calculateDbgValueHistory(const MachineFunction *MF,
                              const TargetRegisterInfo *TRI,
                              DbgValueHistoryMap &Result) {
  // LiveUserVar - Map physreg numbers to the MDNode they contain.
  std::vector<const MDNode *> LiveUserVar(TRI->getNumRegs());

  for (MachineFunction::const_iterator I = MF->begin(), E = MF->end(); I != E;
       ++I) {
    bool AtBlockEntry = true;
    for (MachineBasicBlock::const_iterator II = I->begin(), IE = I->end();
         II != IE; ++II) {
      const MachineInstr *MI = II;

      if (MI->isDebugValue()) {
        assert(MI->getNumOperands() > 1 && "Invalid machine instruction!");

        // Keep track of user variables.
        const MDNode *Var = MI->getDebugVariable();

        // Variable is in a register, we need to check for clobbers.
        if (isDbgValueInDefinedReg(MI))
          LiveUserVar[MI->getOperand(0).getReg()] = Var;

        // Check the history of this variable.
        SmallVectorImpl<const MachineInstr *> &History = Result[Var];
        if (!History.empty()) {
          // We have seen this variable before. Try to coalesce DBG_VALUEs.
          const MachineInstr *Prev = History.back();
          if (Prev->isDebugValue()) {
            // Coalesce identical entries at the end of History.
            if (History.size() >= 2 &&
                Prev->isIdenticalTo(History[History.size() - 2])) {
              DEBUG(dbgs() << "Coalescing identical DBG_VALUE entries:\n"
                           << "\t" << *Prev << "\t"
                           << *History[History.size() - 2] << "\n");
              History.pop_back();
            }

            // Terminate old register assignments that don't reach MI;
            MachineFunction::const_iterator PrevMBB = Prev->getParent();
            if (PrevMBB != I && (!AtBlockEntry || std::next(PrevMBB) != I) &&
                isDbgValueInDefinedReg(Prev)) {
              // Previous register assignment needs to terminate at the end of
              // its basic block.
              MachineBasicBlock::const_iterator LastMI =
                  PrevMBB->getLastNonDebugInstr();
              if (LastMI == PrevMBB->end()) {
                // Drop DBG_VALUE for empty range.
                DEBUG(dbgs() << "Dropping DBG_VALUE for empty range:\n"
                             << "\t" << *Prev << "\n");
                History.pop_back();
              } else if (std::next(PrevMBB) != PrevMBB->getParent()->end())
                // Terminate after LastMI.
                History.push_back(LastMI);
            }
          }
        }
        History.push_back(MI);
      } else {
        // Not a DBG_VALUE instruction.
        if (!MI->isPosition())
          AtBlockEntry = false;

        // Check if the instruction clobbers any registers with debug vars.
        for (const MachineOperand &MO : MI->operands()) {
          if (!MO.isReg() || !MO.isDef() || !MO.getReg())
            continue;
          for (MCRegAliasIterator AI(MO.getReg(), TRI, true); AI.isValid();
               ++AI) {
            unsigned Reg = *AI;
            const MDNode *Var = LiveUserVar[Reg];
            if (!Var)
              continue;
            // Reg is now clobbered.
            LiveUserVar[Reg] = nullptr;

            // Was MD last defined by a DBG_VALUE referring to Reg?
            auto HistI = Result.find(Var);
            if (HistI == Result.end())
              continue;
            SmallVectorImpl<const MachineInstr *> &History = HistI->second;
            if (History.empty())
              continue;
            const MachineInstr *Prev = History.back();
            // Sanity-check: Register assignments are terminated at the end of
            // their block.
            if (!Prev->isDebugValue() || Prev->getParent() != MI->getParent())
              continue;
            // Is the variable still in Reg?
            if (!isDbgValueInDefinedReg(Prev) ||
                Prev->getOperand(0).getReg() != Reg)
              continue;
            // Var is clobbered. Make sure the next instruction gets a label.
            History.push_back(MI);
          }
        }
      }
    }
  }

  // Make sure the final register assignments are terminated.
  for (auto &I : Result) {
    SmallVectorImpl<const MachineInstr *> &History = I.second;
    if (History.empty())
      continue;

    const MachineInstr *Prev = History.back();
    if (Prev->isDebugValue() && isDbgValueInDefinedReg(Prev)) {
      const MachineBasicBlock *PrevMBB = Prev->getParent();
      MachineBasicBlock::const_iterator LastMI =
          PrevMBB->getLastNonDebugInstr();
      if (LastMI == PrevMBB->end())
        // Drop DBG_VALUE for empty range.
        History.pop_back();
      else if (PrevMBB != &PrevMBB->getParent()->back()) {
        // Terminate after LastMI.
        History.push_back(LastMI);
      }
    }
  }
}

}
