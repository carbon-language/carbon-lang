//===-- llvm/CodeGen/AsmPrinter/DbgValueHistoryCalculator.cpp -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DbgValueHistoryCalculator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include <algorithm>
#include <map>

#define DEBUG_TYPE "dwarfdebug"

namespace llvm {

namespace {
// Maps physreg numbers to the variables they describe.
typedef std::map<unsigned, SmallVector<const MDNode *, 1>> RegDescribedVarsMap;
}

// \brief If @MI is a DBG_VALUE with debug value described by a
// defined register, returns the number of this register.
// In the other case, returns 0.
static unsigned isDescribedByReg(const MachineInstr &MI) {
  assert(MI.isDebugValue());
  assert(MI.getNumOperands() == 3);
  // If location of variable is described using a register (directly or
  // indirecltly), this register is always a first operand.
  return MI.getOperand(0).isReg() ? MI.getOperand(0).getReg() : 0;
}

// \brief Claim that @Var is not described by @RegNo anymore.
static void dropRegDescribedVar(RegDescribedVarsMap &RegVars,
                                unsigned RegNo, const MDNode *Var) {
  const auto &I = RegVars.find(RegNo);
  assert(RegNo != 0U && I != RegVars.end());
  auto &VarSet = I->second;
  const auto &VarPos = std::find(VarSet.begin(), VarSet.end(), Var);
  assert(VarPos != VarSet.end());
  VarSet.erase(VarPos);
  // Don't keep empty sets in a map to keep it as small as possible.
  if (VarSet.empty())
    RegVars.erase(I);
}

// \brief Claim that @Var is now described by @RegNo.
static void addRegDescribedVar(RegDescribedVarsMap &RegVars,
                               unsigned RegNo, const MDNode *Var) {
  assert(RegNo != 0U);
  RegVars[RegNo].push_back(Var);
}

static void clobberVariableLocation(SmallVectorImpl<const MachineInstr *> &VarHistory,
                                    const MachineInstr &ClobberingInstr) {
  assert(!VarHistory.empty());
  // DBG_VALUE we're clobbering should belong to the same MBB.
  assert(VarHistory.back()->isDebugValue());
  assert(VarHistory.back()->getParent() == ClobberingInstr.getParent());
  VarHistory.push_back(&ClobberingInstr);
}

// \brief Terminate the location range for variables described by register
// @RegNo by inserting @ClobberingInstr to their history.
static void clobberRegisterUses(RegDescribedVarsMap &RegVars, unsigned RegNo,
                                DbgValueHistoryMap &HistMap,
                                const MachineInstr &ClobberingInstr) {
  const auto &I = RegVars.find(RegNo);
  if (I == RegVars.end())
    return;
  // Iterate over all variables described by this register and add this
  // instruction to their history, clobbering it.
  for (const auto &Var : I->second)
    clobberVariableLocation(HistMap[Var], ClobberingInstr);
  RegVars.erase(I);
}

// \brief Terminate the location range for all variables, described by registers
// clobbered by @MI.
static void clobberRegisterUses(RegDescribedVarsMap &RegVars,
                                const MachineInstr &MI,
                                const TargetRegisterInfo *TRI,
                                DbgValueHistoryMap &HistMap) {
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || !MO.isDef() || !MO.getReg())
      continue;
    for (MCRegAliasIterator AI(MO.getReg(), TRI, true); AI.isValid();
         ++AI) {
      unsigned RegNo = *AI;
      clobberRegisterUses(RegVars, RegNo, HistMap, MI);
    }
  }
}

// \brief Terminate the location range for all register-described variables
// by inserting @ClobberingInstr to their history.
static void clobberAllRegistersUses(RegDescribedVarsMap &RegVars,
                                    DbgValueHistoryMap &HistMap,
                                    const MachineInstr &ClobberingInstr) {
  for (const auto &I : RegVars)
    for (const auto &Var : I.second)
      clobberVariableLocation(HistMap[Var], ClobberingInstr);
  RegVars.clear();
}

// \brief Update the register that describes location of @Var in @RegVars map.
static void
updateRegForVariable(RegDescribedVarsMap &RegVars, const MDNode *Var,
                     const SmallVectorImpl<const MachineInstr *> &VarHistory,
                     const MachineInstr &MI) {
  if (!VarHistory.empty()) {
     const MachineInstr &Prev = *VarHistory.back();
     // Check if Var is currently described by a register by instruction in the
     // same basic block.
     if (Prev.isDebugValue() && Prev.getDebugVariable() == Var &&
         Prev.getParent() == MI.getParent()) {
       if (unsigned PrevReg = isDescribedByReg(Prev))
         dropRegDescribedVar(RegVars, PrevReg, Var);
     }
  }

  assert(MI.getDebugVariable() == Var);
  if (unsigned MIReg = isDescribedByReg(MI))
    addRegDescribedVar(RegVars, MIReg, Var);
}

void calculateDbgValueHistory(const MachineFunction *MF,
                              const TargetRegisterInfo *TRI,
                              DbgValueHistoryMap &Result) {
  RegDescribedVarsMap RegVars;

  for (const auto &MBB : *MF) {
    for (const auto &MI : MBB) {
      if (!MI.isDebugValue()) {
        // Not a DBG_VALUE instruction. It may clobber registers which describe
        // some variables.
        clobberRegisterUses(RegVars, MI, TRI, Result);
        continue;
      }

      const MDNode *Var = MI.getDebugVariable();
      auto &History = Result[Var];

      if (!History.empty() && History.back()->isIdenticalTo(&MI)) {
        DEBUG(dbgs() << "Coalescing identical DBG_VALUE entries:\n"
                     << "\t" << History.back() << "\t" << MI << "\n");
        continue;
      }

      updateRegForVariable(RegVars, Var, History, MI);
      History.push_back(&MI);
    }

    // Make sure locations for register-described variables are valid only
    // until the end of the basic block (unless it's the last basic block, in
    // which case let their liveness run off to the end of the function).
    if (!MBB.empty() &&  &MBB != &MF->back())
      clobberAllRegistersUses(RegVars, Result, MBB.back());
  }
}

}
