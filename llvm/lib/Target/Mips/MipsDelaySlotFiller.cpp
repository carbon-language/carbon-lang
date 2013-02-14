//===-- MipsDelaySlotFiller.cpp - Mips Delay Slot Filler ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Simple pass to fill delay slots with useful instructions.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "delay-slot-filler"

#include "Mips.h"
#include "MipsTargetMachine.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"

using namespace llvm;

STATISTIC(FilledSlots, "Number of delay slots filled");
STATISTIC(UsefulSlots, "Number of delay slots filled with instructions that"
                       " are not NOP.");

static cl::opt<bool> DisableDelaySlotFiller(
  "disable-mips-delay-filler",
  cl::init(false),
  cl::desc("Fill all delay slots with NOPs."),
  cl::Hidden);

// This option can be used to silence complaints by machine verifier passes.
static cl::opt<bool> SkipDelaySlotFiller(
  "skip-mips-delay-filler",
  cl::init(false),
  cl::desc("Skip MIPS' delay slot filling pass."),
  cl::Hidden);

namespace {
  class Filler : public MachineFunctionPass {
  public:
    Filler(TargetMachine &tm)
      : MachineFunctionPass(ID), TM(tm), TII(tm.getInstrInfo()) { }

    virtual const char *getPassName() const {
      return "Mips Delay Slot Filler";
    }

    bool runOnMachineFunction(MachineFunction &F) {
      if (SkipDelaySlotFiller)
        return false;

      bool Changed = false;
      for (MachineFunction::iterator FI = F.begin(), FE = F.end();
           FI != FE; ++FI)
        Changed |= runOnMachineBasicBlock(*FI);
      return Changed;
    }

  private:
    typedef MachineBasicBlock::iterator Iter;
    typedef MachineBasicBlock::reverse_iterator ReverseIter;

    bool runOnMachineBasicBlock(MachineBasicBlock &MBB);

    void insertDefsUses(const MachineInstr &MI, SmallSet<unsigned, 32> &RegDefs,
                        SmallSet<unsigned, 32> &RegUses) const;

    bool isRegInSet(const SmallSet<unsigned, 32> &RegSet, unsigned Reg) const;

    bool delayHasHazard(const MachineInstr &Candidate, bool &SawLoad,
                        bool &SawStore, const SmallSet<unsigned, 32> &RegDefs,
                        const SmallSet<unsigned, 32> &RegUses) const;

    bool findDelayInstr(MachineBasicBlock &MBB, Iter slot, Iter &Filler) const;

    bool terminateSearch(const MachineInstr &Candidate) const;

    TargetMachine &TM;
    const TargetInstrInfo *TII;

    static char ID;
  };
  char Filler::ID = 0;
} // end of anonymous namespace

/// runOnMachineBasicBlock - Fill in delay slots for the given basic block.
/// We assume there is only one delay slot per delayed instruction.
bool Filler::runOnMachineBasicBlock(MachineBasicBlock &MBB) {
  bool Changed = false;

  for (Iter I = MBB.begin(); I != MBB.end(); ++I) {
    if (!I->hasDelaySlot())
      continue;

    ++FilledSlots;
    Changed = true;
    Iter D;

    // Delay slot filling is disabled at -O0.
    if (!DisableDelaySlotFiller && (TM.getOptLevel() != CodeGenOpt::None) &&
        findDelayInstr(MBB, I, D)) {
      MBB.splice(llvm::next(I), &MBB, D);
      ++UsefulSlots;
    } else
      BuildMI(MBB, llvm::next(I), I->getDebugLoc(), TII->get(Mips::NOP));

    // Bundle the delay slot filler to the instruction with the delay slot.
    MIBundleBuilder(MBB, I, llvm::next(llvm::next(I)));
  }

  return Changed;
}

/// createMipsDelaySlotFillerPass - Returns a pass that fills in delay
/// slots in Mips MachineFunctions
FunctionPass *llvm::createMipsDelaySlotFillerPass(MipsTargetMachine &tm) {
  return new Filler(tm);
}

bool Filler::findDelayInstr(MachineBasicBlock &MBB, Iter Slot,
                            Iter &Filler) const {
  SmallSet<unsigned, 32> RegDefs;
  SmallSet<unsigned, 32> RegUses;

  insertDefsUses(*Slot, RegDefs, RegUses);

  bool SawLoad = false;
  bool SawStore = false;

  for (ReverseIter I(Slot); I != MBB.rend(); ++I) {
    // skip debug value
    if (I->isDebugValue())
      continue;

    if (terminateSearch(*I))
      break;

    if (delayHasHazard(*I, SawLoad, SawStore, RegDefs, RegUses)) {
      insertDefsUses(*I, RegDefs, RegUses);
      continue;
    }

    Filler = llvm::next(I).base();
    return true;
  }

  return false;
}

bool Filler::delayHasHazard(const MachineInstr &Candidate, bool &SawLoad,
                            bool &SawStore,
                            const SmallSet<unsigned, 32> &RegDefs,
                            const SmallSet<unsigned, 32> &RegUses) const {
  if (Candidate.isImplicitDef() || Candidate.isKill())
    return true;

  // Loads or stores cannot be moved past a store to the delay slot
  // and stores cannot be moved past a load.
  if (Candidate.mayLoad()) {
    if (SawStore)
      return true;
    SawLoad = true;
  }

  if (Candidate.mayStore()) {
    if (SawStore)
      return true;
    SawStore = true;
    if (SawLoad)
      return true;
  }

  assert((!Candidate.isCall() && !Candidate.isReturn()) &&
         "Cannot put calls or returns in delay slot.");

  for (unsigned I = 0, E = Candidate.getNumOperands(); I != E; ++I) {
    const MachineOperand &MO = Candidate.getOperand(I);
    unsigned Reg;

    if (!MO.isReg() || !(Reg = MO.getReg()))
      continue; // skip

    if (MO.isDef()) {
      // check whether Reg is defined or used before delay slot.
      if (isRegInSet(RegDefs, Reg) || isRegInSet(RegUses, Reg))
        return true;
    }
    if (MO.isUse()) {
      // check whether Reg is defined before delay slot.
      if (isRegInSet(RegDefs, Reg))
        return true;
    }
  }
  return false;
}

// Helper function for getting a MachineOperand's register number and adding it
// to RegDefs or RegUses.
static void insertDefUse(const MachineOperand &MO,
                         SmallSet<unsigned, 32> &RegDefs,
                         SmallSet<unsigned, 32> &RegUses,
                         unsigned ExcludedReg = 0) {
  unsigned Reg;

  if (!MO.isReg() || !(Reg = MO.getReg()) || (Reg == ExcludedReg))
    return;

  if (MO.isDef())
    RegDefs.insert(Reg);
  else if (MO.isUse())
    RegUses.insert(Reg);
}

// Insert Defs and Uses of MI into the sets RegDefs and RegUses.
void Filler::insertDefsUses(const MachineInstr &MI,
                            SmallSet<unsigned, 32> &RegDefs,
                            SmallSet<unsigned, 32> &RegUses) const {
  unsigned I, E = MI.getDesc().getNumOperands();

  for (I = 0; I != E; ++I)
    insertDefUse(MI.getOperand(I), RegDefs, RegUses);

  // If MI is a call, add RA to RegDefs to prevent users of RA from going into
  // delay slot.
  if (MI.isCall()) {
    RegDefs.insert(Mips::RA);
    return;
  }

  // Return if MI is a return.
  if (MI.isReturn())
    return;

  // Examine the implicit operands. Exclude register AT which is in the list of
  // clobbered registers of branch instructions.
  E = MI.getNumOperands();
  for (; I != E; ++I)
    insertDefUse(MI.getOperand(I), RegDefs, RegUses, Mips::AT);
}

//returns true if the Reg or its alias is in the RegSet.
bool Filler::isRegInSet(const SmallSet<unsigned, 32> &RegSet,
                        unsigned Reg) const {
  // Check Reg and all aliased Registers.
  for (MCRegAliasIterator AI(Reg, TM.getRegisterInfo(), true);
       AI.isValid(); ++AI)
    if (RegSet.count(*AI))
      return true;
  return false;
}

bool Filler::terminateSearch(const MachineInstr &Candidate) const {
  return (Candidate.isTerminator() || Candidate.isCall() ||
          Candidate.isLabel() || Candidate.isInlineAsm() ||
          Candidate.hasUnmodeledSideEffects());
}
