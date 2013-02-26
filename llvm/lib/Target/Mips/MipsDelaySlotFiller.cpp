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
#include "llvm/ADT/BitVector.h"
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
  class RegDefsUses {
  public:
    RegDefsUses(TargetMachine &TM);
    void init(const MachineInstr &MI);
    bool update(const MachineInstr &MI, unsigned Begin, unsigned End);

  private:
    bool checkRegDefsUses(BitVector &NewDefs, BitVector &NewUses, unsigned Reg,
                          bool IsDef) const;

    /// Returns true if Reg or its alias is in RegSet.
    bool isRegInSet(const BitVector &RegSet, unsigned Reg) const;

    const TargetRegisterInfo &TRI;
    BitVector Defs, Uses;
  };

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

    /// This function checks if it is valid to move Candidate to the delay slot
    /// and returns true if it isn't. It also updates load and store flags and
    /// register defs and uses.
    bool delayHasHazard(const MachineInstr &Candidate, bool &SawLoad,
                        bool &SawStore, RegDefsUses &RegDU) const;

    bool findDelayInstr(MachineBasicBlock &MBB, Iter slot, Iter &Filler) const;

    bool terminateSearch(const MachineInstr &Candidate) const;

    TargetMachine &TM;
    const TargetInstrInfo *TII;

    static char ID;
  };
  char Filler::ID = 0;
} // end of anonymous namespace

RegDefsUses::RegDefsUses(TargetMachine &TM)
  : TRI(*TM.getRegisterInfo()), Defs(TRI.getNumRegs(), false),
    Uses(TRI.getNumRegs(), false) {}

void RegDefsUses::init(const MachineInstr &MI) {
  // Add all register operands which are explicit and non-variadic.
  update(MI, 0, MI.getDesc().getNumOperands());

  // If MI is a call, add RA to Defs to prevent users of RA from going into
  // delay slot.
  if (MI.isCall())
    Defs.set(Mips::RA);

  // Add all implicit register operands of branch instructions except
  // register AT.
  if (MI.isBranch()) {
    update(MI, MI.getDesc().getNumOperands(), MI.getNumOperands());
    Defs.reset(Mips::AT);
  }
}

bool RegDefsUses::update(const MachineInstr &MI, unsigned Begin, unsigned End) {
  BitVector NewDefs(TRI.getNumRegs()), NewUses(TRI.getNumRegs());
  bool HasHazard = false;

  for (unsigned I = Begin; I != End; ++I) {
    const MachineOperand &MO = MI.getOperand(I);

    if (MO.isReg() && MO.getReg())
      HasHazard |= checkRegDefsUses(NewDefs, NewUses, MO.getReg(), MO.isDef());
  }

  Defs |= NewDefs;
  Uses |= NewUses;

  return HasHazard;
}

bool RegDefsUses::checkRegDefsUses(BitVector &NewDefs, BitVector &NewUses,
                                   unsigned Reg, bool IsDef) const {
  if (IsDef) {
    NewDefs.set(Reg);
    // check whether Reg has already been defined or used.
    return (isRegInSet(Defs, Reg) || isRegInSet(Uses, Reg));
  }

  NewUses.set(Reg);
  // check whether Reg has already been defined.
  return isRegInSet(Defs, Reg);
}

bool RegDefsUses::isRegInSet(const BitVector &RegSet, unsigned Reg) const {
  // Check Reg and all aliased Registers.
  for (MCRegAliasIterator AI(Reg, &TRI, true); AI.isValid(); ++AI)
    if (RegSet.test(*AI))
      return true;
  return false;
}

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
  RegDefsUses RegDU(TM);

  RegDU.init(*Slot);

  bool SawLoad = false;
  bool SawStore = false;

  for (ReverseIter I(Slot); I != MBB.rend(); ++I) {
    // skip debug value
    if (I->isDebugValue())
      continue;

    if (terminateSearch(*I))
      break;

    if (delayHasHazard(*I, SawLoad, SawStore, RegDU))
      continue;

    Filler = llvm::next(I).base();
    return true;
  }

  return false;
}

bool Filler::delayHasHazard(const MachineInstr &Candidate, bool &SawLoad,
                            bool &SawStore, RegDefsUses &RegDU) const {
  bool HasHazard = (Candidate.isImplicitDef() || Candidate.isKill());

  // Loads or stores cannot be moved past a store to the delay slot
  // and stores cannot be moved past a load.
  if (Candidate.mayStore() || Candidate.hasOrderedMemoryRef()) {
    HasHazard |= SawStore | SawLoad;
    SawStore = true;
  } else if (Candidate.mayLoad()) {
    HasHazard |= SawStore;
    SawLoad = true;
  }

  assert((!Candidate.isCall() && !Candidate.isReturn()) &&
         "Cannot put calls or returns in delay slot.");

  HasHazard |= RegDU.update(Candidate, 0, Candidate.getNumOperands());

  return HasHazard;
}

bool Filler::terminateSearch(const MachineInstr &Candidate) const {
  return (Candidate.isTerminator() || Candidate.isCall() ||
          Candidate.isLabel() || Candidate.isInlineAsm() ||
          Candidate.hasUnmodeledSideEffects());
}
