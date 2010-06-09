//===-- Thumb2ITBlockPass.cpp - Insert Thumb IT blocks ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "thumb2-it"
#include "ARM.h"
#include "ARMMachineFunctionInfo.h"
#include "Thumb2InstrInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumITs,        "Number of IT blocks inserted");
STATISTIC(NumMovedInsts, "Number of predicated instructions moved");

namespace {
  class Thumb2ITBlockPass : public MachineFunctionPass {
    bool PreRegAlloc;

  public:
    static char ID;
    Thumb2ITBlockPass(bool PreRA) :
      MachineFunctionPass(&ID), PreRegAlloc(PreRA) {}

    const Thumb2InstrInfo *TII;
    ARMFunctionInfo *AFI;

    virtual bool runOnMachineFunction(MachineFunction &Fn);

    virtual const char *getPassName() const {
      return "Thumb IT blocks insertion pass";
    }

  private:
    bool MoveCPSRUseUp(MachineBasicBlock &MBB,
                       MachineBasicBlock::iterator MBBI,
                       MachineBasicBlock::iterator E,
                       unsigned PredReg,
                       ARMCC::CondCodes CC, ARMCC::CondCodes OCC,
                       bool &Done);

    void FindITBlockRanges(MachineBasicBlock &MBB,
                           SmallVector<MachineInstr*,4> &FirstUses,
                           SmallVector<MachineInstr*,4> &LastUses);
    bool InsertITBlock(MachineInstr *First, MachineInstr *Last);
    bool InsertITBlocks(MachineBasicBlock &MBB);
    bool InsertITInstructions(MachineBasicBlock &MBB);
  };
  char Thumb2ITBlockPass::ID = 0;
}

static ARMCC::CondCodes getPredicate(const MachineInstr *MI, unsigned &PredReg){
  unsigned Opc = MI->getOpcode();
  if (Opc == ARM::tBcc || Opc == ARM::t2Bcc)
    return ARMCC::AL;

  int PIdx = MI->findFirstPredOperandIdx();
  if (PIdx == -1) {
    PredReg = 0;
    return ARMCC::AL;
  }

  PredReg = MI->getOperand(PIdx+1).getReg();
  return (ARMCC::CondCodes)MI->getOperand(PIdx).getImm();
}

bool
Thumb2ITBlockPass::MoveCPSRUseUp(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MBBI,
                                 MachineBasicBlock::iterator E,
                                 unsigned PredReg,
                                 ARMCC::CondCodes CC, ARMCC::CondCodes OCC,
                                 bool &Done) {
  SmallSet<unsigned, 4> Defs, Uses;
  MachineBasicBlock::iterator I = MBBI;
  // Look for next CPSR use by scanning up to 4 instructions.
  for (unsigned i = 0; i < 4; ++i) {
    MachineInstr *MI = &*I;
    unsigned MPredReg = 0;
    ARMCC::CondCodes MCC = getPredicate(MI, MPredReg);
    if (MCC != ARMCC::AL) {
      if (MPredReg != PredReg || (MCC != CC && MCC != OCC))
        return false;

      // Check if the instruction is using any register that's defined
      // below the previous predicated instruction. Also return false if
      // it defines any register which is used in between.
      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
        const MachineOperand &MO = MI->getOperand(i);
        if (!MO.isReg())
          continue;
        unsigned Reg = MO.getReg();
        if (!Reg)
          continue;
        if (MO.isDef()) {
          if (Reg == PredReg || Uses.count(Reg))
            return false;
        } else {
          if (Defs.count(Reg))
            return false;
        }
      }

      Done = (I == E);
      MBB.remove(MI);
      MBB.insert(MBBI, MI);
      ++NumMovedInsts;
      return true;
    }

    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      const MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg())
        continue;
      unsigned Reg = MO.getReg();
      if (!Reg)
        continue;
      if (MO.isDef()) {
        if (Reg == PredReg)
          return false;
        Defs.insert(Reg);
      } else
        Uses.insert(Reg);
    }

    if (I == E)
      break;
    ++I;
  }
  return false;
}

static bool isCPSRLiveout(MachineBasicBlock &MBB) {
  for (MachineBasicBlock::succ_iterator I = MBB.succ_begin(),
         E = MBB.succ_end(); I != E; ++I) {
    if ((*I)->isLiveIn(ARM::CPSR))
      return true;
  }
  return false;
}

void Thumb2ITBlockPass::FindITBlockRanges(MachineBasicBlock &MBB,
                                       SmallVector<MachineInstr*,4> &FirstUses,
                                       SmallVector<MachineInstr*,4> &LastUses) {
  bool SeenUse = false;
  MachineOperand *LastDef = 0;
  MachineOperand *LastUse = 0;
  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    MachineInstr *MI = &*MBBI;
    ++MBBI;

    MachineOperand *Def = 0;
    MachineOperand *Use = 0;
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg() || MO.getReg() != ARM::CPSR)
        continue;
      if (MO.isDef()) {
        assert(Def == 0 && "Multiple defs of CPSR?");
        Def = &MO;
      } else {
        assert(Use == 0 && "Multiple uses of CPSR?");
        Use = &MO;
      }
    }

    if (Use) {
      LastUse = Use;
      if (!SeenUse) {
        FirstUses.push_back(MI);
        SeenUse = true;
      }
    }
    if (Def) {
      if (LastUse) {
        LastUses.push_back(LastUse->getParent());
        LastUse = 0;
      }
      LastDef = Def;
      SeenUse = false;
    }
  }

  if (LastUse) {
    // Is the last use a kill?
    if (isCPSRLiveout(MBB))
      LastUses.push_back(0);
    else
      LastUses.push_back(LastUse->getParent());
  }
}

bool Thumb2ITBlockPass::InsertITBlock(MachineInstr *First, MachineInstr *Last) {
  if (First == Last)
    return false;

  bool Modified = false;
  MachineBasicBlock *MBB = First->getParent();
  MachineBasicBlock::iterator MBBI = First;
  MachineBasicBlock::iterator E = Last;

  if (First->getDesc().isBranch() || First->getDesc().isReturn())
    return false;

  unsigned PredReg = 0;
  ARMCC::CondCodes CC = getPredicate(First, PredReg);
  if (CC == ARMCC::AL)
    return Modified;

  // Move uses of the CPSR together if possible.
  ARMCC::CondCodes OCC = ARMCC::getOppositeCondition(CC);

  do {
    ++MBBI;
    if (MBBI->getDesc().isBranch() || MBBI->getDesc().isReturn())
      return Modified;
    MachineInstr *NMI = &*MBBI;
    unsigned NPredReg = 0;
    ARMCC::CondCodes NCC = getPredicate(NMI, NPredReg);
    if (NCC != CC && NCC != OCC) {
      if (NCC != ARMCC::AL)
        return Modified;
      assert(MBBI != E);
      bool Done = false;
      if (!MoveCPSRUseUp(*MBB, MBBI, E, PredReg, CC, OCC, Done))
        return Modified;
      Modified = true;
      if (Done)
        MBBI = E;
    }
  } while (MBBI != E);

  // Insert a new block for consecutive predicated instructions.
  MachineFunction *MF = MBB->getParent();
  MachineBasicBlock *NewMBB = MF->CreateMachineBasicBlock(MBB->getBasicBlock());
  MachineFunction::iterator Pos = MBB;
  MF->insert(++Pos, NewMBB);

  // Move all the successors of this block to the specified block.
  NewMBB->transferSuccessors(MBB);

  // Add an edge from CurMBB to NewMBB for the fall-through.
  MBB->addSuccessor(NewMBB);
  NewMBB->splice(NewMBB->end(), MBB, ++MBBI, MBB->end());  
  return true;
}

bool Thumb2ITBlockPass::InsertITBlocks(MachineBasicBlock &MBB) {
  SmallVector<MachineInstr*, 4> FirstUses;
  SmallVector<MachineInstr*, 4> LastUses;
  FindITBlockRanges(MBB, FirstUses, LastUses);
  assert(FirstUses.size() == LastUses.size() && "Incorrect range information!");

  bool Modified = false;
  for (unsigned i = 0, e = FirstUses.size(); i != e; ++i) {
    if (LastUses[i] == 0)
      // Must be the last pair where CPSR is live out of the block.
      return Modified;
    Modified |= InsertITBlock(FirstUses[i], LastUses[i]);
  }
  return Modified;
}

static void TrackDefUses(MachineInstr *MI, SmallSet<unsigned, 4> &Defs,
                         SmallSet<unsigned, 4> &Uses) {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (!Reg)
      continue;
    if (MO.isDef())
      Defs.insert(Reg);
    else
      Uses.insert(Reg);
  }
}

bool Thumb2ITBlockPass::InsertITInstructions(MachineBasicBlock &MBB) {
  bool Modified = false;

  SmallSet<unsigned, 4> Defs;
  SmallSet<unsigned, 4> Uses;
  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    MachineInstr *MI = &*MBBI;
    DebugLoc dl = MI->getDebugLoc();
    unsigned PredReg = 0;
    ARMCC::CondCodes CC = getPredicate(MI, PredReg);
    if (CC == ARMCC::AL) {
      ++MBBI;
      continue;
    }

    Defs.clear();
    Uses.clear();
    TrackDefUses(MI, Defs, Uses);

    // Insert an IT instruction.
    MachineInstrBuilder MIB = BuildMI(MBB, MBBI, dl, TII->get(ARM::t2IT))
      .addImm(CC);
    MachineBasicBlock::iterator InsertPos = MIB;
    ++MBBI;

    // Finalize IT mask.
    ARMCC::CondCodes OCC = ARMCC::getOppositeCondition(CC);
    unsigned Mask = 0, Pos = 3;
    // Branches, including tricky ones like LDM_RET, need to end an IT
    // block so check the instruction we just put in the block.
    for (; MBBI != E && Pos &&
           (!MI->getDesc().isBranch() && !MI->getDesc().isReturn()) ; ++MBBI) {
      if (MBBI->isDebugValue())
        continue;

      MachineInstr *NMI = &*MBBI;
      MI = NMI;

      unsigned NPredReg = 0;
      ARMCC::CondCodes NCC = getPredicate(NMI, NPredReg);
      if (NCC == CC || NCC == OCC)
        Mask |= (NCC & 1) << Pos;
      else {
        unsigned SrcReg, DstReg, SrcSubIdx, DstSubIdx;
        if (NCC == ARMCC::AL &&
            TII->isMoveInstr(*NMI, SrcReg, DstReg, SrcSubIdx, DstSubIdx)) {
          assert(SrcSubIdx == 0 && DstSubIdx == 0 &&
                 "Sub-register indices still around?");
          // llvm models select's as two-address instructions. That means a copy
          // is inserted before a t2MOVccr, etc. If the copy is scheduled in
          // between selects we would end up creating multiple IT blocks.
          if (!Uses.count(DstReg) && !Defs.count(SrcReg)) {
            --MBBI;
            MBB.remove(NMI);
            MBB.insert(InsertPos, NMI);
            ++NumMovedInsts;
            continue;
          }
        }
        break;
      }
      TrackDefUses(NMI, Defs, Uses);
      --Pos;
    }

    Mask |= (1 << Pos);
    // Tag along (firstcond[0] << 4) with the mask.
    Mask |= (CC & 1) << 4;
    MIB.addImm(Mask);
    Modified = true;
    ++NumITs;
  }

  return Modified;
}

bool Thumb2ITBlockPass::runOnMachineFunction(MachineFunction &Fn) {
  const TargetMachine &TM = Fn.getTarget();
  AFI = Fn.getInfo<ARMFunctionInfo>();
  TII = static_cast<const Thumb2InstrInfo*>(TM.getInstrInfo());

  if (!AFI->isThumbFunction())
    return false;

  bool Modified = false;
  for (MachineFunction::iterator MFI = Fn.begin(), E = Fn.end(); MFI != E; ) {
    MachineBasicBlock &MBB = *MFI;
    ++MFI;
    if (PreRegAlloc)
      Modified |= InsertITBlocks(MBB);
    else
      Modified |= InsertITInstructions(MBB);
  }

  return Modified;
}

/// createThumb2ITBlockPass - Returns an instance of the Thumb2 IT blocks
/// insertion pass.
FunctionPass *llvm::createThumb2ITBlockPass(bool PreAlloc) {
  return new Thumb2ITBlockPass(PreAlloc);
}
