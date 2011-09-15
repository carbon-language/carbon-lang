//===-- llvm/CodeGen/VirtRegMap.cpp - Virtual Register Map ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the VirtRegMap class.
//
// It also contains implementations of the Spiller interface, which, given a
// virtual register map and a machine function, eliminates all virtual
// references by replacing them with physical register references - adding spill
// code as necessary.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "virtregmap"
#include "VirtRegMap.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include <algorithm>
using namespace llvm;

STATISTIC(NumSpillSlots, "Number of spill slots allocated");
STATISTIC(NumIdCopies,   "Number of identity moves eliminated after rewriting");

//===----------------------------------------------------------------------===//
//  VirtRegMap implementation
//===----------------------------------------------------------------------===//

char VirtRegMap::ID = 0;

INITIALIZE_PASS(VirtRegMap, "virtregmap", "Virtual Register Map", false, false)

bool VirtRegMap::runOnMachineFunction(MachineFunction &mf) {
  MRI = &mf.getRegInfo();
  TII = mf.getTarget().getInstrInfo();
  TRI = mf.getTarget().getRegisterInfo();
  MF = &mf;

  ReMatId = MAX_STACK_SLOT+1;
  LowSpillSlot = HighSpillSlot = NO_STACK_SLOT;
  
  Virt2PhysMap.clear();
  Virt2StackSlotMap.clear();
  Virt2ReMatIdMap.clear();
  Virt2SplitMap.clear();
  Virt2SplitKillMap.clear();
  ReMatMap.clear();
  ImplicitDefed.clear();
  SpillSlotToUsesMap.clear();
  MI2VirtMap.clear();
  SpillPt2VirtMap.clear();
  RestorePt2VirtMap.clear();
  EmergencySpillMap.clear();
  EmergencySpillSlots.clear();
  
  SpillSlotToUsesMap.resize(8);
  ImplicitDefed.resize(MF->getRegInfo().getNumVirtRegs());

  allocatableRCRegs.clear();
  for (TargetRegisterInfo::regclass_iterator I = TRI->regclass_begin(),
         E = TRI->regclass_end(); I != E; ++I)
    allocatableRCRegs.insert(std::make_pair(*I,
                                            TRI->getAllocatableSet(mf, *I)));

  grow();
  
  return false;
}

void VirtRegMap::grow() {
  unsigned NumRegs = MF->getRegInfo().getNumVirtRegs();
  Virt2PhysMap.resize(NumRegs);
  Virt2StackSlotMap.resize(NumRegs);
  Virt2ReMatIdMap.resize(NumRegs);
  Virt2SplitMap.resize(NumRegs);
  Virt2SplitKillMap.resize(NumRegs);
  ReMatMap.resize(NumRegs);
  ImplicitDefed.resize(NumRegs);
}

unsigned VirtRegMap::createSpillSlot(const TargetRegisterClass *RC) {
  int SS = MF->getFrameInfo()->CreateSpillStackObject(RC->getSize(),
                                                      RC->getAlignment());
  if (LowSpillSlot == NO_STACK_SLOT)
    LowSpillSlot = SS;
  if (HighSpillSlot == NO_STACK_SLOT || SS > HighSpillSlot)
    HighSpillSlot = SS;
  assert(SS >= LowSpillSlot && "Unexpected low spill slot");
  unsigned Idx = SS-LowSpillSlot;
  while (Idx >= SpillSlotToUsesMap.size())
    SpillSlotToUsesMap.resize(SpillSlotToUsesMap.size()*2);
  ++NumSpillSlots;
  return SS;
}

unsigned VirtRegMap::getRegAllocPref(unsigned virtReg) {
  std::pair<unsigned, unsigned> Hint = MRI->getRegAllocationHint(virtReg);
  unsigned physReg = Hint.second;
  if (TargetRegisterInfo::isVirtualRegister(physReg) && hasPhys(physReg))
    physReg = getPhys(physReg);
  if (Hint.first == 0)
    return (TargetRegisterInfo::isPhysicalRegister(physReg))
      ? physReg : 0;
  return TRI->ResolveRegAllocHint(Hint.first, physReg, *MF);
}

int VirtRegMap::assignVirt2StackSlot(unsigned virtReg) {
  assert(TargetRegisterInfo::isVirtualRegister(virtReg));
  assert(Virt2StackSlotMap[virtReg] == NO_STACK_SLOT &&
         "attempt to assign stack slot to already spilled register");
  const TargetRegisterClass* RC = MF->getRegInfo().getRegClass(virtReg);
  return Virt2StackSlotMap[virtReg] = createSpillSlot(RC);
}

void VirtRegMap::assignVirt2StackSlot(unsigned virtReg, int SS) {
  assert(TargetRegisterInfo::isVirtualRegister(virtReg));
  assert(Virt2StackSlotMap[virtReg] == NO_STACK_SLOT &&
         "attempt to assign stack slot to already spilled register");
  assert((SS >= 0 ||
          (SS >= MF->getFrameInfo()->getObjectIndexBegin())) &&
         "illegal fixed frame index");
  Virt2StackSlotMap[virtReg] = SS;
}

int VirtRegMap::assignVirtReMatId(unsigned virtReg) {
  assert(TargetRegisterInfo::isVirtualRegister(virtReg));
  assert(Virt2ReMatIdMap[virtReg] == NO_STACK_SLOT &&
         "attempt to assign re-mat id to already spilled register");
  Virt2ReMatIdMap[virtReg] = ReMatId;
  return ReMatId++;
}

void VirtRegMap::assignVirtReMatId(unsigned virtReg, int id) {
  assert(TargetRegisterInfo::isVirtualRegister(virtReg));
  assert(Virt2ReMatIdMap[virtReg] == NO_STACK_SLOT &&
         "attempt to assign re-mat id to already spilled register");
  Virt2ReMatIdMap[virtReg] = id;
}

int VirtRegMap::getEmergencySpillSlot(const TargetRegisterClass *RC) {
  std::map<const TargetRegisterClass*, int>::iterator I =
    EmergencySpillSlots.find(RC);
  if (I != EmergencySpillSlots.end())
    return I->second;
  return EmergencySpillSlots[RC] = createSpillSlot(RC);
}

void VirtRegMap::addSpillSlotUse(int FI, MachineInstr *MI) {
  if (!MF->getFrameInfo()->isFixedObjectIndex(FI)) {
    // If FI < LowSpillSlot, this stack reference was produced by
    // instruction selection and is not a spill
    if (FI >= LowSpillSlot) {
      assert(FI >= 0 && "Spill slot index should not be negative!");
      assert((unsigned)FI-LowSpillSlot < SpillSlotToUsesMap.size()
             && "Invalid spill slot");
      SpillSlotToUsesMap[FI-LowSpillSlot].insert(MI);
    }
  }
}

void VirtRegMap::virtFolded(unsigned VirtReg, MachineInstr *OldMI,
                            MachineInstr *NewMI, ModRef MRInfo) {
  // Move previous memory references folded to new instruction.
  MI2VirtMapTy::iterator IP = MI2VirtMap.lower_bound(NewMI);
  for (MI2VirtMapTy::iterator I = MI2VirtMap.lower_bound(OldMI),
         E = MI2VirtMap.end(); I != E && I->first == OldMI; ) {
    MI2VirtMap.insert(IP, std::make_pair(NewMI, I->second));
    MI2VirtMap.erase(I++);
  }

  // add new memory reference
  MI2VirtMap.insert(IP, std::make_pair(NewMI, std::make_pair(VirtReg, MRInfo)));
}

void VirtRegMap::virtFolded(unsigned VirtReg, MachineInstr *MI, ModRef MRInfo) {
  MI2VirtMapTy::iterator IP = MI2VirtMap.lower_bound(MI);
  MI2VirtMap.insert(IP, std::make_pair(MI, std::make_pair(VirtReg, MRInfo)));
}

void VirtRegMap::RemoveMachineInstrFromMaps(MachineInstr *MI) {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isFI())
      continue;
    int FI = MO.getIndex();
    if (MF->getFrameInfo()->isFixedObjectIndex(FI))
      continue;
    // This stack reference was produced by instruction selection and
    // is not a spill
    if (FI < LowSpillSlot)
      continue;
    assert((unsigned)FI-LowSpillSlot < SpillSlotToUsesMap.size()
           && "Invalid spill slot");
    SpillSlotToUsesMap[FI-LowSpillSlot].erase(MI);
  }
  MI2VirtMap.erase(MI);
  SpillPt2VirtMap.erase(MI);
  RestorePt2VirtMap.erase(MI);
  EmergencySpillMap.erase(MI);
}

/// FindUnusedRegisters - Gather a list of allocatable registers that
/// have not been allocated to any virtual register.
bool VirtRegMap::FindUnusedRegisters(LiveIntervals* LIs) {
  unsigned NumRegs = TRI->getNumRegs();
  UnusedRegs.reset();
  UnusedRegs.resize(NumRegs);

  BitVector Used(NumRegs);
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    unsigned Reg = TargetRegisterInfo::index2VirtReg(i);
    if (Virt2PhysMap[Reg] != (unsigned)VirtRegMap::NO_PHYS_REG)
      Used.set(Virt2PhysMap[Reg]);
  }

  BitVector Allocatable = TRI->getAllocatableSet(*MF);
  bool AnyUnused = false;
  for (unsigned Reg = 1; Reg < NumRegs; ++Reg) {
    if (Allocatable[Reg] && !Used[Reg] && !LIs->hasInterval(Reg)) {
      bool ReallyUnused = true;
      for (const unsigned *AS = TRI->getAliasSet(Reg); *AS; ++AS) {
        if (Used[*AS] || LIs->hasInterval(*AS)) {
          ReallyUnused = false;
          break;
        }
      }
      if (ReallyUnused) {
        AnyUnused = true;
        UnusedRegs.set(Reg);
      }
    }
  }

  return AnyUnused;
}

void VirtRegMap::rewrite(SlotIndexes *Indexes) {
  DEBUG(dbgs() << "********** REWRITE VIRTUAL REGISTERS **********\n"
               << "********** Function: "
               << MF->getFunction()->getName() << '\n');
  DEBUG(dump());
  SmallVector<unsigned, 8> SuperDeads;
  SmallVector<unsigned, 8> SuperDefs;
  SmallVector<unsigned, 8> SuperKills;

  for (MachineFunction::iterator MBBI = MF->begin(), MBBE = MF->end();
       MBBI != MBBE; ++MBBI) {
    DEBUG(MBBI->print(dbgs(), Indexes));
    for (MachineBasicBlock::iterator MII = MBBI->begin(), MIE = MBBI->end();
         MII != MIE;) {
      MachineInstr *MI = MII;
      ++MII;

      for (MachineInstr::mop_iterator MOI = MI->operands_begin(),
           MOE = MI->operands_end(); MOI != MOE; ++MOI) {
        MachineOperand &MO = *MOI;
        if (!MO.isReg() || !TargetRegisterInfo::isVirtualRegister(MO.getReg()))
          continue;
        unsigned VirtReg = MO.getReg();
        unsigned PhysReg = getPhys(VirtReg);
        assert(PhysReg != NO_PHYS_REG && "Instruction uses unmapped VirtReg");

        // Preserve semantics of sub-register operands.
        if (MO.getSubReg()) {
          // A virtual register kill refers to the whole register, so we may
          // have to add <imp-use,kill> operands for the super-register.
          if (MO.isUse()) {
            if (MO.isKill() && !MO.isUndef())
              SuperKills.push_back(PhysReg);
          } else if (MO.isDead())
            SuperDeads.push_back(PhysReg);
          else
            SuperDefs.push_back(PhysReg);

          // PhysReg operands cannot have subregister indexes.
          PhysReg = TRI->getSubReg(PhysReg, MO.getSubReg());
          assert(PhysReg && "Invalid SubReg for physical register");
          MO.setSubReg(0);
        }
        // Rewrite. Note we could have used MachineOperand::substPhysReg(), but
        // we need the inlining here.
        MO.setReg(PhysReg);
      }

      // Add any missing super-register kills after rewriting the whole
      // instruction.
      while (!SuperKills.empty())
        MI->addRegisterKilled(SuperKills.pop_back_val(), TRI, true);

      while (!SuperDeads.empty())
        MI->addRegisterDead(SuperDeads.pop_back_val(), TRI, true);

      while (!SuperDefs.empty())
        MI->addRegisterDefined(SuperDefs.pop_back_val(), TRI);

      DEBUG(dbgs() << "> " << *MI);

      // Finally, remove any identity copies.
      if (MI->isIdentityCopy()) {
        ++NumIdCopies;
        if (MI->getNumOperands() == 2) {
          DEBUG(dbgs() << "Deleting identity copy.\n");
          RemoveMachineInstrFromMaps(MI);
          if (Indexes)
            Indexes->removeMachineInstrFromMaps(MI);
          // It's safe to erase MI because MII has already been incremented.
          MI->eraseFromParent();
        } else {
          // Transform identity copy to a KILL to deal with subregisters.
          MI->setDesc(TII->get(TargetOpcode::KILL));
          DEBUG(dbgs() << "Identity copy: " << *MI);
        }
      }
    }
  }

  // Tell MRI about physical registers in use.
  for (unsigned Reg = 1, RegE = TRI->getNumRegs(); Reg != RegE; ++Reg)
    if (!MRI->reg_nodbg_empty(Reg))
      MRI->setPhysRegUsed(Reg);
}

void VirtRegMap::print(raw_ostream &OS, const Module* M) const {
  const TargetRegisterInfo* TRI = MF->getTarget().getRegisterInfo();
  const MachineRegisterInfo &MRI = MF->getRegInfo();

  OS << "********** REGISTER MAP **********\n";
  for (unsigned i = 0, e = MRI.getNumVirtRegs(); i != e; ++i) {
    unsigned Reg = TargetRegisterInfo::index2VirtReg(i);
    if (Virt2PhysMap[Reg] != (unsigned)VirtRegMap::NO_PHYS_REG) {
      OS << '[' << PrintReg(Reg, TRI) << " -> "
         << PrintReg(Virt2PhysMap[Reg], TRI) << "] "
         << MRI.getRegClass(Reg)->getName() << "\n";
    }
  }

  for (unsigned i = 0, e = MRI.getNumVirtRegs(); i != e; ++i) {
    unsigned Reg = TargetRegisterInfo::index2VirtReg(i);
    if (Virt2StackSlotMap[Reg] != VirtRegMap::NO_STACK_SLOT) {
      OS << '[' << PrintReg(Reg, TRI) << " -> fi#" << Virt2StackSlotMap[Reg]
         << "] " << MRI.getRegClass(Reg)->getName() << "\n";
    }
  }
  OS << '\n';
}

void VirtRegMap::dump() const {
  print(dbgs());
}
