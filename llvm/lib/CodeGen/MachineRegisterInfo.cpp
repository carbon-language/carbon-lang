//===-- lib/Codegen/MachineRegisterInfo.cpp -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of the MachineRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

MachineRegisterInfo::MachineRegisterInfo(const TargetRegisterInfo &TRI) {
  VRegInfo.reserve(256);
  RegAllocHints.reserve(256);
  RegClass2VRegMap = new std::vector<unsigned>[TRI.getNumRegClasses()];
  UsedPhysRegs.resize(TRI.getNumRegs());
  
  // Create the physreg use/def lists.
  PhysRegUseDefLists = new MachineOperand*[TRI.getNumRegs()];
  memset(PhysRegUseDefLists, 0, sizeof(MachineOperand*)*TRI.getNumRegs());
}

MachineRegisterInfo::~MachineRegisterInfo() {
#ifndef NDEBUG
  for (unsigned i = 0, e = VRegInfo.size(); i != e; ++i)
    assert(VRegInfo[i].second == 0 && "Vreg use list non-empty still?");
  for (unsigned i = 0, e = UsedPhysRegs.size(); i != e; ++i)
    assert(!PhysRegUseDefLists[i] &&
           "PhysRegUseDefLists has entries after all instructions are deleted");
#endif
  delete [] PhysRegUseDefLists;
  delete [] RegClass2VRegMap;
}

/// setRegClass - Set the register class of the specified virtual register.
///
void
MachineRegisterInfo::setRegClass(unsigned Reg, const TargetRegisterClass *RC) {
  unsigned VR = Reg;
  Reg -= TargetRegisterInfo::FirstVirtualRegister;
  assert(Reg < VRegInfo.size() && "Invalid vreg!");
  const TargetRegisterClass *OldRC = VRegInfo[Reg].first;
  VRegInfo[Reg].first = RC;

  // Remove from old register class's vregs list. This may be slow but
  // fortunately this operation is rarely needed.
  std::vector<unsigned> &VRegs = RegClass2VRegMap[OldRC->getID()];
  std::vector<unsigned>::iterator I = std::find(VRegs.begin(), VRegs.end(), VR);
  VRegs.erase(I);

  // Add to new register class's vregs list.
  RegClass2VRegMap[RC->getID()].push_back(VR);
}

/// createVirtualRegister - Create and return a new virtual register in the
/// function with the specified register class.
///
unsigned
MachineRegisterInfo::createVirtualRegister(const TargetRegisterClass *RegClass){
  assert(RegClass && "Cannot create register without RegClass!");
  // Add a reg, but keep track of whether the vector reallocated or not.
  void *ArrayBase = VRegInfo.empty() ? 0 : &VRegInfo[0];
  VRegInfo.push_back(std::make_pair(RegClass, (MachineOperand*)0));
  RegAllocHints.push_back(std::make_pair(0, 0));

  if (!((&VRegInfo[0] == ArrayBase || VRegInfo.size() == 1)))
    // The vector reallocated, handle this now.
    HandleVRegListReallocation();
  unsigned VR = getLastVirtReg();
  RegClass2VRegMap[RegClass->getID()].push_back(VR);
  return VR;
}

/// HandleVRegListReallocation - We just added a virtual register to the
/// VRegInfo info list and it reallocated.  Update the use/def lists info
/// pointers.
void MachineRegisterInfo::HandleVRegListReallocation() {
  // The back pointers for the vreg lists point into the previous vector.
  // Update them to point to their correct slots.
  for (unsigned i = 0, e = VRegInfo.size(); i != e; ++i) {
    MachineOperand *List = VRegInfo[i].second;
    if (!List) continue;
    // Update the back-pointer to be accurate once more.
    List->Contents.Reg.Prev = &VRegInfo[i].second;
  }
}

/// replaceRegWith - Replace all instances of FromReg with ToReg in the
/// machine function.  This is like llvm-level X->replaceAllUsesWith(Y),
/// except that it also changes any definitions of the register as well.
void MachineRegisterInfo::replaceRegWith(unsigned FromReg, unsigned ToReg) {
  assert(FromReg != ToReg && "Cannot replace a reg with itself");

  // TODO: This could be more efficient by bulk changing the operands.
  for (reg_iterator I = reg_begin(FromReg), E = reg_end(); I != E; ) {
    MachineOperand &O = I.getOperand();
    ++I;
    O.setReg(ToReg);
  }
}


/// getVRegDef - Return the machine instr that defines the specified virtual
/// register or null if none is found.  This assumes that the code is in SSA
/// form, so there should only be one definition.
MachineInstr *MachineRegisterInfo::getVRegDef(unsigned Reg) const {
  assert(Reg-TargetRegisterInfo::FirstVirtualRegister < VRegInfo.size() &&
         "Invalid vreg!");
  // Since we are in SSA form, we can use the first definition.
  if (!def_empty(Reg))
    return &*def_begin(Reg);
  return 0;
}

bool MachineRegisterInfo::hasOneUse(unsigned RegNo) const {
  use_iterator UI = use_begin(RegNo);
  if (UI == use_end())
    return false;
  return ++UI == use_end();
}

bool MachineRegisterInfo::hasOneNonDBGUse(unsigned RegNo) const {
  use_nodbg_iterator UI = use_nodbg_begin(RegNo);
  if (UI == use_nodbg_end())
    return false;
  return ++UI == use_nodbg_end();
}

/// clearKillFlags - Iterate over all the uses of the given register and
/// clear the kill flag from the MachineOperand. This function is used by
/// optimization passes which extend register lifetimes and need only
/// preserve conservative kill flag information.
void MachineRegisterInfo::clearKillFlags(unsigned Reg) const {
  for (use_iterator UI = use_begin(Reg), UE = use_end(); UI != UE; ++UI)
    UI.getOperand().setIsKill(false);
}

bool MachineRegisterInfo::isLiveIn(unsigned Reg) const {
  for (livein_iterator I = livein_begin(), E = livein_end(); I != E; ++I)
    if (I->first == Reg || I->second == Reg)
      return true;
  return false;
}

bool MachineRegisterInfo::isLiveOut(unsigned Reg) const {
  for (liveout_iterator I = liveout_begin(), E = liveout_end(); I != E; ++I)
    if (*I == Reg)
      return true;
  return false;
}

/// getLiveInPhysReg - If VReg is a live-in virtual register, return the
/// corresponding live-in physical register.
unsigned MachineRegisterInfo::getLiveInPhysReg(unsigned VReg) const {
  for (livein_iterator I = livein_begin(), E = livein_end(); I != E; ++I)
    if (I->second == VReg)
      return I->first;
  return 0;
}

/// getLiveInVirtReg - If PReg is a live-in physical register, return the
/// corresponding live-in physical register.
unsigned MachineRegisterInfo::getLiveInVirtReg(unsigned PReg) const {
  for (livein_iterator I = livein_begin(), E = livein_end(); I != E; ++I)
    if (I->first == PReg)
      return I->second;
  return 0;
}

/// EmitLiveInCopies - Emit copies to initialize livein virtual registers
/// into the given entry block.
void
MachineRegisterInfo::EmitLiveInCopies(MachineBasicBlock *EntryMBB,
                                      const TargetRegisterInfo &TRI,
                                      const TargetInstrInfo &TII) {
  // Emit the copies into the top of the block.
  for (unsigned i = 0, e = LiveIns.size(); i != e; ++i)
    if (LiveIns[i].second) {
      if (use_empty(LiveIns[i].second)) {
        // The livein has no uses. Drop it.
        //
        // It would be preferable to have isel avoid creating live-in
        // records for unused arguments in the first place, but it's
        // complicated by the debug info code for arguments.
        LiveIns.erase(LiveIns.begin() + i);
        --i; --e;
      } else {
        // Emit a copy.
        BuildMI(*EntryMBB, EntryMBB->begin(), DebugLoc(),
                TII.get(TargetOpcode::COPY), LiveIns[i].second)
          .addReg(LiveIns[i].first);

        // Add the register to the entry block live-in set.
        EntryMBB->addLiveIn(LiveIns[i].first);
      }
    } else {
      // Add the register to the entry block live-in set.
      EntryMBB->addLiveIn(LiveIns[i].first);
    }
}

void MachineRegisterInfo::closePhysRegsUsed(const TargetRegisterInfo &TRI) {
  for (int i = UsedPhysRegs.find_first(); i >= 0;
       i = UsedPhysRegs.find_next(i))
         for (const unsigned *SS = TRI.getSubRegisters(i);
              unsigned SubReg = *SS; ++SS)
           if (SubReg > unsigned(i))
             UsedPhysRegs.set(SubReg);
}

#ifndef NDEBUG
void MachineRegisterInfo::dumpUses(unsigned Reg) const {
  for (use_iterator I = use_begin(Reg), E = use_end(); I != E; ++I)
    I.getOperand().getParent()->dump();
}
#endif
