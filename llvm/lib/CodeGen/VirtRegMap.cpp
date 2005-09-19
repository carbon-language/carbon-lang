//===-- llvm/CodeGen/VirtRegMap.cpp - Virtual Register Map ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the VirtRegMap class.
//
// It also contains implementations of the the Spiller interface, which, given a
// virtual register map and a machine function, eliminates all virtual
// references by replacing them with physical register references - adding spill
// code as necessary.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "spiller"
#include "VirtRegMap.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
using namespace llvm;

namespace {
  Statistic<> NumSpills("spiller", "Number of register spills");
  Statistic<> NumStores("spiller", "Number of stores added");
  Statistic<> NumLoads ("spiller", "Number of loads added");
  Statistic<> NumReused("spiller", "Number of values reused");
  Statistic<> NumDSE   ("spiller", "Number of dead stores elided");

  enum SpillerName { simple, local };

  cl::opt<SpillerName>
  SpillerOpt("spiller",
             cl::desc("Spiller to use: (default: local)"),
             cl::Prefix,
             cl::values(clEnumVal(simple, "  simple spiller"),
                        clEnumVal(local,  "  local spiller"),
                        clEnumValEnd),
             cl::init(local));
}

//===----------------------------------------------------------------------===//
//  VirtRegMap implementation
//===----------------------------------------------------------------------===//

void VirtRegMap::grow() {
  Virt2PhysMap.grow(MF.getSSARegMap()->getLastVirtReg());
  Virt2StackSlotMap.grow(MF.getSSARegMap()->getLastVirtReg());
}

int VirtRegMap::assignVirt2StackSlot(unsigned virtReg) {
  assert(MRegisterInfo::isVirtualRegister(virtReg));
  assert(Virt2StackSlotMap[virtReg] == NO_STACK_SLOT &&
         "attempt to assign stack slot to already spilled register");
  const TargetRegisterClass* RC = MF.getSSARegMap()->getRegClass(virtReg);
  int frameIndex = MF.getFrameInfo()->CreateStackObject(RC->getSize(),
                                                        RC->getAlignment());
  Virt2StackSlotMap[virtReg] = frameIndex;
  ++NumSpills;
  return frameIndex;
}

void VirtRegMap::assignVirt2StackSlot(unsigned virtReg, int frameIndex) {
  assert(MRegisterInfo::isVirtualRegister(virtReg));
  assert(Virt2StackSlotMap[virtReg] == NO_STACK_SLOT &&
         "attempt to assign stack slot to already spilled register");
  Virt2StackSlotMap[virtReg] = frameIndex;
}

void VirtRegMap::virtFolded(unsigned VirtReg, MachineInstr *OldMI,
                            unsigned OpNo, MachineInstr *NewMI) {
  // Move previous memory references folded to new instruction.
  MI2VirtMapTy::iterator IP = MI2VirtMap.lower_bound(NewMI);
  for (MI2VirtMapTy::iterator I = MI2VirtMap.lower_bound(OldMI),
         E = MI2VirtMap.end(); I != E && I->first == OldMI; ) {
    MI2VirtMap.insert(IP, std::make_pair(NewMI, I->second));
    MI2VirtMap.erase(I++);
  }

  ModRef MRInfo;
  if (!OldMI->getOperand(OpNo).isDef()) {
    assert(OldMI->getOperand(OpNo).isUse() && "Operand is not use or def?");
    MRInfo = isRef;
  } else {
    MRInfo = OldMI->getOperand(OpNo).isUse() ? isModRef : isMod;
  }

  // add new memory reference
  MI2VirtMap.insert(IP, std::make_pair(NewMI, std::make_pair(VirtReg, MRInfo)));
}

void VirtRegMap::print(std::ostream &OS) const {
  const MRegisterInfo* MRI = MF.getTarget().getRegisterInfo();

  OS << "********** REGISTER MAP **********\n";
  for (unsigned i = MRegisterInfo::FirstVirtualRegister,
         e = MF.getSSARegMap()->getLastVirtReg(); i <= e; ++i) {
    if (Virt2PhysMap[i] != (unsigned)VirtRegMap::NO_PHYS_REG)
      OS << "[reg" << i << " -> " << MRI->getName(Virt2PhysMap[i]) << "]\n";

  }

  for (unsigned i = MRegisterInfo::FirstVirtualRegister,
         e = MF.getSSARegMap()->getLastVirtReg(); i <= e; ++i)
    if (Virt2StackSlotMap[i] != VirtRegMap::NO_STACK_SLOT)
      OS << "[reg" << i << " -> fi#" << Virt2StackSlotMap[i] << "]\n";
  OS << '\n';
}

void VirtRegMap::dump() const { print(std::cerr); }


//===----------------------------------------------------------------------===//
// Simple Spiller Implementation
//===----------------------------------------------------------------------===//

Spiller::~Spiller() {}

namespace {
  struct SimpleSpiller : public Spiller {
    bool runOnMachineFunction(MachineFunction& mf, const VirtRegMap &VRM);
  };
}

bool SimpleSpiller::runOnMachineFunction(MachineFunction &MF,
                                         const VirtRegMap &VRM) {
  DEBUG(std::cerr << "********** REWRITE MACHINE CODE **********\n");
  DEBUG(std::cerr << "********** Function: "
                  << MF.getFunction()->getName() << '\n');
  const TargetMachine &TM = MF.getTarget();
  const MRegisterInfo &MRI = *TM.getRegisterInfo();
  bool *PhysRegsUsed = MF.getUsedPhysregs();

  // LoadedRegs - Keep track of which vregs are loaded, so that we only load
  // each vreg once (in the case where a spilled vreg is used by multiple
  // operands).  This is always smaller than the number of operands to the
  // current machine instr, so it should be small.
  std::vector<unsigned> LoadedRegs;

  for (MachineFunction::iterator MBBI = MF.begin(), E = MF.end();
       MBBI != E; ++MBBI) {
    DEBUG(std::cerr << MBBI->getBasicBlock()->getName() << ":\n");
    MachineBasicBlock &MBB = *MBBI;
    for (MachineBasicBlock::iterator MII = MBB.begin(),
           E = MBB.end(); MII != E; ++MII) {
      MachineInstr &MI = *MII;
      for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
        MachineOperand &MO = MI.getOperand(i);
        if (MO.isRegister() && MO.getReg())
          if (MRegisterInfo::isVirtualRegister(MO.getReg())) {
            unsigned VirtReg = MO.getReg();
            unsigned PhysReg = VRM.getPhys(VirtReg);
            if (VRM.hasStackSlot(VirtReg)) {
              int StackSlot = VRM.getStackSlot(VirtReg);

              if (MO.isUse() &&
                  std::find(LoadedRegs.begin(), LoadedRegs.end(), VirtReg)
                  == LoadedRegs.end()) {
                MRI.loadRegFromStackSlot(MBB, &MI, PhysReg, StackSlot);
                LoadedRegs.push_back(VirtReg);
                ++NumLoads;
                DEBUG(std::cerr << '\t' << *prior(MII));
              }

              if (MO.isDef()) {
                MRI.storeRegToStackSlot(MBB, next(MII), PhysReg, StackSlot);
                ++NumStores;
              }
            }
            PhysRegsUsed[PhysReg] = true;
            MI.SetMachineOperandReg(i, PhysReg);
          } else {
            PhysRegsUsed[MO.getReg()] = true;
          }
      }

      DEBUG(std::cerr << '\t' << MI);
      LoadedRegs.clear();
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
//  Local Spiller Implementation
//===----------------------------------------------------------------------===//

namespace {
  /// LocalSpiller - This spiller does a simple pass over the machine basic
  /// block to attempt to keep spills in registers as much as possible for
  /// blocks that have low register pressure (the vreg may be spilled due to
  /// register pressure in other blocks).
  class LocalSpiller : public Spiller {
    const MRegisterInfo *MRI;
    const TargetInstrInfo *TII;
  public:
    bool runOnMachineFunction(MachineFunction &MF, const VirtRegMap &VRM) {
      MRI = MF.getTarget().getRegisterInfo();
      TII = MF.getTarget().getInstrInfo();
      DEBUG(std::cerr << "\n**** Local spiller rewriting function '"
                      << MF.getFunction()->getName() << "':\n");

      for (MachineFunction::iterator MBB = MF.begin(), E = MF.end();
           MBB != E; ++MBB)
        RewriteMBB(*MBB, VRM);
      return true;
    }
  private:
    void RewriteMBB(MachineBasicBlock &MBB, const VirtRegMap &VRM);
    void ClobberPhysReg(unsigned PR, std::map<int, unsigned> &SpillSlots,
                        std::map<unsigned, int> &PhysRegs);
    void ClobberPhysRegOnly(unsigned PR, std::map<int, unsigned> &SpillSlots,
                            std::map<unsigned, int> &PhysRegs);
  };
}

void LocalSpiller::ClobberPhysRegOnly(unsigned PhysReg,
                                      std::map<int, unsigned> &SpillSlots,
                                      std::map<unsigned, int> &PhysRegs) {
  std::map<unsigned, int>::iterator I = PhysRegs.find(PhysReg);
  if (I != PhysRegs.end()) {
    int Slot = I->second;
    PhysRegs.erase(I);
    assert(SpillSlots[Slot] == PhysReg && "Bidirectional map mismatch!");
    SpillSlots.erase(Slot);
    DEBUG(std::cerr << "PhysReg " << MRI->getName(PhysReg)
          << " clobbered, invalidating SS#" << Slot << "\n");

  }
}

void LocalSpiller::ClobberPhysReg(unsigned PhysReg,
                                  std::map<int, unsigned> &SpillSlots,
                                  std::map<unsigned, int> &PhysRegs) {
  for (const unsigned *AS = MRI->getAliasSet(PhysReg); *AS; ++AS)
    ClobberPhysRegOnly(*AS, SpillSlots, PhysRegs);
  ClobberPhysRegOnly(PhysReg, SpillSlots, PhysRegs);
}


// ReusedOp - For each reused operand, we keep track of a bit of information, in
// case we need to rollback upon processing a new operand.  See comments below.
namespace {
  struct ReusedOp {
    // The MachineInstr operand that reused an available value.
    unsigned Operand;

    // StackSlot - The spill slot of the value being reused.
    unsigned StackSlot;

    // PhysRegReused - The physical register the value was available in.
    unsigned PhysRegReused;

    // AssignedPhysReg - The physreg that was assigned for use by the reload.
    unsigned AssignedPhysReg;

    ReusedOp(unsigned o, unsigned ss, unsigned prr, unsigned apr)
      : Operand(o), StackSlot(ss), PhysRegReused(prr), AssignedPhysReg(apr) {}
  };
}


/// rewriteMBB - Keep track of which spills are available even after the
/// register allocator is done with them.  If possible, avoid reloading vregs.
void LocalSpiller::RewriteMBB(MachineBasicBlock &MBB, const VirtRegMap &VRM) {

  // SpillSlotsAvailable - This map keeps track of all of the spilled virtual
  // register values that are still available, due to being loaded to stored to,
  // but not invalidated yet.
  std::map<int, unsigned> SpillSlotsAvailable;

  // PhysRegsAvailable - This is the inverse of SpillSlotsAvailable, indicating
  // which physregs are in use holding a stack slot value.
  std::map<unsigned, int> PhysRegsAvailable;

  DEBUG(std::cerr << MBB.getBasicBlock()->getName() << ":\n");

  std::vector<ReusedOp> ReusedOperands;

  // DefAndUseVReg - When we see a def&use operand that is spilled, keep track
  // of it.  ".first" is the machine operand index (should always be 0 for now),
  // and ".second" is the virtual register that is spilled.
  std::vector<std::pair<unsigned, unsigned> > DefAndUseVReg;

  // MaybeDeadStores - When we need to write a value back into a stack slot,
  // keep track of the inserted store.  If the stack slot value is never read
  // (because the value was used from some available register, for example), and
  // subsequently stored to, the original store is dead.  This map keeps track
  // of inserted stores that are not used.  If we see a subsequent store to the
  // same stack slot, the original store is deleted.
  std::map<int, MachineInstr*> MaybeDeadStores;

  bool *PhysRegsUsed = MBB.getParent()->getUsedPhysregs();

  for (MachineBasicBlock::iterator MII = MBB.begin(), E = MBB.end();
       MII != E; ) {
    MachineInstr &MI = *MII;
    MachineBasicBlock::iterator NextMII = MII; ++NextMII;

    ReusedOperands.clear();
    DefAndUseVReg.clear();

    // Process all of the spilled uses and all non spilled reg references.
    for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI.getOperand(i);
      if (!MO.isRegister() || MO.getReg() == 0)
        continue;   // Ignore non-register operands.
      
      if (MRegisterInfo::isPhysicalRegister(MO.getReg())) {
        // Ignore physregs for spilling, but remember that it is used by this
        // function.
        PhysRegsUsed[MO.getReg()] = true;
        continue;
      }
      
      assert(MRegisterInfo::isVirtualRegister(MO.getReg()) &&
             "Not a virtual or a physical register?");
      
      unsigned VirtReg = MO.getReg();
      if (!VRM.hasStackSlot(VirtReg)) {
        // This virtual register was assigned a physreg!
        unsigned Phys = VRM.getPhys(VirtReg);
        PhysRegsUsed[Phys] = true;
        MI.SetMachineOperandReg(i, Phys);
        continue;
      }
      
      // This virtual register is now known to be a spilled value.
      if (!MO.isUse())
        continue;  // Handle defs in the loop below (handle use&def here though)

      // If this is both a def and a use, we need to emit a store to the
      // stack slot after the instruction.  Keep track of D&U operands
      // because we are about to change it to a physreg here.
      if (MO.isDef()) {
        // Remember that this was a def-and-use operand, and that the
        // stack slot is live after this instruction executes.
        DefAndUseVReg.push_back(std::make_pair(i, VirtReg));
      }
      
      int StackSlot = VRM.getStackSlot(VirtReg);
      unsigned PhysReg;

      // Check to see if this stack slot is available.
      std::map<int, unsigned>::iterator SSI =
        SpillSlotsAvailable.find(StackSlot);
      if (SSI != SpillSlotsAvailable.end()) {
        DEBUG(std::cerr << "Reusing SS#" << StackSlot << " from physreg "
                        << MRI->getName(SSI->second) << " for vreg"
                        << VirtReg <<" instead of reloading into physreg "
                        << MRI->getName(VRM.getPhys(VirtReg)) << "\n");
        // If this stack slot value is already available, reuse it!
        PhysReg = SSI->second;
        MI.SetMachineOperandReg(i, PhysReg);

        // The only technical detail we have is that we don't know that
        // PhysReg won't be clobbered by a reloaded stack slot that occurs
        // later in the instruction.  In particular, consider 'op V1, V2'.
        // If V1 is available in physreg R0, we would choose to reuse it
        // here, instead of reloading it into the register the allocator
        // indicated (say R1).  However, V2 might have to be reloaded
        // later, and it might indicate that it needs to live in R0.  When
        // this occurs, we need to have information available that
        // indicates it is safe to use R1 for the reload instead of R0.
        //
        // To further complicate matters, we might conflict with an alias,
        // or R0 and R1 might not be compatible with each other.  In this
        // case, we actually insert a reload for V1 in R1, ensuring that
        // we can get at R0 or its alias.
        ReusedOperands.push_back(ReusedOp(i, StackSlot, PhysReg,
                                          VRM.getPhys(VirtReg)));
        ++NumReused;
        continue;
      }
      
      // Otherwise, reload it and remember that we have it.
      PhysReg = VRM.getPhys(VirtReg);

    RecheckRegister:
      // Note that, if we reused a register for a previous operand, the
      // register we want to reload into might not actually be
      // available.  If this occurs, use the register indicated by the
      // reuser.
      if (!ReusedOperands.empty())   // This is most often empty.
        for (unsigned ro = 0, e = ReusedOperands.size(); ro != e; ++ro)
          if (ReusedOperands[ro].PhysRegReused == PhysReg) {
            // Yup, use the reload register that we didn't use before.
            PhysReg = ReusedOperands[ro].AssignedPhysReg;
            goto RecheckRegister;
          } else {
            ReusedOp &Op = ReusedOperands[ro];
            unsigned PRRU = Op.PhysRegReused;
            if (MRI->areAliases(PRRU, PhysReg)) {
              // Okay, we found out that an alias of a reused register
              // was used.  This isn't good because it means we have
              // to undo a previous reuse.
              MRI->loadRegFromStackSlot(MBB, &MI, Op.AssignedPhysReg,
                                        Op.StackSlot);
              ClobberPhysReg(Op.AssignedPhysReg, SpillSlotsAvailable,
                             PhysRegsAvailable);

              // Any stores to this stack slot are not dead anymore.
              MaybeDeadStores.erase(Op.StackSlot);

              MI.SetMachineOperandReg(Op.Operand, Op.AssignedPhysReg);
              PhysRegsAvailable[Op.AssignedPhysReg] = Op.StackSlot;
              SpillSlotsAvailable[Op.StackSlot] = Op.AssignedPhysReg;
              PhysRegsAvailable.erase(Op.PhysRegReused);
              DEBUG(std::cerr << "Remembering SS#" << Op.StackSlot
                    << " in physreg "
                    << MRI->getName(Op.AssignedPhysReg) << "\n");
              ++NumLoads;
              DEBUG(std::cerr << '\t' << *prior(MII));

              DEBUG(std::cerr << "Reuse undone!\n");
              ReusedOperands.erase(ReusedOperands.begin()+ro);
              --NumReused;
              goto ContinueReload;
            }
          }
    ContinueReload:
      PhysRegsUsed[PhysReg] = true;
      MRI->loadRegFromStackSlot(MBB, &MI, PhysReg, StackSlot);
      // This invalidates PhysReg.
      ClobberPhysReg(PhysReg, SpillSlotsAvailable, PhysRegsAvailable);

      // Any stores to this stack slot are not dead anymore.
      MaybeDeadStores.erase(StackSlot);

      MI.SetMachineOperandReg(i, PhysReg);
      PhysRegsAvailable[PhysReg] = StackSlot;
      SpillSlotsAvailable[StackSlot] = PhysReg;
      DEBUG(std::cerr << "Remembering SS#" << StackSlot <<" in physreg "
                      << MRI->getName(PhysReg) << "\n");
      ++NumLoads;
      DEBUG(std::cerr << '\t' << *prior(MII));
    }

    // Loop over all of the implicit defs, clearing them from our available
    // sets.
    for (const unsigned *ImpDef = TII->getImplicitDefs(MI.getOpcode());
         *ImpDef; ++ImpDef) {
      PhysRegsUsed[*ImpDef] = true;
      ClobberPhysReg(*ImpDef, SpillSlotsAvailable, PhysRegsAvailable);
    }

    DEBUG(std::cerr << '\t' << MI);

    // If we have folded references to memory operands, make sure we clear all
    // physical registers that may contain the value of the spilled virtual
    // register
    VirtRegMap::MI2VirtMapTy::const_iterator I, End;
    for (tie(I, End) = VRM.getFoldedVirts(&MI); I != End; ++I) {
      DEBUG(std::cerr << "Folded vreg: " << I->second.first << "  MR: "
                      << I->second.second);
      unsigned VirtReg = I->second.first;
      VirtRegMap::ModRef MR = I->second.second;
      if (!VRM.hasStackSlot(VirtReg)) {
        DEBUG(std::cerr << ": No stack slot!\n");
        continue;
      }
      int SS = VRM.getStackSlot(VirtReg);
      DEBUG(std::cerr << " - StackSlot: " << SS << "\n");
      
      // If this folded instruction is just a use, check to see if it's a
      // straight load from the virt reg slot.
      if ((MR & VirtRegMap::isRef) && !(MR & VirtRegMap::isMod)) {
        int FrameIdx;
        if (unsigned DestReg = MRI->isLoadFromStackSlot(&MI, FrameIdx)) {
          // If this spill slot is available, insert a copy for it!
          std::map<int, unsigned>::iterator It = SpillSlotsAvailable.find(SS);
          if (FrameIdx == SS && It != SpillSlotsAvailable.end()) {
            DEBUG(std::cerr << "Promoted Load To Copy: " << MI);
            MachineFunction &MF = *MBB.getParent();
            if (DestReg != It->second) {
              MRI->copyRegToReg(MBB, &MI, DestReg, It->second,
                                MF.getSSARegMap()->getRegClass(VirtReg));
              // Revisit the copy if the destination is a vreg.
              if (MRegisterInfo::isVirtualRegister(DestReg)) {
                NextMII = &MI;
                --NextMII;  // backtrack to the copy.
              }
            }
            MBB.erase(&MI);
            goto ProcessNextInst;
          }
        }
      }

      // If this reference is not a use, any previous store is now dead.
      // Otherwise, the store to this stack slot is not dead anymore.
      std::map<int, MachineInstr*>::iterator MDSI = MaybeDeadStores.find(SS);
      if (MDSI != MaybeDeadStores.end()) {
        if (MR & VirtRegMap::isRef)   // Previous store is not dead.
          MaybeDeadStores.erase(MDSI);
        else {
          // If we get here, the store is dead, nuke it now.
          assert(MR == VirtRegMap::isMod && "Can't be modref!");
          MBB.erase(MDSI->second);
          MaybeDeadStores.erase(MDSI);
          ++NumDSE;
        }
      }

      // If the spill slot value is available, and this is a new definition of
      // the value, the value is not available anymore.
      if (MR & VirtRegMap::isMod) {
        std::map<int, unsigned>::iterator It = SpillSlotsAvailable.find(SS);
        if (It != SpillSlotsAvailable.end()) {
          PhysRegsAvailable.erase(It->second);
          SpillSlotsAvailable.erase(It);
        }
      }
    }

    // Process all of the spilled defs.
    for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI.getOperand(i);
      if (MO.isRegister() && MO.getReg() && MO.isDef()) {
        unsigned VirtReg = MO.getReg();

        bool TakenCareOf = false;
        if (!MRegisterInfo::isVirtualRegister(VirtReg)) {
          // Check to see if this is a def-and-use vreg operand that we do need
          // to insert a store for.
          bool OpTakenCareOf = false;
          if (MO.isUse() && !DefAndUseVReg.empty()) {
            for (unsigned dau = 0, e = DefAndUseVReg.size(); dau != e; ++dau)
              if (DefAndUseVReg[dau].first == i) {
                VirtReg = DefAndUseVReg[dau].second;
                OpTakenCareOf = true;
                break;
              }
          }

          if (!OpTakenCareOf) {
            ClobberPhysReg(VirtReg, SpillSlotsAvailable, PhysRegsAvailable);
            TakenCareOf = true;
          }
        }

        if (!TakenCareOf) {
          // The only vregs left are stack slot definitions.
          int StackSlot    = VRM.getStackSlot(VirtReg);
          unsigned PhysReg;

          // If this is a def&use operand, and we used a different physreg for
          // it than the one assigned, make sure to execute the store from the
          // correct physical register.
          if (MO.getReg() == VirtReg)
            PhysReg = VRM.getPhys(VirtReg);
          else
            PhysReg = MO.getReg();

          PhysRegsUsed[PhysReg] = true;
          MRI->storeRegToStackSlot(MBB, next(MII), PhysReg, StackSlot);
          DEBUG(std::cerr << "Store:\t" << *next(MII));
          MI.SetMachineOperandReg(i, PhysReg);

          // If there is a dead store to this stack slot, nuke it now.
          MachineInstr *&LastStore = MaybeDeadStores[StackSlot];
          if (LastStore) {
            DEBUG(std::cerr << " Killed store:\t" << *LastStore);
            ++NumDSE;
            MBB.erase(LastStore);
          }
          LastStore = next(MII);

          // If the stack slot value was previously available in some other
          // register, change it now.  Otherwise, make the register available,
          // in PhysReg.
          std::map<int, unsigned>::iterator SSA =
            SpillSlotsAvailable.find(StackSlot);
          if (SSA != SpillSlotsAvailable.end()) {
            // Remove the record for physreg.
            PhysRegsAvailable.erase(SSA->second);
            SpillSlotsAvailable.erase(SSA);
          }
          ClobberPhysReg(PhysReg, SpillSlotsAvailable, PhysRegsAvailable);

          PhysRegsAvailable[PhysReg] = StackSlot;
          SpillSlotsAvailable[StackSlot] = PhysReg;
          DEBUG(std::cerr << "Updating SS#" << StackSlot <<" in physreg "
                          << MRI->getName(PhysReg) << " for virtreg #"
                          << VirtReg << "\n");

          ++NumStores;
          VirtReg = PhysReg;
        }
      }
    }
  ProcessNextInst:
    MII = NextMII;
  }
}



llvm::Spiller* llvm::createSpiller() {
  switch (SpillerOpt) {
  default: assert(0 && "Unreachable!");
  case local:
    return new LocalSpiller();
  case simple:
    return new SimpleSpiller();
  }
}
