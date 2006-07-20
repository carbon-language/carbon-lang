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
#include "llvm/Support/Visibility.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <iostream>
using namespace llvm;

namespace {
  static Statistic<> NumSpills("spiller", "Number of register spills");
  static Statistic<> NumStores("spiller", "Number of stores added");
  static Statistic<> NumLoads ("spiller", "Number of loads added");
  static Statistic<> NumReused("spiller", "Number of values reused");
  static Statistic<> NumDSE   ("spiller", "Number of dead stores elided");
  static Statistic<> NumDCE   ("spiller", "Number of copies elided");

  enum SpillerName { simple, local };

  static cl::opt<SpillerName>
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
  struct VISIBILITY_HIDDEN SimpleSpiller : public Spiller {
    bool runOnMachineFunction(MachineFunction& mf, VirtRegMap &VRM);
  };
}

bool SimpleSpiller::runOnMachineFunction(MachineFunction &MF, VirtRegMap &VRM) {
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
              const TargetRegisterClass* RC =
                MF.getSSARegMap()->getRegClass(VirtReg);

              if (MO.isUse() &&
                  std::find(LoadedRegs.begin(), LoadedRegs.end(), VirtReg)
                  == LoadedRegs.end()) {
                MRI.loadRegFromStackSlot(MBB, &MI, PhysReg, StackSlot, RC);
                LoadedRegs.push_back(VirtReg);
                ++NumLoads;
                DEBUG(std::cerr << '\t' << *prior(MII));
              }

              if (MO.isDef()) {
                MRI.storeRegToStackSlot(MBB, next(MII), PhysReg, StackSlot, RC);
                ++NumStores;
              }
            }
            PhysRegsUsed[PhysReg] = true;
            MI.getOperand(i).setReg(PhysReg);
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
  class VISIBILITY_HIDDEN LocalSpiller : public Spiller {
    const MRegisterInfo *MRI;
    const TargetInstrInfo *TII;
  public:
    bool runOnMachineFunction(MachineFunction &MF, VirtRegMap &VRM) {
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
    void RewriteMBB(MachineBasicBlock &MBB, VirtRegMap &VRM);
    void ClobberPhysReg(unsigned PR, std::map<int, unsigned> &SpillSlots,
                        std::multimap<unsigned, int> &PhysRegs);
    void ClobberPhysRegOnly(unsigned PR, std::map<int, unsigned> &SpillSlots,
                            std::multimap<unsigned, int> &PhysRegs);
    void ModifyStackSlot(int Slot, std::map<int, unsigned> &SpillSlots,
                         std::multimap<unsigned, int> &PhysRegs);
  };
}

/// AvailableSpills - As the local spiller is scanning and rewriting an MBB from
/// top down, keep track of which spills slots are available in each register.
///
/// Note that not all physregs are created equal here.  In particular, some
/// physregs are reloads that we are allowed to clobber or ignore at any time.
/// Other physregs are values that the register allocated program is using that
/// we cannot CHANGE, but we can read if we like.  We keep track of this on a 
/// per-stack-slot basis as the low bit in the value of the SpillSlotsAvailable
/// entries.  The predicate 'canClobberPhysReg()' checks this bit and
/// addAvailable sets it if.
namespace {
class VISIBILITY_HIDDEN AvailableSpills {
  const MRegisterInfo *MRI;
  const TargetInstrInfo *TII;

  // SpillSlotsAvailable - This map keeps track of all of the spilled virtual
  // register values that are still available, due to being loaded or stored to,
  // but not invalidated yet.
  std::map<int, unsigned> SpillSlotsAvailable;
    
  // PhysRegsAvailable - This is the inverse of SpillSlotsAvailable, indicating
  // which stack slot values are currently held by a physreg.  This is used to
  // invalidate entries in SpillSlotsAvailable when a physreg is modified.
  std::multimap<unsigned, int> PhysRegsAvailable;
  
  void ClobberPhysRegOnly(unsigned PhysReg);
public:
  AvailableSpills(const MRegisterInfo *mri, const TargetInstrInfo *tii)
    : MRI(mri), TII(tii) {
  }
  
  /// getSpillSlotPhysReg - If the specified stack slot is available in a 
  /// physical register, return that PhysReg, otherwise return 0.
  unsigned getSpillSlotPhysReg(int Slot) const {
    std::map<int, unsigned>::const_iterator I = SpillSlotsAvailable.find(Slot);
    if (I != SpillSlotsAvailable.end())
      return I->second >> 1;  // Remove the CanClobber bit.
    return 0;
  }
  
  const MRegisterInfo *getRegInfo() const { return MRI; }

  /// addAvailable - Mark that the specified stack slot is available in the
  /// specified physreg.  If CanClobber is true, the physreg can be modified at
  /// any time without changing the semantics of the program.
  void addAvailable(int Slot, unsigned Reg, bool CanClobber = true) {
    // If this stack slot is thought to be available in some other physreg, 
    // remove its record.
    ModifyStackSlot(Slot);
    
    PhysRegsAvailable.insert(std::make_pair(Reg, Slot));
    SpillSlotsAvailable[Slot] = (Reg << 1) | (unsigned)CanClobber;
  
    DEBUG(std::cerr << "Remembering SS#" << Slot << " in physreg "
                    << MRI->getName(Reg) << "\n");
  }
  
  /// canClobberPhysReg - Return true if the spiller is allowed to change the 
  /// value of the specified stackslot register if it desires.  The specified
  /// stack slot must be available in a physreg for this query to make sense.
  bool canClobberPhysReg(int Slot) const {
    assert(SpillSlotsAvailable.count(Slot) && "Slot not available!");
    return SpillSlotsAvailable.find(Slot)->second & 1;
  }
  
  /// ClobberPhysReg - This is called when the specified physreg changes
  /// value.  We use this to invalidate any info about stuff we thing lives in
  /// it and any of its aliases.
  void ClobberPhysReg(unsigned PhysReg);

  /// ModifyStackSlot - This method is called when the value in a stack slot
  /// changes.  This removes information about which register the previous value
  /// for this slot lives in (as the previous value is dead now).
  void ModifyStackSlot(int Slot);
};
}

/// ClobberPhysRegOnly - This is called when the specified physreg changes
/// value.  We use this to invalidate any info about stuff we thing lives in it.
void AvailableSpills::ClobberPhysRegOnly(unsigned PhysReg) {
  std::multimap<unsigned, int>::iterator I =
    PhysRegsAvailable.lower_bound(PhysReg);
  while (I != PhysRegsAvailable.end() && I->first == PhysReg) {
    int Slot = I->second;
    PhysRegsAvailable.erase(I++);
    assert((SpillSlotsAvailable[Slot] >> 1) == PhysReg &&
           "Bidirectional map mismatch!");
    SpillSlotsAvailable.erase(Slot);
    DEBUG(std::cerr << "PhysReg " << MRI->getName(PhysReg)
                    << " clobbered, invalidating SS#" << Slot << "\n");
  }
}

/// ClobberPhysReg - This is called when the specified physreg changes
/// value.  We use this to invalidate any info about stuff we thing lives in
/// it and any of its aliases.
void AvailableSpills::ClobberPhysReg(unsigned PhysReg) {
  for (const unsigned *AS = MRI->getAliasSet(PhysReg); *AS; ++AS)
    ClobberPhysRegOnly(*AS);
  ClobberPhysRegOnly(PhysReg);
}

/// ModifyStackSlot - This method is called when the value in a stack slot
/// changes.  This removes information about which register the previous value
/// for this slot lives in (as the previous value is dead now).
void AvailableSpills::ModifyStackSlot(int Slot) {
  std::map<int, unsigned>::iterator It = SpillSlotsAvailable.find(Slot);
  if (It == SpillSlotsAvailable.end()) return;
  unsigned Reg = It->second >> 1;
  SpillSlotsAvailable.erase(It);
  
  // This register may hold the value of multiple stack slots, only remove this
  // stack slot from the set of values the register contains.
  std::multimap<unsigned, int>::iterator I = PhysRegsAvailable.lower_bound(Reg);
  for (; ; ++I) {
    assert(I != PhysRegsAvailable.end() && I->first == Reg &&
           "Map inverse broken!");
    if (I->second == Slot) break;
  }
  PhysRegsAvailable.erase(I);
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
    
    // VirtReg - The virtual register itself.
    unsigned VirtReg;

    ReusedOp(unsigned o, unsigned ss, unsigned prr, unsigned apr,
             unsigned vreg)
      : Operand(o), StackSlot(ss), PhysRegReused(prr), AssignedPhysReg(apr),
      VirtReg(vreg) {}
  };
  
  /// ReuseInfo - This maintains a collection of ReuseOp's for each operand that
  /// is reused instead of reloaded.
  class VISIBILITY_HIDDEN ReuseInfo {
    MachineInstr &MI;
    std::vector<ReusedOp> Reuses;
  public:
    ReuseInfo(MachineInstr &mi) : MI(mi) {}
    
    bool hasReuses() const {
      return !Reuses.empty();
    }
    
    /// addReuse - If we choose to reuse a virtual register that is already
    /// available instead of reloading it, remember that we did so.
    void addReuse(unsigned OpNo, unsigned StackSlot,
                  unsigned PhysRegReused, unsigned AssignedPhysReg,
                  unsigned VirtReg) {
      // If the reload is to the assigned register anyway, no undo will be
      // required.
      if (PhysRegReused == AssignedPhysReg) return;
      
      // Otherwise, remember this.
      Reuses.push_back(ReusedOp(OpNo, StackSlot, PhysRegReused, 
                                AssignedPhysReg, VirtReg));
    }
    
    /// GetRegForReload - We are about to emit a reload into PhysReg.  If there
    /// is some other operand that is using the specified register, either pick
    /// a new register to use, or evict the previous reload and use this reg. 
    unsigned GetRegForReload(unsigned PhysReg, MachineInstr *MI,
                             AvailableSpills &Spills,
                             std::map<int, MachineInstr*> &MaybeDeadStores) {
      if (Reuses.empty()) return PhysReg;  // This is most often empty.

      for (unsigned ro = 0, e = Reuses.size(); ro != e; ++ro) {
        ReusedOp &Op = Reuses[ro];
        // If we find some other reuse that was supposed to use this register
        // exactly for its reload, we can change this reload to use ITS reload
        // register.
        if (Op.PhysRegReused == PhysReg) {
          // Yup, use the reload register that we didn't use before.
          unsigned NewReg = Op.AssignedPhysReg;
          
          // Remove the record for the previous reuse.  We know it can never be
          // invalidated now.
          Reuses.erase(Reuses.begin()+ro);
          return GetRegForReload(NewReg, MI, Spills, MaybeDeadStores);
        } else {
          // Otherwise, we might also have a problem if a previously reused
          // value aliases the new register.  If so, codegen the previous reload
          // and use this one.          
          unsigned PRRU = Op.PhysRegReused;
          const MRegisterInfo *MRI = Spills.getRegInfo();
          if (MRI->areAliases(PRRU, PhysReg)) {
            // Okay, we found out that an alias of a reused register
            // was used.  This isn't good because it means we have
            // to undo a previous reuse.
            MachineBasicBlock *MBB = MI->getParent();
            const TargetRegisterClass *AliasRC =
              MBB->getParent()->getSSARegMap()->getRegClass(Op.VirtReg);

            // Copy Op out of the vector and remove it, we're going to insert an
            // explicit load for it.
            ReusedOp NewOp = Op;
            Reuses.erase(Reuses.begin()+ro);

            // Ok, we're going to try to reload the assigned physreg into the
            // slot that we were supposed to in the first place.  However, that
            // register could hold a reuse.  Check to see if it conflicts or
            // would prefer us to use a different register.
            unsigned NewPhysReg = GetRegForReload(NewOp.AssignedPhysReg,
                                                  MI, Spills, MaybeDeadStores);
            
            MRI->loadRegFromStackSlot(*MBB, MI, NewPhysReg,
                                      NewOp.StackSlot, AliasRC);
            Spills.ClobberPhysReg(NewPhysReg);
            Spills.ClobberPhysReg(NewOp.PhysRegReused);
            
            // Any stores to this stack slot are not dead anymore.
            MaybeDeadStores.erase(NewOp.StackSlot);
            
            MI->getOperand(NewOp.Operand).setReg(NewPhysReg);
            
            Spills.addAvailable(NewOp.StackSlot, NewPhysReg);
            ++NumLoads;
            DEBUG(MachineBasicBlock::iterator MII = MI;
                  std::cerr << '\t' << *prior(MII));
            
            DEBUG(std::cerr << "Reuse undone!\n");
            --NumReused;
            
            // Finally, PhysReg is now available, go ahead and use it.
            return PhysReg;
          }
        }
      }
      return PhysReg;
    }
  };
}


/// rewriteMBB - Keep track of which spills are available even after the
/// register allocator is done with them.  If possible, avoid reloading vregs.
void LocalSpiller::RewriteMBB(MachineBasicBlock &MBB, VirtRegMap &VRM) {

  DEBUG(std::cerr << MBB.getBasicBlock()->getName() << ":\n");

  // Spills - Keep track of which spilled values are available in physregs so
  // that we can choose to reuse the physregs instead of emitting reloads.
  AvailableSpills Spills(MRI, TII);
  
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

    /// ReusedOperands - Keep track of operand reuse in case we need to undo
    /// reuse.
    ReuseInfo ReusedOperands(MI);
    
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
        MI.getOperand(i).setReg(Phys);
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
      if ((PhysReg = Spills.getSpillSlotPhysReg(StackSlot))) {

        // Don't reuse it for a def&use operand if we aren't allowed to change
        // the physreg!
        if (!MO.isDef() || Spills.canClobberPhysReg(StackSlot)) {
          // If this stack slot value is already available, reuse it!
          DEBUG(std::cerr << "Reusing SS#" << StackSlot << " from physreg "
                          << MRI->getName(PhysReg) << " for vreg"
                          << VirtReg <<" instead of reloading into physreg "
                          << MRI->getName(VRM.getPhys(VirtReg)) << "\n");
          MI.getOperand(i).setReg(PhysReg);

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
          ReusedOperands.addReuse(i, StackSlot, PhysReg,
                                  VRM.getPhys(VirtReg), VirtReg);
          ++NumReused;
          continue;
        }
        
        // Otherwise we have a situation where we have a two-address instruction
        // whose mod/ref operand needs to be reloaded.  This reload is already
        // available in some register "PhysReg", but if we used PhysReg as the
        // operand to our 2-addr instruction, the instruction would modify
        // PhysReg.  This isn't cool if something later uses PhysReg and expects
        // to get its initial value.
        //
        // To avoid this problem, and to avoid doing a load right after a store,
        // we emit a copy from PhysReg into the designated register for this
        // operand.
        unsigned DesignatedReg = VRM.getPhys(VirtReg);
        assert(DesignatedReg && "Must map virtreg to physreg!");

        // Note that, if we reused a register for a previous operand, the
        // register we want to reload into might not actually be
        // available.  If this occurs, use the register indicated by the
        // reuser.
        if (ReusedOperands.hasReuses())
          DesignatedReg = ReusedOperands.GetRegForReload(DesignatedReg, &MI, 
                                                      Spills, MaybeDeadStores);
        
        // If the mapped designated register is actually the physreg we have
        // incoming, we don't need to inserted a dead copy.
        if (DesignatedReg == PhysReg) {
          // If this stack slot value is already available, reuse it!
          DEBUG(std::cerr << "Reusing SS#" << StackSlot << " from physreg "
                          << MRI->getName(PhysReg) << " for vreg"
                          << VirtReg
                          << " instead of reloading into same physreg.\n");
          MI.getOperand(i).setReg(PhysReg);
          ++NumReused;
          continue;
        }
        
        const TargetRegisterClass* RC =
          MBB.getParent()->getSSARegMap()->getRegClass(VirtReg);

        PhysRegsUsed[DesignatedReg] = true;
        MRI->copyRegToReg(MBB, &MI, DesignatedReg, PhysReg, RC);
        
        // This invalidates DesignatedReg.
        Spills.ClobberPhysReg(DesignatedReg);
        
        Spills.addAvailable(StackSlot, DesignatedReg);
        MI.getOperand(i).setReg(DesignatedReg);
        DEBUG(std::cerr << '\t' << *prior(MII));
        ++NumReused;
        continue;
      }
      
      // Otherwise, reload it and remember that we have it.
      PhysReg = VRM.getPhys(VirtReg);
      assert(PhysReg && "Must map virtreg to physreg!");
      const TargetRegisterClass* RC =
        MBB.getParent()->getSSARegMap()->getRegClass(VirtReg);

      // Note that, if we reused a register for a previous operand, the
      // register we want to reload into might not actually be
      // available.  If this occurs, use the register indicated by the
      // reuser.
      if (ReusedOperands.hasReuses())
        PhysReg = ReusedOperands.GetRegForReload(PhysReg, &MI, 
                                                 Spills, MaybeDeadStores);
      
      PhysRegsUsed[PhysReg] = true;
      MRI->loadRegFromStackSlot(MBB, &MI, PhysReg, StackSlot, RC);
      // This invalidates PhysReg.
      Spills.ClobberPhysReg(PhysReg);

      // Any stores to this stack slot are not dead anymore.
      MaybeDeadStores.erase(StackSlot);
      Spills.addAvailable(StackSlot, PhysReg);
      ++NumLoads;
      MI.getOperand(i).setReg(PhysReg);
      DEBUG(std::cerr << '\t' << *prior(MII));
    }

    // Loop over all of the implicit defs, clearing them from our available
    // sets.
    for (const unsigned *ImpDef = TII->getImplicitDefs(MI.getOpcode());
         *ImpDef; ++ImpDef) {
      PhysRegsUsed[*ImpDef] = true;
      Spills.ClobberPhysReg(*ImpDef);
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
        if (unsigned DestReg = TII->isLoadFromStackSlot(&MI, FrameIdx)) {
          // If this spill slot is available, turn it into a copy (or nothing)
          // instead of leaving it as a load!
          unsigned InReg;
          if (FrameIdx == SS && (InReg = Spills.getSpillSlotPhysReg(SS))) {
            DEBUG(std::cerr << "Promoted Load To Copy: " << MI);
            MachineFunction &MF = *MBB.getParent();
            if (DestReg != InReg) {
              MRI->copyRegToReg(MBB, &MI, DestReg, InReg,
                                MF.getSSARegMap()->getRegClass(VirtReg));
              // Revisit the copy so we make sure to notice the effects of the
              // operation on the destreg (either needing to RA it if it's 
              // virtual or needing to clobber any values if it's physical).
              NextMII = &MI;
              --NextMII;  // backtrack to the copy.
            }
            VRM.RemoveFromFoldedVirtMap(&MI);
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
          assert(VirtRegMap::isMod && "Can't be modref!");
          DEBUG(std::cerr << "Removed dead store:\t" << *MDSI->second);
          MBB.erase(MDSI->second);
          VRM.RemoveFromFoldedVirtMap(MDSI->second);
          MaybeDeadStores.erase(MDSI);
          ++NumDSE;
        }
      }

      // If the spill slot value is available, and this is a new definition of
      // the value, the value is not available anymore.
      if (MR & VirtRegMap::isMod) {
        // Notice that the value in this stack slot has been modified.
        Spills.ModifyStackSlot(SS);
        
        // If this is *just* a mod of the value, check to see if this is just a
        // store to the spill slot (i.e. the spill got merged into the copy). If
        // so, realize that the vreg is available now, and add the store to the
        // MaybeDeadStore info.
        int StackSlot;
        if (!(MR & VirtRegMap::isRef)) {
          if (unsigned SrcReg = TII->isStoreToStackSlot(&MI, StackSlot)) {
            assert(MRegisterInfo::isPhysicalRegister(SrcReg) &&
                   "Src hasn't been allocated yet?");
            // Okay, this is certainly a store of SrcReg to [StackSlot].  Mark
            // this as a potentially dead store in case there is a subsequent
            // store into the stack slot without a read from it.
            MaybeDeadStores[StackSlot] = &MI;

            // If the stack slot value was previously available in some other
            // register, change it now.  Otherwise, make the register available,
            // in PhysReg.
            Spills.addAvailable(StackSlot, SrcReg, false /*don't clobber*/);
          }
        }
      }
    }

    // Process all of the spilled defs.
    for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI.getOperand(i);
      if (MO.isRegister() && MO.getReg() && MO.isDef()) {
        unsigned VirtReg = MO.getReg();

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
            // Check to see if this is a noop copy.  If so, eliminate the
            // instruction before considering the dest reg to be changed.
            unsigned Src, Dst;
            if (TII->isMoveInstr(MI, Src, Dst) && Src == Dst) {
              ++NumDCE;
              DEBUG(std::cerr << "Removing now-noop copy: " << MI);
              MBB.erase(&MI);
              VRM.RemoveFromFoldedVirtMap(&MI);
              goto ProcessNextInst;
            }
            Spills.ClobberPhysReg(VirtReg);
            continue;
          }
        }

        // The only vregs left are stack slot definitions.
        int StackSlot = VRM.getStackSlot(VirtReg);
        const TargetRegisterClass *RC =
          MBB.getParent()->getSSARegMap()->getRegClass(VirtReg);
        unsigned PhysReg;

        // If this is a def&use operand, and we used a different physreg for
        // it than the one assigned, make sure to execute the store from the
        // correct physical register.
        if (MO.getReg() == VirtReg)
          PhysReg = VRM.getPhys(VirtReg);
        else
          PhysReg = MO.getReg();

        PhysRegsUsed[PhysReg] = true;
        MRI->storeRegToStackSlot(MBB, next(MII), PhysReg, StackSlot, RC);
        DEBUG(std::cerr << "Store:\t" << *next(MII));
        MI.getOperand(i).setReg(PhysReg);

        // Check to see if this is a noop copy.  If so, eliminate the
        // instruction before considering the dest reg to be changed.
        {
          unsigned Src, Dst;
          if (TII->isMoveInstr(MI, Src, Dst) && Src == Dst) {
            ++NumDCE;
            DEBUG(std::cerr << "Removing now-noop copy: " << MI);
            MBB.erase(&MI);
            VRM.RemoveFromFoldedVirtMap(&MI);
            goto ProcessNextInst;
          }
        }
        
        // If there is a dead store to this stack slot, nuke it now.
        MachineInstr *&LastStore = MaybeDeadStores[StackSlot];
        if (LastStore) {
          DEBUG(std::cerr << "Removed dead store:\t" << *LastStore);
          ++NumDSE;
          MBB.erase(LastStore);
          VRM.RemoveFromFoldedVirtMap(LastStore);
        }
        LastStore = next(MII);

        // If the stack slot value was previously available in some other
        // register, change it now.  Otherwise, make the register available,
        // in PhysReg.
        Spills.ModifyStackSlot(StackSlot);
        Spills.ClobberPhysReg(PhysReg);
        Spills.addAvailable(StackSlot, PhysReg);
        ++NumStores;
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
