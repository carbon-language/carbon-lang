//===-- llvm/CodeGen/VirtRegMap.h - Virtual Register Map -*- C++ -*--------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a virtual register map. This maps virtual registers to
// physical registers and virtual registers to stack slots. It is created and
// updated by a register allocator and then used by a machine code rewriter that
// adds spill code and rewrites virtual into physical register references.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_VIRTREGMAP_H
#define LLVM_CODEGEN_VIRTREGMAP_H

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include <map>

namespace llvm {
  class LiveIntervals;
  class MachineInstr;
  class MachineFunction;
  class MachineRegisterInfo;
  class TargetInstrInfo;
  class TargetRegisterInfo;
  class raw_ostream;

  class VirtRegMap : public MachineFunctionPass {
  public:
    enum {
      NO_PHYS_REG = 0,
      NO_STACK_SLOT = (1L << 30)-1,
      MAX_STACK_SLOT = (1L << 18)-1
    };

    enum ModRef { isRef = 1, isMod = 2, isModRef = 3 };
    typedef std::multimap<MachineInstr*,
                          std::pair<unsigned, ModRef> > MI2VirtMapTy;

  private:
    MachineRegisterInfo *MRI;
    const TargetInstrInfo *TII;
    const TargetRegisterInfo *TRI;
    MachineFunction *MF;

    DenseMap<const TargetRegisterClass*, BitVector> allocatableRCRegs;

    /// Virt2PhysMap - This is a virtual to physical register
    /// mapping. Each virtual register is required to have an entry in
    /// it; even spilled virtual registers (the register mapped to a
    /// spilled register is the temporary used to load it from the
    /// stack).
    IndexedMap<unsigned, VirtReg2IndexFunctor> Virt2PhysMap;

    /// Virt2StackSlotMap - This is virtual register to stack slot
    /// mapping. Each spilled virtual register has an entry in it
    /// which corresponds to the stack slot this register is spilled
    /// at.
    IndexedMap<int, VirtReg2IndexFunctor> Virt2StackSlotMap;

    /// Virt2ReMatIdMap - This is virtual register to rematerialization id
    /// mapping. Each spilled virtual register that should be remat'd has an
    /// entry in it which corresponds to the remat id.
    IndexedMap<int, VirtReg2IndexFunctor> Virt2ReMatIdMap;

    /// Virt2SplitMap - This is virtual register to splitted virtual register
    /// mapping.
    IndexedMap<unsigned, VirtReg2IndexFunctor> Virt2SplitMap;

    /// Virt2SplitKillMap - This is splitted virtual register to its last use
    /// (kill) index mapping.
    IndexedMap<LiveIndex> Virt2SplitKillMap;

    /// ReMatMap - This is virtual register to re-materialized instruction
    /// mapping. Each virtual register whose definition is going to be
    /// re-materialized has an entry in it.
    IndexedMap<MachineInstr*, VirtReg2IndexFunctor> ReMatMap;

    /// MI2VirtMap - This is MachineInstr to virtual register
    /// mapping. In the case of memory spill code being folded into
    /// instructions, we need to know which virtual register was
    /// read/written by this instruction.
    MI2VirtMapTy MI2VirtMap;

    /// SpillPt2VirtMap - This records the virtual registers which should
    /// be spilled right after the MachineInstr due to live interval
    /// splitting.
    std::map<MachineInstr*, std::vector<std::pair<unsigned,bool> > >
    SpillPt2VirtMap;

    /// RestorePt2VirtMap - This records the virtual registers which should
    /// be restored right before the MachineInstr due to live interval
    /// splitting.
    std::map<MachineInstr*, std::vector<unsigned> > RestorePt2VirtMap;

    /// EmergencySpillMap - This records the physical registers that should
    /// be spilled / restored around the MachineInstr since the register
    /// allocator has run out of registers.
    std::map<MachineInstr*, std::vector<unsigned> > EmergencySpillMap;

    /// EmergencySpillSlots - This records emergency spill slots used to
    /// spill physical registers when the register allocator runs out of
    /// registers. Ideally only one stack slot is used per function per
    /// register class.
    std::map<const TargetRegisterClass*, int> EmergencySpillSlots;

    /// ReMatId - Instead of assigning a stack slot to a to be rematerialized
    /// virtual register, an unique id is being assigned. This keeps track of
    /// the highest id used so far. Note, this starts at (1<<18) to avoid
    /// conflicts with stack slot numbers.
    int ReMatId;

    /// LowSpillSlot, HighSpillSlot - Lowest and highest spill slot indexes.
    int LowSpillSlot, HighSpillSlot;

    /// SpillSlotToUsesMap - Records uses for each register spill slot.
    SmallVector<SmallPtrSet<MachineInstr*, 4>, 8> SpillSlotToUsesMap;

    /// ImplicitDefed - One bit for each virtual register. If set it indicates
    /// the register is implicitly defined.
    BitVector ImplicitDefed;

    /// UnusedRegs - A list of physical registers that have not been used.
    BitVector UnusedRegs;

    VirtRegMap(const VirtRegMap&);     // DO NOT IMPLEMENT
    void operator=(const VirtRegMap&); // DO NOT IMPLEMENT

  public:
    static char ID;
    VirtRegMap() : MachineFunctionPass(&ID), Virt2PhysMap(NO_PHYS_REG),
                   Virt2StackSlotMap(NO_STACK_SLOT), 
                   Virt2ReMatIdMap(NO_STACK_SLOT), Virt2SplitMap(0),
                   Virt2SplitKillMap(LiveIndex()), ReMatMap(NULL),
                   ReMatId(MAX_STACK_SLOT+1),
                   LowSpillSlot(NO_STACK_SLOT), HighSpillSlot(NO_STACK_SLOT) { }
    virtual bool runOnMachineFunction(MachineFunction &MF);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    void grow();

    /// @brief returns true if the specified virtual register is
    /// mapped to a physical register
    bool hasPhys(unsigned virtReg) const {
      return getPhys(virtReg) != NO_PHYS_REG;
    }

    /// @brief returns the physical register mapped to the specified
    /// virtual register
    unsigned getPhys(unsigned virtReg) const {
      assert(TargetRegisterInfo::isVirtualRegister(virtReg));
      return Virt2PhysMap[virtReg];
    }

    /// @brief creates a mapping for the specified virtual register to
    /// the specified physical register
    void assignVirt2Phys(unsigned virtReg, unsigned physReg) {
      assert(TargetRegisterInfo::isVirtualRegister(virtReg) &&
             TargetRegisterInfo::isPhysicalRegister(physReg));
      assert(Virt2PhysMap[virtReg] == NO_PHYS_REG &&
             "attempt to assign physical register to already mapped "
             "virtual register");
      Virt2PhysMap[virtReg] = physReg;
    }

    /// @brief clears the specified virtual register's, physical
    /// register mapping
    void clearVirt(unsigned virtReg) {
      assert(TargetRegisterInfo::isVirtualRegister(virtReg));
      assert(Virt2PhysMap[virtReg] != NO_PHYS_REG &&
             "attempt to clear a not assigned virtual register");
      Virt2PhysMap[virtReg] = NO_PHYS_REG;
    }

    /// @brief clears all virtual to physical register mappings
    void clearAllVirt() {
      Virt2PhysMap.clear();
      grow();
    }

    /// @brief returns the register allocation preference.
    unsigned getRegAllocPref(unsigned virtReg);

    /// @brief records virtReg is a split live interval from SReg.
    void setIsSplitFromReg(unsigned virtReg, unsigned SReg) {
      Virt2SplitMap[virtReg] = SReg;
    }

    /// @brief returns the live interval virtReg is split from.
    unsigned getPreSplitReg(unsigned virtReg) {
      return Virt2SplitMap[virtReg];
    }

    /// @brief returns true if the specified virtual register is not
    /// mapped to a stack slot or rematerialized.
    bool isAssignedReg(unsigned virtReg) const {
      if (getStackSlot(virtReg) == NO_STACK_SLOT &&
          getReMatId(virtReg) == NO_STACK_SLOT)
        return true;
      // Split register can be assigned a physical register as well as a
      // stack slot or remat id.
      return (Virt2SplitMap[virtReg] && Virt2PhysMap[virtReg] != NO_PHYS_REG);
    }

    /// @brief returns the stack slot mapped to the specified virtual
    /// register
    int getStackSlot(unsigned virtReg) const {
      assert(TargetRegisterInfo::isVirtualRegister(virtReg));
      return Virt2StackSlotMap[virtReg];
    }

    /// @brief returns the rematerialization id mapped to the specified virtual
    /// register
    int getReMatId(unsigned virtReg) const {
      assert(TargetRegisterInfo::isVirtualRegister(virtReg));
      return Virt2ReMatIdMap[virtReg];
    }

    /// @brief create a mapping for the specifed virtual register to
    /// the next available stack slot
    int assignVirt2StackSlot(unsigned virtReg);
    /// @brief create a mapping for the specified virtual register to
    /// the specified stack slot
    void assignVirt2StackSlot(unsigned virtReg, int frameIndex);

    /// @brief assign an unique re-materialization id to the specified
    /// virtual register.
    int assignVirtReMatId(unsigned virtReg);
    /// @brief assign an unique re-materialization id to the specified
    /// virtual register.
    void assignVirtReMatId(unsigned virtReg, int id);

    /// @brief returns true if the specified virtual register is being
    /// re-materialized.
    bool isReMaterialized(unsigned virtReg) const {
      return ReMatMap[virtReg] != NULL;
    }

    /// @brief returns the original machine instruction being re-issued
    /// to re-materialize the specified virtual register.
    MachineInstr *getReMaterializedMI(unsigned virtReg) const {
      return ReMatMap[virtReg];
    }

    /// @brief records the specified virtual register will be
    /// re-materialized and the original instruction which will be re-issed
    /// for this purpose.  If parameter all is true, then all uses of the
    /// registers are rematerialized and it's safe to delete the definition.
    void setVirtIsReMaterialized(unsigned virtReg, MachineInstr *def) {
      ReMatMap[virtReg] = def;
    }

    /// @brief record the last use (kill) of a split virtual register.
    void addKillPoint(unsigned virtReg, LiveIndex index) {
      Virt2SplitKillMap[virtReg] = index;
    }

    LiveIndex getKillPoint(unsigned virtReg) const {
      return Virt2SplitKillMap[virtReg];
    }

    /// @brief remove the last use (kill) of a split virtual register.
    void removeKillPoint(unsigned virtReg) {
      Virt2SplitKillMap[virtReg] = LiveIndex();
    }

    /// @brief returns true if the specified MachineInstr is a spill point.
    bool isSpillPt(MachineInstr *Pt) const {
      return SpillPt2VirtMap.find(Pt) != SpillPt2VirtMap.end();
    }

    /// @brief returns the virtual registers that should be spilled due to
    /// splitting right after the specified MachineInstr.
    std::vector<std::pair<unsigned,bool> > &getSpillPtSpills(MachineInstr *Pt) {
      return SpillPt2VirtMap[Pt];
    }

    /// @brief records the specified MachineInstr as a spill point for virtReg.
    void addSpillPoint(unsigned virtReg, bool isKill, MachineInstr *Pt) {
      std::map<MachineInstr*, std::vector<std::pair<unsigned,bool> > >::iterator
        I = SpillPt2VirtMap.find(Pt);
      if (I != SpillPt2VirtMap.end())
        I->second.push_back(std::make_pair(virtReg, isKill));
      else {
        std::vector<std::pair<unsigned,bool> > Virts;
        Virts.push_back(std::make_pair(virtReg, isKill));
        SpillPt2VirtMap.insert(std::make_pair(Pt, Virts));
      }
    }

    /// @brief - transfer spill point information from one instruction to
    /// another.
    void transferSpillPts(MachineInstr *Old, MachineInstr *New) {
      std::map<MachineInstr*, std::vector<std::pair<unsigned,bool> > >::iterator
        I = SpillPt2VirtMap.find(Old);
      if (I == SpillPt2VirtMap.end())
        return;
      while (!I->second.empty()) {
        unsigned virtReg = I->second.back().first;
        bool isKill = I->second.back().second;
        I->second.pop_back();
        addSpillPoint(virtReg, isKill, New);
      }
      SpillPt2VirtMap.erase(I);
    }

    /// @brief returns true if the specified MachineInstr is a restore point.
    bool isRestorePt(MachineInstr *Pt) const {
      return RestorePt2VirtMap.find(Pt) != RestorePt2VirtMap.end();
    }

    /// @brief returns the virtual registers that should be restoreed due to
    /// splitting right after the specified MachineInstr.
    std::vector<unsigned> &getRestorePtRestores(MachineInstr *Pt) {
      return RestorePt2VirtMap[Pt];
    }

    /// @brief records the specified MachineInstr as a restore point for virtReg.
    void addRestorePoint(unsigned virtReg, MachineInstr *Pt) {
      std::map<MachineInstr*, std::vector<unsigned> >::iterator I =
        RestorePt2VirtMap.find(Pt);
      if (I != RestorePt2VirtMap.end())
        I->second.push_back(virtReg);
      else {
        std::vector<unsigned> Virts;
        Virts.push_back(virtReg);
        RestorePt2VirtMap.insert(std::make_pair(Pt, Virts));
      }
    }

    /// @brief - transfer restore point information from one instruction to
    /// another.
    void transferRestorePts(MachineInstr *Old, MachineInstr *New) {
      std::map<MachineInstr*, std::vector<unsigned> >::iterator I =
        RestorePt2VirtMap.find(Old);
      if (I == RestorePt2VirtMap.end())
        return;
      while (!I->second.empty()) {
        unsigned virtReg = I->second.back();
        I->second.pop_back();
        addRestorePoint(virtReg, New);
      }
      RestorePt2VirtMap.erase(I);
    }

    /// @brief records that the specified physical register must be spilled
    /// around the specified machine instr.
    void addEmergencySpill(unsigned PhysReg, MachineInstr *MI) {
      if (EmergencySpillMap.find(MI) != EmergencySpillMap.end())
        EmergencySpillMap[MI].push_back(PhysReg);
      else {
        std::vector<unsigned> PhysRegs;
        PhysRegs.push_back(PhysReg);
        EmergencySpillMap.insert(std::make_pair(MI, PhysRegs));
      }
    }

    /// @brief returns true if one or more physical registers must be spilled
    /// around the specified instruction.
    bool hasEmergencySpills(MachineInstr *MI) const {
      return EmergencySpillMap.find(MI) != EmergencySpillMap.end();
    }

    /// @brief returns the physical registers to be spilled and restored around
    /// the instruction.
    std::vector<unsigned> &getEmergencySpills(MachineInstr *MI) {
      return EmergencySpillMap[MI];
    }

    /// @brief - transfer emergency spill information from one instruction to
    /// another.
    void transferEmergencySpills(MachineInstr *Old, MachineInstr *New) {
      std::map<MachineInstr*,std::vector<unsigned> >::iterator I =
        EmergencySpillMap.find(Old);
      if (I == EmergencySpillMap.end())
        return;
      while (!I->second.empty()) {
        unsigned virtReg = I->second.back();
        I->second.pop_back();
        addEmergencySpill(virtReg, New);
      }
      EmergencySpillMap.erase(I);
    }

    /// @brief return or get a emergency spill slot for the register class.
    int getEmergencySpillSlot(const TargetRegisterClass *RC);

    /// @brief Return lowest spill slot index.
    int getLowSpillSlot() const {
      return LowSpillSlot;
    }

    /// @brief Return highest spill slot index.
    int getHighSpillSlot() const {
      return HighSpillSlot;
    }

    /// @brief Records a spill slot use.
    void addSpillSlotUse(int FrameIndex, MachineInstr *MI);

    /// @brief Returns true if spill slot has been used.
    bool isSpillSlotUsed(int FrameIndex) const {
      assert(FrameIndex >= 0 && "Spill slot index should not be negative!");
      return !SpillSlotToUsesMap[FrameIndex-LowSpillSlot].empty();
    }

    /// @brief Mark the specified register as being implicitly defined.
    void setIsImplicitlyDefined(unsigned VirtReg) {
      ImplicitDefed.set(VirtReg-TargetRegisterInfo::FirstVirtualRegister);
    }

    /// @brief Returns true if the virtual register is implicitly defined.
    bool isImplicitlyDefined(unsigned VirtReg) const {
      return ImplicitDefed[VirtReg-TargetRegisterInfo::FirstVirtualRegister];
    }

    /// @brief Updates information about the specified virtual register's value
    /// folded into newMI machine instruction.
    void virtFolded(unsigned VirtReg, MachineInstr *OldMI, MachineInstr *NewMI,
                    ModRef MRInfo);

    /// @brief Updates information about the specified virtual register's value
    /// folded into the specified machine instruction.
    void virtFolded(unsigned VirtReg, MachineInstr *MI, ModRef MRInfo);

    /// @brief returns the virtual registers' values folded in memory
    /// operands of this instruction
    std::pair<MI2VirtMapTy::const_iterator, MI2VirtMapTy::const_iterator>
    getFoldedVirts(MachineInstr* MI) const {
      return MI2VirtMap.equal_range(MI);
    }
    
    /// RemoveMachineInstrFromMaps - MI is being erased, remove it from the
    /// the folded instruction map and spill point map.
    void RemoveMachineInstrFromMaps(MachineInstr *MI);

    /// FindUnusedRegisters - Gather a list of allocatable registers that
    /// have not been allocated to any virtual register.
    bool FindUnusedRegisters(LiveIntervals* LIs);

    /// HasUnusedRegisters - Return true if there are any allocatable registers
    /// that have not been allocated to any virtual register.
    bool HasUnusedRegisters() const {
      return !UnusedRegs.none();
    }

    /// setRegisterUsed - Remember the physical register is now used.
    void setRegisterUsed(unsigned Reg) {
      UnusedRegs.reset(Reg);
    }

    /// isRegisterUnused - Return true if the physical register has not been
    /// used.
    bool isRegisterUnused(unsigned Reg) const {
      return UnusedRegs[Reg];
    }

    /// getFirstUnusedRegister - Return the first physical register that has not
    /// been used.
    unsigned getFirstUnusedRegister(const TargetRegisterClass *RC) {
      int Reg = UnusedRegs.find_first();
      while (Reg != -1) {
        if (allocatableRCRegs[RC][Reg])
          return (unsigned)Reg;
        Reg = UnusedRegs.find_next(Reg);
      }
      return 0;
    }

    void print(raw_ostream &OS, const Module* M = 0) const;
    void dump() const;
  };

  inline raw_ostream &operator<<(raw_ostream &OS, const VirtRegMap &VRM) {
    VRM.print(OS);
    return OS;
  }
} // End llvm namespace

#endif
