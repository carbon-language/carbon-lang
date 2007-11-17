//===-- llvm/CodeGen/VirtRegMap.h - Virtual Register Map -*- C++ -*--------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
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

#include "llvm/Target/MRegisterInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/Support/Streams.h"
#include <map>

namespace llvm {
  class MachineInstr;
  class MachineFunction;
  class TargetInstrInfo;

  class VirtRegMap {
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
    const TargetInstrInfo &TII;

    MachineFunction &MF;
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

    /// Virt2StackSlotMap - This is virtual register to rematerialization id
    /// mapping. Each spilled virtual register that should be remat'd has an
    /// entry in it which corresponds to the remat id.
    IndexedMap<int, VirtReg2IndexFunctor> Virt2ReMatIdMap;

    /// Virt2SplitMap - This is virtual register to splitted virtual register
    /// mapping.
    IndexedMap<unsigned, VirtReg2IndexFunctor> Virt2SplitMap;

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
    DenseMap<MachineInstr*, std::vector<unsigned> > SpillPt2VirtMap;

    /// Virt2SplitMap - This records the MachineInstrs where a virtual
    /// register should be spilled due to live interval splitting.
    IndexedMap<std::vector<MachineInstr*>, VirtReg2IndexFunctor>
    Virt2SpillPtsMap;

    /// ReMatId - Instead of assigning a stack slot to a to be rematerialized
    /// virtual register, an unique id is being assigned. This keeps track of
    /// the highest id used so far. Note, this starts at (1<<18) to avoid
    /// conflicts with stack slot numbers.
    int ReMatId;

    VirtRegMap(const VirtRegMap&);     // DO NOT IMPLEMENT
    void operator=(const VirtRegMap&); // DO NOT IMPLEMENT

  public:
    explicit VirtRegMap(MachineFunction &mf);

    void grow();

    /// @brief returns true if the specified virtual register is
    /// mapped to a physical register
    bool hasPhys(unsigned virtReg) const {
      return getPhys(virtReg) != NO_PHYS_REG;
    }

    /// @brief returns the physical register mapped to the specified
    /// virtual register
    unsigned getPhys(unsigned virtReg) const {
      assert(MRegisterInfo::isVirtualRegister(virtReg));
      return Virt2PhysMap[virtReg];
    }

    /// @brief creates a mapping for the specified virtual register to
    /// the specified physical register
    void assignVirt2Phys(unsigned virtReg, unsigned physReg) {
      assert(MRegisterInfo::isVirtualRegister(virtReg) &&
             MRegisterInfo::isPhysicalRegister(physReg));
      assert(Virt2PhysMap[virtReg] == NO_PHYS_REG &&
             "attempt to assign physical register to already mapped "
             "virtual register");
      Virt2PhysMap[virtReg] = physReg;
    }

    /// @brief clears the specified virtual register's, physical
    /// register mapping
    void clearVirt(unsigned virtReg) {
      assert(MRegisterInfo::isVirtualRegister(virtReg));
      assert(Virt2PhysMap[virtReg] != NO_PHYS_REG &&
             "attempt to clear a not assigned virtual register");
      Virt2PhysMap[virtReg] = NO_PHYS_REG;
    }

    /// @brief clears all virtual to physical register mappings
    void clearAllVirt() {
      Virt2PhysMap.clear();
      grow();
    }

    /// @brief records virtReg is a split live interval from SReg.
    void setIsSplitFromReg(unsigned virtReg, unsigned SReg) {
      Virt2SplitMap[virtReg] = SReg;
    }

    /// @brief returns the live interval virtReg is split from.
    unsigned getPreSplitReg(unsigned virtReg) {
      return Virt2SplitMap[virtReg];
    }

    /// @brief returns true is the specified virtual register is not
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
      assert(MRegisterInfo::isVirtualRegister(virtReg));
      return Virt2StackSlotMap[virtReg];
    }

    /// @brief returns the rematerialization id mapped to the specified virtual
    /// register
    int getReMatId(unsigned virtReg) const {
      assert(MRegisterInfo::isVirtualRegister(virtReg));
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

    /// @brief returns the virtual registers that should be spilled due to
    /// splitting right after the specified MachineInstr.
    std::vector<unsigned> &getSpillPtSpills(MachineInstr *Pt) {
      return SpillPt2VirtMap[Pt];
    }

    /// @brief records the specified MachineInstr as a spill point for virtReg.
    void addSpillPoint(unsigned virtReg, MachineInstr *Pt) {
      SpillPt2VirtMap[Pt].push_back(virtReg);
      Virt2SpillPtsMap[virtReg].push_back(Pt);
    }

    /// @brief remove the virtReg from the list of registers that should be
    /// spilled (due to splitting) right after the specified MachineInstr.
    void removeRegFromSpillPt(MachineInstr *Pt, unsigned virtReg) {
      std::vector<unsigned> &Regs = SpillPt2VirtMap[Pt];
      if (Regs.back() == virtReg) // Most common case.
        Regs.pop_back();
      for (unsigned i = 0, e = Regs.size(); i != e; ++i)
        if (Regs[i] == virtReg) {
          Regs.erase(Regs.begin()+i-1);
          break;
        }
    }

    /// @brief specify virtReg is no longer being spilled due to splitting.
    void removeAllSpillPtsForReg(unsigned virtReg) {
      std::vector<MachineInstr*> &SpillPts = Virt2SpillPtsMap[virtReg];
      for (unsigned i = 0, e = SpillPts.size(); i != e; ++i)
        removeRegFromSpillPt(SpillPts[i], virtReg);
      Virt2SpillPtsMap[virtReg].clear();
    }

    /// @brief remove the specified MachineInstr as a spill point for the
    /// specified register.
    void removeRegSpillPt(unsigned virtReg, MachineInstr *Pt) {
      std::vector<MachineInstr*> &SpillPts = Virt2SpillPtsMap[virtReg];
      if (SpillPts.back() == Pt) // Most common case.
        SpillPts.pop_back();
      for (unsigned i = 0, e = SpillPts.size(); i != e; ++i)
        if (SpillPts[i] == Pt) {
          SpillPts.erase(SpillPts.begin()+i-1);
          break;
        }
    }

    void transferSpillPts(MachineInstr *Old, MachineInstr *New) {
      std::vector<unsigned> &OldRegs = SpillPt2VirtMap[Old];
      while (!OldRegs.empty()) {
        unsigned virtReg = OldRegs.back();
        OldRegs.pop_back();
        removeRegSpillPt(virtReg, Old);
        addSpillPoint(virtReg, New);
      }
    }

    /// @brief Updates information about the specified virtual register's value
    /// folded into newMI machine instruction.  The OpNum argument indicates the
    /// operand number of OldMI that is folded.
    void virtFolded(unsigned VirtReg, MachineInstr *OldMI, unsigned OpNum,
                    MachineInstr *NewMI);

    /// @brief Updates information about the specified virtual register's value
    /// folded into the specified machine instruction.
    void virtFolded(unsigned VirtReg, MachineInstr *MI, ModRef MRInfo);

    /// @brief returns the virtual registers' values folded in memory
    /// operands of this instruction
    std::pair<MI2VirtMapTy::const_iterator, MI2VirtMapTy::const_iterator>
    getFoldedVirts(MachineInstr* MI) const {
      return MI2VirtMap.equal_range(MI);
    }
    
    /// RemoveFromFoldedVirtMap - If the specified machine instruction is in
    /// the folded instruction map, remove its entry from the map.
    void RemoveFromFoldedVirtMap(MachineInstr *MI) {
      MI2VirtMap.erase(MI);
    }

    void print(std::ostream &OS) const;
    void print(std::ostream *OS) const { if (OS) print(*OS); }
    void dump() const;
  };

  inline std::ostream *operator<<(std::ostream *OS, const VirtRegMap &VRM) {
    VRM.print(OS);
    return OS;
  }
  inline std::ostream &operator<<(std::ostream &OS, const VirtRegMap &VRM) {
    VRM.print(OS);
    return OS;
  }

  /// Spiller interface: Implementations of this interface assign spilled
  /// virtual registers to stack slots, rewriting the code.
  struct Spiller {
    virtual ~Spiller();
    virtual bool runOnMachineFunction(MachineFunction &MF,
                                      VirtRegMap &VRM) = 0;
  };

  /// createSpiller - Create an return a spiller object, as specified on the
  /// command line.
  Spiller* createSpiller();

} // End llvm namespace

#endif
