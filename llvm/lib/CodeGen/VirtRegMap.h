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
#include <map>

namespace llvm {
  class MachineInstr;

  class VirtRegMap {
  public:
    enum ModRef { isRef = 1, isMod = 2, isModRef = 3 };
    typedef std::multimap<MachineInstr*,
                          std::pair<unsigned, ModRef> > MI2VirtMapTy;

  private:
    MachineFunction &MF;
    /// Virt2PhysMap - This is a virtual to physical register
    /// mapping. Each virtual register is required to have an entry in
    /// it; even spilled virtual registers (the register mapped to a
    /// spilled register is the temporary used to load it from the
    /// stack).
    DenseMap<unsigned, VirtReg2IndexFunctor> Virt2PhysMap;
    /// Virt2StackSlotMap - This is virtual register to stack slot
    /// mapping. Each spilled virtual register has an entry in it
    /// which corresponds to the stack slot this register is spilled
    /// at.
    DenseMap<int, VirtReg2IndexFunctor> Virt2StackSlotMap;
    /// MI2VirtMap - This is MachineInstr to virtual register
    /// mapping. In the case of memory spill code being folded into
    /// instructions, we need to know which virtual register was
    /// read/written by this instruction.
    MI2VirtMapTy MI2VirtMap;

    VirtRegMap(const VirtRegMap&);     // DO NOT IMPLEMENT
    void operator=(const VirtRegMap&); // DO NOT IMPLEMENT

    enum {
      NO_PHYS_REG = 0,
      NO_STACK_SLOT = ~0 >> 1
    };

  public:
    VirtRegMap(MachineFunction &mf)
      : MF(mf), Virt2PhysMap(NO_PHYS_REG), Virt2StackSlotMap(NO_STACK_SLOT) {
      grow();
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

    /// @brief returns true is the specified virtual register is
    /// mapped to a stack slot
    bool hasStackSlot(unsigned virtReg) const {
      return getStackSlot(virtReg) != NO_STACK_SLOT;
    }

    /// @brief returns the stack slot mapped to the specified virtual
    /// register
    int getStackSlot(unsigned virtReg) const {
      assert(MRegisterInfo::isVirtualRegister(virtReg));
      return Virt2StackSlotMap[virtReg];
    }

    /// @brief create a mapping for the specifed virtual register to
    /// the next available stack slot
    int assignVirt2StackSlot(unsigned virtReg);
    /// @brief create a mapping for the specified virtual register to
    /// the specified stack slot
    void assignVirt2StackSlot(unsigned virtReg, int frameIndex);

    /// @brief Updates information about the specified virtual register's value
    /// folded into newMI machine instruction.  The OpNum argument indicates the
    /// operand number of OldMI that is folded.
    void virtFolded(unsigned VirtReg, MachineInstr *OldMI, unsigned OpNum,
                    MachineInstr *NewMI);

    /// @brief returns the virtual registers' values folded in memory
    /// operands of this instruction
    std::pair<MI2VirtMapTy::const_iterator, MI2VirtMapTy::const_iterator>
    getFoldedVirts(MachineInstr* MI) const {
      return MI2VirtMap.equal_range(MI);
    }
    
    /// RemoveFromFoldedVirtMap - Given a machine instruction in the folded
    /// instruction map, remove the entry in the folded instruction map.
    void RemoveFromFoldedVirtMap(MachineInstr *MI) {
      bool ErasedAny = MI2VirtMap.erase(MI);
      assert(ErasedAny && "Machine instr not in folded vreg map!");
    }

    void print(std::ostream &OS) const;
    void dump() const;
  };

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
