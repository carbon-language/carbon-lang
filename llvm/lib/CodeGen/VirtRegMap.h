//===-- llvm/CodeGen/VirtRegMap.h - Virtual Register Map -*- C++ -*--------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a virtual register map. This maps virtual
// registers to physical registers and virtual registers to stack
// slots. It is created and updated by a register allocator and then
// used by a machine code rewriter that adds spill code and rewrites
// virtual into physical register references.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_VIRTREGMAP_H
#define LLVM_CODEGEN_VIRTREGMAP_H

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "Support/DenseMap.h"
#include <climits>

namespace llvm {

    class VirtRegMap {
    public:
        typedef DenseMap<unsigned, VirtReg2IndexFunctor> Virt2PhysMap;
        typedef DenseMap<int, VirtReg2IndexFunctor> Virt2StackSlotMap;

    private:
        MachineFunction* mf_;
        Virt2PhysMap v2pMap_;
        Virt2StackSlotMap v2ssMap_;

        // do not implement
        VirtRegMap(const VirtRegMap& rhs);
        const VirtRegMap& operator=(const VirtRegMap& rhs);

        enum {
            NO_PHYS_REG   = 0,
            NO_STACK_SLOT = INT_MAX
        };

    public:
        VirtRegMap(MachineFunction& mf)
            : mf_(&mf),
              v2pMap_(NO_PHYS_REG),
              v2ssMap_(NO_STACK_SLOT) {
            v2pMap_.grow(mf.getSSARegMap()->getLastVirtReg());
            v2ssMap_.grow(mf.getSSARegMap()->getLastVirtReg());
        }

        bool hasPhys(unsigned virtReg) const {
            return getPhys(virtReg) != NO_PHYS_REG;
        }

        unsigned getPhys(unsigned virtReg) const {
            assert(MRegisterInfo::isVirtualRegister(virtReg));
            return v2pMap_[virtReg];
        }

        void assignVirt2Phys(unsigned virtReg, unsigned physReg) {
            assert(MRegisterInfo::isVirtualRegister(virtReg) &&
                   MRegisterInfo::isPhysicalRegister(physReg));
            assert(v2pMap_[virtReg] == NO_PHYS_REG &&
                   "attempt to assign physical register to already mapped "
                   "virtual register");
            v2pMap_[virtReg] = physReg;
        }

        void clearVirt(unsigned virtReg) {
            assert(MRegisterInfo::isVirtualRegister(virtReg));
            assert(v2pMap_[virtReg] != NO_PHYS_REG &&
                   "attempt to clear a not assigned virtual register");
            v2pMap_[virtReg] = NO_PHYS_REG;
        }

        bool hasStackSlot(unsigned virtReg) const {
            return getStackSlot(virtReg) != NO_STACK_SLOT;
        }

        int getStackSlot(unsigned virtReg) const {
            assert(MRegisterInfo::isVirtualRegister(virtReg));
            return v2ssMap_[virtReg];
        }

        int assignVirt2StackSlot(unsigned virtReg);

        friend std::ostream& operator<<(std::ostream& os, const VirtRegMap& li);
    };

    std::ostream& operator<<(std::ostream& os, const VirtRegMap& li);

    void eliminateVirtRegs(MachineFunction& mf, const VirtRegMap& vrm);

} // End llvm namespace

#endif
