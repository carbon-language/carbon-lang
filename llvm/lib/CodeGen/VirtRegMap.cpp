//===-- llvm/CodeGen/VirtRegMap.cpp - Virtual Register Map ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the virtual register map. It also implements
// the eliminateVirtRegs() function that given a virtual register map
// and a machine function it eliminates all virtual references by
// replacing them with physical register references and adds spill
// code as necessary.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "regalloc"
#include "VirtRegMap.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "Support/Statistic.h"
#include "Support/Debug.h"
#include "Support/STLExtras.h"
#include <iostream>

using namespace llvm;

namespace {
    Statistic<> numSpills("spiller", "Number of register spills");
    Statistic<> numStores("spiller", "Number of stores added");
    Statistic<> numLoads ("spiller", "Number of loads added");

}

int VirtRegMap::assignVirt2StackSlot(unsigned virtReg)
{
    assert(MRegisterInfo::isVirtualRegister(virtReg));
    assert(v2ssMap_[toIndex(virtReg)] == NO_STACK_SLOT &&
           "attempt to assign stack slot to already spilled register");
    const TargetRegisterClass* rc =
        mf_->getSSARegMap()->getRegClass(virtReg);
    int frameIndex = mf_->getFrameInfo()->CreateStackObject(rc);
    v2ssMap_[toIndex(virtReg)] = frameIndex;
    ++numSpills;
    return frameIndex;
}

std::ostream& llvm::operator<<(std::ostream& os, const VirtRegMap& vrm)
{
    const MRegisterInfo* mri = vrm.mf_->getTarget().getRegisterInfo();

    std::cerr << "********** REGISTER MAP **********\n";
    for (unsigned i = 0, e = vrm.v2pMap_.size(); i != e; ++i) {
        if (vrm.v2pMap_[i] != VirtRegMap::NO_PHYS_REG)
            std::cerr << "[reg" << VirtRegMap::fromIndex(i) << " -> "
                      << mri->getName(vrm.v2pMap_[i]) << "]\n";
    }
    for (unsigned i = 0, e = vrm.v2ssMap_.size(); i != e; ++i) {
        if (vrm.v2ssMap_[i] != VirtRegMap::NO_STACK_SLOT)
            std::cerr << "[reg" << VirtRegMap::fromIndex(i) << " -> fi#"
                      << vrm.v2ssMap_[i] << "]\n";
    }
    return std::cerr << '\n';
}

namespace {

    class Spiller {
        typedef std::vector<unsigned> Phys2VirtMap;
        typedef std::vector<bool> PhysFlag;

        MachineFunction& mf_;
        const TargetMachine& tm_;
        const TargetInstrInfo& tii_;
        const MRegisterInfo& mri_;
        const VirtRegMap& vrm_;
        Phys2VirtMap p2vMap_;
        PhysFlag dirty_;

    public:
        Spiller(MachineFunction& mf, const VirtRegMap& vrm)
            : mf_(mf),
              tm_(mf_.getTarget()),
              tii_(tm_.getInstrInfo()),
              mri_(*tm_.getRegisterInfo()),
              vrm_(vrm),
              p2vMap_(mri_.getNumRegs()),
              dirty_(mri_.getNumRegs()) {
            DEBUG(std::cerr << "********** REWRITE MACHINE CODE **********\n");
            DEBUG(std::cerr << "********** Function: "
                  << mf_.getFunction()->getName() << '\n');
        }

        void eliminateVirtRegs() {
            for (MachineFunction::iterator mbbi = mf_.begin(),
                     mbbe = mf_.end(); mbbi != mbbe; ++mbbi) {
                // clear map and dirty flag
                p2vMap_.assign(p2vMap_.size(), 0);
                dirty_.assign(dirty_.size(), false);
                DEBUG(std::cerr << mbbi->getBasicBlock()->getName() << ":\n");
                eliminateVirtRegsInMbb(*mbbi);
            }
        }

    private:
        void vacateJustPhysReg(MachineBasicBlock& mbb,
                               MachineBasicBlock::iterator mii,
                               unsigned physReg) {
            unsigned virtReg = p2vMap_[physReg];
            if (dirty_[physReg] && vrm_.hasStackSlot(virtReg)) {
                mri_.storeRegToStackSlot(mbb, mii, physReg,
                                         vrm_.getStackSlot(virtReg),
                                         mri_.getRegClass(physReg));
                ++numStores;
                DEBUG(std::cerr << "*\t"; prior(mii)->print(std::cerr, tm_));
            }
            p2vMap_[physReg] = 0;
            dirty_[physReg] = false;
        }

        void vacatePhysReg(MachineBasicBlock& mbb,
                           MachineBasicBlock::iterator mii,
                           unsigned physReg) {
            vacateJustPhysReg(mbb, mii, physReg);
            for (const unsigned* as = mri_.getAliasSet(physReg); *as; ++as)
                vacateJustPhysReg(mbb, mii, *as);
        }

        void handleUse(MachineBasicBlock& mbb,
                       MachineBasicBlock::iterator mii,
                       unsigned virtReg,
                       unsigned physReg) {
            // check if we are replacing a previous mapping
            if (p2vMap_[physReg] != virtReg) {
                vacatePhysReg(mbb, mii, physReg);
                p2vMap_[physReg] = virtReg;
                // load if necessary
                if (vrm_.hasStackSlot(virtReg)) {
                    mri_.loadRegFromStackSlot(mbb, mii, physReg,
                                              vrm_.getStackSlot(virtReg),
                                              mri_.getRegClass(physReg));
                    ++numLoads;
                    DEBUG(std::cerr << "*\t"; prior(mii)->print(std::cerr,tm_));
                }
            }
        }

        void handleDef(MachineBasicBlock& mbb,
                       MachineBasicBlock::iterator mii,
                       unsigned virtReg,
                       unsigned physReg) {
            // check if we are replacing a previous mapping
            if (p2vMap_[physReg] != virtReg)
                vacatePhysReg(mbb, mii, physReg);

            p2vMap_[physReg] = virtReg;
            dirty_[physReg] = true;
        }

        void eliminateVirtRegsInMbb(MachineBasicBlock& mbb) {
            for (MachineBasicBlock::iterator mii = mbb.begin(),
                     mie = mbb.end(); mii != mie; ++mii) {
                // rewrite all used operands
                for (unsigned i = 0, e = mii->getNumOperands(); i != e; ++i) {
                    MachineOperand& op = mii->getOperand(i);
                    if (op.isRegister() && op.isUse() &&
                        MRegisterInfo::isVirtualRegister(op.getReg())) {
                        unsigned physReg = vrm_.getPhys(op.getReg());
                        handleUse(mbb, mii, op.getReg(), physReg);
                        mii->SetMachineOperandReg(i, physReg);
                        // mark as dirty if this is def&use
                        if (op.isDef()) dirty_[physReg] = true;
                    }
                }

                // spill implicit defs
                const TargetInstrDescriptor& tid =tii_.get(mii->getOpcode());
                for (const unsigned* id = tid.ImplicitDefs; *id; ++id)
                    vacatePhysReg(mbb, mii, *id);

                // rewrite def operands (def&use was handled with the
                // uses so don't check for those here)
                for (unsigned i = 0, e = mii->getNumOperands(); i != e; ++i) {
                    MachineOperand& op = mii->getOperand(i);
                    if (op.isRegister() && !op.isUse())
                        if (MRegisterInfo::isPhysicalRegister(op.getReg()))
                            vacatePhysReg(mbb, mii, op.getReg());
                        else {
                            unsigned physReg = vrm_.getPhys(op.getReg());
                            handleDef(mbb, mii, op.getReg(), physReg);
                            mii->SetMachineOperandReg(i, physReg);
                        }
                }

                DEBUG(std::cerr << '\t'; mii->print(std::cerr, tm_));
            }

            for (unsigned i = 1, e = p2vMap_.size(); i != e; ++i)
                vacateJustPhysReg(mbb, mbb.getFirstTerminator(), i);

        }
    };
}


void llvm::eliminateVirtRegs(MachineFunction& mf, const VirtRegMap& vrm)
{
    Spiller(mf, vrm).eliminateVirtRegs();
}
