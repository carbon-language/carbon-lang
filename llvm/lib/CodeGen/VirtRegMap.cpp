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
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "Support/CommandLine.h"
#include "Support/Debug.h"
#include "Support/DenseMap.h"
#include "Support/Statistic.h"
#include "Support/STLExtras.h"
#include <iostream>

using namespace llvm;

namespace {
    Statistic<> numSpills("spiller", "Number of register spills");
    Statistic<> numStores("spiller", "Number of stores added");
    Statistic<> numLoads ("spiller", "Number of loads added");

    enum SpillerName { simple, local };

    cl::opt<SpillerName>
    SpillerOpt("spiller",
               cl::desc("Spiller to use: (default: local)"),
               cl::Prefix,
               cl::values(clEnumVal(simple, "  simple spiller"),
                          clEnumVal(local,  "  local spiller"),
                          0),
               cl::init(local));
}

int VirtRegMap::assignVirt2StackSlot(unsigned virtReg)
{
    assert(MRegisterInfo::isVirtualRegister(virtReg));
    assert(v2ssMap_[virtReg] == NO_STACK_SLOT &&
           "attempt to assign stack slot to already spilled register");
    const TargetRegisterClass* rc =
        mf_->getSSARegMap()->getRegClass(virtReg);
    int frameIndex = mf_->getFrameInfo()->CreateStackObject(rc);
    v2ssMap_[virtReg] = frameIndex;
    ++numSpills;
    return frameIndex;
}

void VirtRegMap::assignVirt2StackSlot(unsigned virtReg, int frameIndex)
{
    assert(MRegisterInfo::isVirtualRegister(virtReg));
    assert(v2ssMap_[virtReg] == NO_STACK_SLOT &&
           "attempt to assign stack slot to already spilled register");
     v2ssMap_[virtReg] = frameIndex;
}

void VirtRegMap::virtFolded(unsigned virtReg,
                            MachineInstr* oldMI,
                            MachineInstr* newMI)
{
    // move previous memory references folded to new instruction
    MI2VirtMap::iterator i, e;
    std::vector<MI2VirtMap::mapped_type> regs;
    for (tie(i, e) = mi2vMap_.equal_range(oldMI); i != e; ) {
        regs.push_back(i->second);
        mi2vMap_.erase(i++);
    }
    for (unsigned i = 0, e = regs.size(); i != e; ++i)
        mi2vMap_.insert(std::make_pair(newMI, i));

    // add new memory reference
    mi2vMap_.insert(std::make_pair(newMI, virtReg));
}

std::ostream& llvm::operator<<(std::ostream& os, const VirtRegMap& vrm)
{
    const MRegisterInfo* mri = vrm.mf_->getTarget().getRegisterInfo();

    std::cerr << "********** REGISTER MAP **********\n";
    for (unsigned i = MRegisterInfo::FirstVirtualRegister,
             e = vrm.mf_->getSSARegMap()->getLastVirtReg(); i <= e; ++i) {
        if (vrm.v2pMap_[i] != VirtRegMap::NO_PHYS_REG)
            std::cerr << "[reg" << i << " -> "
                      << mri->getName(vrm.v2pMap_[i]) << "]\n";
    }
    for (unsigned i = MRegisterInfo::FirstVirtualRegister,
             e = vrm.mf_->getSSARegMap()->getLastVirtReg(); i <= e; ++i) {
        if (vrm.v2ssMap_[i] != VirtRegMap::NO_STACK_SLOT)
            std::cerr << "[reg" << i << " -> fi#"
                      << vrm.v2ssMap_[i] << "]\n";
    }
    return std::cerr << '\n';
}

Spiller::~Spiller()
{

}

namespace {

    class SimpleSpiller : public Spiller {
    public:
        bool runOnMachineFunction(MachineFunction& mf, const VirtRegMap& vrm) {
            DEBUG(std::cerr << "********** REWRITE MACHINE CODE **********\n");
            DEBUG(std::cerr << "********** Function: "
              << mf.getFunction()->getName() << '\n');
            const TargetMachine& tm = mf.getTarget();
            const MRegisterInfo& mri = *tm.getRegisterInfo();

            typedef DenseMap<bool, VirtReg2IndexFunctor> Loaded;
            Loaded loaded;

            for (MachineFunction::iterator mbbi = mf.begin(),
                     mbbe = mf.end(); mbbi != mbbe; ++mbbi) {
                DEBUG(std::cerr << mbbi->getBasicBlock()->getName() << ":\n");
                for (MachineBasicBlock::iterator mii = mbbi->begin(),
                         mie = mbbi->end(); mii != mie; ++mii) {
                    loaded.grow(mf.getSSARegMap()->getLastVirtReg());
                    for (unsigned i = 0,e = mii->getNumOperands(); i != e; ++i){
                        MachineOperand& mop = mii->getOperand(i);
                        if (mop.isRegister() && mop.getReg() &&
                            MRegisterInfo::isVirtualRegister(mop.getReg())) {
                            unsigned virtReg = mop.getReg();
                            unsigned physReg = vrm.getPhys(virtReg);
                            if (mop.isUse() &&
                                vrm.hasStackSlot(mop.getReg()) &&
                                !loaded[virtReg]) {
                                mri.loadRegFromStackSlot(
                                    *mbbi,
                                    mii,
                                    physReg,
                                    vrm.getStackSlot(virtReg),
                                    mf.getSSARegMap()->getRegClass(virtReg));
                                loaded[virtReg] = true;
                                DEBUG(std::cerr << '\t';
                                      prior(mii)->print(std::cerr, tm));
                                ++numLoads;
                            }
                            if (mop.isDef() &&
                                vrm.hasStackSlot(mop.getReg())) {
                                mri.storeRegToStackSlot(
                                    *mbbi,
                                    next(mii),
                                    physReg,
                                    vrm.getStackSlot(virtReg),
                                    mf.getSSARegMap()->getRegClass(virtReg));
                                ++numStores;
                            }
                            mii->SetMachineOperandReg(i, physReg);
                        }
                    }
                    DEBUG(std::cerr << '\t'; mii->print(std::cerr, tm));
                    loaded.clear();
                }
            }
            return true;
        }
    };

    class LocalSpiller : public Spiller {
        typedef std::vector<unsigned> Phys2VirtMap;
        typedef std::vector<bool> PhysFlag;
        typedef DenseMap<MachineInstr*, VirtReg2IndexFunctor> Virt2MI;

        MachineFunction* mf_;
        const TargetMachine* tm_;
        const TargetInstrInfo* tii_;
        const MRegisterInfo* mri_;
        const VirtRegMap* vrm_;
        Phys2VirtMap p2vMap_;
        PhysFlag dirty_;
        Virt2MI lastDef_;

    public:
        bool runOnMachineFunction(MachineFunction& mf, const VirtRegMap& vrm) {
            mf_ = &mf;
            tm_ = &mf_->getTarget();
            tii_ = tm_->getInstrInfo();
            mri_ = tm_->getRegisterInfo();
            vrm_ = &vrm;
            p2vMap_.assign(mri_->getNumRegs(), 0);
            dirty_.assign(mri_->getNumRegs(), false);

            DEBUG(std::cerr << "********** REWRITE MACHINE CODE **********\n");
            DEBUG(std::cerr << "********** Function: "
                  << mf_->getFunction()->getName() << '\n');

            for (MachineFunction::iterator mbbi = mf_->begin(),
                     mbbe = mf_->end(); mbbi != mbbe; ++mbbi) {
                lastDef_.grow(mf_->getSSARegMap()->getLastVirtReg());
                DEBUG(std::cerr << mbbi->getBasicBlock()->getName() << ":\n");
                eliminateVirtRegsInMbb(*mbbi);
                // clear map, dirty flag and last ref
                p2vMap_.assign(p2vMap_.size(), 0);
                dirty_.assign(dirty_.size(), false);
                lastDef_.clear();
            }
            return true;
        }

    private:
        void vacateJustPhysReg(MachineBasicBlock& mbb,
                               MachineBasicBlock::iterator mii,
                               unsigned physReg) {
            unsigned virtReg = p2vMap_[physReg];
            if (dirty_[physReg] && vrm_->hasStackSlot(virtReg)) {
                assert(lastDef_[virtReg] && "virtual register is mapped "
                       "to a register and but was not defined!");
                MachineBasicBlock::iterator lastDef = lastDef_[virtReg];
                MachineBasicBlock::iterator nextLastRef = next(lastDef);
                mri_->storeRegToStackSlot(*lastDef->getParent(),
                                          nextLastRef,
                                          physReg,
                                          vrm_->getStackSlot(virtReg),
                                          mri_->getRegClass(physReg));
                ++numStores;
                DEBUG(std::cerr << "added: ";
                      prior(nextLastRef)->print(std::cerr, *tm_);
                      std::cerr << "after: ";
                      lastDef->print(std::cerr, *tm_));
                lastDef_[virtReg] = 0;
            }
            p2vMap_[physReg] = 0;
            dirty_[physReg] = false;
        }

        void vacatePhysReg(MachineBasicBlock& mbb,
                           MachineBasicBlock::iterator mii,
                           unsigned physReg) {
            vacateJustPhysReg(mbb, mii, physReg);
            for (const unsigned* as = mri_->getAliasSet(physReg); *as; ++as)
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
                if (vrm_->hasStackSlot(virtReg)) {
                    mri_->loadRegFromStackSlot(mbb, mii, physReg,
                                               vrm_->getStackSlot(virtReg),
                                               mri_->getRegClass(physReg));
                    ++numLoads;
                    DEBUG(std::cerr << "added: ";
                          prior(mii)->print(std::cerr, *tm_));
                    lastDef_[virtReg] = mii;
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
            lastDef_[virtReg] = mii;
        }

        void eliminateVirtRegsInMbb(MachineBasicBlock& mbb) {
            for (MachineBasicBlock::iterator mii = mbb.begin(),
                     mie = mbb.end(); mii != mie; ++mii) {

                // if we have references to memory operands make sure
                // we clear all physical registers that may contain
                // the value of the spilled virtual register
                VirtRegMap::MI2VirtMap::const_iterator i, e;
                for (tie(i, e) = vrm_->getFoldedVirts(mii); i != e; ++i) {
                    if (vrm_->hasPhys(i->second))
                        vacateJustPhysReg(mbb, mii, vrm_->getPhys(i->second));
                }

                // rewrite all used operands
                for (unsigned i = 0, e = mii->getNumOperands(); i != e; ++i) {
                    MachineOperand& op = mii->getOperand(i);
                    if (op.isRegister() && op.getReg() && op.isUse() &&
                        MRegisterInfo::isVirtualRegister(op.getReg())) {
                        unsigned virtReg = op.getReg();
                        unsigned physReg = vrm_->getPhys(virtReg);
                        handleUse(mbb, mii, virtReg, physReg);
                        mii->SetMachineOperandReg(i, physReg);
                        // mark as dirty if this is def&use
                        if (op.isDef()) {
                            dirty_[physReg] = true;
                            lastDef_[virtReg] = mii;
                        }
                    }
                }

                // spill implicit physical register defs
                const TargetInstrDescriptor& tid = tii_->get(mii->getOpcode());
                for (const unsigned* id = tid.ImplicitDefs; *id; ++id)
                    vacatePhysReg(mbb, mii, *id);

                // spill explicit physical register defs
                for (unsigned i = 0, e = mii->getNumOperands(); i != e; ++i) {
                    MachineOperand& op = mii->getOperand(i);
                    if (op.isRegister() && op.getReg() && !op.isUse() &&
                        MRegisterInfo::isPhysicalRegister(op.getReg()))
                        vacatePhysReg(mbb, mii, op.getReg());
                }

                // rewrite def operands (def&use was handled with the
                // uses so don't check for those here)
                for (unsigned i = 0, e = mii->getNumOperands(); i != e; ++i) {
                    MachineOperand& op = mii->getOperand(i);
                    if (op.isRegister() && op.getReg() && !op.isUse())
                        if (MRegisterInfo::isPhysicalRegister(op.getReg()))
                            vacatePhysReg(mbb, mii, op.getReg());
                        else {
                            unsigned physReg = vrm_->getPhys(op.getReg());
                            handleDef(mbb, mii, op.getReg(), physReg);
                            mii->SetMachineOperandReg(i, physReg);
                        }
                }

                DEBUG(std::cerr << '\t'; mii->print(std::cerr, *tm_));
            }

            for (unsigned i = 1, e = p2vMap_.size(); i != e; ++i)
                vacateJustPhysReg(mbb, mbb.getFirstTerminator(), i);

        }
    };
}

llvm::Spiller* llvm::createSpiller()
{
    switch (SpillerOpt) {
    default:
        std::cerr << "no spiller selected";
        abort();
    case local:
        return new LocalSpiller();
    case simple:
        return new SimpleSpiller();
    }
}
