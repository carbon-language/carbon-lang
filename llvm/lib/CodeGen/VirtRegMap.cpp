//===-- llvm/CodeGen/VirtRegMap.cpp - Virtual Register Map ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the virtual register map.
//
//===----------------------------------------------------------------------===//

#include "VirtRegMap.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "Support/Statistic.h"
#include <iostream>

using namespace llvm;

namespace {
    Statistic<> numSpills("ra-linearscan", "Number of register spills");
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
