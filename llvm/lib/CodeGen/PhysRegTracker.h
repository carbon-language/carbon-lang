//===-- llvm/CodeGen/PhysRegTracker.h - Physical Register Tracker -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a physical register tracker. The tracker
// tracks physical register usage through addRegUse and
// delRegUse. isRegAvail checks if a physical register is available or
// not taking into consideration register aliases.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PHYSREGTRACKER_H
#define LLVM_CODEGEN_PHYSREGTRACKER_H

#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

    class PhysRegTracker {
        const MRegisterInfo* mri_;
        std::vector<unsigned> regUse_;

    public:
        PhysRegTracker(MachineFunction* mf)
            : mri_(mf ? mf->getTarget().getRegisterInfo() : NULL) {
            if (mri_) {
                regUse_.assign(mri_->getNumRegs(), 0);
            }
        }

        PhysRegTracker(const PhysRegTracker& rhs)
            : mri_(rhs.mri_),
              regUse_(rhs.regUse_) {
        }

        const PhysRegTracker& operator=(const PhysRegTracker& rhs) {
            mri_ = rhs.mri_;
            regUse_ = rhs.regUse_;
            return *this;
        }

        void addRegUse(unsigned physReg) {
            assert(MRegisterInfo::isPhysicalRegister(physReg) &&
                   "should be physical register!");
            ++regUse_[physReg];
            for (const unsigned* as = mri_->getAliasSet(physReg); *as; ++as)
                ++regUse_[*as];
        }

        void delRegUse(unsigned physReg) {
            assert(MRegisterInfo::isPhysicalRegister(physReg) &&
                   "should be physical register!");
            assert(regUse_[physReg] != 0);
            --regUse_[physReg];
            for (const unsigned* as = mri_->getAliasSet(physReg); *as; ++as) {
                assert(regUse_[*as] != 0);
                --regUse_[*as];
            }
        }

        bool isRegAvail(unsigned physReg) const {
            assert(MRegisterInfo::isPhysicalRegister(physReg) &&
                   "should be physical register!");
            return regUse_[physReg] == 0;
        }
    };

} // End llvm namespace

#endif
