//===- SparcV9RegisterInfo.h - SparcV9 Register Information Impl -*- C++ -*-==//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains the SparcV9 implementation of the MRegisterInfo class.
// It also contains stuff needed to instantiate that class, which would
// ordinarily be provided by TableGen.
//
//===----------------------------------------------------------------------===//

#ifndef SPARCV9REGISTERINFO_H
#define SPARCV9REGISTERINFO_H

#include "llvm/Target/MRegisterInfo.h"

namespace llvm {

struct SparcV9RegisterInfo : public MRegisterInfo {
  SparcV9RegisterInfo ();
  const unsigned *getCalleeSaveRegs() const;

  // The rest of these are stubs... for now.
  int storeRegToStackSlot (MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MI,
                           unsigned SrcReg, int FrameIndex,
                           const TargetRegisterClass *RC) const;
  int loadRegFromStackSlot (MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MI,
                            unsigned DestReg, int FrameIndex,
                            const TargetRegisterClass *RC) const;
  int copyRegToReg (MachineBasicBlock &MBB,
                    MachineBasicBlock::iterator MI,
                    unsigned DestReg, unsigned SrcReg,
                    const TargetRegisterClass *RC) const;
  void eliminateFrameIndex (MachineFunction &MF,
                            MachineBasicBlock::iterator MI) const;
  void emitPrologue (MachineFunction &MF) const;
  void emitEpilogue (MachineFunction &MF, MachineBasicBlock &MBB) const;
};

} // End llvm namespace

//===----------------------------------------------------------------------===//
//
// The second section of this file (immediately following) contains
// a *handwritten* SparcV9 unified register number enumeration, which
// provides a flat namespace containing all the SparcV9 unified
// register numbers.
//
// It would ordinarily be contained in the file SparcV9GenRegisterNames.inc
// if we were using TableGen to generate the register file description
// automatically.
//
//===----------------------------------------------------------------------===//

namespace llvm {
  namespace SparcV9 {
    enum {
    // FIXME - Register 0 is not a "non-register" like it is on other targets!!

    // SparcV9IntRegClass(IntRegClassID)
    // - unified register numbers 0 ... 31 (32 regs)
    /* 0  */ o0, o1, o2, o3, o4,
    /* 5  */ o5, o7, l0, l1, l2,
    /* 10 */ l3, l4, l5, l6, l7,
    /* 15 */ i0, i1, i2, i3, i4,
    /* 20 */ i5, i6, i7, g0, g1,
    /* 25 */ g2, g3, g4, g5, g6,
    /* 30 */ g7, o6,

    // SparcV9FloatRegClass(FloatRegClassID)
    // - unified register numbers 32 ... 95 (64 regs)
    /* 32 */ f0,  f1,  f2,
    /* 35 */ f3,  f4,  f5,  f6,  f7,
    /* 40 */ f8,  f9,  f10, f11, f12,
    /* 45 */ f13, f14, f15, f16, f17,
    /* 50 */ f18, f19, f20, f21, f22,
    /* 55 */ f23, f24, f25, f26, f27,
    /* 60 */ f28, f29, f30, f31, f32,
    /* 65 */ f33, f34, f35, f36, f37,
    /* 70 */ f38, f39, f40, f41, f42,
    /* 75 */ f43, f44, f45, f46, f47,
    /* 80 */ f48, f49, f50, f51, f52,
    /* 85 */ f53, f54, f55, f56, f57,
    /* 90 */ f58, f59, f60, f61, f62,
    /* 95 */ f63,

    // SparcV9IntCCRegClass(IntCCRegClassID) 
    // - unified register numbers 96 ... 98 (3 regs)
    /* 96 */ xcc, icc, ccr,

    // SparcV9FloatCCRegClass(FloatCCRegClassID)
    // - unified register numbers 99 ... 102 (4 regs)
    /* 99 */ fcc0, fcc1, fcc2, fcc3,

    // SparcV9SpecialRegClass(SpecialRegClassID)
    // - unified register number 103  (1 reg)
    /* 103 */ fsr
    };
  } // end namespace SparcV9
} // end namespace llvm

#endif // SPARCV9REGISTERINFO_H
