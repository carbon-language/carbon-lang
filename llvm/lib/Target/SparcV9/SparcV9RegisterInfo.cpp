//===- SparcV9RegisterInfo.cpp - SparcV9 Register Information ---*- C++ -*-===//
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
// This is NOT used by the SparcV9 backend to do register allocation, yet.
//
//===----------------------------------------------------------------------===//
//
// The first section of this file (immediately following) is what
// you would find in SparcV9GenRegisterInfo.inc, if we were using
// TableGen to generate the register file description automatically.
// It consists of register classes and register class instances
// for the SparcV9 target.
// 
// FIXME: the alignments listed here are wild guesses.
//
//===----------------------------------------------------------------------===//

#include "SparcV9RegisterInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
using namespace llvm;

namespace llvm {

namespace {
  // IR Register Class...
  const unsigned IR[] = {
    SparcV9::o0, SparcV9::o1, SparcV9::o2, SparcV9::o3, SparcV9::o4,
    SparcV9::o5, SparcV9::o7, SparcV9::l0, SparcV9::l1, SparcV9::l2,
    SparcV9::l3, SparcV9::l4, SparcV9::l5, SparcV9::l6, SparcV9::l7,
    SparcV9::i0, SparcV9::i1, SparcV9::i2, SparcV9::i3, SparcV9::i4,
    SparcV9::i5, SparcV9::i6, SparcV9::i7, SparcV9::g0, SparcV9::g1,
    SparcV9::g2, SparcV9::g3, SparcV9::g4, SparcV9::g5, SparcV9::g6,
    SparcV9::g7, SparcV9::o6
  };
  struct IRClass : public TargetRegisterClass {
    IRClass() : TargetRegisterClass(8, 8, IR, IR + 32) {}
  } IRInstance;


  // FR Register Class...
  const unsigned FR[] = {
    SparcV9::f0, SparcV9::f1, SparcV9::f2, SparcV9::f3, SparcV9::f4,
    SparcV9::f5, SparcV9::f6, SparcV9::f7, SparcV9::f8, SparcV9::f9,
    SparcV9::f10, SparcV9::f11, SparcV9::f12, SparcV9::f13,
    SparcV9::f14, SparcV9::f15, SparcV9::f16, SparcV9::f17,
    SparcV9::f18, SparcV9::f19, SparcV9::f20, SparcV9::f21,
    SparcV9::f22, SparcV9::f23, SparcV9::f24, SparcV9::f25,
    SparcV9::f26, SparcV9::f27, SparcV9::f28, SparcV9::f29,
    SparcV9::f30, SparcV9::f31, SparcV9::f32, SparcV9::f33,
    SparcV9::f34, SparcV9::f35, SparcV9::f36, SparcV9::f37,
    SparcV9::f38, SparcV9::f39, SparcV9::f40, SparcV9::f41,
    SparcV9::f42, SparcV9::f43, SparcV9::f44, SparcV9::f45,
    SparcV9::f46, SparcV9::f47, SparcV9::f48, SparcV9::f49,
    SparcV9::f50, SparcV9::f51, SparcV9::f52, SparcV9::f53,
    SparcV9::f54, SparcV9::f55, SparcV9::f56, SparcV9::f57,
    SparcV9::f58, SparcV9::f59, SparcV9::f60, SparcV9::f61,
    SparcV9::f62, SparcV9::f63
  };
  // FIXME: The size is correct for the first 32 registers. The
  // latter 32 do not all really exist; you can only access every other
  // one (32, 34, ...), and they must contain double-fp or quad-fp
  // values... see below about the aliasing problems.
  struct FRClass : public TargetRegisterClass {
    FRClass() : TargetRegisterClass(4, 8, FR, FR + 64) {}
  } FRInstance;


  // ICCR Register Class...
  const unsigned ICCR[] = {
    SparcV9::xcc, SparcV9::icc, SparcV9::ccr
  };
  struct ICCRClass : public TargetRegisterClass {
    ICCRClass() : TargetRegisterClass(1, 8, ICCR, ICCR + 3) {}
  } ICCRInstance;


  // FCCR Register Class...
  const unsigned FCCR[] = {
    SparcV9::fcc0, SparcV9::fcc1, SparcV9::fcc2, SparcV9::fcc3
  };
  struct FCCRClass : public TargetRegisterClass {
    FCCRClass() : TargetRegisterClass(1, 8, FCCR, FCCR + 4) {}
  } FCCRInstance;


  // SR Register Class...
  const unsigned SR[] = {
    SparcV9::fsr
  };
  struct SRClass : public TargetRegisterClass {
    SRClass() : TargetRegisterClass(8, 8, SR, SR + 1) {}
  } SRInstance;


  // Register Classes...
  const TargetRegisterClass* const RegisterClasses[] = {
    &IRInstance,
    &FRInstance,
    &ICCRInstance,
    &FCCRInstance,
    &SRInstance
  };


  // Register Alias Sets...
  // FIXME: Note that the SparcV9 backend does not currently abstract
  // very well over the way that double-fp and quad-fp values may alias
  // single-fp values in registers. Therefore those aliases are NOT
  // reflected here.
  const unsigned Empty_AliasSet[] = { 0 };
  const unsigned fcc3_AliasSet[] = { SparcV9::fsr, 0 };
  const unsigned fcc2_AliasSet[] = { SparcV9::fsr, 0 };
  const unsigned fcc1_AliasSet[] = { SparcV9::fsr, 0 };
  const unsigned fcc0_AliasSet[] = { SparcV9::fsr, 0 };
  const unsigned fsr_AliasSet[] = { SparcV9::fcc3, SparcV9::fcc2,
                                    SparcV9::fcc1, SparcV9::fcc0, 0 };
  const unsigned xcc_AliasSet[] = { SparcV9::ccr, 0 };
  const unsigned icc_AliasSet[] = { SparcV9::ccr, 0 };
  const unsigned ccr_AliasSet[] = { SparcV9::xcc, SparcV9::icc, 0 };

const MRegisterDesc RegisterDescriptors[] = { // Descriptors
  { "o0", Empty_AliasSet, 0, 0 },
  { "o1", Empty_AliasSet, 0, 0 },
  { "o2", Empty_AliasSet, 0, 0 },
  { "o3", Empty_AliasSet, 0, 0 },
  { "o4", Empty_AliasSet, 0, 0 },
  { "o5", Empty_AliasSet, 0, 0 },
  { "o7", Empty_AliasSet, 0, 0 },
  { "l0", Empty_AliasSet, 0, 0 },
  { "l1", Empty_AliasSet, 0, 0 },
  { "l2", Empty_AliasSet, 0, 0 },
  { "l3", Empty_AliasSet, 0, 0 },
  { "l4", Empty_AliasSet, 0, 0 },
  { "l5", Empty_AliasSet, 0, 0 },
  { "l6", Empty_AliasSet, 0, 0 },
  { "l7", Empty_AliasSet, 0, 0 },
  { "i0", Empty_AliasSet, 0, 0 },
  { "i1", Empty_AliasSet, 0, 0 },
  { "i2", Empty_AliasSet, 0, 0 },
  { "i3", Empty_AliasSet, 0, 0 },
  { "i4", Empty_AliasSet, 0, 0 },
  { "i5", Empty_AliasSet, 0, 0 },
  { "i6", Empty_AliasSet, 0, 0 },
  { "i7", Empty_AliasSet, 0, 0 },
  { "g0", Empty_AliasSet, 0, 0 },
  { "g1", Empty_AliasSet, 0, 0 },
  { "g2", Empty_AliasSet, 0, 0 },
  { "g3", Empty_AliasSet, 0, 0 },
  { "g4", Empty_AliasSet, 0, 0 },
  { "g5", Empty_AliasSet, 0, 0 },
  { "g6", Empty_AliasSet, 0, 0 },
  { "g7", Empty_AliasSet, 0, 0 },
  { "o6", Empty_AliasSet, 0, 0 },
  { "f0", Empty_AliasSet, 0, 0 },
  { "f1", Empty_AliasSet, 0, 0 },
  { "f2", Empty_AliasSet, 0, 0 },
  { "f3", Empty_AliasSet, 0, 0 },
  { "f4", Empty_AliasSet, 0, 0 },
  { "f5", Empty_AliasSet, 0, 0 },
  { "f6", Empty_AliasSet, 0, 0 },
  { "f7", Empty_AliasSet, 0, 0 },
  { "f8", Empty_AliasSet, 0, 0 },
  { "f9", Empty_AliasSet, 0, 0 },
  { "f10", Empty_AliasSet, 0, 0 },
  { "f11", Empty_AliasSet, 0, 0 },
  { "f12", Empty_AliasSet, 0, 0 },
  { "f13", Empty_AliasSet, 0, 0 },
  { "f14", Empty_AliasSet, 0, 0 },
  { "f15", Empty_AliasSet, 0, 0 },
  { "f16", Empty_AliasSet, 0, 0 },
  { "f17", Empty_AliasSet, 0, 0 },
  { "f18", Empty_AliasSet, 0, 0 },
  { "f19", Empty_AliasSet, 0, 0 },
  { "f20", Empty_AliasSet, 0, 0 },
  { "f21", Empty_AliasSet, 0, 0 },
  { "f22", Empty_AliasSet, 0, 0 },
  { "f23", Empty_AliasSet, 0, 0 },
  { "f24", Empty_AliasSet, 0, 0 },
  { "f25", Empty_AliasSet, 0, 0 },
  { "f26", Empty_AliasSet, 0, 0 },
  { "f27", Empty_AliasSet, 0, 0 },
  { "f28", Empty_AliasSet, 0, 0 },
  { "f29", Empty_AliasSet, 0, 0 },
  { "f30", Empty_AliasSet, 0, 0 },
  { "f31", Empty_AliasSet, 0, 0 },
  { "f32", Empty_AliasSet, 0, 0 },
  { "f33", Empty_AliasSet, 0, 0 },
  { "f34", Empty_AliasSet, 0, 0 },
  { "f35", Empty_AliasSet, 0, 0 },
  { "f36", Empty_AliasSet, 0, 0 },
  { "f37", Empty_AliasSet, 0, 0 },
  { "f38", Empty_AliasSet, 0, 0 },
  { "f39", Empty_AliasSet, 0, 0 },
  { "f40", Empty_AliasSet, 0, 0 },
  { "f41", Empty_AliasSet, 0, 0 },
  { "f42", Empty_AliasSet, 0, 0 },
  { "f43", Empty_AliasSet, 0, 0 },
  { "f44", Empty_AliasSet, 0, 0 },
  { "f45", Empty_AliasSet, 0, 0 },
  { "f46", Empty_AliasSet, 0, 0 },
  { "f47", Empty_AliasSet, 0, 0 },
  { "f48", Empty_AliasSet, 0, 0 },
  { "f49", Empty_AliasSet, 0, 0 },
  { "f50", Empty_AliasSet, 0, 0 },
  { "f51", Empty_AliasSet, 0, 0 },
  { "f52", Empty_AliasSet, 0, 0 },
  { "f53", Empty_AliasSet, 0, 0 },
  { "f54", Empty_AliasSet, 0, 0 },
  { "f55", Empty_AliasSet, 0, 0 },
  { "f56", Empty_AliasSet, 0, 0 },
  { "f57", Empty_AliasSet, 0, 0 },
  { "f58", Empty_AliasSet, 0, 0 },
  { "f59", Empty_AliasSet, 0, 0 },
  { "f60", Empty_AliasSet, 0, 0 },
  { "f61", Empty_AliasSet, 0, 0 },
  { "f62", Empty_AliasSet, 0, 0 },
  { "f63", Empty_AliasSet, 0, 0 },
  { "xcc", xcc_AliasSet, 0, 0 },
  { "icc", icc_AliasSet, 0, 0 },
  { "ccr", ccr_AliasSet, 0, 0 },
  { "fcc0", fcc0_AliasSet, 0, 0 },
  { "fcc1", fcc1_AliasSet, 0, 0 },
  { "fcc2", fcc2_AliasSet, 0, 0 },
  { "fcc3", fcc3_AliasSet, 0, 0 },
  { "fsr", fsr_AliasSet, 0, 0 },
};

} // end anonymous namespace

namespace SparcV9 { // Register classes
  TargetRegisterClass *IRRegisterClass = &IRInstance;
  TargetRegisterClass *FRRegisterClass = &FRInstance;
  TargetRegisterClass *ICCRRegisterClass = &ICCRInstance;
  TargetRegisterClass *FCCRRegisterClass = &FCCRInstance;
  TargetRegisterClass *SRRegisterClass = &SRInstance;
} // end namespace SparcV9

const unsigned *SparcV9RegisterInfo::getCalleeSaveRegs() const {
  // FIXME: This should be verified against the SparcV9 ABI at some point.
  // These are the registers which the SparcV9 backend considers
  // "non-volatile".
  static const unsigned CalleeSaveRegs[] = {
     SparcV9::l0, SparcV9::l1, SparcV9::l2, SparcV9::l3, SparcV9::l4,
     SparcV9::l5, SparcV9::l6, SparcV9::l7, SparcV9::i0, SparcV9::i1,
     SparcV9::i2, SparcV9::i3, SparcV9::i4, SparcV9::i5, SparcV9::i6,
     SparcV9::i7, SparcV9::g0, SparcV9::g1, SparcV9::g2, SparcV9::g3,
     SparcV9::g4, SparcV9::g5, SparcV9::g6, SparcV9::g7, SparcV9::o6,
     0
  };
  return CalleeSaveRegs;
}

} // end namespace llvm

//===----------------------------------------------------------------------===//
//
// The second section of this file (immediately following) contains the
// SparcV9 implementation of the MRegisterInfo class. It currently consists
// entirely of stub functions, because the SparcV9 target does not use the
// same register allocator that the X86 target uses.
//
//===----------------------------------------------------------------------===//

SparcV9RegisterInfo::SparcV9RegisterInfo ()
  : MRegisterInfo (RegisterDescriptors, 104, RegisterClasses,
                   RegisterClasses + 5) {
}

int SparcV9RegisterInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator MI,
                                         unsigned SrcReg, int FrameIndex,
                                         const TargetRegisterClass *RC) const {
  abort ();
}

int SparcV9RegisterInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator MI,
                                          unsigned DestReg, int FrameIndex,
                                          const TargetRegisterClass *RC) const {
  abort ();
}

int SparcV9RegisterInfo::copyRegToReg(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator MI,
                                      unsigned DestReg, unsigned SrcReg,
                                      const TargetRegisterClass *RC) const {
  abort ();
}

void SparcV9RegisterInfo::eliminateFrameIndex(MachineFunction &MF,
                                         MachineBasicBlock::iterator MI) const {
  abort ();
}

void SparcV9RegisterInfo::emitPrologue(MachineFunction &MF) const {
  abort ();
}

void SparcV9RegisterInfo::emitEpilogue(MachineFunction &MF,
                                       MachineBasicBlock &MBB) const {
  abort ();
}
