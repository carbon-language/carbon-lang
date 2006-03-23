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
#include "llvm/CodeGen/ValueTypes.h"
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
  const MVT::ValueType IRVTs[] = { MVT::i64, MVT::Other };
  struct IRClass : public TargetRegisterClass {
    IRClass() : TargetRegisterClass(IRVTs, 8, 8, IR, IR + 32) {}
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
  const MVT::ValueType FRVTs[] = { MVT::f32, MVT::Other };
  // FIXME: The size is correct for the first 32 registers. The
  // latter 32 do not all really exist; you can only access every other
  // one (32, 34, ...), and they must contain double-fp or quad-fp
  // values... see below about the aliasing problems.
  struct FRClass : public TargetRegisterClass {
    FRClass() : TargetRegisterClass(FRVTs, 4, 8, FR, FR + 64) {}
  } FRInstance;


  // ICCR Register Class...
  const unsigned ICCR[] = {
    SparcV9::xcc, SparcV9::icc, SparcV9::ccr
  };
  const MVT::ValueType ICCRVTs[] = { MVT::i1, MVT::Other };
  struct ICCRClass : public TargetRegisterClass {
    ICCRClass() : TargetRegisterClass(ICCRVTs, 1, 8, ICCR, ICCR + 3) {}
  } ICCRInstance;


  // FCCR Register Class...
  const unsigned FCCR[] = {
    SparcV9::fcc0, SparcV9::fcc1, SparcV9::fcc2, SparcV9::fcc3
  };
  const MVT::ValueType FCCRVTs[] = { MVT::i1, MVT::Other };
  struct FCCRClass : public TargetRegisterClass {
    FCCRClass() : TargetRegisterClass(FCCRVTs, 1, 8, FCCR, FCCR + 4) {}
  } FCCRInstance;


  // SR Register Class...
  const unsigned SR[] = {
    SparcV9::fsr
  };
  const MVT::ValueType SRVTs[] = { MVT::i64, MVT::Other };
  struct SRClass : public TargetRegisterClass {
    SRClass() : TargetRegisterClass(SRVTs, 8, 8, SR, SR + 1) {}
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

const TargetRegisterDesc RegisterDescriptors[] = { // Descriptors
  { "o0", Empty_AliasSet },
  { "o1", Empty_AliasSet },
  { "o2", Empty_AliasSet },
  { "o3", Empty_AliasSet },
  { "o4", Empty_AliasSet },
  { "o5", Empty_AliasSet },
  { "o7", Empty_AliasSet },
  { "l0", Empty_AliasSet },
  { "l1", Empty_AliasSet },
  { "l2", Empty_AliasSet },
  { "l3", Empty_AliasSet },
  { "l4", Empty_AliasSet },
  { "l5", Empty_AliasSet },
  { "l6", Empty_AliasSet },
  { "l7", Empty_AliasSet },
  { "i0", Empty_AliasSet },
  { "i1", Empty_AliasSet },
  { "i2", Empty_AliasSet },
  { "i3", Empty_AliasSet },
  { "i4", Empty_AliasSet },
  { "i5", Empty_AliasSet },
  { "i6", Empty_AliasSet },
  { "i7", Empty_AliasSet },
  { "g0", Empty_AliasSet },
  { "g1", Empty_AliasSet },
  { "g2", Empty_AliasSet },
  { "g3", Empty_AliasSet },
  { "g4", Empty_AliasSet },
  { "g5", Empty_AliasSet },
  { "g6", Empty_AliasSet },
  { "g7", Empty_AliasSet },
  { "o6", Empty_AliasSet },
  { "f0", Empty_AliasSet },
  { "f1", Empty_AliasSet },
  { "f2", Empty_AliasSet },
  { "f3", Empty_AliasSet },
  { "f4", Empty_AliasSet },
  { "f5", Empty_AliasSet },
  { "f6", Empty_AliasSet },
  { "f7", Empty_AliasSet },
  { "f8", Empty_AliasSet },
  { "f9", Empty_AliasSet },
  { "f10", Empty_AliasSet },
  { "f11", Empty_AliasSet },
  { "f12", Empty_AliasSet },
  { "f13", Empty_AliasSet },
  { "f14", Empty_AliasSet },
  { "f15", Empty_AliasSet },
  { "f16", Empty_AliasSet },
  { "f17", Empty_AliasSet },
  { "f18", Empty_AliasSet },
  { "f19", Empty_AliasSet },
  { "f20", Empty_AliasSet },
  { "f21", Empty_AliasSet },
  { "f22", Empty_AliasSet },
  { "f23", Empty_AliasSet },
  { "f24", Empty_AliasSet },
  { "f25", Empty_AliasSet },
  { "f26", Empty_AliasSet },
  { "f27", Empty_AliasSet },
  { "f28", Empty_AliasSet },
  { "f29", Empty_AliasSet },
  { "f30", Empty_AliasSet },
  { "f31", Empty_AliasSet },
  { "f32", Empty_AliasSet },
  { "f33", Empty_AliasSet },
  { "f34", Empty_AliasSet },
  { "f35", Empty_AliasSet },
  { "f36", Empty_AliasSet },
  { "f37", Empty_AliasSet },
  { "f38", Empty_AliasSet },
  { "f39", Empty_AliasSet },
  { "f40", Empty_AliasSet },
  { "f41", Empty_AliasSet },
  { "f42", Empty_AliasSet },
  { "f43", Empty_AliasSet },
  { "f44", Empty_AliasSet },
  { "f45", Empty_AliasSet },
  { "f46", Empty_AliasSet },
  { "f47", Empty_AliasSet },
  { "f48", Empty_AliasSet },
  { "f49", Empty_AliasSet },
  { "f50", Empty_AliasSet },
  { "f51", Empty_AliasSet },
  { "f52", Empty_AliasSet },
  { "f53", Empty_AliasSet },
  { "f54", Empty_AliasSet },
  { "f55", Empty_AliasSet },
  { "f56", Empty_AliasSet },
  { "f57", Empty_AliasSet },
  { "f58", Empty_AliasSet },
  { "f59", Empty_AliasSet },
  { "f60", Empty_AliasSet },
  { "f61", Empty_AliasSet },
  { "f62", Empty_AliasSet },
  { "f63", Empty_AliasSet },
  { "xcc", xcc_AliasSet },
  { "icc", icc_AliasSet },
  { "ccr", ccr_AliasSet },
  { "fcc0", fcc0_AliasSet },
  { "fcc1", fcc1_AliasSet },
  { "fcc2", fcc2_AliasSet },
  { "fcc3", fcc3_AliasSet },
  { "fsr", fsr_AliasSet },
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

void SparcV9RegisterInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator MI,
                                         unsigned SrcReg, int FrameIndex,
                                         const TargetRegisterClass *RC) const {
  abort ();
}

void SparcV9RegisterInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator MI,
                                          unsigned DestReg, int FrameIndex,
                                          const TargetRegisterClass *RC) const {
  abort ();
}

void SparcV9RegisterInfo::copyRegToReg(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator MI,
                                      unsigned DestReg, unsigned SrcReg,
                                      const TargetRegisterClass *RC) const {
  abort ();
}

void SparcV9RegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator MI)
  const {
  abort ();
}

void SparcV9RegisterInfo::emitPrologue(MachineFunction &MF) const {
  abort ();
}

void SparcV9RegisterInfo::emitEpilogue(MachineFunction &MF,
                                       MachineBasicBlock &MBB) const {
  abort ();
}


void SparcV9RegisterInfo::getLocation(MachineFunction &MF, unsigned Index,
                                      MachineLocation &ML) const {
  abort ();
}
