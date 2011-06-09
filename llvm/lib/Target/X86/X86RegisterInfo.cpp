//===- X86RegisterInfo.cpp - X86 Register Information -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the X86 implementation of the TargetRegisterInfo class.
// This file is responsible for the frame pointer elimination optimization
// on X86.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86RegisterInfo.h"
#include "X86InstrBuilder.h"
#include "X86MachineFunctionInfo.h"
#include "X86Subtarget.h"
#include "X86TargetMachine.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Type.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

cl::opt<bool>
ForceStackAlign("force-align-stack",
                 cl::desc("Force align the stack to the minimum alignment"
                           " needed for the function."),
                 cl::init(false), cl::Hidden);

X86RegisterInfo::X86RegisterInfo(X86TargetMachine &tm,
                                 const TargetInstrInfo &tii)
  : X86GenRegisterInfo(tm.getSubtarget<X86Subtarget>().is64Bit() ?
                         X86::ADJCALLSTACKDOWN64 :
                         X86::ADJCALLSTACKDOWN32,
                       tm.getSubtarget<X86Subtarget>().is64Bit() ?
                         X86::ADJCALLSTACKUP64 :
                         X86::ADJCALLSTACKUP32),
    TM(tm), TII(tii) {
  // Cache some information.
  const X86Subtarget *Subtarget = &TM.getSubtarget<X86Subtarget>();
  Is64Bit = Subtarget->is64Bit();
  IsWin64 = Subtarget->isTargetWin64();
  StackAlign = TM.getFrameLowering()->getStackAlignment();

  if (Is64Bit) {
    SlotSize = 8;
    StackPtr = X86::RSP;
    FramePtr = X86::RBP;
  } else {
    SlotSize = 4;
    StackPtr = X86::ESP;
    FramePtr = X86::EBP;
  }
}

static unsigned getFlavour(const X86Subtarget *Subtarget, bool isEH) {
  if (!Subtarget->is64Bit()) {
    if (Subtarget->isTargetDarwin()) {
      if (isEH)
        return DWARFFlavour::X86_32_DarwinEH;
      else
        return DWARFFlavour::X86_32_Generic;
    } else if (Subtarget->isTargetCygMing()) {
      // Unsupported by now, just quick fallback
      return DWARFFlavour::X86_32_Generic;
    } else {
      return DWARFFlavour::X86_32_Generic;
    }
  }
  return DWARFFlavour::X86_64;
}

/// getDwarfRegNum - This function maps LLVM register identifiers to the DWARF
/// specific numbering, used in debug info and exception tables.
int X86RegisterInfo::getDwarfRegNum(unsigned RegNo, bool isEH) const {
  const X86Subtarget *Subtarget = &TM.getSubtarget<X86Subtarget>();
  unsigned Flavour = getFlavour(Subtarget, isEH);

  return X86GenRegisterInfo::getDwarfRegNumFull(RegNo, Flavour);
}

/// getLLVMRegNum - This function maps DWARF register numbers to LLVM register.
int X86RegisterInfo::getLLVMRegNum(unsigned DwarfRegNo, bool isEH) const {
  const X86Subtarget *Subtarget = &TM.getSubtarget<X86Subtarget>();
  unsigned Flavour = getFlavour(Subtarget, isEH);

  return X86GenRegisterInfo::getLLVMRegNumFull(DwarfRegNo, Flavour);
}

int
X86RegisterInfo::getSEHRegNum(unsigned i) const {
  int reg = getX86RegNum(i);
  switch (i) {
  case X86::R8:  case X86::R8D:  case X86::R8W:  case X86::R8B:
  case X86::R9:  case X86::R9D:  case X86::R9W:  case X86::R9B:
  case X86::R10: case X86::R10D: case X86::R10W: case X86::R10B:
  case X86::R11: case X86::R11D: case X86::R11W: case X86::R11B:
  case X86::R12: case X86::R12D: case X86::R12W: case X86::R12B:
  case X86::R13: case X86::R13D: case X86::R13W: case X86::R13B:
  case X86::R14: case X86::R14D: case X86::R14W: case X86::R14B:
  case X86::R15: case X86::R15D: case X86::R15W: case X86::R15B:
  case X86::XMM8: case X86::XMM9: case X86::XMM10: case X86::XMM11:
  case X86::XMM12: case X86::XMM13: case X86::XMM14: case X86::XMM15:
  case X86::YMM8: case X86::YMM9: case X86::YMM10: case X86::YMM11:
  case X86::YMM12: case X86::YMM13: case X86::YMM14: case X86::YMM15:
    reg += 8;
  }
  return reg;
}

/// getX86RegNum - This function maps LLVM register identifiers to their X86
/// specific numbering, which is used in various places encoding instructions.
unsigned X86RegisterInfo::getX86RegNum(unsigned RegNo) {
  switch(RegNo) {
  case X86::RAX: case X86::EAX: case X86::AX: case X86::AL: return N86::EAX;
  case X86::RCX: case X86::ECX: case X86::CX: case X86::CL: return N86::ECX;
  case X86::RDX: case X86::EDX: case X86::DX: case X86::DL: return N86::EDX;
  case X86::RBX: case X86::EBX: case X86::BX: case X86::BL: return N86::EBX;
  case X86::RSP: case X86::ESP: case X86::SP: case X86::SPL: case X86::AH:
    return N86::ESP;
  case X86::RBP: case X86::EBP: case X86::BP: case X86::BPL: case X86::CH:
    return N86::EBP;
  case X86::RSI: case X86::ESI: case X86::SI: case X86::SIL: case X86::DH:
    return N86::ESI;
  case X86::RDI: case X86::EDI: case X86::DI: case X86::DIL: case X86::BH:
    return N86::EDI;

  case X86::R8:  case X86::R8D:  case X86::R8W:  case X86::R8B:
    return N86::EAX;
  case X86::R9:  case X86::R9D:  case X86::R9W:  case X86::R9B:
    return N86::ECX;
  case X86::R10: case X86::R10D: case X86::R10W: case X86::R10B:
    return N86::EDX;
  case X86::R11: case X86::R11D: case X86::R11W: case X86::R11B:
    return N86::EBX;
  case X86::R12: case X86::R12D: case X86::R12W: case X86::R12B:
    return N86::ESP;
  case X86::R13: case X86::R13D: case X86::R13W: case X86::R13B:
    return N86::EBP;
  case X86::R14: case X86::R14D: case X86::R14W: case X86::R14B:
    return N86::ESI;
  case X86::R15: case X86::R15D: case X86::R15W: case X86::R15B:
    return N86::EDI;

  case X86::ST0: case X86::ST1: case X86::ST2: case X86::ST3:
  case X86::ST4: case X86::ST5: case X86::ST6: case X86::ST7:
    return RegNo-X86::ST0;

  case X86::XMM0: case X86::XMM8:
  case X86::YMM0: case X86::YMM8: case X86::MM0:
    return 0;
  case X86::XMM1: case X86::XMM9:
  case X86::YMM1: case X86::YMM9: case X86::MM1:
    return 1;
  case X86::XMM2: case X86::XMM10:
  case X86::YMM2: case X86::YMM10: case X86::MM2:
    return 2;
  case X86::XMM3: case X86::XMM11:
  case X86::YMM3: case X86::YMM11: case X86::MM3:
    return 3;
  case X86::XMM4: case X86::XMM12:
  case X86::YMM4: case X86::YMM12: case X86::MM4:
    return 4;
  case X86::XMM5: case X86::XMM13:
  case X86::YMM5: case X86::YMM13: case X86::MM5:
    return 5;
  case X86::XMM6: case X86::XMM14:
  case X86::YMM6: case X86::YMM14: case X86::MM6:
    return 6;
  case X86::XMM7: case X86::XMM15:
  case X86::YMM7: case X86::YMM15: case X86::MM7:
    return 7;

  case X86::ES: return 0;
  case X86::CS: return 1;
  case X86::SS: return 2;
  case X86::DS: return 3;
  case X86::FS: return 4;
  case X86::GS: return 5;

  case X86::CR0: case X86::CR8 : case X86::DR0: return 0;
  case X86::CR1: case X86::CR9 : case X86::DR1: return 1;
  case X86::CR2: case X86::CR10: case X86::DR2: return 2;
  case X86::CR3: case X86::CR11: case X86::DR3: return 3;
  case X86::CR4: case X86::CR12: case X86::DR4: return 4;
  case X86::CR5: case X86::CR13: case X86::DR5: return 5;
  case X86::CR6: case X86::CR14: case X86::DR6: return 6;
  case X86::CR7: case X86::CR15: case X86::DR7: return 7;

  // Pseudo index registers are equivalent to a "none"
  // scaled index (See Intel Manual 2A, table 2-3)
  case X86::EIZ:
  case X86::RIZ:
    return 4;

  default:
    assert(isVirtualRegister(RegNo) && "Unknown physical register!");
    llvm_unreachable("Register allocator hasn't allocated reg correctly yet!");
    return 0;
  }
}

const TargetRegisterClass *
X86RegisterInfo::getMatchingSuperRegClass(const TargetRegisterClass *A,
                                          const TargetRegisterClass *B,
                                          unsigned SubIdx) const {
  switch (SubIdx) {
  default: return 0;
  case X86::sub_8bit:
    if (B == &X86::GR8RegClass) {
      if (A->getSize() == 2 || A->getSize() == 4 || A->getSize() == 8)
        return A;
    } else if (B == &X86::GR8_ABCD_LRegClass || B == &X86::GR8_ABCD_HRegClass) {
      if (A == &X86::GR64RegClass || A == &X86::GR64_ABCDRegClass ||
          A == &X86::GR64_NOREXRegClass ||
          A == &X86::GR64_NOSPRegClass ||
          A == &X86::GR64_NOREX_NOSPRegClass)
        return &X86::GR64_ABCDRegClass;
      else if (A == &X86::GR32RegClass || A == &X86::GR32_ABCDRegClass ||
               A == &X86::GR32_NOREXRegClass ||
               A == &X86::GR32_NOSPRegClass)
        return &X86::GR32_ABCDRegClass;
      else if (A == &X86::GR16RegClass || A == &X86::GR16_ABCDRegClass ||
               A == &X86::GR16_NOREXRegClass)
        return &X86::GR16_ABCDRegClass;
    } else if (B == &X86::GR8_NOREXRegClass) {
      if (A == &X86::GR64RegClass || A == &X86::GR64_NOREXRegClass ||
          A == &X86::GR64_NOSPRegClass || A == &X86::GR64_NOREX_NOSPRegClass)
        return &X86::GR64_NOREXRegClass;
      else if (A == &X86::GR64_ABCDRegClass)
        return &X86::GR64_ABCDRegClass;
      else if (A == &X86::GR32RegClass || A == &X86::GR32_NOREXRegClass ||
               A == &X86::GR32_NOSPRegClass)
        return &X86::GR32_NOREXRegClass;
      else if (A == &X86::GR32_ABCDRegClass)
        return &X86::GR32_ABCDRegClass;
      else if (A == &X86::GR16RegClass || A == &X86::GR16_NOREXRegClass)
        return &X86::GR16_NOREXRegClass;
      else if (A == &X86::GR16_ABCDRegClass)
        return &X86::GR16_ABCDRegClass;
    }
    break;
  case X86::sub_8bit_hi:
    if (B->hasSubClassEq(&X86::GR8_ABCD_HRegClass))
      switch (A->getSize()) {
        case 2: return getCommonSubClass(A, &X86::GR16_ABCDRegClass);
        case 4: return getCommonSubClass(A, &X86::GR32_ABCDRegClass);
        case 8: return getCommonSubClass(A, &X86::GR64_ABCDRegClass);
        default: return 0;
      }
    break;
  case X86::sub_16bit:
    if (B == &X86::GR16RegClass) {
      if (A->getSize() == 4 || A->getSize() == 8)
        return A;
    } else if (B == &X86::GR16_ABCDRegClass) {
      if (A == &X86::GR64RegClass || A == &X86::GR64_ABCDRegClass ||
          A == &X86::GR64_NOREXRegClass ||
          A == &X86::GR64_NOSPRegClass ||
          A == &X86::GR64_NOREX_NOSPRegClass)
        return &X86::GR64_ABCDRegClass;
      else if (A == &X86::GR32RegClass || A == &X86::GR32_ABCDRegClass ||
               A == &X86::GR32_NOREXRegClass || A == &X86::GR32_NOSPRegClass)
        return &X86::GR32_ABCDRegClass;
    } else if (B == &X86::GR16_NOREXRegClass) {
      if (A == &X86::GR64RegClass || A == &X86::GR64_NOREXRegClass ||
          A == &X86::GR64_NOSPRegClass || A == &X86::GR64_NOREX_NOSPRegClass)
        return &X86::GR64_NOREXRegClass;
      else if (A == &X86::GR64_ABCDRegClass)
        return &X86::GR64_ABCDRegClass;
      else if (A == &X86::GR32RegClass || A == &X86::GR32_NOREXRegClass ||
               A == &X86::GR32_NOSPRegClass)
        return &X86::GR32_NOREXRegClass;
      else if (A == &X86::GR32_ABCDRegClass)
        return &X86::GR64_ABCDRegClass;
    }
    break;
  case X86::sub_32bit:
    if (B == &X86::GR32RegClass) {
      if (A->getSize() == 8)
        return A;
    } else if (B == &X86::GR32_NOSPRegClass) {
      if (A == &X86::GR64RegClass || A == &X86::GR64_NOSPRegClass)
        return &X86::GR64_NOSPRegClass;
      if (A->getSize() == 8)
        return getCommonSubClass(A, &X86::GR64_NOSPRegClass);
    } else if (B == &X86::GR32_ABCDRegClass) {
      if (A == &X86::GR64RegClass || A == &X86::GR64_ABCDRegClass ||
          A == &X86::GR64_NOREXRegClass ||
          A == &X86::GR64_NOSPRegClass ||
          A == &X86::GR64_NOREX_NOSPRegClass)
        return &X86::GR64_ABCDRegClass;
    } else if (B == &X86::GR32_NOREXRegClass) {
      if (A == &X86::GR64RegClass || A == &X86::GR64_NOREXRegClass)
        return &X86::GR64_NOREXRegClass;
      else if (A == &X86::GR64_NOSPRegClass || A == &X86::GR64_NOREX_NOSPRegClass)
        return &X86::GR64_NOREX_NOSPRegClass;
      else if (A == &X86::GR64_ABCDRegClass)
        return &X86::GR64_ABCDRegClass;
    } else if (B == &X86::GR32_NOREX_NOSPRegClass) {
      if (A == &X86::GR64RegClass || A == &X86::GR64_NOREXRegClass ||
          A == &X86::GR64_NOSPRegClass || A == &X86::GR64_NOREX_NOSPRegClass)
        return &X86::GR64_NOREX_NOSPRegClass;
      else if (A == &X86::GR64_ABCDRegClass)
        return &X86::GR64_ABCDRegClass;
    }
    break;
  case X86::sub_ss:
    if (B == &X86::FR32RegClass)
      return A;
    break;
  case X86::sub_sd:
    if (B == &X86::FR64RegClass)
      return A;
    break;
  case X86::sub_xmm:
    if (B == &X86::VR128RegClass)
      return A;
    break;
  }
  return 0;
}

const TargetRegisterClass*
X86RegisterInfo::getLargestLegalSuperClass(const TargetRegisterClass *RC) const{
  const TargetRegisterClass *Super = RC;
  TargetRegisterClass::sc_iterator I = RC->superclasses_begin();
  do {
    switch (Super->getID()) {
    case X86::GR8RegClassID:
    case X86::GR16RegClassID:
    case X86::GR32RegClassID:
    case X86::GR64RegClassID:
    case X86::FR32RegClassID:
    case X86::FR64RegClassID:
    case X86::RFP32RegClassID:
    case X86::RFP64RegClassID:
    case X86::RFP80RegClassID:
    case X86::VR128RegClassID:
    case X86::VR256RegClassID:
      // Don't return a super-class that would shrink the spill size.
      // That can happen with the vector and float classes.
      if (Super->getSize() == RC->getSize())
        return Super;
    }
    Super = *I++;
  } while (Super);
  return RC;
}

const TargetRegisterClass *
X86RegisterInfo::getPointerRegClass(unsigned Kind) const {
  switch (Kind) {
  default: llvm_unreachable("Unexpected Kind in getPointerRegClass!");
  case 0: // Normal GPRs.
    if (TM.getSubtarget<X86Subtarget>().is64Bit())
      return &X86::GR64RegClass;
    return &X86::GR32RegClass;
  case 1: // Normal GPRs except the stack pointer (for encoding reasons).
    if (TM.getSubtarget<X86Subtarget>().is64Bit())
      return &X86::GR64_NOSPRegClass;
    return &X86::GR32_NOSPRegClass;
  case 2: // Available for tailcall (not callee-saved GPRs).
    if (TM.getSubtarget<X86Subtarget>().isTargetWin64())
      return &X86::GR64_TCW64RegClass;
    if (TM.getSubtarget<X86Subtarget>().is64Bit())
      return &X86::GR64_TCRegClass;
    return &X86::GR32_TCRegClass;
  }
}

const TargetRegisterClass *
X86RegisterInfo::getCrossCopyRegClass(const TargetRegisterClass *RC) const {
  if (RC == &X86::CCRRegClass) {
    if (Is64Bit)
      return &X86::GR64RegClass;
    else
      return &X86::GR32RegClass;
  }
  return RC;
}

unsigned
X86RegisterInfo::getRegPressureLimit(const TargetRegisterClass *RC,
                                     MachineFunction &MF) const {
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();

  unsigned FPDiff = TFI->hasFP(MF) ? 1 : 0;
  switch (RC->getID()) {
  default:
    return 0;
  case X86::GR32RegClassID:
    return 4 - FPDiff;
  case X86::GR64RegClassID:
    return 12 - FPDiff;
  case X86::VR128RegClassID:
    return TM.getSubtarget<X86Subtarget>().is64Bit() ? 10 : 4;
  case X86::VR64RegClassID:
    return 4;
  }
}

const unsigned *
X86RegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  bool callsEHReturn = false;
  bool ghcCall = false;

  if (MF) {
    callsEHReturn = MF->getMMI().callsEHReturn();
    const Function *F = MF->getFunction();
    ghcCall = (F ? F->getCallingConv() == CallingConv::GHC : false);
  }

  static const unsigned GhcCalleeSavedRegs[] = {
    0
  };

  static const unsigned CalleeSavedRegs32Bit[] = {
    X86::ESI, X86::EDI, X86::EBX, X86::EBP,  0
  };

  static const unsigned CalleeSavedRegs32EHRet[] = {
    X86::EAX, X86::EDX, X86::ESI, X86::EDI, X86::EBX, X86::EBP,  0
  };

  static const unsigned CalleeSavedRegs64Bit[] = {
    X86::RBX, X86::R12, X86::R13, X86::R14, X86::R15, X86::RBP, 0
  };

  static const unsigned CalleeSavedRegs64EHRet[] = {
    X86::RAX, X86::RDX, X86::RBX, X86::R12,
    X86::R13, X86::R14, X86::R15, X86::RBP, 0
  };

  static const unsigned CalleeSavedRegsWin64[] = {
    X86::RBX,   X86::RBP,   X86::RDI,   X86::RSI,
    X86::R12,   X86::R13,   X86::R14,   X86::R15,
    X86::XMM6,  X86::XMM7,  X86::XMM8,  X86::XMM9,
    X86::XMM10, X86::XMM11, X86::XMM12, X86::XMM13,
    X86::XMM14, X86::XMM15, 0
  };

  if (ghcCall) {
    return GhcCalleeSavedRegs;
  } else if (Is64Bit) {
    if (IsWin64)
      return CalleeSavedRegsWin64;
    else
      return (callsEHReturn ? CalleeSavedRegs64EHRet : CalleeSavedRegs64Bit);
  } else {
    return (callsEHReturn ? CalleeSavedRegs32EHRet : CalleeSavedRegs32Bit);
  }
}

BitVector X86RegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();

  // Set the stack-pointer register and its aliases as reserved.
  Reserved.set(X86::RSP);
  Reserved.set(X86::ESP);
  Reserved.set(X86::SP);
  Reserved.set(X86::SPL);

  // Set the instruction pointer register and its aliases as reserved.
  Reserved.set(X86::RIP);
  Reserved.set(X86::EIP);
  Reserved.set(X86::IP);

  // Set the frame-pointer register and its aliases as reserved if needed.
  if (TFI->hasFP(MF)) {
    Reserved.set(X86::RBP);
    Reserved.set(X86::EBP);
    Reserved.set(X86::BP);
    Reserved.set(X86::BPL);
  }

  // Mark the x87 stack registers as reserved, since they don't behave normally
  // with respect to liveness. We don't fully model the effects of x87 stack
  // pushes and pops after stackification.
  Reserved.set(X86::ST0);
  Reserved.set(X86::ST1);
  Reserved.set(X86::ST2);
  Reserved.set(X86::ST3);
  Reserved.set(X86::ST4);
  Reserved.set(X86::ST5);
  Reserved.set(X86::ST6);
  Reserved.set(X86::ST7);

  // Mark the segment registers as reserved.
  Reserved.set(X86::CS);
  Reserved.set(X86::SS);
  Reserved.set(X86::DS);
  Reserved.set(X86::ES);
  Reserved.set(X86::FS);
  Reserved.set(X86::GS);

  // Reserve the registers that only exist in 64-bit mode.
  if (!Is64Bit) {
    for (unsigned n = 0; n != 8; ++n) {
      const unsigned GPR64[] = {
        X86::R8,  X86::R9,  X86::R10, X86::R11,
        X86::R12, X86::R13, X86::R14, X86::R15
      };
      for (const unsigned *AI = getOverlaps(GPR64[n]); unsigned Reg = *AI;
           ++AI)
        Reserved.set(Reg);

      // XMM8, XMM9, ...
      assert(X86::XMM15 == X86::XMM8+7);
      for (const unsigned *AI = getOverlaps(X86::XMM8 + n); unsigned Reg = *AI;
           ++AI)
        Reserved.set(Reg);
    }
  }

  return Reserved;
}

//===----------------------------------------------------------------------===//
// Stack Frame Processing methods
//===----------------------------------------------------------------------===//

bool X86RegisterInfo::canRealignStack(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  return (RealignStack &&
          !MFI->hasVarSizedObjects());
}

bool X86RegisterInfo::needsStackRealignment(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  const Function *F = MF.getFunction();
  bool requiresRealignment = ((MFI->getMaxAlignment() > StackAlign) ||
                               F->hasFnAttr(Attribute::StackAlignment));

  // FIXME: Currently we don't support stack realignment for functions with
  //        variable-sized allocas.
  // FIXME: It's more complicated than this...
  if (0 && requiresRealignment && MFI->hasVarSizedObjects())
    report_fatal_error(
      "Stack realignment in presence of dynamic allocas is not supported");

  // If we've requested that we force align the stack do so now.
  if (ForceStackAlign)
    return canRealignStack(MF);

  return requiresRealignment && canRealignStack(MF);
}

bool X86RegisterInfo::hasReservedSpillSlot(const MachineFunction &MF,
                                           unsigned Reg, int &FrameIdx) const {
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();

  if (Reg == FramePtr && TFI->hasFP(MF)) {
    FrameIdx = MF.getFrameInfo()->getObjectIndexBegin();
    return true;
  }
  return false;
}

static unsigned getSUBriOpcode(unsigned is64Bit, int64_t Imm) {
  if (is64Bit) {
    if (isInt<8>(Imm))
      return X86::SUB64ri8;
    return X86::SUB64ri32;
  } else {
    if (isInt<8>(Imm))
      return X86::SUB32ri8;
    return X86::SUB32ri;
  }
}

static unsigned getADDriOpcode(unsigned is64Bit, int64_t Imm) {
  if (is64Bit) {
    if (isInt<8>(Imm))
      return X86::ADD64ri8;
    return X86::ADD64ri32;
  } else {
    if (isInt<8>(Imm))
      return X86::ADD32ri8;
    return X86::ADD32ri;
  }
}

void X86RegisterInfo::
eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I) const {
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();
  bool reseveCallFrame = TFI->hasReservedCallFrame(MF);
  int Opcode = I->getOpcode();
  bool isDestroy = Opcode == getCallFrameDestroyOpcode();
  DebugLoc DL = I->getDebugLoc();
  uint64_t Amount = !reseveCallFrame ? I->getOperand(0).getImm() : 0;
  uint64_t CalleeAmt = isDestroy ? I->getOperand(1).getImm() : 0;
  I = MBB.erase(I);

  if (!reseveCallFrame) {
    // If the stack pointer can be changed after prologue, turn the
    // adjcallstackup instruction into a 'sub ESP, <amt>' and the
    // adjcallstackdown instruction into 'add ESP, <amt>'
    // TODO: consider using push / pop instead of sub + store / add
    if (Amount == 0)
      return;

    // We need to keep the stack aligned properly.  To do this, we round the
    // amount of space needed for the outgoing arguments up to the next
    // alignment boundary.
    Amount = (Amount + StackAlign - 1) / StackAlign * StackAlign;

    MachineInstr *New = 0;
    if (Opcode == getCallFrameSetupOpcode()) {
      New = BuildMI(MF, DL, TII.get(getSUBriOpcode(Is64Bit, Amount)),
                    StackPtr)
        .addReg(StackPtr)
        .addImm(Amount);
    } else {
      assert(Opcode == getCallFrameDestroyOpcode());

      // Factor out the amount the callee already popped.
      Amount -= CalleeAmt;

      if (Amount) {
        unsigned Opc = getADDriOpcode(Is64Bit, Amount);
        New = BuildMI(MF, DL, TII.get(Opc), StackPtr)
          .addReg(StackPtr).addImm(Amount);
      }
    }

    if (New) {
      // The EFLAGS implicit def is dead.
      New->getOperand(3).setIsDead();

      // Replace the pseudo instruction with a new instruction.
      MBB.insert(I, New);
    }

    return;
  }

  if (Opcode == getCallFrameDestroyOpcode() && CalleeAmt) {
    // If we are performing frame pointer elimination and if the callee pops
    // something off the stack pointer, add it back.  We do this until we have
    // more advanced stack pointer tracking ability.
    unsigned Opc = getSUBriOpcode(Is64Bit, CalleeAmt);
    MachineInstr *New = BuildMI(MF, DL, TII.get(Opc), StackPtr)
      .addReg(StackPtr).addImm(CalleeAmt);

    // The EFLAGS implicit def is dead.
    New->getOperand(3).setIsDead();
    MBB.insert(I, New);
  }
}

void
X86RegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                     int SPAdj, RegScavenger *RS) const{
  assert(SPAdj == 0 && "Unexpected");

  unsigned i = 0;
  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getParent()->getParent();
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();

  while (!MI.getOperand(i).isFI()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }

  int FrameIndex = MI.getOperand(i).getIndex();
  unsigned BasePtr;

  unsigned Opc = MI.getOpcode();
  bool AfterFPPop = Opc == X86::TAILJMPm64 || Opc == X86::TAILJMPm;
  if (needsStackRealignment(MF))
    BasePtr = (FrameIndex < 0 ? FramePtr : StackPtr);
  else if (AfterFPPop)
    BasePtr = StackPtr;
  else
    BasePtr = (TFI->hasFP(MF) ? FramePtr : StackPtr);

  // This must be part of a four operand memory reference.  Replace the
  // FrameIndex with base register with EBP.  Add an offset to the offset.
  MI.getOperand(i).ChangeToRegister(BasePtr, false);

  // Now add the frame object offset to the offset from EBP.
  int FIOffset;
  if (AfterFPPop) {
    // Tail call jmp happens after FP is popped.
    const MachineFrameInfo *MFI = MF.getFrameInfo();
    FIOffset = MFI->getObjectOffset(FrameIndex) - TFI->getOffsetOfLocalArea();
  } else
    FIOffset = TFI->getFrameIndexOffset(MF, FrameIndex);

  if (MI.getOperand(i+3).isImm()) {
    // Offset is a 32-bit integer.
    int Offset = FIOffset + (int)(MI.getOperand(i + 3).getImm());
    MI.getOperand(i + 3).ChangeToImmediate(Offset);
  } else {
    // Offset is symbolic. This is extremely rare.
    uint64_t Offset = FIOffset + (uint64_t)MI.getOperand(i+3).getOffset();
    MI.getOperand(i+3).setOffset(Offset);
  }
}

unsigned X86RegisterInfo::getRARegister() const {
  return Is64Bit ? X86::RIP     // Should have dwarf #16.
                 : X86::EIP;    // Should have dwarf #8.
}

unsigned X86RegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  const TargetFrameLowering *TFI = MF.getTarget().getFrameLowering();
  return TFI->hasFP(MF) ? FramePtr : StackPtr;
}

unsigned X86RegisterInfo::getEHExceptionRegister() const {
  llvm_unreachable("What is the exception register");
  return 0;
}

unsigned X86RegisterInfo::getEHHandlerRegister() const {
  llvm_unreachable("What is the exception handler register");
  return 0;
}

namespace llvm {
unsigned getX86SubSuperRegister(unsigned Reg, EVT VT, bool High) {
  switch (VT.getSimpleVT().SimpleTy) {
  default: return Reg;
  case MVT::i8:
    if (High) {
      switch (Reg) {
      default: return 0;
      case X86::AH: case X86::AL: case X86::AX: case X86::EAX: case X86::RAX:
        return X86::AH;
      case X86::DH: case X86::DL: case X86::DX: case X86::EDX: case X86::RDX:
        return X86::DH;
      case X86::CH: case X86::CL: case X86::CX: case X86::ECX: case X86::RCX:
        return X86::CH;
      case X86::BH: case X86::BL: case X86::BX: case X86::EBX: case X86::RBX:
        return X86::BH;
      }
    } else {
      switch (Reg) {
      default: return 0;
      case X86::AH: case X86::AL: case X86::AX: case X86::EAX: case X86::RAX:
        return X86::AL;
      case X86::DH: case X86::DL: case X86::DX: case X86::EDX: case X86::RDX:
        return X86::DL;
      case X86::CH: case X86::CL: case X86::CX: case X86::ECX: case X86::RCX:
        return X86::CL;
      case X86::BH: case X86::BL: case X86::BX: case X86::EBX: case X86::RBX:
        return X86::BL;
      case X86::SIL: case X86::SI: case X86::ESI: case X86::RSI:
        return X86::SIL;
      case X86::DIL: case X86::DI: case X86::EDI: case X86::RDI:
        return X86::DIL;
      case X86::BPL: case X86::BP: case X86::EBP: case X86::RBP:
        return X86::BPL;
      case X86::SPL: case X86::SP: case X86::ESP: case X86::RSP:
        return X86::SPL;
      case X86::R8B: case X86::R8W: case X86::R8D: case X86::R8:
        return X86::R8B;
      case X86::R9B: case X86::R9W: case X86::R9D: case X86::R9:
        return X86::R9B;
      case X86::R10B: case X86::R10W: case X86::R10D: case X86::R10:
        return X86::R10B;
      case X86::R11B: case X86::R11W: case X86::R11D: case X86::R11:
        return X86::R11B;
      case X86::R12B: case X86::R12W: case X86::R12D: case X86::R12:
        return X86::R12B;
      case X86::R13B: case X86::R13W: case X86::R13D: case X86::R13:
        return X86::R13B;
      case X86::R14B: case X86::R14W: case X86::R14D: case X86::R14:
        return X86::R14B;
      case X86::R15B: case X86::R15W: case X86::R15D: case X86::R15:
        return X86::R15B;
      }
    }
  case MVT::i16:
    switch (Reg) {
    default: return Reg;
    case X86::AH: case X86::AL: case X86::AX: case X86::EAX: case X86::RAX:
      return X86::AX;
    case X86::DH: case X86::DL: case X86::DX: case X86::EDX: case X86::RDX:
      return X86::DX;
    case X86::CH: case X86::CL: case X86::CX: case X86::ECX: case X86::RCX:
      return X86::CX;
    case X86::BH: case X86::BL: case X86::BX: case X86::EBX: case X86::RBX:
      return X86::BX;
    case X86::SIL: case X86::SI: case X86::ESI: case X86::RSI:
      return X86::SI;
    case X86::DIL: case X86::DI: case X86::EDI: case X86::RDI:
      return X86::DI;
    case X86::BPL: case X86::BP: case X86::EBP: case X86::RBP:
      return X86::BP;
    case X86::SPL: case X86::SP: case X86::ESP: case X86::RSP:
      return X86::SP;
    case X86::R8B: case X86::R8W: case X86::R8D: case X86::R8:
      return X86::R8W;
    case X86::R9B: case X86::R9W: case X86::R9D: case X86::R9:
      return X86::R9W;
    case X86::R10B: case X86::R10W: case X86::R10D: case X86::R10:
      return X86::R10W;
    case X86::R11B: case X86::R11W: case X86::R11D: case X86::R11:
      return X86::R11W;
    case X86::R12B: case X86::R12W: case X86::R12D: case X86::R12:
      return X86::R12W;
    case X86::R13B: case X86::R13W: case X86::R13D: case X86::R13:
      return X86::R13W;
    case X86::R14B: case X86::R14W: case X86::R14D: case X86::R14:
      return X86::R14W;
    case X86::R15B: case X86::R15W: case X86::R15D: case X86::R15:
      return X86::R15W;
    }
  case MVT::i32:
    switch (Reg) {
    default: return Reg;
    case X86::AH: case X86::AL: case X86::AX: case X86::EAX: case X86::RAX:
      return X86::EAX;
    case X86::DH: case X86::DL: case X86::DX: case X86::EDX: case X86::RDX:
      return X86::EDX;
    case X86::CH: case X86::CL: case X86::CX: case X86::ECX: case X86::RCX:
      return X86::ECX;
    case X86::BH: case X86::BL: case X86::BX: case X86::EBX: case X86::RBX:
      return X86::EBX;
    case X86::SIL: case X86::SI: case X86::ESI: case X86::RSI:
      return X86::ESI;
    case X86::DIL: case X86::DI: case X86::EDI: case X86::RDI:
      return X86::EDI;
    case X86::BPL: case X86::BP: case X86::EBP: case X86::RBP:
      return X86::EBP;
    case X86::SPL: case X86::SP: case X86::ESP: case X86::RSP:
      return X86::ESP;
    case X86::R8B: case X86::R8W: case X86::R8D: case X86::R8:
      return X86::R8D;
    case X86::R9B: case X86::R9W: case X86::R9D: case X86::R9:
      return X86::R9D;
    case X86::R10B: case X86::R10W: case X86::R10D: case X86::R10:
      return X86::R10D;
    case X86::R11B: case X86::R11W: case X86::R11D: case X86::R11:
      return X86::R11D;
    case X86::R12B: case X86::R12W: case X86::R12D: case X86::R12:
      return X86::R12D;
    case X86::R13B: case X86::R13W: case X86::R13D: case X86::R13:
      return X86::R13D;
    case X86::R14B: case X86::R14W: case X86::R14D: case X86::R14:
      return X86::R14D;
    case X86::R15B: case X86::R15W: case X86::R15D: case X86::R15:
      return X86::R15D;
    }
  case MVT::i64:
    switch (Reg) {
    default: return Reg;
    case X86::AH: case X86::AL: case X86::AX: case X86::EAX: case X86::RAX:
      return X86::RAX;
    case X86::DH: case X86::DL: case X86::DX: case X86::EDX: case X86::RDX:
      return X86::RDX;
    case X86::CH: case X86::CL: case X86::CX: case X86::ECX: case X86::RCX:
      return X86::RCX;
    case X86::BH: case X86::BL: case X86::BX: case X86::EBX: case X86::RBX:
      return X86::RBX;
    case X86::SIL: case X86::SI: case X86::ESI: case X86::RSI:
      return X86::RSI;
    case X86::DIL: case X86::DI: case X86::EDI: case X86::RDI:
      return X86::RDI;
    case X86::BPL: case X86::BP: case X86::EBP: case X86::RBP:
      return X86::RBP;
    case X86::SPL: case X86::SP: case X86::ESP: case X86::RSP:
      return X86::RSP;
    case X86::R8B: case X86::R8W: case X86::R8D: case X86::R8:
      return X86::R8;
    case X86::R9B: case X86::R9W: case X86::R9D: case X86::R9:
      return X86::R9;
    case X86::R10B: case X86::R10W: case X86::R10D: case X86::R10:
      return X86::R10;
    case X86::R11B: case X86::R11W: case X86::R11D: case X86::R11:
      return X86::R11;
    case X86::R12B: case X86::R12W: case X86::R12D: case X86::R12:
      return X86::R12;
    case X86::R13B: case X86::R13W: case X86::R13D: case X86::R13:
      return X86::R13;
    case X86::R14B: case X86::R14W: case X86::R14D: case X86::R14:
      return X86::R14;
    case X86::R15B: case X86::R15W: case X86::R15D: case X86::R15:
      return X86::R15;
    }
  }

  return Reg;
}
}

#include "X86GenRegisterInfo.inc"

namespace {
  struct MSAH : public MachineFunctionPass {
    static char ID;
    MSAH() : MachineFunctionPass(ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF) {
      const X86TargetMachine *TM =
        static_cast<const X86TargetMachine *>(&MF.getTarget());
      const X86RegisterInfo *X86RI = TM->getRegisterInfo();
      MachineRegisterInfo &RI = MF.getRegInfo();
      X86MachineFunctionInfo *FuncInfo = MF.getInfo<X86MachineFunctionInfo>();
      unsigned StackAlignment = X86RI->getStackAlignment();

      // Be over-conservative: scan over all vreg defs and find whether vector
      // registers are used. If yes, there is a possibility that vector register
      // will be spilled and thus require dynamic stack realignment.
      for (unsigned i = 0, e = RI.getNumVirtRegs(); i != e; ++i) {
        unsigned Reg = TargetRegisterInfo::index2VirtReg(i);
        if (RI.getRegClass(Reg)->getAlignment() > StackAlignment) {
          FuncInfo->setReserveFP(true);
          return true;
        }
      }
      // Nothing to do
      return false;
    }

    virtual const char *getPassName() const {
      return "X86 Maximal Stack Alignment Check";
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      MachineFunctionPass::getAnalysisUsage(AU);
    }
  };

  char MSAH::ID = 0;
}

FunctionPass*
llvm::createX86MaxStackAlignmentHeuristicPass() { return new MSAH(); }
