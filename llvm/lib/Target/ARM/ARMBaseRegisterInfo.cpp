//===- ARMBaseRegisterInfo.cpp - ARM Register Information -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the base ARM implementation of TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMAddressingModes.h"
#include "ARMBaseRegisterInfo.h"
#include "ARMInstrInfo.h"
#include "ARMMachineFunctionInfo.h"
#include "ARMSubtarget.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
using namespace llvm;

unsigned ARMBaseRegisterInfo::getRegisterNumbering(unsigned RegEnum) {
  using namespace ARM;
  switch (RegEnum) {
  case R0:  case S0:  case D0:  return 0;
  case R1:  case S1:  case D1:  return 1;
  case R2:  case S2:  case D2:  return 2;
  case R3:  case S3:  case D3:  return 3;
  case R4:  case S4:  case D4:  return 4;
  case R5:  case S5:  case D5:  return 5;
  case R6:  case S6:  case D6:  return 6;
  case R7:  case S7:  case D7:  return 7;
  case R8:  case S8:  case D8:  return 8;
  case R9:  case S9:  case D9:  return 9;
  case R10: case S10: case D10: return 10;
  case R11: case S11: case D11: return 11;
  case R12: case S12: case D12: return 12;
  case SP:  case S13: case D13: return 13;
  case LR:  case S14: case D14: return 14;
  case PC:  case S15: case D15: return 15;
  case S16: return 16;
  case S17: return 17;
  case S18: return 18;
  case S19: return 19;
  case S20: return 20;
  case S21: return 21;
  case S22: return 22;
  case S23: return 23;
  case S24: return 24;
  case S25: return 25;
  case S26: return 26;
  case S27: return 27;
  case S28: return 28;
  case S29: return 29;
  case S30: return 30;
  case S31: return 31;
  default:
    assert(0 && "Unknown ARM register!");
    abort();
  }
}

unsigned ARMBaseRegisterInfo::getRegisterNumbering(unsigned RegEnum,
                                                   bool &isSPVFP) {
  isSPVFP = false;

  using namespace ARM;
  switch (RegEnum) {
  default:
    assert(0 && "Unknown ARM register!");
    abort();
  case R0:  case D0:  return 0;
  case R1:  case D1:  return 1;
  case R2:  case D2:  return 2;
  case R3:  case D3:  return 3;
  case R4:  case D4:  return 4;
  case R5:  case D5:  return 5;
  case R6:  case D6:  return 6;
  case R7:  case D7:  return 7;
  case R8:  case D8:  return 8;
  case R9:  case D9:  return 9;
  case R10: case D10: return 10;
  case R11: case D11: return 11;
  case R12: case D12: return 12;
  case SP:  case D13: return 13;
  case LR:  case D14: return 14;
  case PC:  case D15: return 15;

  case S0: case S1: case S2: case S3:
  case S4: case S5: case S6: case S7:
  case S8: case S9: case S10: case S11:
  case S12: case S13: case S14: case S15:
  case S16: case S17: case S18: case S19:
  case S20: case S21: case S22: case S23:
  case S24: case S25: case S26: case S27:
  case S28: case S29: case S30: case S31:  {
    isSPVFP = true;
    switch (RegEnum) {
    default: return 0; // Avoid compile time warning.
    case S0: return 0;
    case S1: return 1;
    case S2: return 2;
    case S3: return 3;
    case S4: return 4;
    case S5: return 5;
    case S6: return 6;
    case S7: return 7;
    case S8: return 8;
    case S9: return 9;
    case S10: return 10;
    case S11: return 11;
    case S12: return 12;
    case S13: return 13;
    case S14: return 14;
    case S15: return 15;
    case S16: return 16;
    case S17: return 17;
    case S18: return 18;
    case S19: return 19;
    case S20: return 20;
    case S21: return 21;
    case S22: return 22;
    case S23: return 23;
    case S24: return 24;
    case S25: return 25;
    case S26: return 26;
    case S27: return 27;
    case S28: return 28;
    case S29: return 29;
    case S30: return 30;
    case S31: return 31;
    }
  }
  }
}

ARMBaseRegisterInfo::ARMBaseRegisterInfo(const TargetInstrInfo &tii,
                                         const ARMSubtarget &sti)
  : ARMGenRegisterInfo(ARM::ADJCALLSTACKDOWN, ARM::ADJCALLSTACKUP),
    TII(tii), STI(sti),
    FramePtr((STI.isTargetDarwin() || STI.isThumb()) ? ARM::R7 : ARM::R11) {
}

const unsigned*
ARMBaseRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  static const unsigned CalleeSavedRegs[] = {
    ARM::LR, ARM::R11, ARM::R10, ARM::R9, ARM::R8,
    ARM::R7, ARM::R6,  ARM::R5,  ARM::R4,

    ARM::D15, ARM::D14, ARM::D13, ARM::D12,
    ARM::D11, ARM::D10, ARM::D9,  ARM::D8,
    0
  };

  static const unsigned DarwinCalleeSavedRegs[] = {
    // Darwin ABI deviates from ARM standard ABI. R9 is not a callee-saved
    // register.
    ARM::LR,  ARM::R7,  ARM::R6, ARM::R5, ARM::R4,
    ARM::R11, ARM::R10, ARM::R8,

    ARM::D15, ARM::D14, ARM::D13, ARM::D12,
    ARM::D11, ARM::D10, ARM::D9,  ARM::D8,
    0
  };
  return STI.isTargetDarwin() ? DarwinCalleeSavedRegs : CalleeSavedRegs;
}

const TargetRegisterClass* const *
ARMBaseRegisterInfo::getCalleeSavedRegClasses(const MachineFunction *MF) const {
  static const TargetRegisterClass * const CalleeSavedRegClasses[] = {
    &ARM::GPRRegClass, &ARM::GPRRegClass, &ARM::GPRRegClass,
    &ARM::GPRRegClass, &ARM::GPRRegClass, &ARM::GPRRegClass,
    &ARM::GPRRegClass, &ARM::GPRRegClass, &ARM::GPRRegClass,

    &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass,
    &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass,
    0
  };

  static const TargetRegisterClass * const ThumbCalleeSavedRegClasses[] = {
    &ARM::GPRRegClass, &ARM::GPRRegClass, &ARM::GPRRegClass,
    &ARM::GPRRegClass, &ARM::GPRRegClass, &ARM::tGPRRegClass,
    &ARM::tGPRRegClass,&ARM::tGPRRegClass,&ARM::tGPRRegClass,

    &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass,
    &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass,
    0
  };

  static const TargetRegisterClass * const DarwinCalleeSavedRegClasses[] = {
    &ARM::GPRRegClass, &ARM::GPRRegClass, &ARM::GPRRegClass,
    &ARM::GPRRegClass, &ARM::GPRRegClass, &ARM::GPRRegClass,
    &ARM::GPRRegClass, &ARM::GPRRegClass,

    &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass,
    &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass,
    0
  };

  static const TargetRegisterClass * const DarwinThumbCalleeSavedRegClasses[] ={
    &ARM::GPRRegClass,  &ARM::tGPRRegClass, &ARM::tGPRRegClass,
    &ARM::tGPRRegClass, &ARM::tGPRRegClass, &ARM::GPRRegClass,
    &ARM::GPRRegClass,  &ARM::GPRRegClass,

    &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass,
    &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass, &ARM::DPRRegClass,
    0
  };

  if (STI.isThumb()) {
    return STI.isTargetDarwin()
      ? DarwinThumbCalleeSavedRegClasses : ThumbCalleeSavedRegClasses;
  }
  return STI.isTargetDarwin()
    ? DarwinCalleeSavedRegClasses : CalleeSavedRegClasses;
}

BitVector ARMBaseRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  // FIXME: avoid re-calculating this everytime.
  BitVector Reserved(getNumRegs());
  Reserved.set(ARM::SP);
  Reserved.set(ARM::PC);
  if (STI.isTargetDarwin() || hasFP(MF))
    Reserved.set(FramePtr);
  // Some targets reserve R9.
  if (STI.isR9Reserved())
    Reserved.set(ARM::R9);
  return Reserved;
}

bool
ARMBaseRegisterInfo::isReservedReg(const MachineFunction &MF, unsigned Reg) const {
  switch (Reg) {
  default: break;
  case ARM::SP:
  case ARM::PC:
    return true;
  case ARM::R7:
  case ARM::R11:
    if (FramePtr == Reg && (STI.isTargetDarwin() || hasFP(MF)))
      return true;
    break;
  case ARM::R9:
    return STI.isR9Reserved();
  }

  return false;
}

const TargetRegisterClass *ARMBaseRegisterInfo::getPointerRegClass() const {
  return &ARM::GPRRegClass;
}

/// getAllocationOrder - Returns the register allocation order for a specified
/// register class in the form of a pair of TargetRegisterClass iterators.
std::pair<TargetRegisterClass::iterator,TargetRegisterClass::iterator>
ARMBaseRegisterInfo::getAllocationOrder(const TargetRegisterClass *RC,
                                        unsigned HintType, unsigned HintReg,
                                        const MachineFunction &MF) const {
  // Alternative register allocation orders when favoring even / odd registers
  // of register pairs.

  // No FP, R9 is available.
  static const unsigned GPREven1[] = {
    ARM::R0, ARM::R2, ARM::R4, ARM::R6, ARM::R8, ARM::R10,
    ARM::R1, ARM::R3, ARM::R12,ARM::LR, ARM::R5, ARM::R7,
    ARM::R9, ARM::R11
  };
  static const unsigned GPROdd1[] = {
    ARM::R1, ARM::R3, ARM::R5, ARM::R7, ARM::R9, ARM::R11,
    ARM::R0, ARM::R2, ARM::R12,ARM::LR, ARM::R4, ARM::R6,
    ARM::R8, ARM::R10
  };

  // FP is R7, R9 is available.
  static const unsigned GPREven2[] = {
    ARM::R0, ARM::R2, ARM::R4,          ARM::R8, ARM::R10,
    ARM::R1, ARM::R3, ARM::R12,ARM::LR, ARM::R5, ARM::R6,
    ARM::R9, ARM::R11
  };
  static const unsigned GPROdd2[] = {
    ARM::R1, ARM::R3, ARM::R5,          ARM::R9, ARM::R11,
    ARM::R0, ARM::R2, ARM::R12,ARM::LR, ARM::R4, ARM::R6,
    ARM::R8, ARM::R10
  };

  // FP is R11, R9 is available.
  static const unsigned GPREven3[] = {
    ARM::R0, ARM::R2, ARM::R4, ARM::R6, ARM::R8,
    ARM::R1, ARM::R3, ARM::R10,ARM::R12,ARM::LR, ARM::R5, ARM::R7,
    ARM::R9
  };
  static const unsigned GPROdd3[] = {
    ARM::R1, ARM::R3, ARM::R5, ARM::R6, ARM::R9,
    ARM::R0, ARM::R2, ARM::R10,ARM::R12,ARM::LR, ARM::R4, ARM::R7,
    ARM::R8
  };

  // No FP, R9 is not available.
  static const unsigned GPREven4[] = {
    ARM::R0, ARM::R2, ARM::R4, ARM::R6,          ARM::R10,
    ARM::R1, ARM::R3, ARM::R12,ARM::LR, ARM::R5, ARM::R7, ARM::R8,
    ARM::R11
  };
  static const unsigned GPROdd4[] = {
    ARM::R1, ARM::R3, ARM::R5, ARM::R7,          ARM::R11,
    ARM::R0, ARM::R2, ARM::R12,ARM::LR, ARM::R4, ARM::R6, ARM::R8,
    ARM::R10
  };

  // FP is R7, R9 is not available.
  static const unsigned GPREven5[] = {
    ARM::R0, ARM::R2, ARM::R4,                   ARM::R10,
    ARM::R1, ARM::R3, ARM::R12,ARM::LR, ARM::R5, ARM::R6, ARM::R8,
    ARM::R11
  };
  static const unsigned GPROdd5[] = {
    ARM::R1, ARM::R3, ARM::R5,                   ARM::R11,
    ARM::R0, ARM::R2, ARM::R12,ARM::LR, ARM::R4, ARM::R6, ARM::R8,
    ARM::R10
  };

  // FP is R11, R9 is not available.
  static const unsigned GPREven6[] = {
    ARM::R0, ARM::R2, ARM::R4, ARM::R6,
    ARM::R1, ARM::R3, ARM::R10,ARM::R12,ARM::LR, ARM::R5, ARM::R7, ARM::R8
  };
  static const unsigned GPROdd6[] = {
    ARM::R1, ARM::R3, ARM::R5, ARM::R7,
    ARM::R0, ARM::R2, ARM::R10,ARM::R12,ARM::LR, ARM::R4, ARM::R6, ARM::R8
  };


  if (HintType == ARMRI::RegPairEven) {
    if (isPhysicalRegister(HintReg) && getRegisterPairEven(HintReg, MF) == 0)
      // It's no longer possible to fulfill this hint. Return the default
      // allocation order.
      return std::make_pair(RC->allocation_order_begin(MF),
                            RC->allocation_order_end(MF));

    if (!STI.isTargetDarwin() && !hasFP(MF)) {
      if (!STI.isR9Reserved())
        return std::make_pair(GPREven1,
                              GPREven1 + (sizeof(GPREven1)/sizeof(unsigned)));
      else
        return std::make_pair(GPREven4,
                              GPREven4 + (sizeof(GPREven4)/sizeof(unsigned)));
    } else if (FramePtr == ARM::R7) {
      if (!STI.isR9Reserved())
        return std::make_pair(GPREven2,
                              GPREven2 + (sizeof(GPREven2)/sizeof(unsigned)));
      else
        return std::make_pair(GPREven5,
                              GPREven5 + (sizeof(GPREven5)/sizeof(unsigned)));
    } else { // FramePtr == ARM::R11
      if (!STI.isR9Reserved())
        return std::make_pair(GPREven3,
                              GPREven3 + (sizeof(GPREven3)/sizeof(unsigned)));
      else
        return std::make_pair(GPREven6,
                              GPREven6 + (sizeof(GPREven6)/sizeof(unsigned)));
    }
  } else if (HintType == ARMRI::RegPairOdd) {
    if (isPhysicalRegister(HintReg) && getRegisterPairOdd(HintReg, MF) == 0)
      // It's no longer possible to fulfill this hint. Return the default
      // allocation order.
      return std::make_pair(RC->allocation_order_begin(MF),
                            RC->allocation_order_end(MF));

    if (!STI.isTargetDarwin() && !hasFP(MF)) {
      if (!STI.isR9Reserved())
        return std::make_pair(GPROdd1,
                              GPROdd1 + (sizeof(GPROdd1)/sizeof(unsigned)));
      else
        return std::make_pair(GPROdd4,
                              GPROdd4 + (sizeof(GPROdd4)/sizeof(unsigned)));
    } else if (FramePtr == ARM::R7) {
      if (!STI.isR9Reserved())
        return std::make_pair(GPROdd2,
                              GPROdd2 + (sizeof(GPROdd2)/sizeof(unsigned)));
      else
        return std::make_pair(GPROdd5,
                              GPROdd5 + (sizeof(GPROdd5)/sizeof(unsigned)));
    } else { // FramePtr == ARM::R11
      if (!STI.isR9Reserved())
        return std::make_pair(GPROdd3,
                              GPROdd3 + (sizeof(GPROdd3)/sizeof(unsigned)));
      else
        return std::make_pair(GPROdd6,
                              GPROdd6 + (sizeof(GPROdd6)/sizeof(unsigned)));
    }
  }
  return std::make_pair(RC->allocation_order_begin(MF),
                        RC->allocation_order_end(MF));
}

/// ResolveRegAllocHint - Resolves the specified register allocation hint
/// to a physical register. Returns the physical register if it is successful.
unsigned
ARMBaseRegisterInfo::ResolveRegAllocHint(unsigned Type, unsigned Reg,
                                         const MachineFunction &MF) const {
  if (Reg == 0 || !isPhysicalRegister(Reg))
    return 0;
  if (Type == 0)
    return Reg;
  else if (Type == (unsigned)ARMRI::RegPairOdd)
    // Odd register.
    return getRegisterPairOdd(Reg, MF);
  else if (Type == (unsigned)ARMRI::RegPairEven)
    // Even register.
    return getRegisterPairEven(Reg, MF);
  return 0;
}

void
ARMBaseRegisterInfo::UpdateRegAllocHint(unsigned Reg, unsigned NewReg,
                                        MachineFunction &MF) const {
  MachineRegisterInfo *MRI = &MF.getRegInfo();
  std::pair<unsigned, unsigned> Hint = MRI->getRegAllocationHint(Reg);
  if ((Hint.first == (unsigned)ARMRI::RegPairOdd ||
       Hint.first == (unsigned)ARMRI::RegPairEven) &&
      Hint.second && TargetRegisterInfo::isVirtualRegister(Hint.second)) {
    // If 'Reg' is one of the even / odd register pair and it's now changed
    // (e.g. coalesced) into a different register. The other register of the
    // pair allocation hint must be updated to reflect the relationship
    // change.
    unsigned OtherReg = Hint.second;
    Hint = MRI->getRegAllocationHint(OtherReg);
    if (Hint.second == Reg)
      // Make sure the pair has not already divorced.
      MRI->setRegAllocationHint(OtherReg, Hint.first, NewReg);
  }
}

/// hasFP - Return true if the specified function should have a dedicated frame
/// pointer register.  This is true if the function has variable sized allocas
/// or if frame pointer elimination is disabled.
///
bool ARMBaseRegisterInfo::hasFP(const MachineFunction &MF) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  return (NoFramePointerElim ||
          MFI->hasVarSizedObjects() ||
          MFI->isFrameAddressTaken());
}

static unsigned estimateStackSize(MachineFunction &MF, MachineFrameInfo *MFI) {
  const MachineFrameInfo *FFI = MF.getFrameInfo();
  int Offset = 0;
  for (int i = FFI->getObjectIndexBegin(); i != 0; ++i) {
    int FixedOff = -FFI->getObjectOffset(i);
    if (FixedOff > Offset) Offset = FixedOff;
  }
  for (unsigned i = 0, e = FFI->getObjectIndexEnd(); i != e; ++i) {
    if (FFI->isDeadObjectIndex(i))
      continue;
    Offset += FFI->getObjectSize(i);
    unsigned Align = FFI->getObjectAlignment(i);
    // Adjust to alignment boundary
    Offset = (Offset+Align-1)/Align*Align;
  }
  return (unsigned)Offset;
}

void
ARMBaseRegisterInfo::processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                                          RegScavenger *RS) const {
  // This tells PEI to spill the FP as if it is any other callee-save register
  // to take advantage the eliminateFrameIndex machinery. This also ensures it
  // is spilled in the order specified by getCalleeSavedRegs() to make it easier
  // to combine multiple loads / stores.
  bool CanEliminateFrame = true;
  bool CS1Spilled = false;
  bool LRSpilled = false;
  unsigned NumGPRSpills = 0;
  SmallVector<unsigned, 4> UnspilledCS1GPRs;
  SmallVector<unsigned, 4> UnspilledCS2GPRs;
  ARMFunctionInfo *AFI = MF.getInfo<ARMFunctionInfo>();

  // Don't spill FP if the frame can be eliminated. This is determined
  // by scanning the callee-save registers to see if any is used.
  const unsigned *CSRegs = getCalleeSavedRegs();
  const TargetRegisterClass* const *CSRegClasses = getCalleeSavedRegClasses();
  for (unsigned i = 0; CSRegs[i]; ++i) {
    unsigned Reg = CSRegs[i];
    bool Spilled = false;
    if (MF.getRegInfo().isPhysRegUsed(Reg)) {
      AFI->setCSRegisterIsSpilled(Reg);
      Spilled = true;
      CanEliminateFrame = false;
    } else {
      // Check alias registers too.
      for (const unsigned *Aliases = getAliasSet(Reg); *Aliases; ++Aliases) {
        if (MF.getRegInfo().isPhysRegUsed(*Aliases)) {
          Spilled = true;
          CanEliminateFrame = false;
        }
      }
    }

    if (CSRegClasses[i] == &ARM::GPRRegClass) {
      if (Spilled) {
        NumGPRSpills++;

        if (!STI.isTargetDarwin()) {
          if (Reg == ARM::LR)
            LRSpilled = true;
          CS1Spilled = true;
          continue;
        }

        // Keep track if LR and any of R4, R5, R6, and R7 is spilled.
        switch (Reg) {
        case ARM::LR:
          LRSpilled = true;
          // Fallthrough
        case ARM::R4:
        case ARM::R5:
        case ARM::R6:
        case ARM::R7:
          CS1Spilled = true;
          break;
        default:
          break;
        }
      } else {
        if (!STI.isTargetDarwin()) {
          UnspilledCS1GPRs.push_back(Reg);
          continue;
        }

        switch (Reg) {
        case ARM::R4:
        case ARM::R5:
        case ARM::R6:
        case ARM::R7:
        case ARM::LR:
          UnspilledCS1GPRs.push_back(Reg);
          break;
        default:
          UnspilledCS2GPRs.push_back(Reg);
          break;
        }
      }
    }
  }

  bool ForceLRSpill = false;
  if (!LRSpilled && AFI->isThumbFunction()) {
    unsigned FnSize = TII.GetFunctionSizeInBytes(MF);
    // Force LR to be spilled if the Thumb function size is > 2048. This enables
    // use of BL to implement far jump. If it turns out that it's not needed
    // then the branch fix up path will undo it.
    if (FnSize >= (1 << 11)) {
      CanEliminateFrame = false;
      ForceLRSpill = true;
    }
  }

  bool ExtraCSSpill = false;
  if (!CanEliminateFrame || hasFP(MF)) {
    AFI->setHasStackFrame(true);

    // If LR is not spilled, but at least one of R4, R5, R6, and R7 is spilled.
    // Spill LR as well so we can fold BX_RET to the registers restore (LDM).
    if (!LRSpilled && CS1Spilled) {
      MF.getRegInfo().setPhysRegUsed(ARM::LR);
      AFI->setCSRegisterIsSpilled(ARM::LR);
      NumGPRSpills++;
      UnspilledCS1GPRs.erase(std::find(UnspilledCS1GPRs.begin(),
                                    UnspilledCS1GPRs.end(), (unsigned)ARM::LR));
      ForceLRSpill = false;
      ExtraCSSpill = true;
    }

    // Darwin ABI requires FP to point to the stack slot that contains the
    // previous FP.
    if (STI.isTargetDarwin() || hasFP(MF)) {
      MF.getRegInfo().setPhysRegUsed(FramePtr);
      NumGPRSpills++;
    }

    // If stack and double are 8-byte aligned and we are spilling an odd number
    // of GPRs. Spill one extra callee save GPR so we won't have to pad between
    // the integer and double callee save areas.
    unsigned TargetAlign = MF.getTarget().getFrameInfo()->getStackAlignment();
    if (TargetAlign == 8 && (NumGPRSpills & 1)) {
      if (CS1Spilled && !UnspilledCS1GPRs.empty()) {
        for (unsigned i = 0, e = UnspilledCS1GPRs.size(); i != e; ++i) {
          unsigned Reg = UnspilledCS1GPRs[i];
          // Don't spiil high register if the function is thumb
          if (!AFI->isThumbFunction() ||
              isARMLowRegister(Reg) || Reg == ARM::LR) {
            MF.getRegInfo().setPhysRegUsed(Reg);
            AFI->setCSRegisterIsSpilled(Reg);
            if (!isReservedReg(MF, Reg))
              ExtraCSSpill = true;
            break;
          }
        }
      } else if (!UnspilledCS2GPRs.empty() &&
                 !AFI->isThumbFunction()) {
        unsigned Reg = UnspilledCS2GPRs.front();
        MF.getRegInfo().setPhysRegUsed(Reg);
        AFI->setCSRegisterIsSpilled(Reg);
        if (!isReservedReg(MF, Reg))
          ExtraCSSpill = true;
      }
    }

    // Estimate if we might need to scavenge a register at some point in order
    // to materialize a stack offset. If so, either spill one additional
    // callee-saved register or reserve a special spill slot to facilitate
    // register scavenging.
    if (RS && !ExtraCSSpill && !AFI->isThumbFunction()) {
      MachineFrameInfo  *MFI = MF.getFrameInfo();
      unsigned Size = estimateStackSize(MF, MFI);
      unsigned Limit = (1 << 12) - 1;
      for (MachineFunction::iterator BB = MF.begin(),E = MF.end();BB != E; ++BB)
        for (MachineBasicBlock::iterator I= BB->begin(); I != BB->end(); ++I) {
          for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
            if (I->getOperand(i).isFI()) {
              unsigned Opcode = I->getOpcode();
              const TargetInstrDesc &Desc = TII.get(Opcode);
              unsigned AddrMode = (Desc.TSFlags & ARMII::AddrModeMask);
              if (AddrMode == ARMII::AddrMode3) {
                Limit = (1 << 8) - 1;
                goto DoneEstimating;
              } else if (AddrMode == ARMII::AddrMode5) {
                unsigned ThisLimit = ((1 << 8) - 1) * 4;
                if (ThisLimit < Limit)
                  Limit = ThisLimit;
              }
            }
        }
    DoneEstimating:
      if (Size >= Limit) {
        // If any non-reserved CS register isn't spilled, just spill one or two
        // extra. That should take care of it!
        unsigned NumExtras = TargetAlign / 4;
        SmallVector<unsigned, 2> Extras;
        while (NumExtras && !UnspilledCS1GPRs.empty()) {
          unsigned Reg = UnspilledCS1GPRs.back();
          UnspilledCS1GPRs.pop_back();
          if (!isReservedReg(MF, Reg)) {
            Extras.push_back(Reg);
            NumExtras--;
          }
        }
        while (NumExtras && !UnspilledCS2GPRs.empty()) {
          unsigned Reg = UnspilledCS2GPRs.back();
          UnspilledCS2GPRs.pop_back();
          if (!isReservedReg(MF, Reg)) {
            Extras.push_back(Reg);
            NumExtras--;
          }
        }
        if (Extras.size() && NumExtras == 0) {
          for (unsigned i = 0, e = Extras.size(); i != e; ++i) {
            MF.getRegInfo().setPhysRegUsed(Extras[i]);
            AFI->setCSRegisterIsSpilled(Extras[i]);
          }
        } else {
          // Reserve a slot closest to SP or frame pointer.
          const TargetRegisterClass *RC = &ARM::GPRRegClass;
          RS->setScavengingFrameIndex(MFI->CreateStackObject(RC->getSize(),
                                                           RC->getAlignment()));
        }
      }
    }
  }

  if (ForceLRSpill) {
    MF.getRegInfo().setPhysRegUsed(ARM::LR);
    AFI->setCSRegisterIsSpilled(ARM::LR);
    AFI->setLRIsSpilledForFarJump(true);
  }
}

unsigned ARMBaseRegisterInfo::getRARegister() const {
  return ARM::LR;
}

unsigned ARMBaseRegisterInfo::getFrameRegister(MachineFunction &MF) const {
  if (STI.isTargetDarwin() || hasFP(MF))
    return FramePtr;
  return ARM::SP;
}

unsigned ARMBaseRegisterInfo::getEHExceptionRegister() const {
  assert(0 && "What is the exception register");
  return 0;
}

unsigned ARMBaseRegisterInfo::getEHHandlerRegister() const {
  assert(0 && "What is the exception handler register");
  return 0;
}

int ARMBaseRegisterInfo::getDwarfRegNum(unsigned RegNum, bool isEH) const {
  return ARMGenRegisterInfo::getDwarfRegNumFull(RegNum, 0);
}

unsigned ARMBaseRegisterInfo::getRegisterPairEven(unsigned Reg,
                                               const MachineFunction &MF) const {
  switch (Reg) {
  default: break;
  // Return 0 if either register of the pair is a special register.
  // So no R12, etc.
  case ARM::R1:
    return ARM::R0;
  case ARM::R3:
    // FIXME!
    return STI.isThumb() ? 0 : ARM::R2;
  case ARM::R5:
    return ARM::R4;
  case ARM::R7:
    return isReservedReg(MF, ARM::R7)  ? 0 : ARM::R6;
  case ARM::R9:
    return isReservedReg(MF, ARM::R9)  ? 0 :ARM::R8;
  case ARM::R11:
    return isReservedReg(MF, ARM::R11) ? 0 : ARM::R10;

  case ARM::S1:
    return ARM::S0;
  case ARM::S3:
    return ARM::S2;
  case ARM::S5:
    return ARM::S4;
  case ARM::S7:
    return ARM::S6;
  case ARM::S9:
    return ARM::S8;
  case ARM::S11:
    return ARM::S10;
  case ARM::S13:
    return ARM::S12;
  case ARM::S15:
    return ARM::S14;
  case ARM::S17:
    return ARM::S16;
  case ARM::S19:
    return ARM::S18;
  case ARM::S21:
    return ARM::S20;
  case ARM::S23:
    return ARM::S22;
  case ARM::S25:
    return ARM::S24;
  case ARM::S27:
    return ARM::S26;
  case ARM::S29:
    return ARM::S28;
  case ARM::S31:
    return ARM::S30;

  case ARM::D1:
    return ARM::D0;
  case ARM::D3:
    return ARM::D2;
  case ARM::D5:
    return ARM::D4;
  case ARM::D7:
    return ARM::D6;
  case ARM::D9:
    return ARM::D8;
  case ARM::D11:
    return ARM::D10;
  case ARM::D13:
    return ARM::D12;
  case ARM::D15:
    return ARM::D14;
  }

  return 0;
}

unsigned ARMBaseRegisterInfo::getRegisterPairOdd(unsigned Reg,
                                             const MachineFunction &MF) const {
  switch (Reg) {
  default: break;
  // Return 0 if either register of the pair is a special register.
  // So no R12, etc.
  case ARM::R0:
    return ARM::R1;
  case ARM::R2:
    // FIXME!
    return STI.isThumb() ? 0 : ARM::R3;
  case ARM::R4:
    return ARM::R5;
  case ARM::R6:
    return isReservedReg(MF, ARM::R7)  ? 0 : ARM::R7;
  case ARM::R8:
    return isReservedReg(MF, ARM::R9)  ? 0 :ARM::R9;
  case ARM::R10:
    return isReservedReg(MF, ARM::R11) ? 0 : ARM::R11;

  case ARM::S0:
    return ARM::S1;
  case ARM::S2:
    return ARM::S3;
  case ARM::S4:
    return ARM::S5;
  case ARM::S6:
    return ARM::S7;
  case ARM::S8:
    return ARM::S9;
  case ARM::S10:
    return ARM::S11;
  case ARM::S12:
    return ARM::S13;
  case ARM::S14:
    return ARM::S15;
  case ARM::S16:
    return ARM::S17;
  case ARM::S18:
    return ARM::S19;
  case ARM::S20:
    return ARM::S21;
  case ARM::S22:
    return ARM::S23;
  case ARM::S24:
    return ARM::S25;
  case ARM::S26:
    return ARM::S27;
  case ARM::S28:
    return ARM::S29;
  case ARM::S30:
    return ARM::S31;

  case ARM::D0:
    return ARM::D1;
  case ARM::D2:
    return ARM::D3;
  case ARM::D4:
    return ARM::D5;
  case ARM::D6:
    return ARM::D7;
  case ARM::D8:
    return ARM::D9;
  case ARM::D10:
    return ARM::D11;
  case ARM::D12:
    return ARM::D13;
  case ARM::D14:
    return ARM::D15;
  }

  return 0;
}

#include "ARMGenRegisterInfo.inc"
