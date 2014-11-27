//=== ARMCallingConv.h - ARM Custom Calling Convention Routines -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the custom routines for the ARM Calling Convention that
// aren't done by tablegen.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ARM_ARMCALLINGCONV_H
#define LLVM_LIB_TARGET_ARM_ARMCALLINGCONV_H

#include "ARM.h"
#include "ARMBaseInstrInfo.h"
#include "ARMSubtarget.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/Target/TargetInstrInfo.h"

namespace llvm {

// APCS f64 is in register pairs, possibly split to stack
static bool f64AssignAPCS(unsigned &ValNo, MVT &ValVT, MVT &LocVT,
                          CCValAssign::LocInfo &LocInfo,
                          CCState &State, bool CanFail) {
  static const MCPhysReg RegList[] = { ARM::R0, ARM::R1, ARM::R2, ARM::R3 };

  // Try to get the first register.
  if (unsigned Reg = State.AllocateReg(RegList, 4))
    State.addLoc(CCValAssign::getCustomReg(ValNo, ValVT, Reg, LocVT, LocInfo));
  else {
    // For the 2nd half of a v2f64, do not fail.
    if (CanFail)
      return false;

    // Put the whole thing on the stack.
    State.addLoc(CCValAssign::getCustomMem(ValNo, ValVT,
                                           State.AllocateStack(8, 4),
                                           LocVT, LocInfo));
    return true;
  }

  // Try to get the second register.
  if (unsigned Reg = State.AllocateReg(RegList, 4))
    State.addLoc(CCValAssign::getCustomReg(ValNo, ValVT, Reg, LocVT, LocInfo));
  else
    State.addLoc(CCValAssign::getCustomMem(ValNo, ValVT,
                                           State.AllocateStack(4, 4),
                                           LocVT, LocInfo));
  return true;
}

static bool CC_ARM_APCS_Custom_f64(unsigned &ValNo, MVT &ValVT, MVT &LocVT,
                                   CCValAssign::LocInfo &LocInfo,
                                   ISD::ArgFlagsTy &ArgFlags,
                                   CCState &State) {
  if (!f64AssignAPCS(ValNo, ValVT, LocVT, LocInfo, State, true))
    return false;
  if (LocVT == MVT::v2f64 &&
      !f64AssignAPCS(ValNo, ValVT, LocVT, LocInfo, State, false))
    return false;
  return true;  // we handled it
}

// AAPCS f64 is in aligned register pairs
static bool f64AssignAAPCS(unsigned &ValNo, MVT &ValVT, MVT &LocVT,
                           CCValAssign::LocInfo &LocInfo,
                           CCState &State, bool CanFail) {
  static const MCPhysReg HiRegList[] = { ARM::R0, ARM::R2 };
  static const MCPhysReg LoRegList[] = { ARM::R1, ARM::R3 };
  static const MCPhysReg ShadowRegList[] = { ARM::R0, ARM::R1 };
  static const MCPhysReg GPRArgRegs[] = { ARM::R0, ARM::R1, ARM::R2, ARM::R3 };

  unsigned Reg = State.AllocateReg(HiRegList, ShadowRegList, 2);
  if (Reg == 0) {

    // If we had R3 unallocated only, now we still must to waste it.
    Reg = State.AllocateReg(GPRArgRegs, 4);
    assert((!Reg || Reg == ARM::R3) && "Wrong GPRs usage for f64");

    // For the 2nd half of a v2f64, do not just fail.
    if (CanFail)
      return false;

    // Put the whole thing on the stack.
    State.addLoc(CCValAssign::getCustomMem(ValNo, ValVT,
                                           State.AllocateStack(8, 8),
                                           LocVT, LocInfo));
    return true;
  }

  unsigned i;
  for (i = 0; i < 2; ++i)
    if (HiRegList[i] == Reg)
      break;

  unsigned T = State.AllocateReg(LoRegList[i]);
  (void)T;
  assert(T == LoRegList[i] && "Could not allocate register");

  State.addLoc(CCValAssign::getCustomReg(ValNo, ValVT, Reg, LocVT, LocInfo));
  State.addLoc(CCValAssign::getCustomReg(ValNo, ValVT, LoRegList[i],
                                         LocVT, LocInfo));
  return true;
}

static bool CC_ARM_AAPCS_Custom_f64(unsigned &ValNo, MVT &ValVT, MVT &LocVT,
                                    CCValAssign::LocInfo &LocInfo,
                                    ISD::ArgFlagsTy &ArgFlags,
                                    CCState &State) {
  if (!f64AssignAAPCS(ValNo, ValVT, LocVT, LocInfo, State, true))
    return false;
  if (LocVT == MVT::v2f64 &&
      !f64AssignAAPCS(ValNo, ValVT, LocVT, LocInfo, State, false))
    return false;
  return true;  // we handled it
}

static bool f64RetAssign(unsigned &ValNo, MVT &ValVT, MVT &LocVT,
                         CCValAssign::LocInfo &LocInfo, CCState &State) {
  static const MCPhysReg HiRegList[] = { ARM::R0, ARM::R2 };
  static const MCPhysReg LoRegList[] = { ARM::R1, ARM::R3 };

  unsigned Reg = State.AllocateReg(HiRegList, LoRegList, 2);
  if (Reg == 0)
    return false; // we didn't handle it

  unsigned i;
  for (i = 0; i < 2; ++i)
    if (HiRegList[i] == Reg)
      break;

  State.addLoc(CCValAssign::getCustomReg(ValNo, ValVT, Reg, LocVT, LocInfo));
  State.addLoc(CCValAssign::getCustomReg(ValNo, ValVT, LoRegList[i],
                                         LocVT, LocInfo));
  return true;
}

static bool RetCC_ARM_APCS_Custom_f64(unsigned &ValNo, MVT &ValVT, MVT &LocVT,
                                      CCValAssign::LocInfo &LocInfo,
                                      ISD::ArgFlagsTy &ArgFlags,
                                      CCState &State) {
  if (!f64RetAssign(ValNo, ValVT, LocVT, LocInfo, State))
    return false;
  if (LocVT == MVT::v2f64 && !f64RetAssign(ValNo, ValVT, LocVT, LocInfo, State))
    return false;
  return true;  // we handled it
}

static bool RetCC_ARM_AAPCS_Custom_f64(unsigned &ValNo, MVT &ValVT, MVT &LocVT,
                                       CCValAssign::LocInfo &LocInfo,
                                       ISD::ArgFlagsTy &ArgFlags,
                                       CCState &State) {
  return RetCC_ARM_APCS_Custom_f64(ValNo, ValVT, LocVT, LocInfo, ArgFlags,
                                   State);
}

static const uint16_t SRegList[] = { ARM::S0,  ARM::S1,  ARM::S2,  ARM::S3,
                                     ARM::S4,  ARM::S5,  ARM::S6,  ARM::S7,
                                     ARM::S8,  ARM::S9,  ARM::S10, ARM::S11,
                                     ARM::S12, ARM::S13, ARM::S14,  ARM::S15 };
static const uint16_t DRegList[] = { ARM::D0, ARM::D1, ARM::D2, ARM::D3,
                                     ARM::D4, ARM::D5, ARM::D6, ARM::D7 };
static const uint16_t QRegList[] = { ARM::Q0, ARM::Q1, ARM::Q2, ARM::Q3 };

// Allocate part of an AAPCS HFA or HVA. We assume that each member of the HA
// has InConsecutiveRegs set, and that the last member also has
// InConsecutiveRegsLast set. We must process all members of the HA before
// we can allocate it, as we need to know the total number of registers that
// will be needed in order to (attempt to) allocate a contiguous block.
static bool CC_ARM_AAPCS_Custom_HA(unsigned &ValNo, MVT &ValVT, MVT &LocVT,
                                   CCValAssign::LocInfo &LocInfo,
                                   ISD::ArgFlagsTy &ArgFlags, CCState &State) {
  SmallVectorImpl<CCValAssign> &PendingHAMembers = State.getPendingLocs();

  // AAPCS HFAs must have 1-4 elements, all of the same type
  assert(PendingHAMembers.size() < 4);
  if (PendingHAMembers.size() > 0)
    assert(PendingHAMembers[0].getLocVT() == LocVT);

  // Add the argument to the list to be allocated once we know the size of the
  // HA
  PendingHAMembers.push_back(
      CCValAssign::getPending(ValNo, ValVT, LocVT, LocInfo));

  if (ArgFlags.isInConsecutiveRegsLast()) {
    assert(PendingHAMembers.size() > 0 && PendingHAMembers.size() <= 4 &&
           "Homogeneous aggregates must have between 1 and 4 members");

    // Try to allocate a contiguous block of registers, each of the correct
    // size to hold one member.
    ArrayRef<uint16_t> RegList;
    switch (LocVT.SimpleTy) {
    case MVT::f32:
      RegList = SRegList;
      break;
    case MVT::f64:
      RegList = DRegList;
      break;
    case MVT::v2f64:
      RegList = QRegList;
      break;
    default:
      llvm_unreachable("Unexpected member type for HA");
      break;
    }

    unsigned RegResult =
        State.AllocateRegBlock(RegList, PendingHAMembers.size());

    if (RegResult) {
      for (SmallVectorImpl<CCValAssign>::iterator It = PendingHAMembers.begin();
           It != PendingHAMembers.end(); ++It) {
        It->convertToReg(RegResult);
        State.addLoc(*It);
        ++RegResult;
      }
      PendingHAMembers.clear();
      return true;
    }

    // Register allocation failed, fall back to the stack

    // Mark all VFP regs as unavailable (AAPCS rule C.2.vfp)
    for (unsigned regNo = 0; regNo < 16; ++regNo)
      State.AllocateReg(SRegList[regNo]);

    unsigned Size = LocVT.getSizeInBits() / 8;
    unsigned Align = std::min(Size, 8U);

    for (auto It : PendingHAMembers) {
      It.convertToMem(State.AllocateStack(Size, Align));
      State.addLoc(It);
    }

    // All pending members have now been allocated
    PendingHAMembers.clear();
  }

  // This will be allocated by the last member of the HA
  return true;
}

} // End llvm namespace

#endif
