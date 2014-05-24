//=== AArch64CallingConv.h - Custom Calling Convention Routines -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the custom routines for the AArch64 Calling Convention that
// aren't done by tablegen.
//
//===----------------------------------------------------------------------===//

#ifndef AArch64CALLINGCONV_H
#define AArch64CALLINGCONV_H

#include "AArch64InstrInfo.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/Target/TargetInstrInfo.h"

namespace llvm {

/// CC_AArch64_Custom_i1i8i16_Reg - customized handling of passing i1/i8/i16 via
/// register. Here, ValVT can be i1/i8/i16 or i32 depending on whether the
/// argument is already promoted and LocVT is i1/i8/i16. We only promote the
/// argument to i32 if we are sure this argument will be passed in register.
static bool CC_AArch64_Custom_i1i8i16_Reg(unsigned ValNo, MVT ValVT, MVT LocVT,
                                        CCValAssign::LocInfo LocInfo,
                                        ISD::ArgFlagsTy ArgFlags,
                                        CCState &State,
                                        bool IsWebKitJS = false) {
  static const MCPhysReg RegList1[] = { AArch64::W0, AArch64::W1, AArch64::W2,
                                        AArch64::W3, AArch64::W4, AArch64::W5,
                                        AArch64::W6, AArch64::W7 };
  static const MCPhysReg RegList2[] = { AArch64::X0, AArch64::X1, AArch64::X2,
                                        AArch64::X3, AArch64::X4, AArch64::X5,
                                        AArch64::X6, AArch64::X7 };
  static const MCPhysReg WebKitRegList1[] = { AArch64::W0 };
  static const MCPhysReg WebKitRegList2[] = { AArch64::X0 };

  const MCPhysReg *List1 = IsWebKitJS ? WebKitRegList1 : RegList1;
  const MCPhysReg *List2 = IsWebKitJS ? WebKitRegList2 : RegList2;

  if (unsigned Reg = State.AllocateReg(List1, List2, 8)) {
    // Customized extra section for handling i1/i8/i16:
    // We need to promote the argument to i32 if it is not done already.
    if (ValVT != MVT::i32) {
      if (ArgFlags.isSExt())
        LocInfo = CCValAssign::SExt;
      else if (ArgFlags.isZExt())
        LocInfo = CCValAssign::ZExt;
      else
        LocInfo = CCValAssign::AExt;
      ValVT = MVT::i32;
    }
    // Set LocVT to i32 as well if passing via register.
    LocVT = MVT::i32;
    State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
    return true;
  }
  return false;
}

/// CC_AArch64_WebKit_JS_i1i8i16_Reg - customized handling of passing i1/i8/i16
/// via register. This behaves the same as CC_AArch64_Custom_i1i8i16_Reg, but only
/// uses the first register.
static bool CC_AArch64_WebKit_JS_i1i8i16_Reg(unsigned ValNo, MVT ValVT, MVT LocVT,
                                           CCValAssign::LocInfo LocInfo,
                                           ISD::ArgFlagsTy ArgFlags,
                                           CCState &State) {
  return CC_AArch64_Custom_i1i8i16_Reg(ValNo, ValVT, LocVT, LocInfo, ArgFlags,
                                     State, true);
}

/// CC_AArch64_Custom_i1i8i16_Stack: customized handling of passing i1/i8/i16 on
/// stack. Here, ValVT can be i1/i8/i16 or i32 depending on whether the argument
/// is already promoted and LocVT is i1/i8/i16. If ValVT is already promoted,
/// it will be truncated back to i1/i8/i16.
static bool CC_AArch64_Custom_i1i8i16_Stack(unsigned ValNo, MVT ValVT, MVT LocVT,
                                          CCValAssign::LocInfo LocInfo,
                                          ISD::ArgFlagsTy ArgFlags,
                                          CCState &State) {
  unsigned Space = ((LocVT == MVT::i1 || LocVT == MVT::i8) ? 1 : 2);
  unsigned Offset12 = State.AllocateStack(Space, Space);
  ValVT = LocVT;
  State.addLoc(CCValAssign::getMem(ValNo, ValVT, Offset12, LocVT, LocInfo));
  return true;
}

} // End llvm namespace

#endif
