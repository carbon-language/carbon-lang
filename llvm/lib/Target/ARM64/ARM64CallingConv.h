//=== ARM64CallingConv.h - Custom Calling Convention Routines -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the custom routines for the ARM64 Calling Convention that
// aren't done by tablegen.
//
//===----------------------------------------------------------------------===//

#ifndef ARM64CALLINGCONV_H
#define ARM64CALLINGCONV_H

#include "ARM64InstrInfo.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/Target/TargetInstrInfo.h"

namespace llvm {

/// CC_ARM64_Custom_i1i8i16_Reg - customized handling of passing i1/i8/i16 via
/// register. Here, ValVT can be i1/i8/i16 or i32 depending on whether the
/// argument is already promoted and LocVT is i1/i8/i16. We only promote the
/// argument to i32 if we are sure this argument will be passed in register.
static bool CC_ARM64_Custom_i1i8i16_Reg(unsigned ValNo, MVT ValVT, MVT LocVT,
                                        CCValAssign::LocInfo LocInfo,
                                        ISD::ArgFlagsTy ArgFlags,
                                        CCState &State,
                                        bool IsWebKitJS = false) {
  static const uint16_t RegList1[] = { ARM64::W0, ARM64::W1, ARM64::W2,
                                       ARM64::W3, ARM64::W4, ARM64::W5,
                                       ARM64::W6, ARM64::W7 };
  static const uint16_t RegList2[] = { ARM64::X0, ARM64::X1, ARM64::X2,
                                       ARM64::X3, ARM64::X4, ARM64::X5,
                                       ARM64::X6, ARM64::X7 };
  static const uint16_t WebKitRegList1[] = { ARM64::W0 };
  static const uint16_t WebKitRegList2[] = { ARM64::X0 };

  const uint16_t *List1 = IsWebKitJS ? WebKitRegList1 : RegList1;
  const uint16_t *List2 = IsWebKitJS ? WebKitRegList2 : RegList2;

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

/// CC_ARM64_WebKit_JS_i1i8i16_Reg - customized handling of passing i1/i8/i16
/// via register. This behaves the same as CC_ARM64_Custom_i1i8i16_Reg, but only
/// uses the first register.
static bool CC_ARM64_WebKit_JS_i1i8i16_Reg(unsigned ValNo, MVT ValVT, MVT LocVT,
                                           CCValAssign::LocInfo LocInfo,
                                           ISD::ArgFlagsTy ArgFlags,
                                           CCState &State) {
  return CC_ARM64_Custom_i1i8i16_Reg(ValNo, ValVT, LocVT, LocInfo, ArgFlags,
                                     State, true);
}

/// CC_ARM64_Custom_i1i8i16_Stack: customized handling of passing i1/i8/i16 on
/// stack. Here, ValVT can be i1/i8/i16 or i32 depending on whether the argument
/// is already promoted and LocVT is i1/i8/i16. If ValVT is already promoted,
/// it will be truncated back to i1/i8/i16.
static bool CC_ARM64_Custom_i1i8i16_Stack(unsigned ValNo, MVT ValVT, MVT LocVT,
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
