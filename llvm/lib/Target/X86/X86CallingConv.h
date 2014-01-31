//=== X86CallingConv.h - X86 Custom Calling Convention Routines -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the custom routines for the X86 Calling Convention that
// aren't done by tablegen.
//
//===----------------------------------------------------------------------===//

#ifndef X86CALLINGCONV_H
#define X86CALLINGCONV_H

#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/IR/CallingConv.h"

namespace llvm {

inline bool CC_X86_AnyReg_Error(unsigned &, MVT &, MVT &,
                                CCValAssign::LocInfo &, ISD::ArgFlagsTy &,
                                CCState &) {
  llvm_unreachable("The AnyReg calling convention is only supported by the " \
                   "stackmap and patchpoint intrinsics.");
  // gracefully fallback to X86 C calling convention on Release builds.
  return false;
}

inline bool CC_X86_CDeclMethod_SRet(unsigned &ValNo, MVT &ValVT, MVT &LocVT,
                                    CCValAssign::LocInfo &LocInfo,
                                    ISD::ArgFlagsTy &ArgFlags, CCState &State) {
  // Swap the order of the first two parameters if the first parameter is sret.
  if (ArgFlags.isSRet()) {
    assert(ValNo == 0);
    assert(ValVT == MVT::i32);
    State.AllocateStack(8, 4);
    State.addLoc(CCValAssign::getCustomMem(ValNo, ValVT, 4, LocVT, LocInfo));

    // Indicate that we need to swap the order of the first and second
    // parameters by "allocating" register zero.  There are no register
    // parameters with cdecl methods, so we can use this to communicate to the
    // next call.
    State.AllocateReg(1);
    return true;
  } else if (ValNo == 1 && State.isAllocated(1)) {
    assert(ValVT == MVT::i32 && "non-i32-sized this param unsupported");
    // Stack was already allocated while processing sret.
    State.addLoc(CCValAssign::getCustomMem(ValNo, ValVT, 0, LocVT, LocInfo));
    return true;
  }

  // All other args use the C calling convention.
  return false;
}

} // End llvm namespace

#endif

