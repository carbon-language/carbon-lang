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

#ifndef LLVM_LIB_TARGET_X86_X86CALLINGCONV_H
#define LLVM_LIB_TARGET_X86_X86CALLINGCONV_H

#include "MCTargetDesc/X86MCTargetDesc.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/IR/CallingConv.h"

namespace llvm {

bool RetCC_X86(unsigned ValNo, MVT ValVT, MVT LocVT,
               CCValAssign::LocInfo LocInfo, ISD::ArgFlagsTy ArgFlags,
               CCState &State);

bool CC_X86(unsigned ValNo, MVT ValVT, MVT LocVT, CCValAssign::LocInfo LocInfo,
            ISD::ArgFlagsTy ArgFlags, CCState &State);

} // End llvm namespace

#endif

