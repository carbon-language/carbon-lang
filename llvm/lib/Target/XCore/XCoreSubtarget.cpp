//===- XCoreSubtarget.cpp - XCore Subtarget Information -----------*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the XCore specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "XCoreSubtarget.h"
#include "XCore.h"

#define GET_SUBTARGETINFO_CTOR
#define GET_SUBTARGETINFO_MC_DESC
#define GET_SUBTARGETINFO_TARGET_DESC
#include "XCoreGenSubtargetInfo.inc"

using namespace llvm;

XCoreSubtarget::XCoreSubtarget(const std::string &TT,
                               const std::string &CPU, const std::string &FS)
  : XCoreGenSubtargetInfo(TT, CPU, FS)
{
}
