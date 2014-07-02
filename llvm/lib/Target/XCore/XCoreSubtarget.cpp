//===-- XCoreSubtarget.cpp - XCore Subtarget Information ------------------===//
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
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define DEBUG_TYPE "xcore-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "XCoreGenSubtargetInfo.inc"

void XCoreSubtarget::anchor() { }

XCoreSubtarget::XCoreSubtarget(const std::string &TT, const std::string &CPU,
                               const std::string &FS, const TargetMachine &TM)
    : XCoreGenSubtargetInfo(TT, CPU, FS),
      DL("e-m:e-p:32:32-i1:8:32-i8:8:32-i16:16:32-i64:32-f64:32-a:0:32-n32"),
      InstrInfo(), FrameLowering(*this), TLInfo(TM), TSInfo(DL) {}
