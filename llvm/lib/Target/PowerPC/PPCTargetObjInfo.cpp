//===-- PPCTargetObjInfo.cpp - Object File Info ----------------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Bill Wendling and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines target object file properties for PowerPC.
//
//===----------------------------------------------------------------------===//

#include "PPCTargetObjInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;

MachOTargetObjInfo::MachOTargetObjInfo(const TargetMachine &tm)
  : TM(tm),
    is64Bit(TM.getTargetData()->getPointerSizeInBits() == 64),
    isLittleEndian(TM.getTargetData()->isLittleEndian()) {}
