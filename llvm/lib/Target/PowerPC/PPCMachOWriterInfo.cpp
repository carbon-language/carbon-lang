//===-- PPCMachOWriterInfo.cpp - Mach-O Writer Info for the PowerPC -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Bill Wendling and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements Mach-O writer information for the PowerPC backend.
//
//===----------------------------------------------------------------------===//

#include "PPCMachOWriterInfo.h"
#include "PPCTargetMachine.h"
using namespace llvm;

PPCMachOWriterInfo::PPCMachOWriterInfo(const PPCTargetMachine &TM)
  : TargetMachOWriterInfo(TM.getTargetData()->getPointerSizeInBits() == 64 ?
                          HDR_CPU_TYPE_POWERPC64 :
                          HDR_CPU_TYPE_POWERPC,
                          HDR_CPU_SUBTYPE_POWERPC_ALL) {}
