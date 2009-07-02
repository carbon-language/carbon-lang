//===-- lib/Target/TargetELFWriterInfo.cpp - ELF Writer Info --0-*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the TargetELFWriterInfo class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Function.h"
#include "llvm/Target/TargetELFWriterInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;

TargetELFWriterInfo::TargetELFWriterInfo(TargetMachine &tm) : TM(tm) {
  is64Bit = TM.getTargetData()->getPointerSizeInBits() == 64;
  isLittleEndian = TM.getTargetData()->isLittleEndian();
}

TargetELFWriterInfo::~TargetELFWriterInfo() {}

