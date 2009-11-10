//===-- TargetSubtarget.cpp - General Target Information -------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the general parts of a Subtarget.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetSubtarget.h"
#include "llvm/ADT/SmallVector.h"
using namespace llvm;

//---------------------------------------------------------------------------
// TargetSubtarget Class
//
TargetSubtarget::TargetSubtarget() {}

TargetSubtarget::~TargetSubtarget() {}

bool TargetSubtarget::enablePostRAScheduler(
          CodeGenOpt::Level OptLevel,
          AntiDepBreakMode& Mode,
          ExcludedRCVector& ExcludedRCs) const {
  Mode = ANTIDEP_NONE;
  ExcludedRCs.clear();
  return false;
}

