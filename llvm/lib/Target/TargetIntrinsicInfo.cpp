//===-- TargetIntrinsicInfo.cpp - Target Instruction Information ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the TargetIntrinsicInfo class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetIntrinsicInfo.h"
using namespace llvm;

TargetIntrinsicInfo::TargetIntrinsicInfo(const char **desc, unsigned count)
  : Intrinsics(desc), NumIntrinsics(count) {
}

TargetIntrinsicInfo::~TargetIntrinsicInfo() {
}
