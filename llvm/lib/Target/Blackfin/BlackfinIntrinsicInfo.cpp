//===- BlackfinIntrinsicInfo.cpp - Intrinsic Information --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Blackfin implementation of TargetIntrinsicInfo.
//
//===----------------------------------------------------------------------===//

#include "BlackfinIntrinsicInfo.h"
#include "llvm/Intrinsics.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>

using namespace llvm;

namespace bfinIntrinsic {

  enum ID {
    last_non_bfin_intrinsic = Intrinsic::num_intrinsics-1,
#define GET_INTRINSIC_ENUM_VALUES
#include "BlackfinGenIntrinsics.inc"
#undef GET_INTRINSIC_ENUM_VALUES
    , num_bfin_intrinsics
  };

}

const char *BlackfinIntrinsicInfo::getName(unsigned IntrID) const {
  static const char *const names[] = {
#define GET_INTRINSIC_NAME_TABLE
#include "BlackfinGenIntrinsics.inc"
#undef GET_INTRINSIC_NAME_TABLE
  };

  if (IntrID < Intrinsic::num_intrinsics)
    return 0;
  assert(IntrID < bfinIntrinsic::num_bfin_intrinsics && "Invalid intrinsic ID");

  return names[IntrID - Intrinsic::num_intrinsics];
}

unsigned
BlackfinIntrinsicInfo::lookupName(const char *Name, unsigned Len) const {
#define GET_FUNCTION_RECOGNIZER
#include "BlackfinGenIntrinsics.inc"
#undef GET_FUNCTION_RECOGNIZER
  return 0;
}
