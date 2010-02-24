//===- MBlazeIntrinsicInfo.cpp - Intrinsic Information -00-------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the MBlaze implementation of TargetIntrinsicInfo.
//
//===----------------------------------------------------------------------===//

#include "MBlazeIntrinsicInfo.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Intrinsics.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>

using namespace llvm;

namespace mblazeIntrinsic {

  enum ID {
    last_non_mblaze_intrinsic = Intrinsic::num_intrinsics-1,
#define GET_INTRINSIC_ENUM_VALUES
#include "MBlazeGenIntrinsics.inc"
#undef GET_INTRINSIC_ENUM_VALUES
    , num_mblaze_intrinsics
  };

#define GET_LLVM_INTRINSIC_FOR_GCC_BUILTIN
#include "MBlazeGenIntrinsics.inc"
#undef GET_LLVM_INTRINSIC_FOR_GCC_BUILTIN
}

std::string MBlazeIntrinsicInfo::getName(unsigned IntrID, const Type **Tys,
                                         unsigned numTys) const {
  static const char *const names[] = {
#define GET_INTRINSIC_NAME_TABLE
#include "MBlazeGenIntrinsics.inc"
#undef GET_INTRINSIC_NAME_TABLE
  };

  assert(!isOverloaded(IntrID) && "MBlaze intrinsics are not overloaded");
  if (IntrID < Intrinsic::num_intrinsics)
    return 0;
  assert(IntrID < mblazeIntrinsic::num_mblaze_intrinsics && 
         "Invalid intrinsic ID");

  std::string Result(names[IntrID - Intrinsic::num_intrinsics]);
  return Result;
}

unsigned MBlazeIntrinsicInfo::
lookupName(const char *Name, unsigned Len) const {
#define GET_FUNCTION_RECOGNIZER
#include "MBlazeGenIntrinsics.inc"
#undef GET_FUNCTION_RECOGNIZER
  return 0;
}

unsigned MBlazeIntrinsicInfo::
lookupGCCName(const char *Name) const {
    return mblazeIntrinsic::getIntrinsicForGCCBuiltin("mblaze",Name);
}

bool MBlazeIntrinsicInfo::isOverloaded(unsigned IntrID) const {
  // Overload Table
  const bool OTable[] = {
#define GET_INTRINSIC_OVERLOAD_TABLE
#include "MBlazeGenIntrinsics.inc"
#undef GET_INTRINSIC_OVERLOAD_TABLE
  };
  if (IntrID == 0)
    return false;
  else
    return OTable[IntrID - Intrinsic::num_intrinsics];
}

/// This defines the "getAttributes(ID id)" method.
#define GET_INTRINSIC_ATTRIBUTES
#include "MBlazeGenIntrinsics.inc"
#undef GET_INTRINSIC_ATTRIBUTES

static const FunctionType *getType(LLVMContext &Context, unsigned id) {
  const Type *ResultTy = NULL;
  std::vector<const Type*> ArgTys;
  bool IsVarArg = false;
  
#define GET_INTRINSIC_GENERATOR
#include "MBlazeGenIntrinsics.inc"
#undef GET_INTRINSIC_GENERATOR

  return FunctionType::get(ResultTy, ArgTys, IsVarArg); 
}

Function *MBlazeIntrinsicInfo::getDeclaration(Module *M, unsigned IntrID,
                                                const Type **Tys,
                                                unsigned numTy) const {
  assert(!isOverloaded(IntrID) && "MBlaze intrinsics are not overloaded");
  AttrListPtr AList = getAttributes((mblazeIntrinsic::ID) IntrID);
  return cast<Function>(M->getOrInsertFunction(getName(IntrID),
                                               getType(M->getContext(), IntrID),
                                               AList));
}
