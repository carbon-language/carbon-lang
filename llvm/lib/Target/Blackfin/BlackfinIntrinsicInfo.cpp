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
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Intrinsics.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
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

std::string BlackfinIntrinsicInfo::getName(unsigned IntrID, const Type **Tys,
                                           unsigned numTys) const {
  static const char *const names[] = {
#define GET_INTRINSIC_NAME_TABLE
#include "BlackfinGenIntrinsics.inc"
#undef GET_INTRINSIC_NAME_TABLE
  };

  assert(!isOverloaded(IntrID) && "Blackfin intrinsics are not overloaded");
  if (IntrID < Intrinsic::num_intrinsics)
    return 0;
  assert(IntrID < bfinIntrinsic::num_bfin_intrinsics && "Invalid intrinsic ID");

  std::string Result(names[IntrID - Intrinsic::num_intrinsics]);
  return Result;
}

unsigned
BlackfinIntrinsicInfo::lookupName(const char *Name, unsigned Len) const {
#define GET_FUNCTION_RECOGNIZER
#include "BlackfinGenIntrinsics.inc"
#undef GET_FUNCTION_RECOGNIZER
  return 0;
}

bool BlackfinIntrinsicInfo::isOverloaded(unsigned IntrID) const {
  // Overload Table
  const bool OTable[] = {
    false,  // illegal intrinsic
#define GET_INTRINSIC_OVERLOAD_TABLE
#include "BlackfinGenIntrinsics.inc"
#undef GET_INTRINSIC_OVERLOAD_TABLE
  };
  if (IntrID == 0)
    return false;
  else
    return OTable[IntrID - Intrinsic::num_intrinsics];
}

/// This defines the "getAttributes(ID id)" method.
#define GET_INTRINSIC_ATTRIBUTES
#include "BlackfinGenIntrinsics.inc"
#undef GET_INTRINSIC_ATTRIBUTES

static const FunctionType *getType(LLVMContext &Context, unsigned id) {
  const Type *ResultTy = NULL;
  std::vector<const Type*> ArgTys;
  bool IsVarArg = false;
  
#define GET_INTRINSIC_GENERATOR
#include "BlackfinGenIntrinsics.inc"
#undef GET_INTRINSIC_GENERATOR

  return FunctionType::get(ResultTy, ArgTys, IsVarArg); 
}

Function *BlackfinIntrinsicInfo::getDeclaration(Module *M, unsigned IntrID,
                                                const Type **Tys,
                                                unsigned numTy) const {
  assert(!isOverloaded(IntrID) && "Blackfin intrinsics are not overloaded");
  AttrListPtr AList = getAttributes((bfinIntrinsic::ID) IntrID);
  return cast<Function>(M->getOrInsertFunction(getName(IntrID),
                                               getType(M->getContext(), IntrID),
                                               AList));
}
