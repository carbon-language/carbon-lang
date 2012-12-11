//===- AMDILIntrinsicInfo.h - AMDGPU Intrinsic Information ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
//
/// \file
/// \brief Interface for the AMDGPU Implementation of the Intrinsic Info class.
//
//===-----------------------------------------------------------------------===//
#ifndef AMDIL_INTRINSICS_H
#define AMDIL_INTRINSICS_H

#include "llvm/Intrinsics.h"
#include "llvm/Target/TargetIntrinsicInfo.h"

namespace llvm {
class TargetMachine;

namespace AMDGPUIntrinsic {
enum ID {
  last_non_AMDGPU_intrinsic = Intrinsic::num_intrinsics - 1,
#define GET_INTRINSIC_ENUM_VALUES
#include "AMDGPUGenIntrinsics.inc"
#undef GET_INTRINSIC_ENUM_VALUES
      , num_AMDGPU_intrinsics
};

} // end namespace AMDGPUIntrinsic

class AMDGPUIntrinsicInfo : public TargetIntrinsicInfo {
public:
  AMDGPUIntrinsicInfo(TargetMachine *tm);
  std::string getName(unsigned int IntrId, Type **Tys = 0,
                      unsigned int numTys = 0) const;
  unsigned int lookupName(const char *Name, unsigned int Len) const;
  bool isOverloaded(unsigned int IID) const;
  Function *getDeclaration(Module *M, unsigned int ID,
                           Type **Tys = 0,
                           unsigned int numTys = 0) const;
};

} // end namespace llvm

#endif // AMDIL_INTRINSICS_H

