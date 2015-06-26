//===- llvm/Transforms/Utils/VectorUtils.h - Vector utilities -*- C++ -*-=====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines some vectorizer utilities.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_VECTORUTILS_H
#define LLVM_TRANSFORMS_UTILS_VECTORUTILS_H

#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"

namespace llvm {

/// \brief Identify if the intrinsic is trivially vectorizable.
/// This method returns true if the intrinsic's argument types are all
/// scalars for the scalar form of the intrinsic and all vectors for
/// the vector form of the intrinsic.
bool isTriviallyVectorizable(Intrinsic::ID ID);

/// \brief Identifies if the intrinsic has a scalar operand. It checks for
/// ctlz,cttz and powi special intrinsics whose argument is scalar.
bool hasVectorInstrinsicScalarOpd(Intrinsic::ID ID, unsigned ScalarOpdIdx);

/// \brief Identify if call has a unary float signature
/// It returns input intrinsic ID if call has a single argument,
/// argument type and call instruction type should be floating
/// point type and call should only reads memory.
/// else return not_intrinsic.
Intrinsic::ID checkUnaryFloatSignature(const CallInst &I,
                                       Intrinsic::ID ValidIntrinsicID);

/// \brief Identify if call has a binary float signature
/// It returns input intrinsic ID if call has two arguments,
/// arguments type and call instruction type should be floating
/// point type and call should only reads memory.
/// else return not_intrinsic.
Intrinsic::ID checkBinaryFloatSignature(const CallInst &I,
                                        Intrinsic::ID ValidIntrinsicID);

/// \brief Returns intrinsic ID for call.
/// For the input call instruction it finds mapping intrinsic and returns
/// its intrinsic ID, in case it does not found it return not_intrinsic.
Intrinsic::ID getIntrinsicIDForCall(CallInst *CI, const TargetLibraryInfo *TLI);

} // llvm namespace

#endif
