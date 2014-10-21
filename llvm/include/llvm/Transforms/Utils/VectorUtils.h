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

#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Target/TargetLibraryInfo.h"

namespace llvm {

/// \brief Identify if the intrinsic is trivially vectorizable.
///
/// This method returns true if the intrinsic's argument types are all
/// scalars for the scalar form of the intrinsic and all vectors for
/// the vector form of the intrinsic.
static inline bool isTriviallyVectorizable(Intrinsic::ID ID) {
  switch (ID) {
  case Intrinsic::sqrt:
  case Intrinsic::sin:
  case Intrinsic::cos:
  case Intrinsic::exp:
  case Intrinsic::exp2:
  case Intrinsic::log:
  case Intrinsic::log10:
  case Intrinsic::log2:
  case Intrinsic::fabs:
  case Intrinsic::minnum:
  case Intrinsic::maxnum:
  case Intrinsic::copysign:
  case Intrinsic::floor:
  case Intrinsic::ceil:
  case Intrinsic::trunc:
  case Intrinsic::rint:
  case Intrinsic::nearbyint:
  case Intrinsic::round:
  case Intrinsic::bswap:
  case Intrinsic::ctpop:
  case Intrinsic::pow:
  case Intrinsic::fma:
  case Intrinsic::fmuladd:
  case Intrinsic::ctlz:
  case Intrinsic::cttz:
  case Intrinsic::powi:
    return true;
  default:
    return false;
  }
}

static bool hasVectorInstrinsicScalarOpd(Intrinsic::ID ID,
                                         unsigned ScalarOpdIdx) {
  switch (ID) {
    case Intrinsic::ctlz:
    case Intrinsic::cttz:
    case Intrinsic::powi:
      return (ScalarOpdIdx == 1);
    default:
      return false;
  }
}

static Intrinsic::ID checkUnaryFloatSignature(const CallInst &I,
                                              Intrinsic::ID ValidIntrinsicID) {
  if (I.getNumArgOperands() != 1 ||
      !I.getArgOperand(0)->getType()->isFloatingPointTy() ||
      I.getType() != I.getArgOperand(0)->getType() ||
      !I.onlyReadsMemory())
    return Intrinsic::not_intrinsic;

  return ValidIntrinsicID;
}

static Intrinsic::ID checkBinaryFloatSignature(const CallInst &I,
                                               Intrinsic::ID ValidIntrinsicID) {
  if (I.getNumArgOperands() != 2 ||
      !I.getArgOperand(0)->getType()->isFloatingPointTy() ||
      !I.getArgOperand(1)->getType()->isFloatingPointTy() ||
      I.getType() != I.getArgOperand(0)->getType() ||
      I.getType() != I.getArgOperand(1)->getType() ||
      !I.onlyReadsMemory())
    return Intrinsic::not_intrinsic;

  return ValidIntrinsicID;
}

static Intrinsic::ID
getIntrinsicIDForCall(CallInst *CI, const TargetLibraryInfo *TLI) {
  // If we have an intrinsic call, check if it is trivially vectorizable.
  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(CI)) {
    Intrinsic::ID ID = II->getIntrinsicID();
    if (isTriviallyVectorizable(ID) || ID == Intrinsic::lifetime_start ||
        ID == Intrinsic::lifetime_end || ID == Intrinsic::assume)
      return ID;
    else
      return Intrinsic::not_intrinsic;
  }

  if (!TLI)
    return Intrinsic::not_intrinsic;

  LibFunc::Func Func;
  Function *F = CI->getCalledFunction();
  // We're going to make assumptions on the semantics of the functions, check
  // that the target knows that it's available in this environment and it does
  // not have local linkage.
  if (!F || F->hasLocalLinkage() || !TLI->getLibFunc(F->getName(), Func))
    return Intrinsic::not_intrinsic;

  // Otherwise check if we have a call to a function that can be turned into a
  // vector intrinsic.
  switch (Func) {
  default:
    break;
  case LibFunc::sin:
  case LibFunc::sinf:
  case LibFunc::sinl:
    return checkUnaryFloatSignature(*CI, Intrinsic::sin);
  case LibFunc::cos:
  case LibFunc::cosf:
  case LibFunc::cosl:
    return checkUnaryFloatSignature(*CI, Intrinsic::cos);
  case LibFunc::exp:
  case LibFunc::expf:
  case LibFunc::expl:
    return checkUnaryFloatSignature(*CI, Intrinsic::exp);
  case LibFunc::exp2:
  case LibFunc::exp2f:
  case LibFunc::exp2l:
    return checkUnaryFloatSignature(*CI, Intrinsic::exp2);
  case LibFunc::log:
  case LibFunc::logf:
  case LibFunc::logl:
    return checkUnaryFloatSignature(*CI, Intrinsic::log);
  case LibFunc::log10:
  case LibFunc::log10f:
  case LibFunc::log10l:
    return checkUnaryFloatSignature(*CI, Intrinsic::log10);
  case LibFunc::log2:
  case LibFunc::log2f:
  case LibFunc::log2l:
    return checkUnaryFloatSignature(*CI, Intrinsic::log2);
  case LibFunc::fabs:
  case LibFunc::fabsf:
  case LibFunc::fabsl:
    return checkUnaryFloatSignature(*CI, Intrinsic::fabs);
  case LibFunc::fmin:
  case LibFunc::fminf:
  case LibFunc::fminl:
    return checkBinaryFloatSignature(*CI, Intrinsic::minnum);
  case LibFunc::fmax:
  case LibFunc::fmaxf:
  case LibFunc::fmaxl:
    return checkBinaryFloatSignature(*CI, Intrinsic::maxnum);
  case LibFunc::copysign:
  case LibFunc::copysignf:
  case LibFunc::copysignl:
    return checkBinaryFloatSignature(*CI, Intrinsic::copysign);
  case LibFunc::floor:
  case LibFunc::floorf:
  case LibFunc::floorl:
    return checkUnaryFloatSignature(*CI, Intrinsic::floor);
  case LibFunc::ceil:
  case LibFunc::ceilf:
  case LibFunc::ceill:
    return checkUnaryFloatSignature(*CI, Intrinsic::ceil);
  case LibFunc::trunc:
  case LibFunc::truncf:
  case LibFunc::truncl:
    return checkUnaryFloatSignature(*CI, Intrinsic::trunc);
  case LibFunc::rint:
  case LibFunc::rintf:
  case LibFunc::rintl:
    return checkUnaryFloatSignature(*CI, Intrinsic::rint);
  case LibFunc::nearbyint:
  case LibFunc::nearbyintf:
  case LibFunc::nearbyintl:
    return checkUnaryFloatSignature(*CI, Intrinsic::nearbyint);
  case LibFunc::round:
  case LibFunc::roundf:
  case LibFunc::roundl:
    return checkUnaryFloatSignature(*CI, Intrinsic::round);
  case LibFunc::pow:
  case LibFunc::powf:
  case LibFunc::powl:
    return checkBinaryFloatSignature(*CI, Intrinsic::pow);
  }

  return Intrinsic::not_intrinsic;
}

} // llvm namespace

#endif
