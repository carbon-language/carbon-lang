//===-- ConstantFolding.cpp - Analyze constant folding possibilities ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This family of functions determines the possibility of performing constant
// folding.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/MathExtras.h"
#include <cerrno>
#include <cmath>
using namespace llvm;

//===----------------------------------------------------------------------===//
//  Constant Folding ...
//


/// canConstantFoldCallTo - Return true if its even possible to fold a call to
/// the specified function.
bool
llvm::canConstantFoldCallTo(Function *F) {
  const std::string &Name = F->getName();

  switch (F->getIntrinsicID()) {
  case Intrinsic::isunordered:
  case Intrinsic::sqrt:
  case Intrinsic::bswap_i16:
  case Intrinsic::bswap_i32:
  case Intrinsic::bswap_i64:
  // FIXME: these should be constant folded as well
  //case Intrinsic::ctpop:
  //case Intrinsic::ctlz:
  //case Intrinsic::cttz:
    return true;
  default: break;
  }

  switch (Name[0])
  {
    case 'a':
      return Name == "acos" || Name == "asin" || Name == "atan" ||
             Name == "atan2";
    case 'c':
      return Name == "ceil" || Name == "cos" || Name == "cosf" ||
             Name == "cosh";
    case 'e':
      return Name == "exp";
    case 'f':
      return Name == "fabs" || Name == "fmod" || Name == "floor";
    case 'l':
      return Name == "log" || Name == "log10";
    case 'p':
      return Name == "pow";
    case 's':
      return Name == "sin" || Name == "sinh" || Name == "sqrt";
    case 't':
      return Name == "tan" || Name == "tanh";
    default:
      return false;
  }
}

Constant *
llvm::ConstantFoldFP(double (*NativeFP)(double), double V, const Type *Ty) {
  errno = 0;
  V = NativeFP(V);
  if (errno == 0)
    return ConstantFP::get(Ty, V);
  return 0;
}

/// ConstantFoldCall - Attempt to constant fold a call to the specified function
/// with the specified arguments, returning null if unsuccessful.
Constant *
llvm::ConstantFoldCall(Function *F, const std::vector<Constant*> &Operands) {
  const std::string &Name = F->getName();
  const Type *Ty = F->getReturnType();

  if (Operands.size() == 1) {
    if (ConstantFP *Op = dyn_cast<ConstantFP>(Operands[0])) {
      double V = Op->getValue();
      switch (Name[0])
      {
        case 'a':
          if (Name == "acos")
            return ConstantFoldFP(acos, V, Ty);
          else if (Name == "asin")
            return ConstantFoldFP(asin, V, Ty);
          else if (Name == "atan")
            return ConstantFP::get(Ty, atan(V));
          break;
        case 'c':
          if (Name == "ceil")
            return ConstantFoldFP(ceil, V, Ty);
          else if (Name == "cos")
            return ConstantFP::get(Ty, cos(V));
          else if (Name == "cosh")
            return ConstantFP::get(Ty, cosh(V));
          break;
        case 'e':
          if (Name == "exp")
            return ConstantFP::get(Ty, exp(V));
          break;
        case 'f':
          if (Name == "fabs")
            return ConstantFP::get(Ty, fabs(V));
          else if (Name == "floor")
            return ConstantFoldFP(floor, V, Ty);
          break;
        case 'l':
          if (Name == "log" && V > 0)
            return ConstantFP::get(Ty, log(V));
          else if (Name == "log10" && V > 0)
            return ConstantFoldFP(log10, V, Ty);
          else if (Name == "llvm.sqrt") {
            if (V >= -0.0)
              return ConstantFP::get(Ty, sqrt(V));
            else // Undefined
              return ConstantFP::get(Ty, 0.0);
          }
          break;
        case 's':
          if (Name == "sin")
            return ConstantFP::get(Ty, sin(V));
          else if (Name == "sinh")
            return ConstantFP::get(Ty, sinh(V));
          else if (Name == "sqrt" && V >= 0)
            return ConstantFP::get(Ty, sqrt(V));
          break;
        case 't':
          if (Name == "tan")
            return ConstantFP::get(Ty, tan(V));
          else if (Name == "tanh")
            return ConstantFP::get(Ty, tanh(V));
          break;
        default:
          break;
      }
    } else if (ConstantUInt *Op = dyn_cast<ConstantUInt>(Operands[0])) {
      uint64_t V = Op->getValue();
      if (Name == "llvm.bswap.i16")
        return ConstantUInt::get(Ty, ByteSwap_16(V));
      else if (Name == "llvm.bswap.i32")
        return ConstantUInt::get(Ty, ByteSwap_32(V));
      else if (Name == "llvm.bswap.i64")
        return ConstantUInt::get(Ty, ByteSwap_64(V));
    }
  } else if (Operands.size() == 2) {
    if (ConstantFP *Op1 = dyn_cast<ConstantFP>(Operands[0])) {
      double Op1V = Op1->getValue();
      if (ConstantFP *Op2 = dyn_cast<ConstantFP>(Operands[1])) {
        double Op2V = Op2->getValue();

        if (Name == "llvm.isunordered")
          return ConstantBool::get(IsNAN(Op1V) || IsNAN(Op2V));
        else
        if (Name == "pow") {
          errno = 0;
          double V = pow(Op1V, Op2V);
          if (errno == 0)
            return ConstantFP::get(Ty, V);
        } else if (Name == "fmod") {
          errno = 0;
          double V = fmod(Op1V, Op2V);
          if (errno == 0)
            return ConstantFP::get(Ty, V);
        } else if (Name == "atan2")
          return ConstantFP::get(Ty, atan2(Op1V,Op2V));
      }
    }
  }
  return 0;
}

