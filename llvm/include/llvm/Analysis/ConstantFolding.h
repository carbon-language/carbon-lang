//===-- ConstantFolding.h - Analyze constant folding possibilities --------===//
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

#include "llvm/Constants.h"
#include "llvm/Function.h"
using namespace llvm;

namespace llvm {

/// canConstantFoldCallTo - Return true if its even possible to fold a call to
/// the specified function.
extern
bool canConstantFoldCallTo(Function *F);

/// ConstantFoldFP - Given a function that evaluates the constant, return an
///                  LLVM Constant that represents the evaluated constant
extern Constant *
ConstantFoldFP(double (*NativeFP)(double), double V, const Type *Ty);

/// ConstantFoldCall - Attempt to constant fold a call to the specified function
/// with the specified arguments, returning null if unsuccessful.
extern Constant *
ConstantFoldCall(Function *F, const std::vector<Constant*> &Operands);
}

