//===-- Transform/Utils/FunctionUtils.h - Function Utils --------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This family of functions perform manipulations on functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_FUNCTION_H
#define LLVM_TRANSFORMS_UTILS_FUNCTION_H

namespace llvm {

class Function;
class Loop;

/// ExtractLoop - rip out a natural loop into a new function
///
Function* ExtractLoop(Loop *L);

/// ExtractBasicBlock - rip out a basic block into a new function
///
Function* ExtractBasicBlock(BasicBlock *BB);

}

#endif
