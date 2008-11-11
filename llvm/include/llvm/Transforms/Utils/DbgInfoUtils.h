//===-- Transform/Utils/DbgInfoUtils.h - DbgInfo Utils ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Utility functions to manipulate debugging information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_DBGINFO_H
#define LLVM_TRANSFORMS_UTILS_DBGINFO_H
namespace llvm {
class BasicBlock;
class Function;

/// RemoveDeadDbgIntrinsics - Remove dead dbg intrinsics from this 
/// basic block.
void RemoveDeadDbgIntrinsics(BasicBlock &BB);

/// RemoveDeadDbgIntrinsics - Remove dead dbg intrinsics from this function.
void RemoveDeadDbgIntrinsics(Function &F);

} // End llvm namespace
#endif
