//===- llvm/Transforms/Utils/LoopUtils.h - Loop utilities -*- C++ -*-=========//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines some loop transformation utilities.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_LOOPUTILS_H
#define LLVM_TRANSFORMS_UTILS_LOOPUTILS_H

namespace llvm {

class Loop;
class Pass;

BasicBlock *InsertPreheaderForLoop(Loop *L, Pass *P);

}

#endif
