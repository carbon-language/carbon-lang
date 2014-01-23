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
class AliasAnalysis;
class BasicBlock;
class DominatorTree;
class Loop;
class LoopInfo;
class Pass;
class ScalarEvolution;

BasicBlock *InsertPreheaderForLoop(Loop *L, Pass *P);

/// \brief Simplify each loop in a loop nest recursively.
///
/// This takes a potentially un-simplified loop L (and its children) and turns
/// it into a simplified loop nest with preheaders and single backedges. It
/// will optionally update \c AliasAnalysis and \c ScalarEvolution analyses if
/// passed into it.
bool simplifyLoop(Loop *L, DominatorTree *DT, LoopInfo *LI, Pass *PP,
                  AliasAnalysis *AA = 0, ScalarEvolution *SE = 0);

}

#endif
