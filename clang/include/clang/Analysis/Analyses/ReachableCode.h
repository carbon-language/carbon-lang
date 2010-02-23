//===- ReachableCode.h -----------------------------------------*- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A flow-sensitive, path-insensitive analysis of unreachable code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_REACHABLECODE_H
#define LLVM_CLANG_REACHABLECODE_H

//===----------------------------------------------------------------------===//
// Forward declarations.
//===----------------------------------------------------------------------===//

namespace llvm {
class BitVector;
} // end llvm namespace

namespace clang {
class CFGBlock;
} // end clang namespace

//===----------------------------------------------------------------------===//
// API.
//===----------------------------------------------------------------------===//

namespace clang {

/// ScanReachableFromBlock - Mark all blocks reachable from Start.
/// Returns the total number of blocks that were marked reachable.
unsigned ScanReachableFromBlock(const CFGBlock &B, llvm::BitVector &Reachable);

} // end clang namespace

#endif
