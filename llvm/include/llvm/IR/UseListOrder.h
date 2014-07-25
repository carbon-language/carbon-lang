//===- llvm/IR/UseListOrder.h - LLVM Use List Order functions ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file has functions to modify the use-list order and to verify that it
// doesn't change after serialization.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_USELISTORDER_H
#define LLVM_IR_USELISTORDER_H

#include "llvm/ADT/ArrayRef.h"

namespace llvm {

class Module;

/// \brief Whether to preserve use-list ordering.
bool shouldPreserveBitcodeUseListOrder();
bool shouldPreserveAssemblyUseListOrder();

/// \brief Shuffle all use-lists in a module.
///
/// Adds \c SeedOffset to the default seed for the random number generator.
void shuffleUseLists(Module &M, unsigned SeedOffset = 0);

} // end namespace llvm

#endif
