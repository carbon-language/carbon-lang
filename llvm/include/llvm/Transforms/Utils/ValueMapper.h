//===- ValueMapper.h - Remapping for constants and metadata -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MapValue interface which is used by various parts of
// the Transforms/Utils library to implement cloning and linking facilities.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_VALUEMAPPER_H
#define LLVM_TRANSFORMS_UTILS_VALUEMAPPER_H

#include "llvm/ADT/ValueMap.h"

namespace llvm {
  class Value;
  class Instruction;
  typedef ValueMap<const Value *, Value *> ValueToValueMapTy;

  Value *MapValue(const Value *V, ValueToValueMapTy &VM);
  void RemapInstruction(Instruction *I, ValueToValueMapTy &VM);
} // End llvm namespace

#endif
