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
  typedef ValueMap<const Value *, TrackingVH<Value> > ValueToValueMapTy;

  /// RemapFlags - These are flags that the value mapping APIs allow.
  enum RemapFlags {
    RF_None = 0,
    
    /// RF_NoModuleLevelChanges - If this flag is set, the remapper knows that
    /// only local values within a function (such as an instruction or argument)
    /// are mapped, not global values like functions and global metadata.
    RF_NoModuleLevelChanges = 1,
    
    /// RF_IgnoreMissingEntries - If this flag is set, the remapper ignores
    /// entries that are not in the value map.  If it is unset, it aborts if an
    /// operand is asked to be remapped which doesn't exist in the mapping.
    RF_IgnoreMissingEntries = 2
  };
  
  static inline RemapFlags operator|(RemapFlags LHS, RemapFlags RHS) {
    return RemapFlags(unsigned(LHS)|unsigned(RHS));
  }
  
  Value *MapValue(const Value *V, ValueToValueMapTy &VM,
                  RemapFlags Flags = RF_None);
  void RemapInstruction(Instruction *I, ValueToValueMapTy &VM,
                        RemapFlags Flags = RF_None);
} // End llvm namespace

#endif
