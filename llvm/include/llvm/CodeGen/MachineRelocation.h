//===-- llvm/CodeGen/MachineRelocation.h - Target Relocation ----*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the MachineRelocation class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINERELOCATION_H
#define LLVM_CODEGEN_MACHINERELOCATION_H

#include "llvm/Support/DataTypes.h"

namespace llvm {
class GlobalValue;

/// MachineRelocation - This represents a target-specific relocation value,
/// produced by the code emitter.  This relocation is resolved after the has
/// been emitted, either to an object file or to memory, when the target of the
/// relocation can be resolved.
///
/// A relocation is made up of the following logical portions:
///   1. An offset in the machine code buffer, the location to modify.
///   2. A target specific relocation type (a number from 0 to 127).
///   3. A symbol being referenced, either as a GlobalValue* or as a string.
///   4. An optional constant value to be added to the reference.
///
class MachineRelocation {
  /// OffsetTypeExternal - The low 24-bits of this value is the offset from the
  /// start of the code buffer of the relocation to perform.  Bit 24 of this is
  /// set if Target should use ExtSym instead of GV, and the high 7 bits are to
  /// hold the relocation type.
  unsigned OffsetTypeExternal;
  union {
    GlobalValue *GV;     // If this is a pointer to an LLVM global
    const char *ExtSym;  // If this is a pointer to a named symbol
    void *Result;        // If this has been resolved to a resolved pointer
  } Target;
  intptr_t ConstantVal;
public:
  MachineRelocation(unsigned Offset, unsigned RelocationType, GlobalValue *GV,
             intptr_t cst = 0)
    : OffsetTypeExternal(Offset + (RelocationType << 25)), ConstantVal(cst) {
    assert((Offset & ~((1 << 24)-1)) == 0 && "Code offset too large!");
    assert((RelocationType & ~127) == 0 && "Relocation type too large!");
    Target.GV = GV;
  }

  MachineRelocation(unsigned Offset, unsigned RelocationType, const char *ES,
             intptr_t cst = 0)
    : OffsetTypeExternal(Offset + (1 << 24) + (RelocationType << 25)),
    ConstantVal(cst) {
    assert((Offset & ~((1 << 24)-1)) == 0 && "Code offset too large!");
    assert((RelocationType & ~127) == 0 && "Relocation type too large!");
    Target.ExtSym = ES;
  }

  /// getMachineCodeOffset - Return the offset into the code buffer that the
  /// relocation should be performed.
  unsigned getMachineCodeOffset() const {
    return OffsetTypeExternal & ((1 << 24)-1);
  }

  /// getConstantVal - Get the constant value associated with this relocation.
  /// This is often an offset from the symbol.
  ///
  intptr_t getConstantVal() const {
    return ConstantVal;
  }

  /// isGlobalValue - Return true if this relocation is a GlobalValue, as
  /// opposed to a constant string.
  bool isGlobalValue() const {
    return (OffsetTypeExternal & (1 << 24)) == 0;
  }

  /// isString - Return true if this is a constant string.
  ///
  bool isString() const {
    return !isGlobalValue();
  }

  /// getGlobalValue - If this is a global value reference, return the
  /// referenced global.
  GlobalValue *getGlobalValue() const {
    assert(isGlobalValue() && "This is not a global value reference!");
    return Target.GV;
  }

  /// getString - If this is a string value, return the string reference.
  ///
  const char *getString() const {
    assert(isString() && "This is not a string reference!");
    return Target.ExtSym;
  }

  /// getResultPointer - Once this has been resolved to point to an actual
  /// address, this returns the pointer.
  void *getResultPointer() const {
    return Target.Result;
  }

  /// setResultPointer - Set the result to the specified pointer value.
  ///
  void setResultPointer(void *Ptr) {
    Target.Result = Ptr;
  }
};

}

#endif
