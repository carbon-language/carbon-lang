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
#include <cassert>

namespace llvm {
class GlobalValue;

/// MachineRelocation - This represents a target-specific relocation value,
/// produced by the code emitter.  This relocation is resolved after the has
/// been emitted, either to an object file or to memory, when the target of the
/// relocation can be resolved.
///
/// A relocation is made up of the following logical portions:
///   1. An offset in the machine code buffer, the location to modify.
///   2. A target specific relocation type (a number from 0 to 63).
///   3. A symbol being referenced, either as a GlobalValue* or as a string.
///   4. An optional constant value to be added to the reference.
///   5. A bit, CanRewrite, which indicates to the JIT that a function stub is
///      not needed for the relocation.
///   6. An index into the GOT, if the target uses a GOT
///
class MachineRelocation {
  /// OffsetTypeExternal - The low 24-bits of this value is the offset from the
  /// start of the code buffer of the relocation to perform.  Bit 24 of this is
  /// set if Target should use ExtSym instead of GV, Bit 25 is the CanRewrite
  /// bit, and the high 6 bits hold the relocation type.
  // FIXME: with the additional types of relocatable things, rearrange the
  // storage of things to be a bit more effiecient
  unsigned OffsetTypeExternal;
  union {
    GlobalValue *GV;     // If this is a pointer to an LLVM global
    const char *ExtSym;  // If this is a pointer to a named symbol
    void *Result;        // If this has been resolved to a resolved pointer
    unsigned GOTIndex;   // Index in the GOT of this symbol/global
    unsigned CPool;      // Index in the Constant Pool
  } Target;
  intptr_t ConstantVal;
  bool GOTRelative; //out of bits in OffsetTypeExternal
  bool isConstPool;

public:
  MachineRelocation(unsigned Offset, unsigned RelocationType, GlobalValue *GV,
                    intptr_t cst = 0, bool DoesntNeedFunctionStub = 0,
                    bool GOTrelative = 0)
    : OffsetTypeExternal(Offset + (RelocationType << 26)), ConstantVal(cst),
      GOTRelative(GOTrelative), isConstPool(0) {
    assert((Offset & ~((1 << 24)-1)) == 0 && "Code offset too large!");
    assert((RelocationType & ~63) == 0 && "Relocation type too large!");
    Target.GV = GV;
    if (DoesntNeedFunctionStub)
      OffsetTypeExternal |= 1 << 25;
  }

  MachineRelocation(unsigned Offset, unsigned RelocationType, const char *ES,
                    intptr_t cst = 0, bool GOTrelative = 0)
    : OffsetTypeExternal(Offset + (1 << 24) + (RelocationType << 26)),
      ConstantVal(cst), GOTRelative(GOTrelative), isConstPool(0) {
    assert((Offset & ~((1 << 24)-1)) == 0 && "Code offset too large!");
    assert((RelocationType & ~63) == 0 && "Relocation type too large!");
    Target.ExtSym = ES;
  }

  MachineRelocation(unsigned Offset, unsigned RelocationType, unsigned CPI,
                    intptr_t cst = 0)
    : OffsetTypeExternal(Offset + (RelocationType << 26)),
      ConstantVal(cst), GOTRelative(0), isConstPool(1) {
    assert((Offset & ~((1 << 24)-1)) == 0 && "Code offset too large!");
    assert((RelocationType & ~63) == 0 && "Relocation type too large!");
    Target.CPool = CPI;
  }

  /// getMachineCodeOffset - Return the offset into the code buffer that the
  /// relocation should be performed.
  unsigned getMachineCodeOffset() const {
    return OffsetTypeExternal & ((1 << 24)-1);
  }

  /// getRelocationType - Return the target-specific relocation ID for this
  /// relocation.
  unsigned getRelocationType() const {
    return OffsetTypeExternal >> 26;
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
    return (OffsetTypeExternal & (1 << 24)) == 0 && !isConstantPoolIndex();
  }

  /// isString - Return true if this is a constant string.
  ///
  bool isString() const {
    return !isGlobalValue() && !isConstantPoolIndex();
  }

  /// isConstantPoolIndex - Return true if this is a constant pool reference.
  ///
  bool isConstantPoolIndex() const {
    return isConstPool;
  }

  /// isGOTRelative - Return true the target wants the index into the GOT of
  /// the symbol rather than the address of the symbol.
  bool isGOTRelative() const {
    return GOTRelative;
  }

  /// doesntNeedFunctionStub - This function returns true if the JIT for this
  /// target is capable of directly handling the relocated instruction without
  /// using a stub function.  It is always conservatively correct for this flag
  /// to be false, but targets can improve their compilation callback functions
  /// to handle more general cases if they want improved performance.
  bool doesntNeedFunctionStub() const {
    return (OffsetTypeExternal & (1 << 25)) != 0;
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

  /// getConstantPoolIndex - If this is a const pool reference, return
  /// the index into the constant pool.
  unsigned getConstantPoolIndex() const {
    assert(isConstantPoolIndex() && "This is not a constant pool reference!");
    return Target.CPool;
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

  /// setGOTIndex - Set the GOT index to a specific value.
  void setGOTIndex(unsigned idx) {
    Target.GOTIndex = idx;
  }

  /// getGOTIndex - Once this has been resolved to an entry in the GOT,
  /// this returns that index.  The index is from the lowest address entry
  /// in the GOT.
  unsigned getGOTIndex() const {
    return Target.GOTIndex;
  }

};

}

#endif
