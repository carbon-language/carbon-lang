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
class MachineBasicBlock;

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
  enum AddressType {
    isResult,         // Relocation has be transformed into its result pointer.
    isGV,             // The Target.GV field is valid.
    isBB,             // Relocation of BB address.
    isExtSym,         // The Target.ExtSym field is valid.
    isConstPool,      // Relocation of constant pool address.
    isJumpTable,      // Relocation of jump table address.
    isGOTIndex        // The Target.GOTIndex field is valid.
  };
  
  /// Offset - This is the offset from the start of the code buffer of the
  /// relocation to perform.
  intptr_t Offset;
  
  /// ConstantVal - A field that may be used by the target relocation type.
  intptr_t ConstantVal;

  union {
    void *Result;           // If this has been resolved to a resolved pointer
    GlobalValue *GV;        // If this is a pointer to an LLVM global
    MachineBasicBlock *MBB; // If this is a pointer to a LLVM BB
    const char *ExtSym;     // If this is a pointer to a named symbol
    unsigned Index;         // Constant pool / jump table index
    unsigned GOTIndex;      // Index in the GOT of this symbol/global
  } Target;

  unsigned TargetReloType : 6; // The target relocation ID.
  AddressType AddrType    : 3; // The field of Target to use.
  bool DoesntNeedFnStub   : 1; // True if we don't need a fn stub.
  bool GOTRelative        : 1; // Should this relocation be relative to the GOT?

public:
  /// MachineRelocation::getGV - Return a relocation entry for a GlobalValue.
  ///
  static MachineRelocation getGV(intptr_t offset, unsigned RelocationType, 
                                 GlobalValue *GV, intptr_t cst = 0,
                                 bool DoesntNeedFunctionStub = 0,
                                 bool GOTrelative = 0) {
    assert((RelocationType & ~63) == 0 && "Relocation type too large!");
    MachineRelocation Result;
    Result.Offset = offset;
    Result.ConstantVal = cst;
    Result.TargetReloType = RelocationType;
    Result.AddrType = isGV;
    Result.DoesntNeedFnStub = DoesntNeedFunctionStub;
    Result.GOTRelative = GOTrelative;
    Result.Target.GV = GV;
    return Result;
  }

  /// MachineRelocation::getBB - Return a relocation entry for a BB.
  ///
  static MachineRelocation getBB(intptr_t offset,unsigned RelocationType,
                                 MachineBasicBlock *MBB, intptr_t cst = 0) {
    assert((RelocationType & ~63) == 0 && "Relocation type too large!");
    MachineRelocation Result;
    Result.Offset = offset;
    Result.ConstantVal = cst;
    Result.TargetReloType = RelocationType;
    Result.AddrType = isBB;
    Result.DoesntNeedFnStub = false;
    Result.GOTRelative = false;
    Result.Target.MBB = MBB;
    return Result;
  }

  /// MachineRelocation::getExtSym - Return a relocation entry for an external
  /// symbol, like "free".
  ///
  static MachineRelocation getExtSym(intptr_t offset, unsigned RelocationType, 
                                     const char *ES, intptr_t cst = 0,
                                     bool GOTrelative = 0) {
    assert((RelocationType & ~63) == 0 && "Relocation type too large!");
    MachineRelocation Result;
    Result.Offset = offset;
    Result.ConstantVal = cst;
    Result.TargetReloType = RelocationType;
    Result.AddrType = isExtSym;
    Result.DoesntNeedFnStub = false;
    Result.GOTRelative = GOTrelative;
    Result.Target.ExtSym = ES;
    return Result;
  }

  /// MachineRelocation::getConstPool - Return a relocation entry for a constant
  /// pool entry.
  ///
  static MachineRelocation getConstPool(intptr_t offset,unsigned RelocationType,
                                        unsigned CPI, intptr_t cst = 0) {
    assert((RelocationType & ~63) == 0 && "Relocation type too large!");
    MachineRelocation Result;
    Result.Offset = offset;
    Result.ConstantVal = cst;
    Result.TargetReloType = RelocationType;
    Result.AddrType = isConstPool;
    Result.DoesntNeedFnStub = false;
    Result.GOTRelative = false;
    Result.Target.Index = CPI;
    return Result;
  }

  /// MachineRelocation::getJumpTable - Return a relocation entry for a jump
  /// table entry.
  ///
  static MachineRelocation getJumpTable(intptr_t offset,unsigned RelocationType,
                                        unsigned JTI, intptr_t cst = 0) {
    assert((RelocationType & ~63) == 0 && "Relocation type too large!");
    MachineRelocation Result;
    Result.Offset = offset;
    Result.ConstantVal = cst;
    Result.TargetReloType = RelocationType;
    Result.AddrType = isJumpTable;
    Result.DoesntNeedFnStub = false;
    Result.GOTRelative = false;
    Result.Target.Index = JTI;
    return Result;
  }

  /// getMachineCodeOffset - Return the offset into the code buffer that the
  /// relocation should be performed.
  intptr_t getMachineCodeOffset() const {
    return Offset;
  }

  /// getRelocationType - Return the target-specific relocation ID for this
  /// relocation.
  unsigned getRelocationType() const {
    return TargetReloType;
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
    return AddrType == isGV;
  }

  /// isBasicBlock - Return true if this relocation is a basic block reference.
  ///
  bool isBasicBlock() const {
    return AddrType == isBB;
  }

  /// isString - Return true if this is a constant string.
  ///
  bool isString() const {
    return AddrType == isExtSym;
  }

  /// isConstantPoolIndex - Return true if this is a constant pool reference.
  ///
  bool isConstantPoolIndex() const {
    return AddrType == isConstPool;
  }

  /// isJumpTableIndex - Return true if this is a jump table reference.
  ///
  bool isJumpTableIndex() const {
    return AddrType == isJumpTable;
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
    return DoesntNeedFnStub;
  }

  /// getGlobalValue - If this is a global value reference, return the
  /// referenced global.
  GlobalValue *getGlobalValue() const {
    assert(isGlobalValue() && "This is not a global value reference!");
    return Target.GV;
  }

  MachineBasicBlock *getBasicBlock() const {
    assert(isBasicBlock() && "This is not a basic block reference!");
    return Target.MBB;
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
    return Target.Index;
  }

  /// getJumpTableIndex - If this is a jump table reference, return
  /// the index into the jump table.
  unsigned getJumpTableIndex() const {
    assert(isJumpTableIndex() && "This is not a jump table reference!");
    return Target.Index;
  }

  /// getResultPointer - Once this has been resolved to point to an actual
  /// address, this returns the pointer.
  void *getResultPointer() const {
    assert(AddrType == isResult && "Result pointer isn't set yet!");
    return Target.Result;
  }

  /// setResultPointer - Set the result to the specified pointer value.
  ///
  void setResultPointer(void *Ptr) {
    Target.Result = Ptr;
    AddrType = isResult;
  }

  /// setGOTIndex - Set the GOT index to a specific value.
  void setGOTIndex(unsigned idx) {
    AddrType = isGOTIndex;
    Target.GOTIndex = idx;
  }

  /// getGOTIndex - Once this has been resolved to an entry in the GOT,
  /// this returns that index.  The index is from the lowest address entry
  /// in the GOT.
  unsigned getGOTIndex() const {
    assert(AddrType == isGOTIndex);
    return Target.GOTIndex;
  }
};
}

#endif
