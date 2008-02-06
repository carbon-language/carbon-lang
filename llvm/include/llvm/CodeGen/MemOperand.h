//===-- llvm/CodeGen/MemOperand.h - MemOperand class ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MemOperand class, which is a
// description of a memory reference. It is used to help track dependencies
// in the backend.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MEMOPERAND_H
#define LLVM_CODEGEN_MEMOPERAND_H

namespace llvm {

class Value;

//===----------------------------------------------------------------------===//
/// MemOperand - A description of a memory reference used in the backend.
/// Instead of holding a StoreInst or LoadInst, this class holds the address
/// Value of the reference along with a byte size and offset. This allows it
/// to describe lowered loads and stores. Also, the special PseudoSourceValue
/// objects can be used to represent loads and stores to memory locations
/// that aren't explicit in the regular LLVM IR.
///
class MemOperand {
  const Value *V;
  unsigned int Flags;
  int Offset;
  int Size;
  unsigned int Alignment;

public:
  /// Flags values. These may be or'd together.
  enum MemOperandFlags {
    /// The memory access reads data.
    MOLoad = 1,
    /// The memory access writes data.
    MOStore = 2,
    /// The memory access is volatile.
    MOVolatile = 4
  };

  /// MemOperand - Construct an MemOperand object with the specified
  /// address Value, flags, offset, size, and alignment.
  MemOperand(const Value *v, unsigned int f, int o, int s, unsigned int a)
    : V(v), Flags(f), Offset(o), Size(s), Alignment(a) {}

  /// getValue - Return the base address of the memory access.
  /// Special values are PseudoSourceValue::FPRel, PseudoSourceValue::SPRel,
  /// and the other PseudoSourceValue members which indicate references to
  /// frame/stack pointer relative references and other special references.
  const Value *getValue() const { return V; }

  /// getFlags - Return the raw flags of the source value, \see MemOperandFlags.
  unsigned int getFlags() const { return Flags; }

  /// getOffset - For normal values, this is a byte offset added to the base
  /// address. For PseudoSourceValue::FPRel values, this is the FrameIndex
  /// number.
  int getOffset() const { return Offset; }

  /// getSize - Return the size in bytes of the memory reference.
  int getSize() const { return Size; }

  /// getAlignment - Return the minimum known alignment in bytes of the
  /// memory reference.
  unsigned int getAlignment() const { return Alignment; }

  bool isLoad() const { return Flags & MOLoad; }
  bool isStore() const { return Flags & MOStore; }
  bool isVolatile() const { return Flags & MOVolatile; }
};

} // End llvm namespace

#endif
