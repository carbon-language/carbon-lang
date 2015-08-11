//===-- llvm/CodeGen/PseudoSourceValue.h ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the PseudoSourceValue class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PSEUDOSOURCEVALUE_H
#define LLVM_CODEGEN_PSEUDOSOURCEVALUE_H

#include "llvm/IR/Value.h"

namespace llvm {

class MachineFrameInfo;
class MachineMemOperand;
class raw_ostream;

raw_ostream &operator<<(raw_ostream &OS, const MachineMemOperand &MMO);

/// Special value supplied for machine level alias analysis. It indicates that
/// a memory access references the functions stack frame (e.g., a spill slot),
/// below the stack frame (e.g., argument space), or constant pool.
class PseudoSourceValue {
public:
  enum PSVKind { Stack, GOT, JumpTable, ConstantPool, FixedStack, MipsPSV };

private:
  PSVKind Kind;

  friend class MachineMemOperand; // For printCustom().

  /// Implement printing for PseudoSourceValue. This is called from
  /// Value::print or Value's operator<<.
  virtual void printCustom(raw_ostream &O) const;

public:
  explicit PseudoSourceValue(PSVKind Kind);

  virtual ~PseudoSourceValue();

  PSVKind kind() const { return Kind; }

  bool isStack() const { return Kind == Stack; }
  bool isGOT() const { return Kind == GOT; }
  bool isConstantPool() const { return Kind == ConstantPool; }
  bool isJumpTable() const { return Kind == JumpTable; }

  /// Test whether the memory pointed to by this PseudoSourceValue has a
  /// constant value.
  virtual bool isConstant(const MachineFrameInfo *) const;

  /// Test whether the memory pointed to by this PseudoSourceValue may also be
  /// pointed to by an LLVM IR Value.
  virtual bool isAliased(const MachineFrameInfo *) const;

  /// Return true if the memory pointed to by this PseudoSourceValue can ever
  /// alias an LLVM IR Value.
  virtual bool mayAlias(const MachineFrameInfo *) const;

  /// A pseudo source value referencing a fixed stack frame entry,
  /// e.g., a spill slot.
  static const PseudoSourceValue *getFixedStack(int FI);

  /// A pseudo source value referencing the area below the stack frame of
  /// a function, e.g., the argument space.
  static const PseudoSourceValue *getStack();

  /// A pseudo source value referencing the global offset table
  /// (or something the like).
  static const PseudoSourceValue *getGOT();

  /// A pseudo source value referencing the constant pool. Since constant
  /// pools are constant, this doesn't need to identify a specific constant
  /// pool entry.
  static const PseudoSourceValue *getConstantPool();

  /// A pseudo source value referencing a jump table. Since jump tables are
  /// constant, this doesn't need to identify a specific jump table.
  static const PseudoSourceValue *getJumpTable();
};

/// A specialized PseudoSourceValue for holding FixedStack values, which must
/// include a frame index.
class FixedStackPseudoSourceValue : public PseudoSourceValue {
  const int FI;

public:
  explicit FixedStackPseudoSourceValue(int FI)
      : PseudoSourceValue(FixedStack), FI(FI) {}

  static inline bool classof(const PseudoSourceValue *V) {
    return V->kind() == FixedStack;
  }

  bool isConstant(const MachineFrameInfo *MFI) const override;

  bool isAliased(const MachineFrameInfo *MFI) const override;

  bool mayAlias(const MachineFrameInfo *) const override;

  void printCustom(raw_ostream &OS) const override;

  int getFrameIndex() const { return FI; }
};

} // end namespace llvm

#endif
