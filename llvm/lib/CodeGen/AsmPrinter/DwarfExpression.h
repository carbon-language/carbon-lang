//===-- llvm/CodeGen/DwarfExpression.h - Dwarf Compile Unit ---*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing dwarf compile unit.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_DWARFEXPRESSION_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_DWARFEXPRESSION_H

#include "llvm/Support/DataTypes.h"

namespace llvm {

class AsmPrinter;
class ByteStreamer;
class TargetRegisterInfo;

/// Base class containing the logic for constructing DWARF expressions
/// independently of whether they are emitted into a DIE or into a .debug_loc
/// entry.
class DwarfExpression {
protected:
  const AsmPrinter &AP;
  // Various convenience accessors that extract things out of AsmPrinter.
  const TargetRegisterInfo *getTRI() const;
  unsigned getDwarfVersion() const;

public:
  DwarfExpression(const AsmPrinter &AP) : AP(AP) {}
  virtual ~DwarfExpression() {}

  virtual void EmitOp(uint8_t Op, const char* Comment = nullptr) = 0;
  virtual void EmitSigned(int Value) = 0;
  virtual void EmitUnsigned(unsigned Value) = 0;

  virtual unsigned getFrameRegister() = 0;

  /// Emit a dwarf register operation.
  void AddReg(int DwarfReg, const char* Comment = nullptr);
  /// Emit an (double-)indirect dwarf register operation.
  void AddRegIndirect(int DwarfReg, int Offset, bool Deref = false);

  /// Emit a dwarf register operation for describing
  /// - a small value occupying only part of a register or
  /// - a register representing only part of a value.
  void AddOpPiece(unsigned SizeInBits, unsigned OffsetInBits = 0);
  /// Emit a shift-right dwarf expression.
  void AddShr(unsigned ShiftBy);

  /// Emit an indirect dwarf register operation for the given machine register.
  /// Returns false if no DWARF register exists for MachineReg.
  bool AddMachineRegIndirect(unsigned MachineReg, int Offset);

  /// \brief Emit a partial DWARF register operation.
  /// \param MLoc             the register
  /// \param PieceSize        size and
  /// \param PieceOffset      offset of the piece in bits, if this is one
  ///                         piece of an aggregate value.
  ///
  /// If size and offset is zero an operation for the entire
  /// register is emitted: Some targets do not provide a DWARF
  /// register number for every register.  If this is the case, this
  /// function will attempt to emit a DWARF register by emitting a
  /// piece of a super-register or by piecing together multiple
  /// subregisters that alias the register.
  void AddMachineRegPiece(unsigned MachineReg,
                          unsigned PieceSizeInBits = 0,
                          unsigned PieceOffsetInBits = 0);

  /// Emit a signed constant.
  void AddSignedConstant(int Value);
  /// Emit an unsigned constant.
  void AddUnsignedConstant(unsigned Value);
};


/// DwarfExpression implementation for .debug_loc entries.
class DebugLocDwarfExpression : public DwarfExpression {
  ByteStreamer &BS;

public:
  DebugLocDwarfExpression(const AsmPrinter &AP, ByteStreamer &BS)
      : DwarfExpression(AP), BS(BS) {}

  void EmitOp(uint8_t Op, const char *Comment) override;
  void EmitSigned(int Value) override;
  void EmitUnsigned(unsigned Value) override;
  unsigned getFrameRegister() override;
};

}

#endif
