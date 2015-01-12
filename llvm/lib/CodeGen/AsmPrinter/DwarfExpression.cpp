//===-- llvm/CodeGen/DwarfExpression.cpp - Dwarf Debug Framework ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing dwarf debug info into asm files.
//
//===----------------------------------------------------------------------===//

#include "DwarfExpression.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"


using namespace llvm;

void DwarfExpression::AddReg(int DwarfReg, const char* Comment) {
  assert(DwarfReg >= 0 && "invalid negative dwarf register number");
  if (DwarfReg < 32) {
    EmitOp(dwarf::DW_OP_reg0 + DwarfReg, Comment);
  } else {
    EmitOp(dwarf::DW_OP_regx, Comment);
    EmitUnsigned(DwarfReg);
  }
}

void DwarfExpression::AddRegIndirect(int DwarfReg, int Offset, bool Deref) {
  assert(DwarfReg >= 0 && "invalid negative dwarf register number");
  if (DwarfReg < 32) {
    EmitOp(dwarf::DW_OP_breg0 + DwarfReg);
  } else {
    EmitOp(dwarf::DW_OP_bregx);
    EmitUnsigned(DwarfReg);
  }
  EmitSigned(Offset);
  if (Deref)
    EmitOp(dwarf::DW_OP_deref);
}

void DwarfExpression::AddOpPiece(unsigned SizeInBits,
                                 unsigned OffsetInBits) {
  assert(SizeInBits > 0 && "piece has size zero");
  const unsigned SizeOfByte = 8;
  if (OffsetInBits > 0 || SizeInBits % SizeOfByte) {
    EmitOp(dwarf::DW_OP_bit_piece);
    EmitUnsigned(SizeInBits);
    EmitUnsigned(OffsetInBits);
  } else {
    EmitOp(dwarf::DW_OP_piece);
    unsigned ByteSize = SizeInBits / SizeOfByte;
    EmitUnsigned(ByteSize);
  }
}

void DwarfExpression::AddShr(unsigned ShiftBy) {
  EmitOp(dwarf::DW_OP_constu);
  EmitUnsigned(ShiftBy);
  EmitOp(dwarf::DW_OP_shr);
}

bool DwarfExpression::AddMachineRegIndirect(unsigned MachineReg, int Offset) {
  const TargetRegisterInfo *TRI = TM.getSubtargetImpl()->getRegisterInfo();
  int DwarfReg = TRI->getDwarfRegNum(MachineReg, false);
  if (DwarfReg < 0)
    return false;

  if (MachineReg == getFrameRegister()) {
    // If variable offset is based in frame register then use fbreg.
    EmitOp(dwarf::DW_OP_fbreg);
    EmitSigned(Offset);
  } else {
    AddRegIndirect(DwarfReg, Offset);
  }
  return true;
}

void DwarfExpression::AddMachineRegPiece(unsigned MachineReg,
                                         unsigned PieceSizeInBits,
                                         unsigned PieceOffsetInBits) {
  const TargetRegisterInfo *TRI = TM.getSubtargetImpl()->getRegisterInfo();
  int Reg = TRI->getDwarfRegNum(MachineReg, false);

  // If this is a valid register number, emit it.
  if (Reg >= 0) {
    AddReg(Reg);
    AddOpPiece(PieceSizeInBits, PieceOffsetInBits);
    return;
  }

  // Walk up the super-register chain until we find a valid number.
  // For example, EAX on x86_64 is a 32-bit piece of RAX with offset 0.
  for (MCSuperRegIterator SR(MachineReg, TRI); SR.isValid(); ++SR) {
    Reg = TRI->getDwarfRegNum(*SR, false);
    if (Reg >= 0) {
      unsigned Idx = TRI->getSubRegIndex(*SR, MachineReg);
      unsigned Size = TRI->getSubRegIdxSize(Idx);
      unsigned RegOffset = TRI->getSubRegIdxOffset(Idx);
      AddReg(Reg, "super-register");
      if (PieceOffsetInBits == RegOffset) {
        AddOpPiece(Size, RegOffset);
      } else {
        // If this is part of a variable in a sub-register at a
        // non-zero offset, we need to manually shift the value into
        // place, since the DW_OP_piece describes the part of the
        // variable, not the position of the subregister.
        if (RegOffset)
          AddShr(RegOffset);
        AddOpPiece(Size, PieceOffsetInBits);
      }
      return;
    }
  }

  // Otherwise, attempt to find a covering set of sub-register numbers.
  // For example, Q0 on ARM is a composition of D0+D1.
  //
  // Keep track of the current position so we can emit the more
  // efficient DW_OP_piece.
  unsigned CurPos = PieceOffsetInBits;
  // The size of the register in bits, assuming 8 bits per byte.
  unsigned RegSize = TRI->getMinimalPhysRegClass(MachineReg)->getSize() * 8;
  // Keep track of the bits in the register we already emitted, so we
  // can avoid emitting redundant aliasing subregs.
  SmallBitVector Coverage(RegSize, false);
  for (MCSubRegIterator SR(MachineReg, TRI); SR.isValid(); ++SR) {
    unsigned Idx = TRI->getSubRegIndex(MachineReg, *SR);
    unsigned Size = TRI->getSubRegIdxSize(Idx);
    unsigned Offset = TRI->getSubRegIdxOffset(Idx);
    Reg = TRI->getDwarfRegNum(*SR, false);

    // Intersection between the bits we already emitted and the bits
    // covered by this subregister.
    SmallBitVector Intersection(RegSize, false);
    Intersection.set(Offset, Offset + Size);
    Intersection ^= Coverage;

    // If this sub-register has a DWARF number and we haven't covered
    // its range, emit a DWARF piece for it.
    if (Reg >= 0 && Intersection.any()) {
      AddReg(Reg, "sub-register");
      AddOpPiece(Size, Offset == CurPos ? 0 : Offset);
      CurPos = Offset + Size;

      // Mark it as emitted.
      Coverage.set(Offset, Offset + Size);
    }
  }

  if (CurPos == PieceOffsetInBits)
    // FIXME: We have no reasonable way of handling errors in here.
    EmitOp(dwarf::DW_OP_nop, "nop (could not find a dwarf register number)");
}
