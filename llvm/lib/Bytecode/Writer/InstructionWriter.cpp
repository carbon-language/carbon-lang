//===-- WriteInst.cpp - Functions for writing instructions -------*- C++ -*--=//
//
// This file implements the routines for encoding instruction opcodes to a 
// bytecode stream.
//
// Note that the performance of this library is not terribly important, because
// it shouldn't be used by JIT type applications... so it is not a huge focus
// at least.  :)
//
//===----------------------------------------------------------------------===//

#include "WriterInternals.h"
#include "llvm/Module.h"
#include "llvm/Method.h"
#include "llvm/BasicBlock.h"
#include "llvm/Instruction.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Tools/DataTypes.h"
#include <algorithm>

typedef unsigned char uchar;

// outputInstructionFormat0 - Output those wierd instructions that have a large
// number of operands or have large operands themselves...
//
// Format: [opcode] [type] [numargs] [arg0] [arg1] ... [arg<numargs-1>]
//
static void outputInstructionFormat0(const Instruction *I,
				     const SlotCalculator &Table,
				     unsigned Type, vector<uchar> &Out) {
  // Opcode must have top two bits clear...
  output_vbr(I->getInstType(), Out);             // Instruction Opcode ID
  output_vbr(Type, Out);                         // Result type

  unsigned NumArgs;  // Count the number of arguments to the instruction
  for (NumArgs = 0; I->getOperand(NumArgs); NumArgs++) /*empty*/;
  output_vbr(NumArgs, Out);

  for (unsigned i = 0; const Value *N = I->getOperand(i); i++) {
    assert(i < NumArgs && "Count of arguments failed!");

    int Slot = Table.getValSlot(N);
    output_vbr((unsigned)Slot, Out);
  }
  align32(Out);    // We must maintain correct alignment!
}


// outputInstructionFormat1 - Output one operand instructions, knowing that no
// operand index is >= 2^12.
//
static void outputInstructionFormat1(const Instruction *I, 
				     const SlotCalculator &Table, int *Slots,
				     unsigned Type, vector<uchar> &Out) {
  unsigned IType = I->getInstType();      // Instruction Opcode ID
  
  // bits   Instruction format:
  // --------------------------
  // 31-30: Opcode type, fixed to 1.
  // 29-24: Opcode
  // 23-12: Resulting type plane
  // 11- 0: Operand #1 (if set to (2^12-1), then zero operands)
  //
  unsigned Opcode = (1 << 30) | (IType << 24) | (Type << 12) | Slots[0];
  //  cerr << "1 " << IType << " " << Type << " " << Slots[0] << endl;
  output(Opcode, Out);
}


// outputInstructionFormat2 - Output two operand instructions, knowing that no
// operand index is >= 2^8.
//
static void outputInstructionFormat2(const Instruction *I, 
				     const SlotCalculator &Table, int *Slots,
				     unsigned Type, vector<uchar> &Out) {
  unsigned IType = I->getInstType();      // Instruction Opcode ID

  // bits   Instruction format:
  // --------------------------
  // 31-30: Opcode type, fixed to 2.
  // 29-24: Opcode
  // 23-16: Resulting type plane
  // 15- 8: Operand #1
  //  7- 0: Operand #2  
  //
  unsigned Opcode = (2 << 30) | (IType << 24) | (Type << 16) |
                    (Slots[0] << 8) | (Slots[1] << 0);
  //  cerr << "2 " << IType << " " << Type << " " << Slots[0] << " " 
  //       << Slots[1] << endl;
  output(Opcode, Out);
}


// outputInstructionFormat3 - Output three operand instructions, knowing that no
// operand index is >= 2^6.
//
static void outputInstructionFormat3(const Instruction *I, 
				     const SlotCalculator &Table, int *Slots,
				     unsigned Type, vector<uchar> &Out) {
  unsigned IType = I->getInstType();      // Instruction Opcode ID

  // bits   Instruction format:
  // --------------------------
  // 31-30: Opcode type, fixed to 3
  // 29-24: Opcode
  // 23-18: Resulting type plane
  // 17-12: Operand #1
  // 11- 6: Operand #2
  //  5- 0: Operand #3
  //
  unsigned Opcode = (3 << 30) | (IType << 24) | (Type << 18) |
                    (Slots[0] << 12) | (Slots[1] << 6) | (Slots[2] << 0);
  //  cerr << "3 " << IType << " " << Type << " " << Slots[0] << " " 
  //       << Slots[1] << " " << Slots[2] << endl;
  output(Opcode, Out);
}

bool BytecodeWriter::processInstruction(const Instruction *I) {
  assert(I->getInstType() < 64 && "Opcode too big???");

  unsigned NumOperands = 0;
  int MaxOpSlot = 0;
  int Slots[3]; Slots[0] = (1 << 12)-1;

  const Value *Def;
  while ((Def = I->getOperand(NumOperands))) {
    int slot = Table.getValSlot(Def);
    assert(slot != -1 && "Broken bytecode!");
    if (slot > MaxOpSlot) MaxOpSlot = slot;
    if (NumOperands < 3) Slots[NumOperands] = slot;
    NumOperands++;
  }

  // Figure out which type to encode with the instruction.  Typically we want
  // the type of the first parameter, as opposed to the type of the instruction
  // (for example, with setcc, we always know it returns bool, but the type of
  // the first param is actually interesting).  But if we have no arguments
  // we take the type of the instruction itself.  
  //

  const Type *Ty;
  if (NumOperands)
    Ty = I->getOperand(0)->getType();
  else
    Ty = I->getType();

  unsigned Type;
  int Slot = Table.getValSlot(Ty);
  assert(Slot != -1 && "Type not available!!?!");
  Type = (unsigned)Slot;


  // Decide which instruction encoding to use.  This is determined primarily by
  // the number of operands, and secondarily by whether or not the max operand
  // will fit into the instruction encoding.  More operands == fewer bits per
  // operand.
  //
  switch (NumOperands) {
  case 0:
  case 1:
    if (MaxOpSlot < (1 << 12)-1) { // -1 because we use 4095 to indicate 0 ops
      outputInstructionFormat1(I, Table, Slots, Type, Out);
      return false;
    }
    break;

  case 2:
    if (MaxOpSlot < (1 << 8)) {
      outputInstructionFormat2(I, Table, Slots, Type, Out);
      return false;
    }
    break;

  case 3:
    if (MaxOpSlot < (1 << 6)) {
      outputInstructionFormat3(I, Table, Slots, Type, Out);
      return false;
    }
    break;
  }

  outputInstructionFormat0(I, Table, Type, Out);
  return false;
}
