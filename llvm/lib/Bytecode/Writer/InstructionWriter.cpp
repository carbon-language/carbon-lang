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
  output_vbr(I->getOpcode(), Out);               // Instruction Opcode ID
  output_vbr(Type, Out);                         // Result type

  unsigned NumArgs = I->getNumOperands();
  output_vbr(NumArgs, Out);

  for (unsigned i = 0; i < NumArgs; ++i) {
    int Slot = Table.getValSlot(I->getOperand(i));
    assert(Slot >= 0 && "No slot number for value!?!?");      
    output_vbr((unsigned)Slot, Out);
  }
  align32(Out);    // We must maintain correct alignment!
}


// outputInstrVarArgsCall - Output the obsurdly annoying varargs method calls.
// This are more annoying than most because the signature of the call does not
// tell us anything about the types of the arguments in the varargs portion.
// Because of this, we encode (as type 0) all of the argument types explicitly
// before the argument value.  This really sucks, but you shouldn't be using
// varargs functions in your code! *death to printf*!
//
// Format: [opcode] [type] [numargs] [arg0] [arg1] ... [arg<numargs-1>]
//
static void outputInstrVarArgsCall(const Instruction *I,
				   const SlotCalculator &Table, unsigned Type,
				   vector<uchar> &Out) {
  assert(I->getOpcode() == Instruction::Call /*|| 
	 I->getOpcode() == Instruction::ICall */);
  // Opcode must have top two bits clear...
  output_vbr(I->getOpcode(), Out);               // Instruction Opcode ID
  output_vbr(Type, Out);                         // Result type (varargs type)

  unsigned NumArgs = I->getNumOperands();
  output_vbr((NumArgs-2)*2+2, Out); // Don't duplicate method & Arg1 types

  // Output the method type without an extra type argument.
  int Slot = Table.getValSlot(I->getOperand(0));
  assert(Slot >= 0 && "No slot number for value!?!?");      
  output_vbr((unsigned)Slot, Out);

  // VarArgs methods must have at least one specified operand
  Slot = Table.getValSlot(I->getOperand(1));
  assert(Slot >= 0 && "No slot number for value!?!?");      
  output_vbr((unsigned)Slot, Out);

  for (unsigned i = 2; i < NumArgs; ++i) {
    // Output Arg Type ID
    Slot = Table.getValSlot(I->getOperand(i)->getType());
    assert(Slot >= 0 && "No slot number for value!?!?");      
    output_vbr((unsigned)Slot, Out);

    // Output arg ID itself
    Slot = Table.getValSlot(I->getOperand(i));
    assert(Slot >= 0 && "No slot number for value!?!?");      
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
  unsigned IType = I->getOpcode();      // Instruction Opcode ID
  
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
  unsigned IType = I->getOpcode();      // Instruction Opcode ID

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
  unsigned IType = I->getOpcode();      // Instruction Opcode ID

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
  //cerr << "3 " << IType << " " << Type << " " << Slots[0] << " " 
  //     << Slots[1] << " " << Slots[2] << endl;
  output(Opcode, Out);
}

bool BytecodeWriter::processInstruction(const Instruction *I) {
  assert(I->getOpcode() < 64 && "Opcode too big???");

  unsigned NumOperands = I->getNumOperands();
  int MaxOpSlot = 0;
  int Slots[3]; Slots[0] = (1 << 12)-1;   // Marker to signify 0 operands

  for (unsigned i = 0; i < NumOperands; ++i) {
    const Value *Def = I->getOperand(i);
    int slot = Table.getValSlot(Def);
    assert(slot != -1 && "Broken bytecode!");
    if (slot > MaxOpSlot) MaxOpSlot = slot;
    if (i < 3) Slots[i] = slot;
  }

  // Figure out which type to encode with the instruction.  Typically we want
  // the type of the first parameter, as opposed to the type of the instruction
  // (for example, with setcc, we always know it returns bool, but the type of
  // the first param is actually interesting).  But if we have no arguments
  // we take the type of the instruction itself.  
  //
  const Type *Ty;
  switch (I->getOpcode()) {
  case Instruction::Malloc:
  case Instruction::Alloca:
    Ty = I->getType();  // Malloc & Alloca ALWAYS want to encode the return type
    break;
  case Instruction::Store:
    Ty = I->getOperand(1)->getType();  // Encode the pointer type...
    break;
  default:              // Otherwise use the default behavior...
    Ty = NumOperands ? I->getOperand(0)->getType() : I->getType();
    break;
  }

  unsigned Type;
  int Slot = Table.getValSlot(Ty);
  assert(Slot != -1 && "Type not available!!?!");
  Type = (unsigned)Slot;

  // Handle the special case for cast...
  if (I->getOpcode() == Instruction::Cast) {
    // Cast has to encode the destination type as the second argument in the
    // packet, or else we won't know what type to cast to!
    Slots[1] = Table.getValSlot(I->getType());
    assert(Slots[1] != -1 && "Cast return type unknown?");
    if (Slots[1] > MaxOpSlot) MaxOpSlot = Slots[1];
    NumOperands++;
  } else if (I->getOpcode() == Instruction::Call &&  // Handle VarArg calls
	     I->getOperand(0)->getType()->isMethodType()->isVarArg()) {
    outputInstrVarArgsCall(I, Table, Type, Out);
    return false;
  }

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

  // If we weren't handled before here, we either have a large number of
  // operands or a large operand index that we are refering to.
  outputInstructionFormat0(I, Table, Type, Out);
  return false;
}
