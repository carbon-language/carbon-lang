//===-- InstructionWriter.cpp - Functions for writing instructions --------===//
//
// This file implements the routines for encoding instruction opcodes to a 
// bytecode stream.
//
//===----------------------------------------------------------------------===//

#include "WriterInternals.h"
#include "llvm/Module.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "Support/Statistic.h"
#include <algorithm>

static Statistic<> 
NumInstrs("bytecodewriter", "Number of instructions");

typedef unsigned char uchar;

// outputInstructionFormat0 - Output those wierd instructions that have a large
// number of operands or have large operands themselves...
//
// Format: [opcode] [type] [numargs] [arg0] [arg1] ... [arg<numargs-1>]
//
static void outputInstructionFormat0(const Instruction *I, unsigned Opcode,
				     const SlotCalculator &Table,
				     unsigned Type, std::deque<uchar> &Out) {
  // Opcode must have top two bits clear...
  output_vbr(Opcode << 2, Out);                  // Instruction Opcode ID
  output_vbr(Type, Out);                         // Result type

  unsigned NumArgs = I->getNumOperands();
  output_vbr(NumArgs + (isa<CastInst>(I) || isa<VarArgInst>(I)), Out);

  for (unsigned i = 0; i < NumArgs; ++i) {
    int Slot = Table.getValSlot(I->getOperand(i));
    assert(Slot >= 0 && "No slot number for value!?!?");      
    output_vbr((unsigned)Slot, Out);
  }

  if (isa<CastInst>(I) || isa<VarArgInst>(I)) {
    int Slot = Table.getValSlot(I->getType());
    assert(Slot != -1 && "Cast/VarArg return type unknown?");
    output_vbr((unsigned)Slot, Out);
  }

  align32(Out);    // We must maintain correct alignment!
}


// outputInstrVarArgsCall - Output the absurdly annoying varargs function calls.
// This are more annoying than most because the signature of the call does not
// tell us anything about the types of the arguments in the varargs portion.
// Because of this, we encode (as type 0) all of the argument types explicitly
// before the argument value.  This really sucks, but you shouldn't be using
// varargs functions in your code! *death to printf*!
//
// Format: [opcode] [type] [numargs] [arg0] [arg1] ... [arg<numargs-1>]
//
static void outputInstrVarArgsCall(const Instruction *I, unsigned Opcode,
				   const SlotCalculator &Table, unsigned Type,
				   std::deque<uchar> &Out) {
  assert(isa<CallInst>(I) || isa<InvokeInst>(I));
  // Opcode must have top two bits clear...
  output_vbr(Opcode << 2, Out);                  // Instruction Opcode ID
  output_vbr(Type, Out);                         // Result type (varargs type)

  unsigned NumArgs = I->getNumOperands();
  output_vbr(NumArgs*2, Out);
  // TODO: Don't need to emit types for the fixed types of the varargs function
  // prototype...

  // The type for the function has already been emitted in the type field of the
  // instruction.  Just emit the slot # now.
  int Slot = Table.getValSlot(I->getOperand(0));
  assert(Slot >= 0 && "No slot number for value!?!?");      
  output_vbr((unsigned)Slot, Out);

  // Output a dummy field to fill Arg#2 in the reader that is currently unused
  // for varargs calls.  This is a gross hack to make the code simpler, but we
  // aren't really doing very small bytecode for varargs calls anyways.
  // FIXME in the future: Smaller bytecode for varargs calls
  output_vbr(0, Out);

  for (unsigned i = 1; i < NumArgs; ++i) {
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
static void outputInstructionFormat1(const Instruction *I, unsigned Opcode,
				     const SlotCalculator &Table, int *Slots,
				     unsigned Type, std::deque<uchar> &Out) {
  // bits   Instruction format:
  // --------------------------
  // 01-00: Opcode type, fixed to 1.
  // 07-02: Opcode
  // 19-08: Resulting type plane
  // 31-20: Operand #1 (if set to (2^12-1), then zero operands)
  //
  unsigned Bits = 1 | (Opcode << 2) | (Type << 8) | (Slots[0] << 20);
  //  cerr << "1 " << IType << " " << Type << " " << Slots[0] << endl;
  output(Bits, Out);
}


// outputInstructionFormat2 - Output two operand instructions, knowing that no
// operand index is >= 2^8.
//
static void outputInstructionFormat2(const Instruction *I, unsigned Opcode,
				     const SlotCalculator &Table, int *Slots,
				     unsigned Type, std::deque<uchar> &Out) {
  // bits   Instruction format:
  // --------------------------
  // 01-00: Opcode type, fixed to 2.
  // 07-02: Opcode
  // 15-08: Resulting type plane
  // 23-16: Operand #1
  // 31-24: Operand #2  
  //
  unsigned Bits = 2 | (Opcode << 2) | (Type << 8) |
                    (Slots[0] << 16) | (Slots[1] << 24);
  //  cerr << "2 " << IType << " " << Type << " " << Slots[0] << " " 
  //       << Slots[1] << endl;
  output(Bits, Out);
}


// outputInstructionFormat3 - Output three operand instructions, knowing that no
// operand index is >= 2^6.
//
static void outputInstructionFormat3(const Instruction *I, unsigned Opcode,
				     const SlotCalculator &Table, int *Slots,
				     unsigned Type, std::deque<uchar> &Out) {
  // bits   Instruction format:
  // --------------------------
  // 01-00: Opcode type, fixed to 3.
  // 07-02: Opcode
  // 13-08: Resulting type plane
  // 19-14: Operand #1
  // 25-20: Operand #2
  // 31-26: Operand #3
  //
  unsigned Bits = 3 | (Opcode << 2) | (Type << 8) |
          (Slots[0] << 14) | (Slots[1] << 20) | (Slots[2] << 26);
  //cerr << "3 " << IType << " " << Type << " " << Slots[0] << " " 
  //     << Slots[1] << " " << Slots[2] << endl;
  output(Bits, Out);
}

void BytecodeWriter::processInstruction(const Instruction &I) {
  assert(I.getOpcode() < 62 && "Opcode too big???");
  unsigned Opcode = I.getOpcode();

  // Encode 'volatile load' as 62 and 'volatile store' as 63.
  if (isa<LoadInst>(I) && cast<LoadInst>(I).isVolatile())
    Opcode = 62;
  if (isa<StoreInst>(I) && cast<StoreInst>(I).isVolatile())
    Opcode = 63;

  unsigned NumOperands = I.getNumOperands();
  int MaxOpSlot = 0;
  int Slots[3]; Slots[0] = (1 << 12)-1;   // Marker to signify 0 operands

  for (unsigned i = 0; i < NumOperands; ++i) {
    const Value *Def = I.getOperand(i);
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
  switch (I.getOpcode()) {
  case Instruction::Malloc:
  case Instruction::Alloca:
    Ty = I.getType();  // Malloc & Alloca ALWAYS want to encode the return type
    break;
  case Instruction::Store:
    Ty = I.getOperand(1)->getType();  // Encode the pointer type...
    assert(isa<PointerType>(Ty) && "Store to nonpointer type!?!?");
    break;
  default:              // Otherwise use the default behavior...
    Ty = NumOperands ? I.getOperand(0)->getType() : I.getType();
    break;
  }

  unsigned Type;
  int Slot = Table.getValSlot(Ty);
  assert(Slot != -1 && "Type not available!!?!");
  Type = (unsigned)Slot;

  // Make sure that we take the type number into consideration.  We don't want
  // to overflow the field size for the instruction format we select.
  //
  if (Slot > MaxOpSlot) MaxOpSlot = Slot;

  // Handle the special case for cast...
  if (isa<CastInst>(I) || isa<VarArgInst>(I)) {
    // Cast has to encode the destination type as the second argument in the
    // packet, or else we won't know what type to cast to!
    Slots[1] = Table.getValSlot(I.getType());
    assert(Slots[1] != -1 && "Cast return type unknown?");
    if (Slots[1] > MaxOpSlot) MaxOpSlot = Slots[1];
    NumOperands++;
  } else if (const CallInst *CI = dyn_cast<CallInst>(&I)){// Handle VarArg calls
    const PointerType *Ty = cast<PointerType>(CI->getCalledValue()->getType());
    if (cast<FunctionType>(Ty->getElementType())->isVarArg()) {
      outputInstrVarArgsCall(CI, Opcode, Table, Type, Out);
      return;
    }
  } else if (const InvokeInst *II = dyn_cast<InvokeInst>(&I)) {// ...  & Invokes
    const PointerType *Ty = cast<PointerType>(II->getCalledValue()->getType());
    if (cast<FunctionType>(Ty->getElementType())->isVarArg()) {
      outputInstrVarArgsCall(II, Opcode, Table, Type, Out);
      return;
    }
  }

  ++NumInstrs;

  // Decide which instruction encoding to use.  This is determined primarily by
  // the number of operands, and secondarily by whether or not the max operand
  // will fit into the instruction encoding.  More operands == fewer bits per
  // operand.
  //
  switch (NumOperands) {
  case 0:
  case 1:
    if (MaxOpSlot < (1 << 12)-1) { // -1 because we use 4095 to indicate 0 ops
      outputInstructionFormat1(&I, Opcode, Table, Slots, Type, Out);
      return;
    }
    break;

  case 2:
    if (MaxOpSlot < (1 << 8)) {
      outputInstructionFormat2(&I, Opcode, Table, Slots, Type, Out);
      return;
    }
    break;

  case 3:
    if (MaxOpSlot < (1 << 6)) {
      outputInstructionFormat3(&I, Opcode, Table, Slots, Type, Out);
      return;
    }
    break;
  }

  // If we weren't handled before here, we either have a large number of
  // operands or a large operand index that we are referring to.
  outputInstructionFormat0(&I, Opcode, Table, Type, Out);
}
