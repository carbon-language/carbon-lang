//===-- InstructionWriter.cpp - Functions for writing instructions --------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the routines for encoding instruction opcodes to a 
// bytecode stream.
//
//===----------------------------------------------------------------------===//

#include "WriterInternals.h"
#include "llvm/Module.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "Support/Statistic.h"
#include <algorithm>
using namespace llvm;

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
  output_vbr(NumArgs + (isa<CastInst>(I) || isa<VANextInst>(I) ||
                        isa<VAArgInst>(I)), Out);

  if (!isa<GetElementPtrInst>(&I)) {
    for (unsigned i = 0; i < NumArgs; ++i) {
      int Slot = Table.getSlot(I->getOperand(i));
      assert(Slot >= 0 && "No slot number for value!?!?");      
      output_vbr((unsigned)Slot, Out);
    }

    if (isa<CastInst>(I) || isa<VAArgInst>(I)) {
      int Slot = Table.getSlot(I->getType());
      assert(Slot != -1 && "Cast return type unknown?");
      output_vbr((unsigned)Slot, Out);
    } else if (const VANextInst *VAI = dyn_cast<VANextInst>(I)) {
      int Slot = Table.getSlot(VAI->getArgType());
      assert(Slot != -1 && "VarArg argument type unknown?");
      output_vbr((unsigned)Slot, Out);
    }

  } else {
    int Slot = Table.getSlot(I->getOperand(0));
    assert(Slot >= 0 && "No slot number for value!?!?");      
    output_vbr(unsigned(Slot), Out);

    // We need to encode the type of sequential type indices into their slot #
    unsigned Idx = 1;
    for (gep_type_iterator TI = gep_type_begin(I), E = gep_type_end(I);
         Idx != NumArgs; ++TI, ++Idx) {
      Slot = Table.getSlot(I->getOperand(Idx));
      assert(Slot >= 0 && "No slot number for value!?!?");      
    
      if (isa<SequentialType>(*TI)) {
        unsigned IdxId;
        switch (I->getOperand(Idx)->getType()->getPrimitiveID()) {
        default: assert(0 && "Unknown index type!");
        case Type::UIntTyID:  IdxId = 0; break;
        case Type::IntTyID:   IdxId = 1; break;
        case Type::ULongTyID: IdxId = 2; break;
        case Type::LongTyID:  IdxId = 3; break;
        }
        Slot = (Slot << 2) | IdxId;
      }
      output_vbr(unsigned(Slot), Out);
    }
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

  const PointerType *PTy = cast<PointerType>(I->getOperand(0)->getType());
  const FunctionType *FTy = cast<FunctionType>(PTy->getElementType());
  unsigned NumParams = FTy->getNumParams();

  unsigned NumFixedOperands;
  if (isa<CallInst>(I)) {
    // Output an operand for the callee and each fixed argument, then two for
    // each variable argument.
    NumFixedOperands = 1+NumParams;
  } else {
    assert(isa<InvokeInst>(I) && "Not call or invoke??");
    // Output an operand for the callee and destinations, then two for each
    // variable argument.
    NumFixedOperands = 3+NumParams;
  }
  output_vbr(2 * I->getNumOperands()-NumFixedOperands, Out);

  // The type for the function has already been emitted in the type field of the
  // instruction.  Just emit the slot # now.
  for (unsigned i = 0; i != NumFixedOperands; ++i) {
    int Slot = Table.getSlot(I->getOperand(i));
    assert(Slot >= 0 && "No slot number for value!?!?");      
    output_vbr((unsigned)Slot, Out);
  }

  for (unsigned i = NumFixedOperands, e = I->getNumOperands(); i != e; ++i) {
    // Output Arg Type ID
    int Slot = Table.getSlot(I->getOperand(i)->getType());
    assert(Slot >= 0 && "No slot number for value!?!?");      
    output_vbr((unsigned)Slot, Out);
    
    // Output arg ID itself
    Slot = Table.getSlot(I->getOperand(i));
    assert(Slot >= 0 && "No slot number for value!?!?");      
    output_vbr((unsigned)Slot, Out);
  }
  align32(Out);    // We must maintain correct alignment!
}


// outputInstructionFormat1 - Output one operand instructions, knowing that no
// operand index is >= 2^12.
//
static void outputInstructionFormat1(const Instruction *I, unsigned Opcode,
				     const SlotCalculator &Table,
                                     unsigned *Slots, unsigned Type, 
                                     std::deque<uchar> &Out) {
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
				     const SlotCalculator &Table,
                                     unsigned *Slots, unsigned Type, 
                                     std::deque<uchar> &Out) {
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
				     const SlotCalculator &Table,
                                     unsigned *Slots, unsigned Type,
                                     std::deque<uchar> &Out) {
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

void BytecodeWriter::outputInstruction(const Instruction &I) {
  assert(I.getOpcode() < 62 && "Opcode too big???");
  unsigned Opcode = I.getOpcode();
  unsigned NumOperands = I.getNumOperands();

  // Encode 'volatile load' as 62 and 'volatile store' as 63.
  if (isa<LoadInst>(I) && cast<LoadInst>(I).isVolatile())
    Opcode = 62;
  if (isa<StoreInst>(I) && cast<StoreInst>(I).isVolatile())
    Opcode = 63;

  // Figure out which type to encode with the instruction.  Typically we want
  // the type of the first parameter, as opposed to the type of the instruction
  // (for example, with setcc, we always know it returns bool, but the type of
  // the first param is actually interesting).  But if we have no arguments
  // we take the type of the instruction itself.  
  //
  const Type *Ty;
  switch (I.getOpcode()) {
  case Instruction::Select:
  case Instruction::Malloc:
  case Instruction::Alloca:
    Ty = I.getType();  // These ALWAYS want to encode the return type
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
  int Slot = Table.getSlot(Ty);
  assert(Slot != -1 && "Type not available!!?!");
  Type = (unsigned)Slot;

  // Varargs calls and invokes are encoded entirely different from any other
  // instructions.
  if (const CallInst *CI = dyn_cast<CallInst>(&I)){
    const PointerType *Ty =cast<PointerType>(CI->getCalledValue()->getType());
    if (cast<FunctionType>(Ty->getElementType())->isVarArg()) {
      outputInstrVarArgsCall(CI, Opcode, Table, Type, Out);
      return;
    }
  } else if (const InvokeInst *II = dyn_cast<InvokeInst>(&I)) {
    const PointerType *Ty =cast<PointerType>(II->getCalledValue()->getType());
    if (cast<FunctionType>(Ty->getElementType())->isVarArg()) {
      outputInstrVarArgsCall(II, Opcode, Table, Type, Out);
      return;
    }
  }

  if (NumOperands <= 3) {
    // Make sure that we take the type number into consideration.  We don't want
    // to overflow the field size for the instruction format we select.
    //
    unsigned MaxOpSlot = Type;
    unsigned Slots[3]; Slots[0] = (1 << 12)-1;   // Marker to signify 0 operands
    
    for (unsigned i = 0; i != NumOperands; ++i) {
      int slot = Table.getSlot(I.getOperand(i));
      assert(slot != -1 && "Broken bytecode!");
      if (unsigned(slot) > MaxOpSlot) MaxOpSlot = unsigned(slot);
      Slots[i] = unsigned(slot);
    }

    // Handle the special cases for various instructions...
    if (isa<CastInst>(I) || isa<VAArgInst>(I)) {
      // Cast has to encode the destination type as the second argument in the
      // packet, or else we won't know what type to cast to!
      Slots[1] = Table.getSlot(I.getType());
      assert(Slots[1] != ~0U && "Cast return type unknown?");
      if (Slots[1] > MaxOpSlot) MaxOpSlot = Slots[1];
      NumOperands++;
    } else if (const VANextInst *VANI = dyn_cast<VANextInst>(&I)) {
      Slots[1] = Table.getSlot(VANI->getArgType());
      assert(Slots[1] != ~0U && "va_next return type unknown?");
      if (Slots[1] > MaxOpSlot) MaxOpSlot = Slots[1];
      NumOperands++;
    } else if (const GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(&I)) {
      // We need to encode the type of sequential type indices into their slot #
      unsigned Idx = 1;
      for (gep_type_iterator I = gep_type_begin(GEP), E = gep_type_end(GEP);
           I != E; ++I, ++Idx)
        if (isa<SequentialType>(*I)) {
          unsigned IdxId;
          switch (GEP->getOperand(Idx)->getType()->getPrimitiveID()) {
          default: assert(0 && "Unknown index type!");
          case Type::UIntTyID:  IdxId = 0; break;
          case Type::IntTyID:   IdxId = 1; break;
          case Type::ULongTyID: IdxId = 2; break;
          case Type::LongTyID:  IdxId = 3; break;
          }
          Slots[Idx] = (Slots[Idx] << 2) | IdxId;
          if (Slots[Idx] > MaxOpSlot) MaxOpSlot = Slots[Idx];
        }
    }

    // Decide which instruction encoding to use.  This is determined primarily
    // by the number of operands, and secondarily by whether or not the max
    // operand will fit into the instruction encoding.  More operands == fewer
    // bits per operand.
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
    default:
      break;
    }
  }

  // If we weren't handled before here, we either have a large number of
  // operands or a large operand index that we are referring to.
  outputInstructionFormat0(&I, Opcode, Table, Type, Out);
}
