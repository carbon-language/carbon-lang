//===- ReadInst.cpp - Code to read an instruction from bytecode -----------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the mechanism to read an instruction from a bytecode 
// stream.
//
// Note that this library should be as fast as possible, reentrant, and 
// threadsafe!!
//
//===----------------------------------------------------------------------===//

#include "ReaderInternals.h"
#include "llvm/iTerminators.h"
#include "llvm/iMemory.h"
#include "llvm/iPHINode.h"
#include "llvm/iOther.h"
#include "llvm/Module.h"
using namespace llvm;

namespace {
  struct RawInst {       // The raw fields out of the bytecode stream...
    unsigned NumOperands;
    unsigned Opcode;
    unsigned Type;
    
    RawInst(const unsigned char *&Buf, const unsigned char *EndBuf,
            std::vector<unsigned> &Args);
  };
}

RawInst::RawInst(const unsigned char *&Buf, const unsigned char *EndBuf,
                 std::vector<unsigned> &Args) {
  unsigned Op = read(Buf, EndBuf);

  // bits   Instruction format:        Common to all formats
  // --------------------------
  // 01-00: Opcode type, fixed to 1.
  // 07-02: Opcode
  Opcode    = (Op >> 2) & 63;
  Args.resize((Op >> 0) & 03);

  switch (Args.size()) {
  case 1:
    // bits   Instruction format:
    // --------------------------
    // 19-08: Resulting type plane
    // 31-20: Operand #1 (if set to (2^12-1), then zero operands)
    //
    Type    = (Op >>  8) & 4095;
    Args[0] = (Op >> 20) & 4095;
    if (Args[0] == 4095)    // Handle special encoding for 0 operands...
      Args.resize(0);
    break;
  case 2:
    // bits   Instruction format:
    // --------------------------
    // 15-08: Resulting type plane
    // 23-16: Operand #1
    // 31-24: Operand #2  
    //
    Type    = (Op >>  8) & 255;
    Args[0] = (Op >> 16) & 255;
    Args[1] = (Op >> 24) & 255;
    break;
  case 3:
    // bits   Instruction format:
    // --------------------------
    // 13-08: Resulting type plane
    // 19-14: Operand #1
    // 25-20: Operand #2
    // 31-26: Operand #3
    //
    Type    = (Op >>  8) & 63;
    Args[0] = (Op >> 14) & 63;
    Args[1] = (Op >> 20) & 63;
    Args[2] = (Op >> 26) & 63;
    break;
  case 0:
    Buf -= 4;  // Hrm, try this again...
    Opcode = read_vbr_uint(Buf, EndBuf);
    Opcode >>= 2;
    Type = read_vbr_uint(Buf, EndBuf);

    unsigned NumOperands = read_vbr_uint(Buf, EndBuf);
    Args.resize(NumOperands);

    if (NumOperands == 0)
      throw std::string("Zero-argument instruction found; this is invalid.");

    for (unsigned i = 0; i != NumOperands; ++i)
      Args[i] = read_vbr_uint(Buf, EndBuf);
    align32(Buf, EndBuf);
    break;
  }
}


void BytecodeParser::ParseInstruction(const unsigned char *&Buf,
                                      const unsigned char *EndBuf,
                                      std::vector<unsigned> &Args,
                                      BasicBlock *BB) {
  Args.clear();
  RawInst RI(Buf, EndBuf, Args);
  const Type *InstTy = getType(RI.Type);

  Instruction *Result = 0;
  if (RI.Opcode >= Instruction::BinaryOpsBegin &&
      RI.Opcode <  Instruction::BinaryOpsEnd  && Args.size() == 2)
    Result = BinaryOperator::create((Instruction::BinaryOps)RI.Opcode,
                                    getValue(RI.Type, Args[0]),
                                    getValue(RI.Type, Args[1]));

  switch (RI.Opcode) {
  default: 
    if (Result == 0) throw std::string("Illegal instruction read!");
    break;
  case Instruction::VAArg:
    Result = new VAArgInst(getValue(RI.Type, Args[0]), getType(Args[1]));
    break;
  case Instruction::VANext:
    Result = new VANextInst(getValue(RI.Type, Args[0]), getType(Args[1]));
    break;
  case Instruction::Cast:
    Result = new CastInst(getValue(RI.Type, Args[0]), getType(Args[1]));
    break;
  case Instruction::Select:
    Result = new SelectInst(getValue(Type::BoolTyID, Args[0]),
                            getValue(RI.Type, Args[1]),
                            getValue(RI.Type, Args[2]));
    break;
  case Instruction::PHI: {
    if (Args.size() == 0 || (Args.size() & 1))
      throw std::string("Invalid phi node encountered!\n");

    PHINode *PN = new PHINode(InstTy);
    PN->op_reserve(Args.size());
    for (unsigned i = 0, e = Args.size(); i != e; i += 2)
      PN->addIncoming(getValue(RI.Type, Args[i]), getBasicBlock(Args[i+1]));
    Result = PN;
    break;
  }

  case Instruction::Shl:
  case Instruction::Shr:
    Result = new ShiftInst((Instruction::OtherOps)RI.Opcode,
                           getValue(RI.Type, Args[0]),
                           getValue(Type::UByteTyID, Args[1]));
    break;
  case Instruction::Ret:
    if (Args.size() == 0)
      Result = new ReturnInst();
    else if (Args.size() == 1)
      Result = new ReturnInst(getValue(RI.Type, Args[0]));
    else
      throw std::string("Unrecognized instruction!");
    break;

  case Instruction::Br:
    if (Args.size() == 1)
      Result = new BranchInst(getBasicBlock(Args[0]));
    else if (Args.size() == 3)
      Result = new BranchInst(getBasicBlock(Args[0]), getBasicBlock(Args[1]),
                              getValue(Type::BoolTyID , Args[2]));
    else
      throw std::string("Invalid number of operands for a 'br' instruction!");
    break;
  case Instruction::Switch: {
    if (Args.size() & 1)
      throw std::string("Switch statement with odd number of arguments!");

    SwitchInst *I = new SwitchInst(getValue(RI.Type, Args[0]),
                                   getBasicBlock(Args[1]));
    for (unsigned i = 2, e = Args.size(); i != e; i += 2)
      I->addCase(cast<Constant>(getValue(RI.Type, Args[i])),
                 getBasicBlock(Args[i+1]));
    Result = I;
    break;
  }

  case Instruction::Call: {
    if (Args.size() == 0)
      throw std::string("Invalid call instruction encountered!");

    Value *F = getValue(RI.Type, Args[0]);

    // Check to make sure we have a pointer to function type
    const PointerType *PTy = dyn_cast<PointerType>(F->getType());
    if (PTy == 0) throw std::string("Call to non function pointer value!");
    const FunctionType *FTy = dyn_cast<FunctionType>(PTy->getElementType());
    if (FTy == 0) throw std::string("Call to non function pointer value!");

    std::vector<Value *> Params;
    if (!FTy->isVarArg()) {
      FunctionType::param_iterator It = FTy->param_begin();

      for (unsigned i = 1, e = Args.size(); i != e; ++i) {
        if (It == FTy->param_end())
          throw std::string("Invalid call instruction!");
        Params.push_back(getValue(getTypeSlot(*It++), Args[i]));
      }
      if (It != FTy->param_end())
        throw std::string("Invalid call instruction!");
    } else {
      Args.erase(Args.begin(), Args.begin()+1);

      unsigned FirstVariableOperand;
      if (Args.size() < FTy->getNumParams())
        throw std::string("Call instruction missing operands!");

      // Read all of the fixed arguments
      for (unsigned i = 0, e = FTy->getNumParams(); i != e; ++i)
        Params.push_back(getValue(getTypeSlot(FTy->getParamType(i)),Args[i]));
      
      FirstVariableOperand = FTy->getNumParams();

      if ((Args.size()-FirstVariableOperand) & 1) // Must be pairs of type/value
        throw std::string("Invalid call instruction!");
        
      for (unsigned i = FirstVariableOperand, e = Args.size(); i != e; i += 2)
        Params.push_back(getValue(Args[i], Args[i+1]));
    }

    Result = new CallInst(F, Params);
    break;
  }
  case Instruction::Invoke: {
    if (Args.size() < 3) throw std::string("Invalid invoke instruction!");
    Value *F = getValue(RI.Type, Args[0]);

    // Check to make sure we have a pointer to function type
    const PointerType *PTy = dyn_cast<PointerType>(F->getType());
    if (PTy == 0) throw std::string("Invoke to non function pointer value!");
    const FunctionType *FTy = dyn_cast<FunctionType>(PTy->getElementType());
    if (FTy == 0) throw std::string("Invoke to non function pointer value!");

    std::vector<Value *> Params;
    BasicBlock *Normal, *Except;

    if (!FTy->isVarArg()) {
      Normal = getBasicBlock(Args[1]);
      Except = getBasicBlock(Args[2]);

      FunctionType::param_iterator It = FTy->param_begin();
      for (unsigned i = 3, e = Args.size(); i != e; ++i) {
        if (It == FTy->param_end())
          throw std::string("Invalid invoke instruction!");
        Params.push_back(getValue(getTypeSlot(*It++), Args[i]));
      }
      if (It != FTy->param_end())
        throw std::string("Invalid invoke instruction!");
    } else {
      Args.erase(Args.begin(), Args.begin()+1);

      Normal = getBasicBlock(Args[0]);
      Except = getBasicBlock(Args[1]);
      
      unsigned FirstVariableArgument = FTy->getNumParams()+2;
      for (unsigned i = 2; i != FirstVariableArgument; ++i)
        Params.push_back(getValue(getTypeSlot(FTy->getParamType(i-2)),
                                  Args[i]));
      
      if (Args.size()-FirstVariableArgument & 1)  // Must be pairs of type/value
        throw std::string("Invalid invoke instruction!");

      for (unsigned i = FirstVariableArgument; i < Args.size(); i += 2)
        Params.push_back(getValue(Args[i], Args[i+1]));
    }

    Result = new InvokeInst(F, Normal, Except, Params);
    break;
  }
  case Instruction::Malloc:
    if (Args.size() > 2) throw std::string("Invalid malloc instruction!");
    if (!isa<PointerType>(InstTy))
      throw std::string("Invalid malloc instruction!");

    Result = new MallocInst(cast<PointerType>(InstTy)->getElementType(),
                            Args.size() ? getValue(Type::UIntTyID,
                                                   Args[0]) : 0);
    break;

  case Instruction::Alloca:
    if (Args.size() > 2) throw std::string("Invalid alloca instruction!");
    if (!isa<PointerType>(InstTy))
      throw std::string("Invalid alloca instruction!");

    Result = new AllocaInst(cast<PointerType>(InstTy)->getElementType(),
                            Args.size() ? getValue(Type::UIntTyID, Args[0]) :0);
    break;
  case Instruction::Free:
    if (!isa<PointerType>(InstTy))
      throw std::string("Invalid free instruction!");
    Result = new FreeInst(getValue(RI.Type, Args[0]));
    break;
  case Instruction::GetElementPtr: {
    if (Args.size() == 0 || !isa<PointerType>(InstTy))
      throw std::string("Invalid getelementptr instruction!");

    std::vector<Value*> Idx;

    const Type *NextTy = InstTy;
    for (unsigned i = 1, e = Args.size(); i != e; ++i) {
      const CompositeType *TopTy = dyn_cast_or_null<CompositeType>(NextTy);
      if (!TopTy) throw std::string("Invalid getelementptr instruction!"); 

      unsigned ValIdx = Args[i];
      unsigned IdxTy;
      if (!hasRestrictedGEPTypes) {
        // Struct indices are always uints, sequential type indices can be any
        // of the 32 or 64-bit integer types.  The actual choice of type is
        // encoded in the low two bits of the slot number.
        if (isa<StructType>(TopTy))
          IdxTy = Type::UIntTyID;
        else {
          switch (ValIdx & 3) {
          default:
          case 0: IdxTy = Type::UIntTyID; break;
          case 1: IdxTy = Type::IntTyID; break;
          case 2: IdxTy = Type::ULongTyID; break;
          case 3: IdxTy = Type::LongTyID; break;
          }
          ValIdx >>= 2;
        }
      } else {
        IdxTy = isa<StructType>(TopTy) ? Type::UByteTyID : Type::LongTyID;
      }

      Idx.push_back(getValue(IdxTy, ValIdx));

      // Convert ubyte struct indices into uint struct indices.
      if (isa<StructType>(TopTy) && hasRestrictedGEPTypes)
        if (ConstantUInt *C = dyn_cast<ConstantUInt>(Idx.back()))
          Idx[Idx.size()-1] = ConstantExpr::getCast(C, Type::UIntTy);

      NextTy = GetElementPtrInst::getIndexedType(InstTy, Idx, true);
    }

    Result = new GetElementPtrInst(getValue(RI.Type, Args[0]), Idx);
    break;
  }

  case 62:   // volatile load
  case Instruction::Load:
    if (Args.size() != 1 || !isa<PointerType>(InstTy))
      throw std::string("Invalid load instruction!");
    Result = new LoadInst(getValue(RI.Type, Args[0]), "", RI.Opcode == 62);
    break;

  case 63:   // volatile store 
  case Instruction::Store: {
    if (!isa<PointerType>(InstTy) || Args.size() != 2)
      throw std::string("Invalid store instruction!");

    Value *Ptr = getValue(RI.Type, Args[1]);
    const Type *ValTy = cast<PointerType>(Ptr->getType())->getElementType();
    Result = new StoreInst(getValue(getTypeSlot(ValTy), Args[0]), Ptr,
                           RI.Opcode == 63);
    break;
  }
  case Instruction::Unwind:
    if (Args.size() != 0) throw std::string("Invalid unwind instruction!");
    Result = new UnwindInst();
    break;
  }  // end switch(RI.Opcode) 

  unsigned TypeSlot;
  if (Result->getType() == InstTy)
    TypeSlot = RI.Type;
  else
    TypeSlot = getTypeSlot(Result->getType());

  insertValue(Result, TypeSlot, Values);
  BB->getInstList().push_back(Result);
  BCR_TRACE(4, *Result);
}
