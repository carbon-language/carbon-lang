//===- ReadInst.cpp - Code to read an instruction from bytecode -----------===//
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
  unsigned Op, Typ;
  if (read(Buf, EndBuf, Op)) 
    throw std::string("Error reading from buffer.");

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
    if (read_vbr(Buf, EndBuf, Opcode))
      throw std::string("Error reading from buffer.");
    Opcode >>= 2;
    if (read_vbr(Buf, EndBuf, Type))
      throw std::string("Error reading from buffer.");

    unsigned NumOperands;
    if (read_vbr(Buf, EndBuf, NumOperands))
      throw std::string("Error reading from buffer.");
    Args.resize(NumOperands);

    if (NumOperands == 0)
      throw std::string("Zero-argument instruction found; this is invalid.");

    for (unsigned i = 0; i != NumOperands; ++i)
      if (read_vbr(Buf, EndBuf, Args[i])) 
        throw std::string("Error reading from buffer");
    if (align32(Buf, EndBuf))
      throw std::string("Unaligned bytecode buffer.");
    break;
  }
}


Instruction *BytecodeParser::ParseInstruction(const unsigned char *&Buf,
                                              const unsigned char *EndBuf,
                                              std::vector<unsigned> &Args) {
  Args.clear();
  RawInst RI(Buf, EndBuf, Args);
  const Type *InstTy = getType(RI.Type);

  if (RI.Opcode >= Instruction::BinaryOpsBegin &&
      RI.Opcode <  Instruction::BinaryOpsEnd  && Args.size() == 2)
    return BinaryOperator::create((Instruction::BinaryOps)RI.Opcode,
                                  getValue(RI.Type, Args[0]),
                                  getValue(RI.Type, Args[1]));

  switch (RI.Opcode) {
  case Instruction::VarArg:
    return new VarArgInst(getValue(RI.Type, Args[0]), getType(Args[1]));
  case Instruction::Cast:
    return new CastInst(getValue(RI.Type, Args[0]), getType(Args[1]));
  case Instruction::PHINode: {
    if (Args.size() == 0 || (Args.size() & 1))
      throw std::string("Invalid phi node encountered!\n");

    PHINode *PN = new PHINode(InstTy);
    for (unsigned i = 0, e = Args.size(); i != e; i += 2)
      PN->addIncoming(getValue(RI.Type, Args[i]), getBasicBlock(Args[i+1]));
    return PN;
  }

  case Instruction::Shl:
  case Instruction::Shr:
    return new ShiftInst((Instruction::OtherOps)RI.Opcode,
                         getValue(RI.Type, Args[0]),
                         getValue(Type::UByteTyID, Args[1]));
  case Instruction::Ret:
    if (Args.size() == 0)
      return new ReturnInst();
    else if (Args.size() == 1)
      return new ReturnInst(getValue(RI.Type, Args[0]));
    break;

  case Instruction::Br:
    if (Args.size() == 1)
      return new BranchInst(getBasicBlock(Args[0]));
    else if (Args.size() == 3)
      return new BranchInst(getBasicBlock(Args[0]), getBasicBlock(Args[1]),
                            getValue(Type::BoolTyID , Args[2]));
    throw std::string("Invalid number of operands for a 'br' instruction!");
    
  case Instruction::Switch: {
    if (Args.size() & 1)
      throw std::string("Switch statement with odd number of arguments!");

    SwitchInst *I = new SwitchInst(getValue(RI.Type, Args[0]),
                                   getBasicBlock(Args[1]));
    for (unsigned i = 2, e = Args.size(); i != e; i += 2)
      I->addCase(cast<Constant>(getValue(RI.Type, Args[i])),
                 getBasicBlock(Args[i+1]));
    return I;
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
    const FunctionType::ParamTypes &PL = FTy->getParamTypes();

    if (!FTy->isVarArg()) {
      FunctionType::ParamTypes::const_iterator It = PL.begin();

      for (unsigned i = 1, e = Args.size(); i != e; ++i) {
        if (It == PL.end()) throw std::string("Invalid call instruction!");
        Params.push_back(getValue(*It++, Args[i]));
      }
      if (It != PL.end()) throw std::string("Invalid call instruction!");
    } else {
      // FIXME: Args[1] is currently just a dummy padding field!

      if (Args.size() & 1)  // Must be pairs of type/value
        throw std::string("Invalid call instruction!");

      for (unsigned i = 2, e = Args.size(); i != e; i += 2)
        Params.push_back(getValue(Args[i], Args[i+1]));
    }

    return new CallInst(F, Params);
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

    const FunctionType::ParamTypes &PL = FTy->getParamTypes();

    if (!FTy->isVarArg()) {
      Normal = getBasicBlock(Args[1]);
      Except = getBasicBlock(Args[2]);

      FunctionType::ParamTypes::const_iterator It = PL.begin();
      for (unsigned i = 3, e = Args.size(); i != e; ++i) {
        if (It == PL.end()) throw std::string("Invalid invoke instruction!");
        Params.push_back(getValue(*It++, Args[i]));
      }
      if (It != PL.end()) throw std::string("Invalid invoke instruction!");
    } else {
      // FIXME: Args[1] is a dummy padding field

      if (Args.size() < 6) throw std::string("Invalid invoke instruction!");
      if (Args[2] != Type::LabelTyID || Args[4] != Type::LabelTyID)
        throw std::string("Invalid invoke instruction!");
          
      Normal = getBasicBlock(Args[3]);
      Except = getBasicBlock(Args[5]);

      if (Args.size() & 1)   // Must be pairs of type/value
        throw std::string("Invalid invoke instruction!");

      for (unsigned i = 6; i < Args.size(); i += 2)
        Params.push_back(getValue(Args[i], Args[i+1]));
    }

    return new InvokeInst(F, Normal, Except, Params);
  }
  case Instruction::Malloc:
    if (Args.size() > 2) throw std::string("Invalid malloc instruction!");
    if (!isa<PointerType>(InstTy))
      throw std::string("Invalid malloc instruction!");

    return new MallocInst(cast<PointerType>(InstTy)->getElementType(),
                          Args.size() ? getValue(Type::UIntTyID,
                                                      Args[0]) : 0);

  case Instruction::Alloca:
    if (Args.size() > 2) throw std::string("Invalid alloca instruction!");
    if (!isa<PointerType>(InstTy))
      throw std::string("Invalid alloca instruction!");

    return new AllocaInst(cast<PointerType>(InstTy)->getElementType(),
                          Args.size() ? getValue(Type::UIntTyID,
                                                      Args[0]) : 0);
  case Instruction::Free:
    if (!isa<PointerType>(InstTy))
      throw std::string("Invalid free instruction!");
    return new FreeInst(getValue(RI.Type, Args[0]));

  case Instruction::GetElementPtr: {
    if (Args.size() == 0 || !isa<PointerType>(InstTy))
      throw std::string("Invalid getelementptr instruction!");

    std::vector<Value*> Idx;

    const Type *NextTy = InstTy;
    for (unsigned i = 1, e = Args.size(); i != e; ++i) {
      const CompositeType *TopTy = dyn_cast_or_null<CompositeType>(NextTy);
      if (!TopTy) throw std::string("Invalid getelementptr instruction!"); 
      Idx.push_back(getValue(TopTy->getIndexType(), Args[i]));
      NextTy = GetElementPtrInst::getIndexedType(InstTy, Idx, true);
    }

    return new GetElementPtrInst(getValue(RI.Type, Args[0]), Idx);
  }

  case 62:   // volatile load
  case Instruction::Load:
    if (Args.size() != 1 || !isa<PointerType>(InstTy))
      throw std::string("Invalid load instruction!");
    return new LoadInst(getValue(RI.Type, Args[0]), "", RI.Opcode == 62);

  case 63:   // volatile store 
  case Instruction::Store: {
    if (!isa<PointerType>(InstTy) || Args.size() != 2)
      throw std::string("Invalid store instruction!");

    Value *Ptr = getValue(RI.Type, Args[1]);
    const Type *ValTy = cast<PointerType>(Ptr->getType())->getElementType();
    return new StoreInst(getValue(ValTy, Args[0]), Ptr, RI.Opcode == 63);
  }
  case Instruction::Unwind:
    if (Args.size() != 0) throw std::string("Invalid unwind instruction!");
    return new UnwindInst();
  }  // end switch(RI.Opcode) 

  std::cerr << "Unrecognized instruction! " << RI.Opcode 
            << " ADDR = 0x" << (void*)Buf << "\n";
  throw std::string("Unrecognized instruction!");
}
