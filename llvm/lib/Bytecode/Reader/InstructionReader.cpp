//===- ReadInst.cpp - Code to read an instruction from bytecode -----------===//
//
// This file defines the mechanism to read an instruction from a bytecode 
// stream.
//
// Note that this library should be as fast as possible, reentrant, and 
// threadsafe!!
//
// TODO: Change from getValue(Raw.Arg1) etc, to getArg(Raw, 1)
//       Make it check type, so that casts are checked.
//
//===----------------------------------------------------------------------===//

#include "ReaderInternals.h"
#include "llvm/iTerminators.h"
#include "llvm/iMemory.h"
#include "llvm/iPHINode.h"
#include "llvm/iOther.h"

std::auto_ptr<RawInst>
BytecodeParser::ParseRawInst(const unsigned char *&Buf,
                             const unsigned char *EndBuf) { 
  unsigned Op, Typ;
  std::auto_ptr<RawInst> Result = std::auto_ptr<RawInst>(new RawInst());
  if (read(Buf, EndBuf, Op)) 
    throw std::string("Error reading from buffer.");

  // bits   Instruction format:        Common to all formats
  // --------------------------
  // 01-00: Opcode type, fixed to 1.
  // 07-02: Opcode
  Result->NumOperands = (Op >> 0) & 03;
  Result->Opcode      = (Op >> 2) & 63;

  switch (Result->NumOperands) {
  case 1:
    // bits   Instruction format:
    // --------------------------
    // 19-08: Resulting type plane
    // 31-20: Operand #1 (if set to (2^12-1), then zero operands)
    //
    Result->Ty   = getType((Op >> 8) & 4095);
    Result->Arg1 = (Op >> 20) & 4095;
    if (Result->Arg1 == 4095)    // Handle special encoding for 0 operands...
      Result->NumOperands = 0;
    break;
  case 2:
    // bits   Instruction format:
    // --------------------------
    // 15-08: Resulting type plane
    // 23-16: Operand #1
    // 31-24: Operand #2  
    //
    Result->Ty   = getType((Op >> 8) & 255);
    Result->Arg1 = (Op >> 16) & 255;
    Result->Arg2 = (Op >> 24) & 255;
    break;
  case 3:
    // bits   Instruction format:
    // --------------------------
    // 13-08: Resulting type plane
    // 19-14: Operand #1
    // 25-20: Operand #2
    // 31-26: Operand #3
    //
    Result->Ty   = getType((Op >> 8) & 63);
    Result->Arg1 = (Op >> 14) & 63;
    Result->Arg2 = (Op >> 20) & 63;
    Result->Arg3 = (Op >> 26) & 63;
    break;
  case 0:
    Buf -= 4;  // Hrm, try this again...
    if (read_vbr(Buf, EndBuf, Result->Opcode))
      throw std::string("Error reading from buffer.");
    Result->Opcode >>= 2;
    if (read_vbr(Buf, EndBuf, Typ))
      throw std::string("Error reading from buffer.");
    Result->Ty = getType(Typ);
    if (Result->Ty == 0) 
      throw std::string("Invalid type read in instruction.");
    if (read_vbr(Buf, EndBuf, Result->NumOperands))
      throw std::string("Error reading from buffer.");

    switch (Result->NumOperands) {
    case 0: 
      throw std::string("Zero-argument instruction found; this is invalid.");
    case 1: 
      if (read_vbr(Buf, EndBuf, Result->Arg1)) 
        throw std::string("Error reading from buffer");
      break;
    case 2:
      if (read_vbr(Buf, EndBuf, Result->Arg1) || 
	  read_vbr(Buf, EndBuf, Result->Arg2))
        throw std::string("Error reading from buffer");
      break;
    case 3:
      if (read_vbr(Buf, EndBuf, Result->Arg1) || 
	  read_vbr(Buf, EndBuf, Result->Arg2) ||
          read_vbr(Buf, EndBuf, Result->Arg3))
        throw std::string("Error reading from buffer");
      break;
    default:
      if (read_vbr(Buf, EndBuf, Result->Arg1) || 
	  read_vbr(Buf, EndBuf, Result->Arg2))
        throw std::string("Error reading from buffer");

      // Allocate a vector to hold arguments 3, 4, 5, 6 ...
      Result->VarArgs = new std::vector<unsigned>(Result->NumOperands-2);
      for (unsigned a = 0; a < Result->NumOperands-2; a++)
	if (read_vbr(Buf, EndBuf, (*Result->VarArgs)[a]))
        throw std::string("Error reading from buffer");          

      break;
    }
    if (align32(Buf, EndBuf)) 
      throw std::string("Unaligned bytecode buffer.");
    break;
  }

#if 0
  std::cerr << "NO: "  << Result->NumOperands << " opcode: " << Result->Opcode 
            << " Ty: "<< Result->Ty->getDescription()<< " arg1: "<< Result->Arg1
            << " arg2: " << Result->Arg2 << " arg3: "   << Result->Arg3 << "\n";
#endif
  return Result;
}


Instruction *BytecodeParser::ParseInstruction(const unsigned char *&Buf,
                                              const unsigned char *EndBuf) {
  std::auto_ptr<RawInst> Raw = ParseRawInst(Buf, EndBuf);

  if (Raw->Opcode >= Instruction::BinaryOpsBegin &&
      Raw->Opcode <  Instruction::BinaryOpsEnd  && Raw->NumOperands == 2)
    return BinaryOperator::create((Instruction::BinaryOps)Raw->Opcode,
                                  getValue(Raw->Ty, Raw->Arg1),
                                  getValue(Raw->Ty, Raw->Arg2));

  switch (Raw->Opcode) {
  case Instruction::VarArg:
  case Instruction::Cast: {
    Value *V = getValue(Raw->Ty, Raw->Arg1);
    const Type *Ty = getType(Raw->Arg2);
    if (Ty == 0) throw std::string("Invalid cast!\n");
    if (Raw->Opcode == Instruction::Cast)
      return new CastInst(V, Ty);
    else
      return new VarArgInst(V, Ty);
  }
  case Instruction::PHINode: {
    PHINode *PN = new PHINode(Raw->Ty);
    switch (Raw->NumOperands) {
    case 0: 
    case 1: 
    case 3:
      delete PN; 
      throw std::string("Invalid phi node encountered!\n");
    case 2:
      PN->addIncoming(getValue(Raw->Ty, Raw->Arg1), getBasicBlock(Raw->Arg2));
      break;
    default:
      PN->addIncoming(getValue(Raw->Ty, Raw->Arg1), getBasicBlock(Raw->Arg2));
      if (Raw->VarArgs->size() & 1) {
	delete PN;
	throw std::string("PHI Node with ODD number of arguments!\n");
      } else {
        std::vector<unsigned> &args = *Raw->VarArgs;
        for (unsigned i = 0; i < args.size(); i+=2)
          PN->addIncoming(getValue(Raw->Ty, args[i]), getBasicBlock(args[i+1]));
      }
      delete Raw->VarArgs; 
      break;
    }
    return PN;
  }

  case Instruction::Shl:
  case Instruction::Shr:
    return new ShiftInst((Instruction::OtherOps)Raw->Opcode,
                         getValue(Raw->Ty, Raw->Arg1),
                         getValue(Type::UByteTyID, Raw->Arg2));
  case Instruction::Ret:
    if (Raw->NumOperands == 0)
      return new ReturnInst();
    else if (Raw->NumOperands == 1)
      return new ReturnInst(getValue(Raw->Ty, Raw->Arg1));
    break;

  case Instruction::Br:
    if (Raw->NumOperands == 1)
      return new BranchInst(getBasicBlock(Raw->Arg1));
    else if (Raw->NumOperands == 3)
      return new BranchInst(getBasicBlock(Raw->Arg1), getBasicBlock(Raw->Arg2),
                            getValue(Type::BoolTyID , Raw->Arg3));
    throw std::string("Invalid number of operands for a 'br' instruction!");
    
  case Instruction::Switch: {
    SwitchInst *I = new SwitchInst(getValue(Raw->Ty, Raw->Arg1),
                                   getBasicBlock(Raw->Arg2));
    if (Raw->NumOperands < 3)
      return I;

    if (Raw->NumOperands == 3 || Raw->VarArgs->size() & 1) {
      delete I;
      throw std::string("Switch statement with odd number of arguments!");
    }
    
    std::vector<unsigned> &args = *Raw->VarArgs;
    for (unsigned i = 0; i < args.size(); i += 2)
      I->addCase(cast<Constant>(getValue(Raw->Ty, args[i])),
                 getBasicBlock(args[i+1]));

    delete Raw->VarArgs;
    return I;
  }

  case Instruction::Call: {
    Value *F = getValue(Raw->Ty, Raw->Arg1);

    // Check to make sure we have a pointer to function type
    const PointerType *PTy = dyn_cast<PointerType>(F->getType());
    if (PTy == 0) throw std::string("Call to non function pointer value!");
    const FunctionType *FTy = dyn_cast<FunctionType>(PTy->getElementType());
    if (FTy == 0) throw std::string("Call to non function pointer value!");

    std::vector<Value *> Params;
    const FunctionType::ParamTypes &PL = FTy->getParamTypes();

    if (!FTy->isVarArg()) {
      FunctionType::ParamTypes::const_iterator It = PL.begin();

      switch (Raw->NumOperands) {
      case 0: throw std::string("Invalid call instruction encountered!");
      case 1: break;
      case 2: Params.push_back(getValue(*It++, Raw->Arg2)); break;
      case 3: Params.push_back(getValue(*It++, Raw->Arg2)); 
	if (It == PL.end()) throw std::string("Invalid call instruction!");
	Params.push_back(getValue(*It++, Raw->Arg3)); break;
      default:
	Params.push_back(getValue(*It++, Raw->Arg2));
	{
	  std::vector<unsigned> &args = *Raw->VarArgs;
	  for (unsigned i = 0; i < args.size(); i++) {
	    if (It == PL.end()) throw std::string("Invalid call instruction!");
	    Params.push_back(getValue(*It++, args[i]));
	  }
	}
	delete Raw->VarArgs;
      }
      if (It != PL.end()) throw std::string("Invalid call instruction!");
    } else {
      if (Raw->NumOperands > 2) {
	std::vector<unsigned> &args = *Raw->VarArgs;
	if (args.size() < 1) throw std::string("Invalid call instruction!");

	if ((args.size() & 1) != 0)  // Must be pairs of type/value
          throw std::string("Invalid call instruction!");
	for (unsigned i = 0; i < args.size(); i+=2) {
	  const Type *Ty = getType(args[i]);
	  if (Ty == 0) throw std::string("Invalid call instruction!");
	  Params.push_back(getValue(Ty, args[i+1]));
	}
	delete Raw->VarArgs;
      }
    }

    return new CallInst(F, Params);
  }
  case Instruction::Invoke: {
    Value *F = getValue(Raw->Ty, Raw->Arg1);

    // Check to make sure we have a pointer to function type
    const PointerType *PTy = dyn_cast<PointerType>(F->getType());
    if (PTy == 0) throw std::string("Invoke to non function pointer value!");
    const FunctionType *FTy = dyn_cast<FunctionType>(PTy->getElementType());
    if (FTy == 0) throw std::string("Invoke to non function pointer value!");

    std::vector<Value *> Params;
    const FunctionType::ParamTypes &PL = FTy->getParamTypes();
    std::vector<unsigned> &args = *Raw->VarArgs;

    BasicBlock *Normal, *Except;

    if (!FTy->isVarArg()) {
      if (Raw->NumOperands < 3) throw std::string("Invalid call instruction!");

      Normal = getBasicBlock(Raw->Arg2);
      if (Raw->NumOperands == 3)
        Except = getBasicBlock(Raw->Arg3);
      else {
        Except = getBasicBlock(args[0]);

        FunctionType::ParamTypes::const_iterator It = PL.begin();
        for (unsigned i = 1; i < args.size(); i++) {
          if (It == PL.end()) throw std::string("Invalid invoke instruction!");
          Params.push_back(getValue(*It++, args[i]));
        }
        if (It != PL.end()) throw std::string("Invalid invoke instruction!");
      }
    } else {
      if (args.size() < 4) throw std::string("Invalid invoke instruction!");
      if (args[0] != Type::LabelTyID || args[2] != Type::LabelTyID)
        throw std::string("Invalid invoke instruction!");
          
      Normal = getBasicBlock(args[1]);
      Except = getBasicBlock(args[3]);

      if ((args.size() & 1) != 0)   // Must be pairs of type/value
        throw std::string("Invalid invoke instruction!");

      for (unsigned i = 4; i < args.size(); i += 2)
        Params.push_back(getValue(args[i], args[i+1]));
    }

    if (Raw->NumOperands > 3)
      delete Raw->VarArgs;
    return new InvokeInst(F, Normal, Except, Params);
  }
  case Instruction::Malloc:
    if (Raw->NumOperands > 2) throw std::string("Invalid malloc instruction!");
    if (!isa<PointerType>(Raw->Ty))
      throw std::string("Invalid malloc instruction!");

    return new MallocInst(cast<PointerType>(Raw->Ty)->getElementType(),
                          Raw->NumOperands ? getValue(Type::UIntTyID,
                                                      Raw->Arg1) : 0);

  case Instruction::Alloca:
    if (Raw->NumOperands > 2) throw std::string("Invalid alloca instruction!");
    if (!isa<PointerType>(Raw->Ty))
      throw std::string("Invalid alloca instruction!");

    return new AllocaInst(cast<PointerType>(Raw->Ty)->getElementType(),
                          Raw->NumOperands ? getValue(Type::UIntTyID,
                                                      Raw->Arg1) : 0);
  case Instruction::Free:
    if (!isa<PointerType>(Raw->Ty))
      throw std::string("Invalid free instruction!");
    return new FreeInst(getValue(Raw->Ty, Raw->Arg1));

  case Instruction::GetElementPtr: {
    std::vector<Value*> Idx;
    if (!isa<PointerType>(Raw->Ty))
      throw std::string("Invalid getelementptr instruction!");
    const CompositeType *TopTy = dyn_cast<CompositeType>(Raw->Ty);

    switch (Raw->NumOperands) {
    case 0: throw std::string("Invalid getelementptr instruction!");
    case 1: break;
    case 2:
      if (!TopTy) throw std::string("Invalid getelementptr instruction!");
      Idx.push_back(getValue(TopTy->getIndexType(), Raw->Arg2));
      break;
    case 3: {
      if (!TopTy) throw std::string("Invalid getelementptr instruction!");
      Idx.push_back(getValue(TopTy->getIndexType(), Raw->Arg2));

      const Type *ETy = GetElementPtrInst::getIndexedType(TopTy, Idx, true);
      const CompositeType *ElTy = dyn_cast_or_null<CompositeType>(ETy);
      if (!ElTy) throw std::string("Invalid getelementptr instruction!");

      Idx.push_back(getValue(ElTy->getIndexType(), Raw->Arg3));
      break;
    }
    default:
      if (!TopTy) throw std::string("Invalid getelementptr instruction!");
      Idx.push_back(getValue(TopTy->getIndexType(), Raw->Arg2));

      std::vector<unsigned> &args = *Raw->VarArgs;
      for (unsigned i = 0, E = args.size(); i != E; ++i) {
        const Type *ETy = GetElementPtrInst::getIndexedType(Raw->Ty, Idx, true);
        const CompositeType *ElTy = dyn_cast_or_null<CompositeType>(ETy);
        if (!ElTy) throw std::string("Invalid getelementptr instruction!");
	Idx.push_back(getValue(ElTy->getIndexType(), args[i]));
      }
      delete Raw->VarArgs; 
      break;
    }

    return new GetElementPtrInst(getValue(Raw->Ty, Raw->Arg1), Idx);
  }

  case 62:   // volatile load
  case Instruction::Load:
    if (Raw->NumOperands != 1 || !isa<PointerType>(Raw->Ty))
      throw std::string("Invalid load instruction!");
    return new LoadInst(getValue(Raw->Ty, Raw->Arg1), "", Raw->Opcode == 62);

  case 63:   // volatile store 
  case Instruction::Store: {
    if (!isa<PointerType>(Raw->Ty) || Raw->NumOperands != 2)
      throw std::string("Invalid store instruction!");

    Value *Ptr = getValue(Raw->Ty, Raw->Arg2);
    const Type *ValTy = cast<PointerType>(Ptr->getType())->getElementType();
    return new StoreInst(getValue(ValTy, Raw->Arg1), Ptr, Raw->Opcode == 63);
  }
  case Instruction::Unwind:
    if (Raw->NumOperands != 0) throw std::string("Invalid unwind instruction!");
    return new UnwindInst();
  }  // end switch(Raw->Opcode) 

  std::cerr << "Unrecognized instruction! " << Raw->Opcode 
            << " ADDR = 0x" << (void*)Buf << "\n";
  throw std::string("Unrecognized instruction!");
}
