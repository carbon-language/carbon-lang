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

bool BytecodeParser::ParseRawInst(const unsigned char *&Buf,
                                  const unsigned char *EndBuf, 
				  RawInst &Result) {
  unsigned Op, Typ;
  if (read(Buf, EndBuf, Op)) return true;

  // bits   Instruction format:        Common to all formats
  // --------------------------
  // 01-00: Opcode type, fixed to 1.
  // 07-02: Opcode
  Result.NumOperands = (Op >> 0) & 03;
  Result.Opcode      = (Op >> 2) & 63;

  switch (Result.NumOperands) {
  case 1:
    // bits   Instruction format:
    // --------------------------
    // 19-08: Resulting type plane
    // 31-20: Operand #1 (if set to (2^12-1), then zero operands)
    //
    Result.Ty   = getType((Op >> 8) & 4095);
    Result.Arg1 = (Op >> 20) & 4095;
    if (Result.Arg1 == 4095)    // Handle special encoding for 0 operands...
      Result.NumOperands = 0;
    break;
  case 2:
    // bits   Instruction format:
    // --------------------------
    // 15-08: Resulting type plane
    // 23-16: Operand #1
    // 31-24: Operand #2  
    //
    Result.Ty   = getType((Op >> 8) & 255);
    Result.Arg1 = (Op >> 16) & 255;
    Result.Arg2 = (Op >> 24) & 255;
    break;
  case 3:
    // bits   Instruction format:
    // --------------------------
    // 13-08: Resulting type plane
    // 19-14: Operand #1
    // 25-20: Operand #2
    // 31-26: Operand #3
    //
    Result.Ty   = getType((Op >> 8) & 63);
    Result.Arg1 = (Op >> 14) & 63;
    Result.Arg2 = (Op >> 20) & 63;
    Result.Arg3 = (Op >> 26) & 63;
    break;
  case 0:
    Buf -= 4;  // Hrm, try this again...
    if (read_vbr(Buf, EndBuf, Result.Opcode)) return true;
    Result.Opcode >>= 2;
    if (read_vbr(Buf, EndBuf, Typ)) return true;
    Result.Ty = getType(Typ);
    if (Result.Ty == 0) return true;
    if (read_vbr(Buf, EndBuf, Result.NumOperands)) return true;

    switch (Result.NumOperands) {
    case 0: 
      std::cerr << "Zero Arg instr found!\n"; 
      return true;  // This encoding is invalid!
    case 1: 
      if (read_vbr(Buf, EndBuf, Result.Arg1)) return true;
      break;
    case 2:
      if (read_vbr(Buf, EndBuf, Result.Arg1) || 
	  read_vbr(Buf, EndBuf, Result.Arg2)) return true;
      break;
    case 3:
      if (read_vbr(Buf, EndBuf, Result.Arg1) || 
	  read_vbr(Buf, EndBuf, Result.Arg2) ||
          read_vbr(Buf, EndBuf, Result.Arg3)) return true;
      break;
    default:
      if (read_vbr(Buf, EndBuf, Result.Arg1) || 
	  read_vbr(Buf, EndBuf, Result.Arg2)) return true;

      // Allocate a vector to hold arguments 3, 4, 5, 6 ...
      Result.VarArgs = new std::vector<unsigned>(Result.NumOperands-2);
      for (unsigned a = 0; a < Result.NumOperands-2; a++)
	if (read_vbr(Buf, EndBuf, (*Result.VarArgs)[a])) return true;
      break;
    }
    if (align32(Buf, EndBuf)) return true;
    break;
  }

#if 0
  std::cerr << "NO: "  << Result.NumOperands   << " opcode: " << Result.Opcode 
            << " Ty: " << Result.Ty->getDescription() << " arg1: "<< Result.Arg1
            << " arg2: "   << Result.Arg2 << " arg3: "   << Result.Arg3 << "\n";
#endif
  return false;
}


bool BytecodeParser::ParseInstruction(const unsigned char *&Buf,
                                      const unsigned char *EndBuf,
				      Instruction *&Res,
                                      BasicBlock *BB /*HACK*/) {
  RawInst Raw;
  if (ParseRawInst(Buf, EndBuf, Raw))
    return true;

  if (Raw.Opcode >= Instruction::BinaryOpsBegin &&
      Raw.Opcode <  Instruction::BinaryOpsEnd  && Raw.NumOperands == 2) {
    Res = BinaryOperator::create((Instruction::BinaryOps)Raw.Opcode,
				 getValue(Raw.Ty, Raw.Arg1),
				 getValue(Raw.Ty, Raw.Arg2));
    return false;
  } 

  Value *V;
  switch (Raw.Opcode) {
  case Instruction::VarArg:
  case Instruction::Cast: {
    V = getValue(Raw.Ty, Raw.Arg1);
    const Type *Ty = getType(Raw.Arg2);
    if (V == 0 || Ty == 0) { std::cerr << "Invalid cast!\n"; return true; }
    if (Raw.Opcode == Instruction::Cast)
      Res = new CastInst(V, Ty);
    else
      Res = new VarArgInst(V, Ty);
    return false;
  }
  case Instruction::PHINode: {
    PHINode *PN = new PHINode(Raw.Ty);
    switch (Raw.NumOperands) {
    case 0: 
    case 1: 
    case 3: std::cerr << "Invalid phi node encountered!\n"; 
            delete PN; 
	    return true;
    case 2: PN->addIncoming(getValue(Raw.Ty, Raw.Arg1),
			    cast<BasicBlock>(getValue(Type::LabelTy,Raw.Arg2)));
      break;
    default:
      PN->addIncoming(getValue(Raw.Ty, Raw.Arg1), 
		      cast<BasicBlock>(getValue(Type::LabelTy, Raw.Arg2)));
      if (Raw.VarArgs->size() & 1) {
	std::cerr << "PHI Node with ODD number of arguments!\n";
	delete PN;
	return true;
      } else {
        std::vector<unsigned> &args = *Raw.VarArgs;
        for (unsigned i = 0; i < args.size(); i+=2)
          PN->addIncoming(getValue(Raw.Ty, args[i]),
			  cast<BasicBlock>(getValue(Type::LabelTy, args[i+1])));
      }
      delete Raw.VarArgs; 
      break;
    }
    Res = PN;
    return false;
  }

  case Instruction::Shl:
  case Instruction::Shr:
    Res = new ShiftInst((Instruction::OtherOps)Raw.Opcode,
			getValue(Raw.Ty, Raw.Arg1),
			getValue(Type::UByteTy, Raw.Arg2));
    return false;
  case Instruction::Ret:
    if (Raw.NumOperands == 0) {
      Res = new ReturnInst(); return false; 
    } else if (Raw.NumOperands == 1) {
      Res = new ReturnInst(getValue(Raw.Ty, Raw.Arg1)); return false; 
    }
    break;

  case Instruction::Br:
    if (Raw.NumOperands == 1) {
      Res = new BranchInst(cast<BasicBlock>(getValue(Type::LabelTy, Raw.Arg1)));
      return false;
    } else if (Raw.NumOperands == 3) {
      Res = new BranchInst(cast<BasicBlock>(getValue(Type::LabelTy, Raw.Arg1)),
			   cast<BasicBlock>(getValue(Type::LabelTy, Raw.Arg2)),
                                            getValue(Type::BoolTy , Raw.Arg3));
      return false;
    }
    break;
    
  case Instruction::Switch: {
    SwitchInst *I = 
      new SwitchInst(getValue(Raw.Ty, Raw.Arg1), 
                     cast<BasicBlock>(getValue(Type::LabelTy, Raw.Arg2)));
    Res = I;
    if (Raw.NumOperands < 3) return false;  // No destinations?  Wierd.

    if (Raw.NumOperands == 3 || Raw.VarArgs->size() & 1) {
      std::cerr << "Switch statement with odd number of arguments!\n";
      delete I;
      return true;
    }      
    
    std::vector<unsigned> &args = *Raw.VarArgs;
    for (unsigned i = 0; i < args.size(); i += 2)
      I->addCase(cast<Constant>(getValue(Raw.Ty, args[i])),
                 cast<BasicBlock>(getValue(Type::LabelTy, args[i+1])));

    delete Raw.VarArgs;
    return false;
  }

  case Instruction::Call: {
    Value *M = getValue(Raw.Ty, Raw.Arg1);
    if (M == 0) return true;

    // Check to make sure we have a pointer to method type
    const PointerType *PTy = dyn_cast<PointerType>(M->getType());
    if (PTy == 0) return true;
    const FunctionType *MTy = dyn_cast<FunctionType>(PTy->getElementType());
    if (MTy == 0) return true;

    std::vector<Value *> Params;
    const FunctionType::ParamTypes &PL = MTy->getParamTypes();

    if (!MTy->isVarArg()) {
      FunctionType::ParamTypes::const_iterator It = PL.begin();

      switch (Raw.NumOperands) {
      case 0: std::cerr << "Invalid call instruction encountered!\n";
	return true;
      case 1: break;
      case 2: Params.push_back(getValue(*It++, Raw.Arg2)); break;
      case 3: Params.push_back(getValue(*It++, Raw.Arg2)); 
	if (It == PL.end()) return true;
	Params.push_back(getValue(*It++, Raw.Arg3)); break;
      default:
	Params.push_back(getValue(*It++, Raw.Arg2));
	{
	  std::vector<unsigned> &args = *Raw.VarArgs;
	  for (unsigned i = 0; i < args.size(); i++) {
	    if (It == PL.end()) return true;
	    // TODO: Check getValue for null!
	    Params.push_back(getValue(*It++, args[i]));
	  }
	}
	delete Raw.VarArgs;
      }
      if (It != PL.end()) return true;
    } else {
      if (Raw.NumOperands > 2) {
	std::vector<unsigned> &args = *Raw.VarArgs;
	if (args.size() < 1) return true;

	if ((args.size() & 1) != 0)
	  return true;  // Must be pairs of type/value
	for (unsigned i = 0; i < args.size(); i+=2) {
	  const Type *Ty = getType(args[i]);
	  if (Ty == 0)
	    return true;
	  
	  Value *V = getValue(Ty, args[i+1]);
	  if (V == 0) return true;
	  Params.push_back(V);
	}
	delete Raw.VarArgs;
      }
    }

    Res = new CallInst(M, Params);
    return false;
  }
  case Instruction::Invoke: {
    Value *M = getValue(Raw.Ty, Raw.Arg1);
    if (M == 0) return true;

    // Check to make sure we have a pointer to method type
    const PointerType *PTy = dyn_cast<PointerType>(M->getType());
    if (PTy == 0) return true;
    const FunctionType *MTy = dyn_cast<FunctionType>(PTy->getElementType());
    if (MTy == 0) return true;

    std::vector<Value *> Params;
    const FunctionType::ParamTypes &PL = MTy->getParamTypes();
    std::vector<unsigned> &args = *Raw.VarArgs;

    BasicBlock *Normal, *Except;

    if (!MTy->isVarArg()) {
      if (Raw.NumOperands < 3) return true;

      Normal = cast<BasicBlock>(getValue(Type::LabelTy, Raw.Arg2));
      if (Raw.NumOperands == 3)
        Except = cast<BasicBlock>(getValue(Type::LabelTy, Raw.Arg3));
      else {
        Except = cast<BasicBlock>(getValue(Type::LabelTy, args[0]));

        FunctionType::ParamTypes::const_iterator It = PL.begin();
        for (unsigned i = 1; i < args.size(); i++) {
          if (It == PL.end()) return true;
          // TODO: Check getValue for null!
          Params.push_back(getValue(*It++, args[i]));
        }
        if (It != PL.end()) return true;
      }
    } else {
      if (args.size() < 4) return true;

      Normal = cast<BasicBlock>(getValue(Type::LabelTy, args[1]));
      Except = cast<BasicBlock>(getValue(Type::LabelTy, args[2]));

      if ((args.size() & 1) != 0)
	return true;  // Must be pairs of type/value
      for (unsigned i = 4; i < args.size(); i+=2) {
	// TODO: Check getValue for null!
	Params.push_back(getValue(getType(args[i]), args[i+1]));
      }
    }

    if (Raw.NumOperands > 3)
      delete Raw.VarArgs;
    Res = new InvokeInst(M, Normal, Except, Params);
    return false;
  }
  case Instruction::Malloc:
    if (Raw.NumOperands > 2) return true;
    V = Raw.NumOperands ? getValue(Type::UIntTy, Raw.Arg1) : 0;
    if (const PointerType *PTy = dyn_cast<PointerType>(Raw.Ty))
      Res = new MallocInst(PTy->getElementType(), V);
    else
      return true;
    return false;

  case Instruction::Alloca:
    if (Raw.NumOperands > 2) return true;
    V = Raw.NumOperands ? getValue(Type::UIntTy, Raw.Arg1) : 0;
    if (const PointerType *PTy = dyn_cast<PointerType>(Raw.Ty))
      Res = new AllocaInst(PTy->getElementType(), V);
    else
      return true;
    return false;

  case Instruction::Free:
    V = getValue(Raw.Ty, Raw.Arg1);
    if (!isa<PointerType>(V->getType())) return true;
    Res = new FreeInst(V);
    return false;

  case Instruction::Load:
  case Instruction::GetElementPtr: {
    std::vector<Value*> Idx;
    if (!isa<PointerType>(Raw.Ty)) return true;
    const CompositeType *TopTy = dyn_cast<CompositeType>(Raw.Ty);

    switch (Raw.NumOperands) {
    case 0: std::cerr << "Invalid load encountered!\n"; return true;
    case 1: break;
    case 2:
      if (!TopTy) return true;
      Idx.push_back(V = getValue(TopTy->getIndexType(), Raw.Arg2));
      if (!V) return true;
      break;
    case 3: {
      if (!TopTy) return true;
      Idx.push_back(V = getValue(TopTy->getIndexType(), Raw.Arg2));
      if (!V) return true;

      const Type *ETy = GetElementPtrInst::getIndexedType(TopTy, Idx, true);
      const CompositeType *ElTy = dyn_cast_or_null<CompositeType>(ETy);
      if (!ElTy) return true;

      Idx.push_back(V = getValue(ElTy->getIndexType(), Raw.Arg3));
      if (!V) return true;
      break;
    }
    default:
      if (!TopTy) return true;
      Idx.push_back(V = getValue(TopTy->getIndexType(), Raw.Arg2));
      if (!V) return true;

      std::vector<unsigned> &args = *Raw.VarArgs;
      for (unsigned i = 0, E = args.size(); i != E; ++i) {
        const Type *ETy = GetElementPtrInst::getIndexedType(Raw.Ty, Idx, true);
        const CompositeType *ElTy = dyn_cast_or_null<CompositeType>(ETy);
        if (!ElTy) return true;
	Idx.push_back(V = getValue(ElTy->getIndexType(), args[i]));
	if (!V) return true;
      }
      delete Raw.VarArgs; 
      break;
    }

    if (Raw.Opcode == Instruction::Load) {
      Value *Src = getValue(Raw.Ty, Raw.Arg1);
      if (!Idx.empty()) {
        std::cerr << "WARNING: Bytecode contains load instruction with indices."
                  << "  Replacing with getelementptr/load pair\n";
        assert(GetElementPtrInst::getIndexedType(Raw.Ty, Idx) && 
               "Bad indices for Load!");
        Src = new GetElementPtrInst(Src, Idx);
        // FIXME: Remove this compatibility code and the BB parameter to this
        // method.
        BB->getInstList().push_back(cast<Instruction>(Src));
      }
      Res = new LoadInst(Src);
    } else if (Raw.Opcode == Instruction::GetElementPtr)
      Res = new GetElementPtrInst(getValue(Raw.Ty, Raw.Arg1), Idx);
    else
      abort();
    return false;
  }
  case Instruction::Store: {
    std::vector<Value*> Idx;
    if (!isa<PointerType>(Raw.Ty)) return true;
    const CompositeType *TopTy = dyn_cast<CompositeType>(Raw.Ty);

    switch (Raw.NumOperands) {
    case 0: 
    case 1: std::cerr << "Invalid store encountered!\n"; return true;
    case 2: break;
    case 3:
      if (!TopTy) return true;
      Idx.push_back(V = getValue(TopTy->getIndexType(), Raw.Arg3));
      if (!V) return true;
      break;
    default:
      std::vector<unsigned> &args = *Raw.VarArgs;
      const CompositeType *ElTy = TopTy;
      unsigned i, E;
      for (i = 0, E = args.size(); ElTy && i != E; ++i) {
	Idx.push_back(V = getValue(ElTy->getIndexType(), args[i]));
	if (!V) return true;

        const Type *ETy = GetElementPtrInst::getIndexedType(Raw.Ty, Idx, true);
        ElTy = dyn_cast_or_null<CompositeType>(ETy);
      }
      if (i != E)
        return true;  // didn't use up all of the indices!

      delete Raw.VarArgs; 
      break;
    }

    Value *Ptr = getValue(Raw.Ty, Raw.Arg2);
    if (!Idx.empty()) {
      std::cerr << "WARNING: Bytecode contains load instruction with indices.  "
                << "Replacing with getelementptr/load pair\n";

      const Type *ElType = GetElementPtrInst::getIndexedType(Raw.Ty, Idx);
      if (ElType == 0) return true;

      Ptr = new GetElementPtrInst(Ptr, Idx);
      // FIXME: Remove this compatibility code and the BB parameter to this
      // method.
      BB->getInstList().push_back(cast<Instruction>(Ptr));
    }

    const Type *ValTy = cast<PointerType>(Ptr->getType())->getElementType();
    Res = new StoreInst(getValue(ValTy, Raw.Arg1), Ptr);
    return false;
  }
  }  // end switch(Raw.Opcode) 

  std::cerr << "Unrecognized instruction! " << Raw.Opcode 
            << " ADDR = 0x" << (void*)Buf << "\n";
  return true;
}
