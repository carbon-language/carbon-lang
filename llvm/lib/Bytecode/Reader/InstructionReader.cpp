//===- ReadInst.cpp - Code to read an instruction from bytecode -------------===
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
//===------------------------------------------------------------------------===

#include "llvm/iOther.h"
#include "llvm/iTerminators.h"
#include "llvm/iMemory.h"
#include "llvm/DerivedTypes.h"
#include "ReaderInternals.h"

bool BytecodeParser::ParseRawInst(const uchar *&Buf, const uchar *EndBuf, 
				  RawInst &Result) {
  unsigned Op, Typ;
  if (read(Buf, EndBuf, Op)) return true;

  Result.NumOperands =  Op >> 30;
  Result.Opcode      = (Op >> 24) & 63;

  switch (Result.NumOperands) {
  case 1:
    Result.Ty   = getType((Op >> 12) & 4095);
    Result.Arg1 = Op & 4095;
    if (Result.Arg1 == 4095)    // Handle special encoding for 0 operands...
      Result.NumOperands = 0;
    break;
  case 2:
    Result.Ty   = getType((Op >> 16) & 255);
    Result.Arg1 = (Op >> 8 ) & 255;
    Result.Arg2 = (Op >> 0 ) & 255;
    break;
  case 3:
    Result.Ty   = getType((Op >> 18) & 63);
    Result.Arg1 = (Op >> 12) & 63;
    Result.Arg2 = (Op >> 6 ) & 63;
    Result.Arg3 = (Op >> 0 ) & 63;
    break;
  case 0:
    Buf -= 4;  // Hrm, try this again...
    if (read_vbr(Buf, EndBuf, Result.Opcode)) return true;
    if (read_vbr(Buf, EndBuf, Typ)) return true;
    Result.Ty = getType(Typ);
    if (read_vbr(Buf, EndBuf, Result.NumOperands)) return true;

    switch (Result.NumOperands) {
    case 0: 
      cerr << "Zero Arg instr found!\n"; 
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
      Result.VarArgs = new vector<unsigned>(Result.NumOperands-2);
      for (unsigned a = 0; a < Result.NumOperands-2; a++)
	if (read_vbr(Buf, EndBuf, (*Result.VarArgs)[a])) return true;
      break;
    }
    if (align32(Buf, EndBuf)) return true;
    break;
  }

  //cerr << "NO: "  << Result.NumOperands   << " opcode: " << Result.Opcode 
  //   << " Ty: " << Result.Ty->getName() << " arg1: "   << Result.Arg1 << endl;
  return false;
}


bool BytecodeParser::ParseInstruction(const uchar *&Buf, const uchar *EndBuf,
				      Instruction *&Res) {
  RawInst Raw;
  if (ParseRawInst(Buf, EndBuf, Raw)) return true;;

  if (Raw.Opcode >= Instruction::FirstUnaryOp && 
      Raw.Opcode <  Instruction::NumUnaryOps  && Raw.NumOperands == 1) {
    Res = UnaryOperator::getUnaryOperator(Raw.Opcode,getValue(Raw.Ty,Raw.Arg1));
    return false;
  } else if (Raw.Opcode >= Instruction::FirstBinaryOp &&
	     Raw.Opcode <  Instruction::NumBinaryOps  && Raw.NumOperands == 2) {
    Res = BinaryOperator::getBinaryOperator(Raw.Opcode, 
					    getValue(Raw.Ty, Raw.Arg1),
					    getValue(Raw.Ty, Raw.Arg2));
    return false;
  } else if (Raw.Opcode == Instruction::PHINode) {
    PHINode *PN = new PHINode(Raw.Ty);
    switch (Raw.NumOperands) {
    case 0: 
    case 1: 
    case 3: cerr << "Invalid phi node encountered!\n"; 
            delete PN; 
	    return true;
    case 2: PN->addIncoming(getValue(Raw.Ty, Raw.Arg1),
			    (BasicBlock*)getValue(Type::LabelTy, Raw.Arg2)); 
      break;
    default:
      PN->addIncoming(getValue(Raw.Ty, Raw.Arg1), 
		      (BasicBlock*)getValue(Type::LabelTy, Raw.Arg2));
      if (Raw.VarArgs->size() & 1) {
	cerr << "PHI Node with ODD number of arguments!\n";
	delete PN;
	return true;
      } else {
        vector<unsigned> &args = *Raw.VarArgs;
        for (unsigned i = 0; i < args.size(); i+=2)
          PN->addIncoming(getValue(Raw.Ty, args[i]),
			  (BasicBlock*)getValue(Type::LabelTy, args[i+1]));
      }
      delete Raw.VarArgs; 
      break;
    }
    Res = PN;
    return false;
  } else if (Raw.Opcode == Instruction::Ret) {
    if (Raw.NumOperands == 0) {
      Res = new ReturnInst(); return false; 
    } else if (Raw.NumOperands == 1) {
      Res = new ReturnInst(getValue(Raw.Ty, Raw.Arg1)); return false; 
    }
  } else if (Raw.Opcode == Instruction::Br) {
    if (Raw.NumOperands == 1) {
      Res = new BranchInst((BasicBlock*)getValue(Type::LabelTy, Raw.Arg1));
      return false;
    } else if (Raw.NumOperands == 3) {
      Res = new BranchInst((BasicBlock*)getValue(Type::LabelTy, Raw.Arg1),
			   (BasicBlock*)getValue(Type::LabelTy, Raw.Arg2),
			                getValue(Type::BoolTy , Raw.Arg3));
      return false;
    }
  } else if (Raw.Opcode == Instruction::Switch) {
    SwitchInst *I = 
      new SwitchInst(getValue(Raw.Ty, Raw.Arg1), 
                     (BasicBlock*)getValue(Type::LabelTy, Raw.Arg2));
    Res = I;
    if (Raw.NumOperands < 3) return false;  // No destinations?  Wierd.

    if (Raw.NumOperands == 3 || Raw.VarArgs->size() & 1) {
      cerr << "Switch statement with odd number of arguments!\n";
      delete I;
      return true;
    }      
    
    vector<unsigned> &args = *Raw.VarArgs;
    for (unsigned i = 0; i < args.size(); i += 2)
      I->dest_push_back((ConstPoolVal*)getValue(Raw.Ty, args[i]),
                        (BasicBlock*)getValue(Type::LabelTy, args[i+1]));

    delete Raw.VarArgs;
    return false;
  } else if (Raw.Opcode == Instruction::Call) {
    Method *M = (Method*)getValue(Raw.Ty, Raw.Arg1);
    if (M == 0) return true;

    const MethodType::ParamTypes &PL = M->getMethodType()->getParamTypes();
    MethodType::ParamTypes::const_iterator It = PL.begin();

    vector<Value *> Params;
    switch (Raw.NumOperands) {
    case 0: cerr << "Invalid call instruction encountered!\n";
	    return true;
    case 1: break;
    case 2: Params.push_back(getValue(*It++, Raw.Arg2)); break;
    case 3: Params.push_back(getValue(*It++, Raw.Arg2)); 
            if (It == PL.end()) return true;
            Params.push_back(getValue(*It++, Raw.Arg3)); break;
    default:
      Params.push_back(getValue(*It++, Raw.Arg2));
      {
        vector<unsigned> &args = *Raw.VarArgs;
        for (unsigned i = 0; i < args.size(); i++) {
	  if (It == PL.end()) return true;
          Params.push_back(getValue(*It++, args[i]));
	}
      }
      delete Raw.VarArgs;
    }
    if (It != PL.end()) return true;

    Res = new CallInst(M, Params);
    return false;
  } else if (Raw.Opcode == Instruction::Malloc) {
    if (Raw.NumOperands > 2) return true;
    Value *Sz = (Raw.NumOperands == 2) ? getValue(Type::UIntTy, Raw.Arg2) : 0;
    Res = new MallocInst((ConstPoolType*)getValue(Type::TypeTy, Raw.Arg1), Sz);
    return false;
  } else if (Raw.Opcode == Instruction::Alloca) {
    if (Raw.NumOperands > 2) return true;
    Value *Sz = (Raw.NumOperands == 2) ? getValue(Type::UIntTy, Raw.Arg2) : 0;
    Res = new AllocaInst((ConstPoolType*)getValue(Type::TypeTy, Raw.Arg1), Sz);
    return false;
  } else if (Raw.Opcode == Instruction::Free) {
    Value *Val = getValue(Raw.Ty, Raw.Arg1);
    if (!Val->getType()->isPointerType()) return true;
    Res = new FreeInst(Val);
    return false;
  }

  cerr << "Unrecognized instruction! " << Raw.Opcode << endl;
  return true;
}
