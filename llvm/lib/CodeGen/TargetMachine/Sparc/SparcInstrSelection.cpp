// $Id$
//***************************************************************************
// File:
//	SparcInstrSelection.cpp
// 
// Purpose:
//	
// History:
//	7/02/01	 -  Vikram Adve  -  Created
//**************************************************************************/

#include "llvm/CodeGen/Sparc.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/InstrForest.h"
#include "llvm/CodeGen/InstrSelection.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/DerivedTypes.h"
#include "llvm/iTerminators.h"
#include "llvm/iMemory.h"
#include "llvm/iOther.h"
#include "llvm/BasicBlock.h"
#include "llvm/Method.h"
#include "llvm/ConstPoolVals.h"


//******************** Internal Data Declarations ************************/

// to be used later
struct BranchPattern {
  bool	        flipCondition; // should the sense of the test be reversed
  BasicBlock*	targetBB;      // which basic block to branch to
  MachineInstr* extraBranch;   // if neither branch is fall-through, then this
                               // BA must be inserted after the cond'l one
};

//************************* Forward Declarations ***************************/


static MachineOpCode ChooseBprInstruction   (const InstructionNode* instrNode);

static MachineOpCode ChooseBccInstruction   (const InstructionNode* instrNode,
					     bool& isFPBranch);

static MachineOpCode ChooseBpccInstruction  (const InstructionNode* instrNode,
					     const BinaryOperator* setCCInst);

static MachineOpCode ChooseBFpccInstruction (const InstructionNode* instrNode,
					     const BinaryOperator* setCCInst);

static MachineOpCode ChooseMovFpccInstruction(const InstructionNode*);

static MachineOpCode ChooseMovpccAfterSub   (const InstructionNode* instrNode,
					     bool& mustClearReg,
					     int& valueToMove);

static MachineOpCode ChooseConvertToFloatInstr(const InstructionNode*,
					       const Type* opType);

static MachineOpCode ChooseConvertToIntInstr(const InstructionNode* instrNode,
					     const Type* opType);

static MachineOpCode ChooseAddInstruction   (const InstructionNode* instrNode);

static MachineOpCode ChooseSubInstruction   (const InstructionNode* instrNode);

static MachineOpCode ChooseFcmpInstruction  (const InstructionNode* instrNode);

static MachineOpCode ChooseMulInstruction   (const InstructionNode* instrNode,
					     bool checkCasts);

static MachineOpCode ChooseDivInstruction   (const InstructionNode* instrNode);

static MachineOpCode ChooseLoadInstruction  (const Type* resultType);

static MachineOpCode ChooseStoreInstruction (const Type* valueType);

static void		SetOperandsForMemInstr(MachineInstr* minstr,
					 const InstructionNode* vmInstrNode,
					 const TargetMachine& target);

static void		SetMemOperands_Internal	(MachineInstr* minstr,
					 const InstructionNode* vmInstrNode,
					 Value* ptrVal,
					 Value* arrayOffsetVal,
					 const vector<ConstPoolVal*>& idxVec,
					 const TargetMachine& target);

static unsigned		FixConstantOperands(const InstructionNode* vmInstrNode,
					    MachineInstr** mvec,
					    unsigned numInstr,
					    TargetMachine& target);

static MachineInstr*	MakeLoadConstInstr(Instruction* vmInstr,
					   Value* val,
					   TmpInstruction*& tmpReg,
					   MachineInstr*& getMinstr2);

static void		ForwardOperand	(InstructionNode* treeNode,
					 InstructionNode* parent,
					 int operandNum);


//************************ Internal Functions ******************************/

// Convenience function to get the value of an integer constant, for an
// appropriate integer or non-integer type that can be held in an integer.
// The type of the argument must be the following:
//   GetConstantValueAsSignedInt: any of the above, but the value
//				  must fit into a int64_t.
// 
// isValidConstant is set to true if a valid constant was found.
// 

static int64_t GetConstantValueAsSignedInt(const Value *V,
					   bool &isValidConstant) {
  if (!V->isConstant()) { isValidConstant = false; return 0; }
  isValidConstant = true;
  
  if (V->getType() == Type::BoolTy)
    return ((ConstPoolBool*)V)->getValue();
  if (V->getType()->isIntegral()) {
    if (V->getType()->isSigned())
      return ((ConstPoolSInt*)V)->getValue();
    
    assert(V->getType()->isUnsigned());
    uint64_t Val = ((ConstPoolUInt*)V)->getValue();

    if (Val < INT64_MAX)     // then safe to cast to signed
      return (int64_t)Val;
  }

  isValidConstant = false;
  return 0;
}



//------------------------------------------------------------------------ 
// External Function: ThisIsAChainRule
//
// Purpose:
//   Check if a given BURG rule is a chain rule.
//------------------------------------------------------------------------ 

extern bool
ThisIsAChainRule(int eruleno)
{
  switch(eruleno)
    {
    case 111:	// stmt:  reg
    case 112:	// stmt:  boolconst
    case 113:	// stmt:  bool
    case 121:
    case 122:
    case 123:
    case 124:
    case 125:
    case 126:
    case 127:
    case 128:
    case 129:
    case 130:
    case 131:
    case 132:
    case 153:
    case 155: return true; break;
      
    default: return false; break;
    }
}


static inline MachineOpCode 
ChooseBprInstruction(const InstructionNode* instrNode)
{
  MachineOpCode opCode;
  
  Instruction* setCCInstr =
    ((InstructionNode*) instrNode->leftChild())->getInstruction();
  
  switch(setCCInstr->getOpcode())
    {
    case Instruction::SetEQ: opCode = BRZ;   break;
    case Instruction::SetNE: opCode = BRNZ;  break;
    case Instruction::SetLE: opCode = BRLEZ; break;
    case Instruction::SetGE: opCode = BRGEZ; break;
    case Instruction::SetLT: opCode = BRLZ;  break;
    case Instruction::SetGT: opCode = BRGZ;  break;
    default:
      assert(0 && "Unrecognized VM instruction!");
      opCode = INVALID_OPCODE;
      break; 
    }
  
  return opCode;
}


static inline MachineOpCode 
ChooseBccInstruction(const InstructionNode* instrNode,
		     bool& isFPBranch)
{
  InstructionNode* setCCNode = (InstructionNode*) instrNode->leftChild();
  BinaryOperator* setCCInstr = (BinaryOperator*) setCCNode->getInstruction();
  const Type* setCCType = setCCInstr->getOperand(0)->getType();
  
  isFPBranch = (setCCType == Type::FloatTy || setCCType == Type::DoubleTy); 
  
  if (isFPBranch) 
    return ChooseBFpccInstruction(instrNode, setCCInstr);
  else
    return ChooseBpccInstruction(instrNode, setCCInstr);
}


static inline MachineOpCode 
ChooseBpccInstruction(const InstructionNode* instrNode,
		      const BinaryOperator* setCCInstr)
{
  MachineOpCode opCode = INVALID_OPCODE;
  
  bool isSigned = setCCInstr->getOperand(0)->getType()->isSigned();
  
  if (isSigned)
    {
      switch(setCCInstr->getOpcode())
	{
	case Instruction::SetEQ: opCode = BE;  break;
	case Instruction::SetNE: opCode = BNE; break;
	case Instruction::SetLE: opCode = BLE; break;
	case Instruction::SetGE: opCode = BGE; break;
	case Instruction::SetLT: opCode = BL;  break;
	case Instruction::SetGT: opCode = BG;  break;
	default:
	  assert(0 && "Unrecognized VM instruction!");
	  break; 
	}
    }
  else
    {
      switch(setCCInstr->getOpcode())
	{
	case Instruction::SetEQ: opCode = BE;   break;
	case Instruction::SetNE: opCode = BNE;  break;
	case Instruction::SetLE: opCode = BLEU; break;
	case Instruction::SetGE: opCode = BCC;  break;
	case Instruction::SetLT: opCode = BCS;  break;
	case Instruction::SetGT: opCode = BGU;  break;
	default:
	  assert(0 && "Unrecognized VM instruction!");
	  break; 
	}
    }
  
  return opCode;
}

static inline MachineOpCode 
ChooseBFpccInstruction(const InstructionNode* instrNode,
		       const BinaryOperator* setCCInstr)
{
  MachineOpCode opCode = INVALID_OPCODE;
  
  switch(setCCInstr->getOpcode())
    {
    case Instruction::SetEQ: opCode = FBE;  break;
    case Instruction::SetNE: opCode = FBNE; break;
    case Instruction::SetLE: opCode = FBLE; break;
    case Instruction::SetGE: opCode = FBGE; break;
    case Instruction::SetLT: opCode = FBL;  break;
    case Instruction::SetGT: opCode = FBG;  break;
    default:
      assert(0 && "Unrecognized VM instruction!");
      break; 
    }
  
  return opCode;
}


static inline MachineOpCode 
ChooseMovFpccInstruction(const InstructionNode* instrNode)
{
  MachineOpCode opCode = INVALID_OPCODE;
  
  switch(instrNode->getInstruction()->getOpcode())
    {
    case Instruction::SetEQ: opCode = MOVFE;  break;
    case Instruction::SetNE: opCode = MOVFNE; break;
    case Instruction::SetLE: opCode = MOVFLE; break;
    case Instruction::SetGE: opCode = MOVFGE; break;
    case Instruction::SetLT: opCode = MOVFL;  break;
    case Instruction::SetGT: opCode = MOVFG;  break;
    default:
      assert(0 && "Unrecognized VM instruction!");
      break; 
    }
  
  return opCode;
}


// Assumes that SUBcc v1, v2 -> v3 has been executed.
// In most cases, we want to clear v3 and then follow it by instruction
// MOVcc 1 -> v3.
// Set mustClearReg=false if v3 need not be cleared before conditional move.
// Set valueToMove=0 if we want to conditionally move 0 instead of 1
//                      (i.e., we want to test inverse of a condition)
//
// 
static MachineOpCode
ChooseMovpccAfterSub(const InstructionNode* instrNode,
		     bool& mustClearReg,
		     int& valueToMove)
{
  MachineOpCode opCode = INVALID_OPCODE;
  mustClearReg = true;
  valueToMove = 1;
  
  switch(instrNode->getInstruction()->getOpcode())
    {
    case Instruction::SetEQ: opCode = MOVNE; mustClearReg = false;
					     valueToMove = 0; break;
    case Instruction::SetLE: opCode = MOVLE; break;
    case Instruction::SetGE: opCode = MOVGE; break;
    case Instruction::SetLT: opCode = MOVL;  break;
    case Instruction::SetGT: opCode = MOVG;  break;
      
    case Instruction::SetNE: assert(0 && "No move required!");
      
    default:
      assert(0 && "Unrecognized VM instruction!");
      break; 
    }
  
  return opCode;
}


static inline MachineOpCode
ChooseConvertToFloatInstr(const InstructionNode* instrNode,
			  const Type* opType)
{
  MachineOpCode opCode = INVALID_OPCODE;
  
  switch(instrNode->getOpLabel())
    {
    case ToFloatTy: 
      if (opType == Type::SByteTy || opType == Type::ShortTy || opType == Type::IntTy)
	opCode = FITOS;
      else if (opType == Type::LongTy)
	opCode = FXTOS;
      else if (opType == Type::DoubleTy)
	opCode = FDTOS;
      else if (opType == Type::FloatTy)
	;
      else
	assert(0 && "Cannot convert this type to FLOAT on SPARC");		
      break;
      
    case ToDoubleTy: 
      if (opType == Type::SByteTy || opType == Type::ShortTy || opType == Type::IntTy)
	opCode = FITOD;
      else if (opType == Type::LongTy)
	opCode = FXTOD;
      else if (opType == Type::FloatTy)
	opCode = FSTOD;
      else if (opType == Type::DoubleTy)
	;
      else
	assert(0 && "Cannot convert this type to DOUBLE on SPARC");		
      break;
      
    default:
      break;
    }
  
  return opCode;
}

static inline MachineOpCode 
ChooseConvertToIntInstr(const InstructionNode* instrNode,
			const Type* opType)
{
  MachineOpCode opCode = INVALID_OPCODE;;
  
  int instrType = (int) instrNode->getOpLabel();
  
  if (instrType == ToSByteTy || instrType == ToShortTy || instrType == ToIntTy)
    {
      switch (opType->getPrimitiveID())
	{
        case Type::FloatTyID:   opCode = FSTOI; break;
        case Type::DoubleTyID:  opCode = FDTOI; break;
	default:
	  assert(0 && "Non-numeric non-bool type cannot be converted to Int");
	  break;
	}
    }
  else if (instrType == ToLongTy)
    {
      switch (opType->getPrimitiveID())
	{
        case Type::FloatTyID:   opCode = FSTOX; break;
        case Type::DoubleTyID:  opCode = FDTOX; break;
	default:
	  assert(0 && "Non-numeric non-bool type cannot be converted to Long");
	  break;
	}
    }
  else
      assert(0 && "Should not get here, Mo!");
  
  return opCode;
}


static inline MachineOpCode 
ChooseAddInstruction(const InstructionNode* instrNode)
{
  MachineOpCode opCode = INVALID_OPCODE;
  
  const Type* resultType = instrNode->getInstruction()->getType();
  
  if (resultType->isIntegral() ||
      resultType->isPointerType() ||
      resultType->isMethodType() ||
      resultType->isLabelType())
    {
      opCode = ADD;
    }
  else
    {
      Value* operand = ((InstrTreeNode*) instrNode->leftChild())->getValue();
      switch(operand->getType()->getPrimitiveID())
	{
	case Type::FloatTyID:  opCode = FADDS; break;
	case Type::DoubleTyID: opCode = FADDD; break;
	default: assert(0 && "Invalid type for ADD instruction"); break; 
	}
    }
  
  return opCode;
}


static inline MachineInstr* 
CreateMovFloatInstruction(const InstructionNode* instrNode,
			  const Type* resultType)
{
  MachineInstr* minstr = new MachineInstr((resultType == Type::FloatTy)
					  ? FMOVS : FMOVD);
  minstr->SetMachineOperand(0, MachineOperand::MO_VirtualRegister,
			    instrNode->leftChild()->getValue());
  minstr->SetMachineOperand(1, MachineOperand::MO_VirtualRegister,
			    instrNode->getValue());
  return minstr;
}

static inline MachineInstr* 
CreateAddConstInstruction(const InstructionNode* instrNode)
{
  MachineInstr* minstr = NULL;
  
  Value* constOp = ((InstrTreeNode*) instrNode->rightChild())->getValue();
  assert(constOp->isConstant());
  
  // Cases worth optimizing are:
  // (1) Add with 0 for float or double: use an FMOV of appropriate type,
  //	 instead of an FADD (1 vs 3 cycles).  There is no integer MOV.
  // 
  const Type* resultType = instrNode->getInstruction()->getType();
  
  if (resultType == Type::FloatTy || resultType == Type::DoubleTy) {
    double dval = ((ConstPoolFP*) constOp)->getValue();
    if (dval == 0.0)
      minstr = CreateMovFloatInstruction(instrNode, resultType);
  }
  
  return minstr;
}


static inline MachineOpCode 
ChooseSubInstruction(const InstructionNode* instrNode)
{
  MachineOpCode opCode = INVALID_OPCODE;
  
  const Type* resultType = instrNode->getInstruction()->getType();
  
  if (resultType->isIntegral() ||
      resultType->isPointerType())
    {
      opCode = SUB;
    }
  else
    {
      Value* operand = ((InstrTreeNode*) instrNode->leftChild())->getValue();
      switch(operand->getType()->getPrimitiveID())
	{
	case Type::FloatTyID:  opCode = FSUBS; break;
	case Type::DoubleTyID: opCode = FSUBD; break;
	default: assert(0 && "Invalid type for SUB instruction"); break; 
	}
    }
  
  return opCode;
}


static inline MachineInstr* 
CreateSubConstInstruction(const InstructionNode* instrNode)
{
  MachineInstr* minstr = NULL;
  
  Value* constOp = ((InstrTreeNode*) instrNode->rightChild())->getValue();
  assert(constOp->isConstant());
  
  // Cases worth optimizing are:
  // (1) Sub with 0 for float or double: use an FMOV of appropriate type,
  //	 instead of an FSUB (1 vs 3 cycles).  There is no integer MOV.
  // 
  const Type* resultType = instrNode->getInstruction()->getType();
  
  if (resultType == Type::FloatTy ||
      resultType == Type::DoubleTy)
    {
      double dval = ((ConstPoolFP*) constOp)->getValue();
      if (dval == 0.0)
	minstr = CreateMovFloatInstruction(instrNode, resultType);
    }
  
  return minstr;
}


static inline MachineOpCode 
ChooseFcmpInstruction(const InstructionNode* instrNode)
{
  MachineOpCode opCode = INVALID_OPCODE;
  
  Value* operand = ((InstrTreeNode*) instrNode->leftChild())->getValue();
  switch(operand->getType()->getPrimitiveID())
    {
    case Type::FloatTyID:  opCode = FCMPS; break;
    case Type::DoubleTyID: opCode = FCMPD; break;
    default: assert(0 && "Invalid type for FCMP instruction"); break; 
    }
  
  return opCode;
}


// Assumes that leftArg and rightArg are both cast instructions.
//
static inline bool
BothFloatToDouble(const InstructionNode* instrNode)
{
  InstrTreeNode* leftArg = instrNode->leftChild();
  InstrTreeNode* rightArg = instrNode->rightChild();
  InstrTreeNode* leftArgArg = leftArg->leftChild();
  InstrTreeNode* rightArgArg = rightArg->leftChild();
  assert(leftArg->getValue()->getType() == rightArg->getValue()->getType());
  
  // Check if both arguments are floats cast to double
  return (leftArg->getValue()->getType() == Type::DoubleTy &&
	  leftArgArg->getValue()->getType() == Type::FloatTy &&
	  rightArgArg->getValue()->getType() == Type::FloatTy);
}


static inline MachineOpCode 
ChooseMulInstruction(const InstructionNode* instrNode,
		     bool checkCasts)
{
  MachineOpCode opCode = INVALID_OPCODE;
  
  if (checkCasts && BothFloatToDouble(instrNode))
    {
      return opCode = FSMULD;
    }
  // else fall through and use the regular multiply instructions
  
  const Type* resultType = instrNode->getInstruction()->getType();
  
  if (resultType->isIntegral())
    {
      opCode = MULX;
    }
  else
    {
      switch(instrNode->leftChild()->getValue()->getType()->getPrimitiveID())
	{
	case Type::FloatTyID:  opCode = FMULS; break;
	case Type::DoubleTyID: opCode = FMULD; break;
	default: assert(0 && "Invalid type for MUL instruction"); break; 
	}
    }
  
  return opCode;
}


static inline MachineInstr*
CreateIntNegInstruction(Value* vreg)
{
  MachineInstr* minstr = new MachineInstr(SUB);
  minstr->SetMachineOperand(0, /*regNum %g0*/(unsigned int) 0);
  minstr->SetMachineOperand(1, MachineOperand::MO_VirtualRegister, vreg);
  minstr->SetMachineOperand(2, MachineOperand::MO_VirtualRegister, vreg);
  return minstr;
}


static inline MachineInstr* 
CreateMulConstInstruction(const InstructionNode* instrNode,
			  MachineInstr*& getMinstr2)
{
  MachineInstr* minstr = NULL;
  getMinstr2 = NULL;
  bool needNeg = false;

  Value* constOp = ((InstrTreeNode*) instrNode->rightChild())->getValue();
  assert(constOp->isConstant());
  
  // Cases worth optimizing are:
  // (1) Multiply by 0 or 1 for any type: replace with copy (ADD or FMOV)
  // (2) Multiply by 2^x for integer types: replace with Shift
  // 
  const Type* resultType = instrNode->getInstruction()->getType();
  
  if (resultType->isIntegral())
    {
      unsigned pow;
      bool isValidConst;
      int64_t C = GetConstantValueAsSignedInt(constOp, isValidConst);
      if (isValidConst)
	{
	  bool needNeg = false;
	  if (C < 0)
	    {
	      needNeg = true;
	      C = -C;
	    }
	  
	  if (C == 0 || C == 1)
	    {
	      minstr = new MachineInstr(ADD);
	      
	      if (C == 0)
		minstr->SetMachineOperand(0, /*regNum %g0*/ (unsigned int) 0);
	      else
		minstr->SetMachineOperand(0,MachineOperand::MO_VirtualRegister,
					  instrNode->leftChild()->getValue());
	      minstr->SetMachineOperand(1, /*regNum %g0*/ (unsigned int) 0);
	    }
	  else if (IsPowerOf2(C, pow))
	    {
	      minstr = new MachineInstr((resultType == Type::LongTy)
					? SLLX : SLL);
	      minstr->SetMachineOperand(0, MachineOperand::MO_VirtualRegister,
					   instrNode->leftChild()->getValue());
	      minstr->SetMachineOperand(1, MachineOperand::MO_UnextendedImmed,
					   pow);
	    }
	  
	  if (minstr && needNeg)
	    { // insert <reg = SUB 0, reg> after the instr to flip the sign
	      getMinstr2 = CreateIntNegInstruction(instrNode->getValue());
	    }
	}
    }
  else
    {
      if (resultType == Type::FloatTy ||
	  resultType == Type::DoubleTy)
	{
	  bool isValidConst;
	  double dval = ((ConstPoolFP*) constOp)->getValue();
	  
	  if (isValidConst)
	    {
	      if (dval == 0)
		{
		  minstr = new MachineInstr((resultType == Type::FloatTy)
					    ? FITOS : FITOD);
		  minstr->SetMachineOperand(0, /*regNum %g0*/(unsigned int) 0);
		}
	      else if (fabs(dval) == 1)
		{
		  bool needNeg = (dval < 0);
		  
		  MachineOpCode opCode = needNeg
		    ? (resultType == Type::FloatTy? FNEGS : FNEGD)
		    : (resultType == Type::FloatTy? FMOVS : FMOVD);
		  
		  minstr = new MachineInstr(opCode);
		  minstr->SetMachineOperand(0,
					   MachineOperand::MO_VirtualRegister,
					   instrNode->leftChild()->getValue());
		} 
	    }
	}
    }
  
  if (minstr != NULL)
    minstr->SetMachineOperand(2, MachineOperand::MO_VirtualRegister,
			      instrNode->getValue());   
  
  return minstr;
}


static inline MachineOpCode 
ChooseDivInstruction(const InstructionNode* instrNode)
{
  MachineOpCode opCode = INVALID_OPCODE;
  
  const Type* resultType = instrNode->getInstruction()->getType();
  
  if (resultType->isIntegral())
    {
      opCode = resultType->isSigned()? SDIVX : UDIVX;
    }
  else
    {
      Value* operand = ((InstrTreeNode*) instrNode->leftChild())->getValue();
      switch(operand->getType()->getPrimitiveID())
	{
	case Type::FloatTyID:  opCode = FDIVS; break;
	case Type::DoubleTyID: opCode = FDIVD; break;
	default: assert(0 && "Invalid type for DIV instruction"); break; 
	}
    }
  
  return opCode;
}


static inline MachineInstr* 
CreateDivConstInstruction(const InstructionNode* instrNode,
			  MachineInstr*& getMinstr2)
{
  MachineInstr* minstr = NULL;
  getMinstr2 = NULL;
  
  Value* constOp = ((InstrTreeNode*) instrNode->rightChild())->getValue();
  assert(constOp->isConstant());
  
  // Cases worth optimizing are:
  // (1) Divide by 1 for any type: replace with copy (ADD or FMOV)
  // (2) Divide by 2^x for integer types: replace with SR[L or A]{X}
  // 
  const Type* resultType = instrNode->getInstruction()->getType();
  
  if (resultType->isIntegral())
    {
      unsigned pow;
      bool isValidConst;
      int64_t C = GetConstantValueAsSignedInt(constOp, isValidConst);
      if (isValidConst)
	{
	  bool needNeg = false;
	  if (C < 0)
	    {
	      needNeg = true;
	      C = -C;
	    }
	  
	  if (C == 1)
	    {
	      minstr = new MachineInstr(ADD);
	      minstr->SetMachineOperand(0,MachineOperand::MO_VirtualRegister,
					  instrNode->leftChild()->getValue());
	      minstr->SetMachineOperand(1, /*regNum %g0*/ (unsigned int) 0);
	    }
	  else if (IsPowerOf2(C, pow))
	    {
	      MachineOpCode opCode= ((resultType->isSigned())
				     ? (resultType==Type::LongTy)? SRAX : SRA
				     : (resultType==Type::LongTy)? SRLX : SRL);
	      minstr = new MachineInstr(opCode);
	      minstr->SetMachineOperand(0, MachineOperand::MO_VirtualRegister,
					   instrNode->leftChild()->getValue());
	      minstr->SetMachineOperand(1, MachineOperand::MO_UnextendedImmed,
					   pow);
	    }
	  
	  if (minstr && needNeg)
	    { // insert <reg = SUB 0, reg> after the instr to flip the sign
	      getMinstr2 = CreateIntNegInstruction(instrNode->getValue());
	    }
	}
    }
  else
    {
      if (resultType == Type::FloatTy ||
	  resultType == Type::DoubleTy)
	{
	  bool isValidConst;
	  double dval = ((ConstPoolFP*) constOp)->getValue();
	  
	  if (isValidConst && fabs(dval) == 1)
	    {
	      bool needNeg = (dval < 0);
	      
	      MachineOpCode opCode = needNeg
		? (resultType == Type::FloatTy? FNEGS : FNEGD)
		: (resultType == Type::FloatTy? FMOVS : FMOVD);
	      
	      minstr = new MachineInstr(opCode);
	      minstr->SetMachineOperand(0, MachineOperand::MO_VirtualRegister,
					   instrNode->leftChild()->getValue());
	    } 
	}
    }
  
  if (minstr != NULL)
    minstr->SetMachineOperand(2, MachineOperand::MO_VirtualRegister,
			      instrNode->getValue());   
  
  return minstr;
}


static inline MachineOpCode 
ChooseLoadInstruction(const Type* resultType)
{
  MachineOpCode opCode = INVALID_OPCODE;
  
  switch (resultType->getPrimitiveID())
    {
    case Type::BoolTyID:	opCode = LDUB; break;
    case Type::UByteTyID:	opCode = LDUB; break;
    case Type::SByteTyID:	opCode = LDSB; break;
    case Type::UShortTyID:	opCode = LDUH; break;
    case Type::ShortTyID:	opCode = LDSH; break;
    case Type::UIntTyID:	opCode = LDUW; break;
    case Type::IntTyID:		opCode = LDSW; break;
    case Type::ULongTyID:
    case Type::LongTyID:	opCode = LDX;  break;
    case Type::FloatTyID:	opCode = LD;   break;
    case Type::DoubleTyID:	opCode = LDD;  break;
    default: assert(0 && "Invalid type for Load instruction"); break; 
    }
  
  return opCode;
}


static inline MachineOpCode 
ChooseStoreInstruction(const Type* valueType)
{
  MachineOpCode opCode = INVALID_OPCODE;
  
  switch (valueType->getPrimitiveID())
    {
    case Type::BoolTyID:
    case Type::UByteTyID:
    case Type::SByteTyID:	opCode = STB; break;
    case Type::UShortTyID:
    case Type::ShortTyID:	opCode = STH; break;
    case Type::UIntTyID:
    case Type::IntTyID:		opCode = STW; break;
    case Type::ULongTyID:
    case Type::LongTyID:	opCode = STX;  break;
    case Type::FloatTyID:	opCode = ST;   break;
    case Type::DoubleTyID:	opCode = STD;  break;
    default: assert(0 && "Invalid type for Store instruction"); break; 
    }
  
  return opCode;
}


//------------------------------------------------------------------------ 
// Function SetOperandsForMemInstr
//
// Choose addressing mode for the given load or store instruction.
// Use [reg+reg] if it is an indexed reference, and the index offset is
//		 not a constant or if it cannot fit in the offset field.
// Use [reg+offset] in all other cases.
// 
// This assumes that all array refs are "lowered" to one of these forms:
//	%x = load (subarray*) ptr, constant	; single constant offset
//	%x = load (subarray*) ptr, offsetVal	; single non-constant offset
// Generally, this should happen via strength reduction + LICM.
// Also, strength reduction should take care of using the same register for
// the loop index variable and an array index, when that is profitable.
//------------------------------------------------------------------------ 

static void
SetOperandsForMemInstr(MachineInstr* minstr,
		       const InstructionNode* vmInstrNode,
		       const TargetMachine& target)
{
  MemAccessInst* memInst = (MemAccessInst*) vmInstrNode->getInstruction();
  
  // Variables to hold the index vector, ptr value, and offset value.
  // The major work here is to extract these for all 3 instruction types
  // and then call the common function SetMemOperands_Internal().
  // 
  const vector<ConstPoolVal*>* idxVec = & memInst->getIndexVec();
  vector<ConstPoolVal*>* newIdxVec = NULL;
  Value* ptrVal;
  Value* arrayOffsetVal = NULL;
  
  // Test if a GetElemPtr instruction is being folded into this mem instrn.
  // If so, it will be in the left child for Load and GetElemPtr,
  // and in the right child for Store instructions.
  // 
  InstrTreeNode* ptrChild = (vmInstrNode->getOpLabel() == Instruction::Store
			     ? vmInstrNode->rightChild()
			     : vmInstrNode->leftChild()); 
  
  if (ptrChild->getOpLabel() == Instruction::GetElementPtr ||
      ptrChild->getOpLabel() == GetElemPtrIdx)
    {
      // There is a GetElemPtr instruction and there may be a chain of
      // more than one.  Use the pointer value of the last one in the chain.
      // Fold the index vectors from the entire chain and from the mem
      // instruction into one single index vector.
      // Finally, we never fold for an array instruction so make that NULL.
      
      newIdxVec = new vector<ConstPoolVal*>;
      ptrVal = FoldGetElemChain((InstructionNode*) ptrChild, *newIdxVec);
      
      newIdxVec->insert(newIdxVec->end(), idxVec->begin(), idxVec->end());
      idxVec = newIdxVec;
      
      assert(! ((PointerType*)ptrVal->getType())->getValueType()->isArrayType()
	     && "GetElemPtr cannot be folded into array refs in selection");
    }
  else
    {
      // There is no GetElemPtr instruction.
      // Use the pointer value and the index vector from the Mem instruction.
      // If it is an array reference, get the array offset value.
      // 
      ptrVal = memInst->getPtrOperand();

      const Type* opType =
	((const PointerType*) ptrVal->getType())->getValueType();
      if (opType->isArrayType())
	{
	  assert((memInst->getNumOperands()
		  == (unsigned) 1 + memInst->getFirstOffsetIdx())
		 && "Array refs must be lowered before Instruction Selection");
	  
	  arrayOffsetVal = memInst->getOperand(memInst->getFirstOffsetIdx());
	}
    }
  
  SetMemOperands_Internal(minstr, vmInstrNode, ptrVal, arrayOffsetVal,
			  *idxVec, target);
  
  if (newIdxVec != NULL)
    delete newIdxVec;
}


static void
SetMemOperands_Internal(MachineInstr* minstr,
			const InstructionNode* vmInstrNode,
			Value* ptrVal,
			Value* arrayOffsetVal,
			const vector<ConstPoolVal*>& idxVec,
			const TargetMachine& target)
{
  MemAccessInst* memInst = (MemAccessInst*) vmInstrNode->getInstruction();
  
  // Initialize so we default to storing the offset in a register.
  int64_t smallConstOffset;
  Value* valueForRegOffset = NULL;
  MachineOperand::MachineOperandType offsetOpType =MachineOperand::MO_VirtualRegister;

  // Check if there is an index vector and if so, if it translates to
  // a small enough constant to fit in the immediate-offset field.
  // 
  if (idxVec.size() > 0)
    {
      bool isConstantOffset = false;
      unsigned offset;
      
      const PointerType* ptrType = (PointerType*) ptrVal->getType();
      
      if (ptrType->getValueType()->isStructType())
	{
	  // the offset is always constant for structs
	  isConstantOffset = true;
	  
	  // Compute the offset value using the index vector
	  offset = target.DataLayout.getIndexedOffset(ptrType, idxVec);
	}
      else
	{
	  // It must be an array ref.  Check if the offset is a constant,
	  // and that the indexing has been lowered to a single offset.
	  // 
	  assert(ptrType->getValueType()->isArrayType());
	  assert(arrayOffsetVal != NULL
		 && "Expect to be given Value* for array offsets");
	  
	  if (ConstPoolVal *CPV = arrayOffsetVal->castConstant())
	    {
	      isConstantOffset = true;	// always constant for structs
	      assert(arrayOffsetVal->getType()->isIntegral());
	      offset = (CPV->getType()->isSigned()
			? ((ConstPoolSInt*)CPV)->getValue()
			: (int64_t) ((ConstPoolUInt*)CPV)->getValue());
	    }
	  else
	    {
	      valueForRegOffset = arrayOffsetVal;
	    }
	}
      
      if (isConstantOffset)
	{
	  // create a virtual register for the constant
	  valueForRegOffset = ConstPoolSInt::get(Type::IntTy, offset);
	}
    }
  else
    {
      offsetOpType = MachineOperand::MO_SignExtendedImmed;
      smallConstOffset = 0;
    }
  
  // Operand 0 is value for STORE, ptr for LOAD or GET_ELEMENT_PTR
  // It is the left child in the instruction tree in all cases.
  Value* leftVal = vmInstrNode->leftChild()->getValue();
  minstr->SetMachineOperand(0, MachineOperand::MO_VirtualRegister, leftVal);
  
  // Operand 1 is ptr for STORE, offset for LOAD or GET_ELEMENT_PTR
  // Operand 3 is offset for STORE, result reg for LOAD or GET_ELEMENT_PTR
  //
  unsigned offsetOpNum = (memInst->getOpcode() == Instruction::Store)? 2 : 1;
  if (offsetOpType == MachineOperand::MO_VirtualRegister)
    {
      assert(valueForRegOffset != NULL);
      minstr->SetMachineOperand(offsetOpNum, offsetOpType, valueForRegOffset); 
    }
  else
    minstr->SetMachineOperand(offsetOpNum, offsetOpType, smallConstOffset);
  
  if (memInst->getOpcode() == Instruction::Store)
    minstr->SetMachineOperand(1, MachineOperand::MO_VirtualRegister, ptrVal);
  else
    minstr->SetMachineOperand(2, MachineOperand::MO_VirtualRegister,
			         vmInstrNode->getValue());
}


// Special handling for constant operands:
// -- if the constant is 0, use the hardwired 0 register, if any;
// -- if the constant is of float or double type but has an integer value,
//    use int-to-float conversion instruction instead of generating a load;
// -- if the constant fits in the IMMEDIATE field, use that field;
// -- else insert instructions to put the constant into a register, either
//    directly or by loading explicitly from the constant pool.
// 
static unsigned
FixConstantOperands(const InstructionNode* vmInstrNode,
		    MachineInstr** mvec,
		    unsigned numInstr,
		    TargetMachine& target)
{
  static MachineInstr* loadConstVec[MAX_INSTR_PER_VMINSTR];

  unsigned numNew = 0;
  Instruction* vmInstr = vmInstrNode->getInstruction();
  
  for (unsigned i=0; i < numInstr; i++)
    {
      MachineInstr* minstr = mvec[i];
      const MachineInstrDescriptor& instrDesc =
	target.getInstrInfo().getDescriptor(minstr->getOpCode());
      
      for (unsigned op=0; op < minstr->getNumOperands(); op++)
	{
	  const MachineOperand& mop = minstr->getOperand(op);
	  
	  // skip the result position (for efficiency below) and any other
	  // positions already marked as not a virtual register
	  if (instrDesc.resultPos == (int) op || 
	      mop.getOperandType() != MachineOperand::MO_VirtualRegister ||
	      mop.getVRegValue() == NULL)
	    {
	      break;
	    }
	  
	  Value* opValue = mop.getVRegValue();
	  
	  if (opValue->isConstant())
	    {
	      unsigned int machineRegNum;
	      int64_t immedValue;
	      MachineOperand::MachineOperandType opType =
		ChooseRegOrImmed(opValue, minstr->getOpCode(), target,
				 /*canUseImmed*/ (op == 1),
				 machineRegNum, immedValue);
	      
	      if (opType == MachineOperand::MO_MachineRegister)
		minstr->SetMachineOperand(op, machineRegNum);
	      else if (opType == MachineOperand::MO_VirtualRegister)
		{
		  // value is constant and must be loaded into a register
		  TmpInstruction* tmpReg;
		  MachineInstr* minstr2;
		  loadConstVec[numNew++] = MakeLoadConstInstr(vmInstr, opValue,
							      tmpReg, minstr2);
		  minstr->SetMachineOperand(op, opType, tmpReg);
		  if (minstr2 != NULL)
		    loadConstVec[numNew++] = minstr2;
		}
	      else
		minstr->SetMachineOperand(op, opType, immedValue);
	    }
	}
    }
  
  if (numNew > 0)
    {
      // Insert the new instructions *before* the old ones by moving
      // the old ones over `numNew' positions (last-to-first, of course!).
      // We do check *after* returning that we did not exceed the vector mvec.
      for (int i=numInstr-1; i >= 0; i--)
	mvec[i+numNew] = mvec[i];
      
      for (unsigned i=0; i < numNew; i++)
	mvec[i] = loadConstVec[i];
    }
  
  return (numInstr + numNew);
}


static inline MachineInstr*
MakeIntSetInstruction(int64_t C, bool isSigned, Value* dest)
{
  MachineInstr* minstr;
  if (isSigned)
    {
      minstr = new MachineInstr(SETSW);
      minstr->SetMachineOperand(0, MachineOperand::MO_SignExtendedImmed, C);
    }
  else
    {
      minstr = new MachineInstr(SETUW);
      minstr->SetMachineOperand(0, MachineOperand::MO_UnextendedImmed, C);
    }
  
  minstr->SetMachineOperand(1, MachineOperand::MO_VirtualRegister, dest);
  
  return minstr;
}


static MachineInstr*
MakeLoadConstInstr(Instruction* vmInstr,
		   Value* val,
		   TmpInstruction*& tmpReg,
		   MachineInstr*& getMinstr2)
{
  assert(val->isConstant());
  
  MachineInstr* minstr;
  
  getMinstr2 = NULL;
  
  // Create a TmpInstruction to mark the hidden register used for the constant
  tmpReg = new TmpInstruction(Instruction::UserOp1, val, NULL);
  vmInstr->getMachineInstrVec().addTempValue(tmpReg);
  
  // Use a "set" instruction for known constants that can go in an integer reg.
  // Use a "set" instruction followed by a int-to-float conversion for known
  // constants that must go in a floating point reg but have an integer value.
  // Use a "load" instruction for all other constants, in particular,
  // floating point constants.
  // 
  const Type* valType = val->getType();
  
  if (valType->isIntegral() ||
      valType->isPointerType() ||
      valType == Type::BoolTy)
    {
      bool isValidConstant;
      int64_t C = GetConstantValueAsSignedInt(val, isValidConstant);
      assert(isValidConstant && "Unrecognized constant");
      
      minstr = MakeIntSetInstruction(C, valType->isSigned(), tmpReg);
    }
  else
    {
      assert(valType == Type::FloatTy || valType == Type::DoubleTy);
      double dval = ((ConstPoolFP*) val)->getValue();
      if (dval == (int64_t) dval)
	{
	  // The constant actually has an integer value, so use a
	  // [set; int-to-float] sequence instead of a load instruction.
	  // 
	  TmpInstruction* tmpReg2 = NULL;
	  if (dval != 0.0)
	    { // First, create an integer constant of the same value as dval
	      ConstPoolSInt* ival = ConstPoolSInt::get(Type::IntTy,
						       (int64_t) dval);
	      // Create another TmpInstruction for the hidden integer register
	      TmpInstruction* tmpReg2 =
		new TmpInstruction(Instruction::UserOp1, ival, NULL);
	      vmInstr->getMachineInstrVec().addTempValue(tmpReg2);
	  
	      // Create the `SET' instruction
	      minstr = MakeIntSetInstruction((int64_t)dval, true, tmpReg2);
	    }
	  
	  // In which variable do we put the second instruction?
	  MachineInstr*& instr2 = (minstr)? getMinstr2 : minstr;
	  
	  // Create the int-to-float instruction
	  instr2 = new MachineInstr(valType == Type::FloatTy? FITOS : FITOD);
	  
	  if (dval == 0.0)
	    instr2->SetMachineOperand(0, /*regNum %g0*/ (unsigned int) 0);
	  else
	    instr2->SetMachineOperand(0,MachineOperand::MO_VirtualRegister,
					  tmpReg2);
	  
	  instr2->SetMachineOperand(1, MachineOperand::MO_VirtualRegister,
					   tmpReg);
	}
      else
	{
	  // Make a Load instruction, and make `val' both the ptr value *and*
	  // the result value, and set the offset field to 0.  Final code
	  // generation will have to generate the base+offset for the constant.
	  // 
	  int64_t zeroOffset = 0; // to avoid ambiguity with (Value*) 0
	  minstr = new MachineInstr(ChooseLoadInstruction(val->getType()));
	  minstr->SetMachineOperand(0, MachineOperand::MO_VirtualRegister,val);
	  minstr->SetMachineOperand(1, MachineOperand::MO_SignExtendedImmed,
				       zeroOffset);
	  minstr->SetMachineOperand(2, MachineOperand::MO_VirtualRegister,
				       tmpReg);
	}
    }
  
  tmpReg->addMachineInstruction(minstr);
  
  assert(minstr);
  return minstr;
}

// 
// Substitute operand `operandNum' of the instruction in node `treeNode'
// in place the use(s) of that instruction in node `parent'.
// 
static void
ForwardOperand(InstructionNode* treeNode,
	       InstructionNode* parent,
	       int operandNum)
{
  Instruction* unusedOp = treeNode->getInstruction();
  Value* fwdOp = unusedOp->getOperand(operandNum);
  Instruction* userInstr = parent->getInstruction();
  MachineCodeForVMInstr& mvec = userInstr->getMachineInstrVec();
  for (unsigned i=0, N=mvec.size(); i < N; i++)
    {
      MachineInstr* minstr = mvec[i];
      for (unsigned i=0, numOps=minstr->getNumOperands(); i < numOps; i++)
	{
	  const MachineOperand& mop = minstr->getOperand(i);
	  if (mop.getOperandType() == MachineOperand::MO_VirtualRegister &&
	      mop.getVRegValue() == unusedOp)
	    {
	      minstr->SetMachineOperand(i, MachineOperand::MO_VirtualRegister,
					   fwdOp);
	    }
	}
    }
}


// This function is currently unused and incomplete but will be 
// used if we have a linear layout of basic blocks in LLVM code.
// It decides which branch should fall-through, and whether an
// extra unconditional branch is needed (when neither falls through).
// 
void
ChooseBranchPattern(Instruction* vmInstr, BranchPattern& brPattern)
{
  BranchInst* brInstr = (BranchInst*) vmInstr;
  
  brPattern.flipCondition = false;
  brPattern.targetBB	  = brInstr->getSuccessor(0);
  brPattern.extraBranch   = NULL;
  
  assert(brInstr->getNumSuccessors() > 1 &&
	 "Unnecessary analysis for unconditional branch");
  
  assert(0 && "Fold branches in peephole optimization");
}


//******************* Externally Visible Functions *************************/


//------------------------------------------------------------------------ 
// External Function: GetInstructionsByRule
//
// Purpose:
//   Choose machine instructions for the SPARC according to the
//   patterns chosen by the BURG-generated parser.
//------------------------------------------------------------------------ 

unsigned
GetInstructionsByRule(InstructionNode* subtreeRoot,
		      int ruleForNode,
		      short* nts,
		      TargetMachine &target,
		      MachineInstr** mvec)
{
  int numInstr = 1;			// initialize for common case
  bool checkCast = false;		// initialize here to use fall-through
  Value *leftVal, *rightVal;
  const Type* opType;
  int nextRule;
  int forwardOperandNum = -1;
  BranchPattern brPattern;
  int64_t s0 = 0;			// variables holding zero to avoid
  uint64_t u0 = 0;			// overloading ambiguities below
  
  mvec[0] = mvec[1] = mvec[2] = mvec[3] = NULL;	// just for safety
  
  switch(ruleForNode) {
  case 1:	// stmt:   Ret
  case 2:	// stmt:   RetValue(reg)
		// NOTE: Prepass of register allocation is responsible
		//	 for moving return value to appropriate register.
		// Mark the return-address register as a hidden virtual reg.
    {		
    Instruction* returnReg = new TmpInstruction(Instruction::UserOp1,
					subtreeRoot->getInstruction(), NULL);
    subtreeRoot->getInstruction()->getMachineInstrVec().addTempValue(returnReg);
    
    mvec[0] = new MachineInstr(RETURN);
    mvec[0]->SetMachineOperand(0,MachineOperand::MO_VirtualRegister,returnReg);
    mvec[0]->SetMachineOperand(1, MachineOperand::MO_SignExtendedImmed, s0);
    
    returnReg->addMachineInstruction(mvec[0]);
    
    mvec[numInstr++] = new MachineInstr(NOP); // delay slot
    break;
    }  
    
  case 3:	// stmt:   Store(reg,reg)
  case 4:	// stmt:   Store(reg,ptrreg)
    mvec[0] = new MachineInstr(ChooseStoreInstruction(subtreeRoot->leftChild()->getValue()->getType()));
    SetOperandsForMemInstr(mvec[0], subtreeRoot, target);
    break;
    
  case 5:	// stmt:   BrUncond
    mvec[0] = new MachineInstr(BA);
    mvec[0]->SetMachineOperand(0, MachineOperand::MO_CCRegister, (Value*)NULL);
    mvec[0]->SetMachineOperand(1, MachineOperand::MO_PCRelativeDisp,
	      ((BranchInst*) subtreeRoot->getInstruction())->getSuccessor(0));
    
    // delay slot
    mvec[numInstr++] = new MachineInstr(NOP);
    break;
    
  case 6:	// stmt:   BrCond(boolconst)
    // boolconst => boolean was computed with `%b = setCC type reg1 constant'
    // If the constant is ZERO, we can use the branch-on-integer-register
    // instructions and avoid the SUBcc instruction entirely.
    // Otherwise this is just the same as case 5, so just fall through.
    {
    InstrTreeNode* constNode = subtreeRoot->leftChild()->rightChild();
    assert(constNode && constNode->getNodeType() ==InstrTreeNode::NTConstNode);
    ConstPoolVal* constVal = (ConstPoolVal*) constNode->getValue();
    bool isValidConst;
    
    if (constVal->getType()->isIntegral()
	&& GetConstantValueAsSignedInt(constVal, isValidConst) == 0
	&& isValidConst)
      {
	// That constant ia a zero after all...
	// Use the left child of the setCC instruction as the first argument!
	mvec[0] = new MachineInstr(ChooseBprInstruction(subtreeRoot));
	mvec[0]->SetMachineOperand(0, MachineOperand::MO_VirtualRegister,
		      subtreeRoot->leftChild()->leftChild()->getValue());
	mvec[0]->SetMachineOperand(1, MachineOperand::MO_PCRelativeDisp,
	      ((BranchInst*) subtreeRoot->getInstruction())->getSuccessor(0));

	// delay slot
	mvec[numInstr++] = new MachineInstr(NOP);
	
	// false branch
	mvec[numInstr++] = new MachineInstr(BA);
	mvec[numInstr-1]->SetMachineOperand(0, MachineOperand::MO_CCRegister,
						(Value*) NULL);
	mvec[numInstr-1]->SetMachineOperand(1, MachineOperand::MO_PCRelativeDisp, ((BranchInst*) subtreeRoot->getInstruction())->getSuccessor(1));

	// delay slot
	mvec[numInstr++] = new MachineInstr(NOP);
	
	break;
      }
    // ELSE FALL THROUGH
    }
    
  case 7:	// stmt:   BrCond(bool)
    // bool => boolean was computed with `%b = setcc type reg1 reg2'
    // Need to check whether the type was a FP, signed int or unsigned int,
    // and check the branching condition in order to choose the branch to use.
    // 
    {
    bool isFPBranch;
    mvec[0] = new MachineInstr(ChooseBccInstruction(subtreeRoot, isFPBranch));
    mvec[0]->SetMachineOperand(0, MachineOperand::MO_CCRegister,
			          subtreeRoot->leftChild()->getValue());
    mvec[0]->SetMachineOperand(1, MachineOperand::MO_PCRelativeDisp,
	      ((BranchInst*) subtreeRoot->getInstruction())->getSuccessor(0));
    
    // delay slot
    mvec[numInstr++] = new MachineInstr(NOP);
    
    // false branch
    mvec[numInstr++] = new MachineInstr(BA);
    mvec[numInstr-1]->SetMachineOperand(0, MachineOperand::MO_CCRegister,
					   (Value*) NULL);
    mvec[numInstr-1]->SetMachineOperand(0, MachineOperand::MO_PCRelativeDisp,
	      ((BranchInst*) subtreeRoot->getInstruction())->getSuccessor(1));
    
    // delay slot
    mvec[numInstr++] = new MachineInstr(NOP);
    break;
    }
    
  case 8:	// stmt:   BrCond(boolreg)
    // bool => boolean is stored in an existing register.
    // Just use the branch-on-integer-register instruction!
    // 
    mvec[0] = new MachineInstr(BRNZ);
    mvec[0]->SetMachineOperand(0, MachineOperand::MO_VirtualRegister,
			          subtreeRoot->leftChild()->getValue());
    mvec[0]->SetMachineOperand(1, MachineOperand::MO_PCRelativeDisp,
	      ((BranchInst*) subtreeRoot->getInstruction())->getSuccessor(0));
    
    // delay slot
    mvec[numInstr++] = new MachineInstr(NOP); // delay slot

    // false branch
    mvec[numInstr++] = new MachineInstr(BA);
    mvec[numInstr-1]->SetMachineOperand(0, MachineOperand::MO_CCRegister,
					   (Value*) NULL);
    mvec[numInstr-1]->SetMachineOperand(0, MachineOperand::MO_PCRelativeDisp,
	      ((BranchInst*) subtreeRoot->getInstruction())->getSuccessor(1));
    
    // delay slot
    mvec[numInstr++] = new MachineInstr(NOP);
    break;
    
  case 9:	// stmt:   Switch(reg)
    assert(0 && "*** SWITCH instruction is not implemented yet.");
    numInstr = 0;
    break;
    
  case 10:	// reg:   VRegList(reg, reg)
    assert(0 && "VRegList should never be the topmost non-chain rule");
    break;
    
  case 21:	// reg:   Not(reg):	Implemented as reg = reg XOR-NOT 0
    mvec[0] = new MachineInstr(XNOR);
    mvec[0]->SetMachineOperand(0, MachineOperand::MO_VirtualRegister,
			          subtreeRoot->leftChild()->getValue());
    mvec[0]->SetMachineOperand(1, /*regNum %g0*/ (unsigned int) 0);
    mvec[0]->SetMachineOperand(2, MachineOperand::MO_VirtualRegister,
				 subtreeRoot->getValue());
    break;
    
  case 22:	// reg:   ToBoolTy(reg):
    opType = subtreeRoot->leftChild()->getValue()->getType();
    assert(opType->isIntegral() || opType == Type::BoolTy);
    numInstr = 0;
    forwardOperandNum = 0;
    break;
    
  case 23:	// reg:   ToUByteTy(reg)
  case 25:	// reg:   ToUShortTy(reg)
  case 27:	// reg:   ToUIntTy(reg)
  case 29:	// reg:   ToULongTy(reg)
    opType = subtreeRoot->leftChild()->getValue()->getType();
    assert(opType->isIntegral() ||
	   opType->isPointerType() ||
	   opType == Type::BoolTy && "Ignoring cast: illegal for other types");
    numInstr = 0;
    forwardOperandNum = 0;
    break;
    
  case 24:	// reg:   ToSByteTy(reg)
  case 26:	// reg:   ToShortTy(reg)
  case 28:	// reg:   ToIntTy(reg)
  case 30:	// reg:   ToLongTy(reg)
    opType = subtreeRoot->leftChild()->getValue()->getType();
    if (opType->isIntegral() || opType == Type::BoolTy)
      {
	numInstr = 0;
	forwardOperandNum = 0;
      }
    else
      {
	mvec[0] =new MachineInstr(ChooseConvertToIntInstr(subtreeRoot,opType));
	Set2OperandsFromInstr(mvec[0], subtreeRoot, target);
      }
    break;
    
  case 31:	// reg:   ToFloatTy(reg):
  case 32:	// reg:   ToDoubleTy(reg):
    
    // If this instruction has a parent (a user) in the tree 
    // and the user is translated as an FsMULd instruction,
    // then the cast is unnecessary.  So check that first.
    // In the future, we'll want to do the same for the FdMULq instruction,
    // so do the check here instead of only for ToFloatTy(reg).
    // 
    if (subtreeRoot->parent() != NULL &&
	((InstructionNode*) subtreeRoot->parent())->getInstruction()->getMachineInstrVec()[0]->getOpCode() == FSMULD)
      {
	numInstr = 0;
	forwardOperandNum = 0;
      }
    else
      {
	opType = subtreeRoot->leftChild()->getValue()->getType();
	MachineOpCode opCode = ChooseConvertToFloatInstr(subtreeRoot, opType);
	if (opCode == INVALID_OPCODE)	// no conversion needed
	  {
	    numInstr = 0;
	    forwardOperandNum = 0;
	  }
	else
	  {
	    mvec[0] = new MachineInstr(opCode);
	    Set2OperandsFromInstr(mvec[0], subtreeRoot, target);
	  }
      }
    break;
    
  case 19:	// reg:   ToArrayTy(reg):
  case 20:	// reg:   ToPointerTy(reg):
    numInstr = 0;
    forwardOperandNum = 0;
    break;
    
  case 233:	// reg:   Add(reg, Constant)
    mvec[0] = CreateAddConstInstruction(subtreeRoot);
    if (mvec[0] != NULL)
      break;
    // ELSE FALL THROUGH
    
  case 33:	// reg:   Add(reg, reg)
    mvec[0] = new MachineInstr(ChooseAddInstruction(subtreeRoot));
    Set3OperandsFromInstr(mvec[0], subtreeRoot, target);
    break;
    
  case 234:	// reg:   Sub(reg, Constant)
    mvec[0] = CreateSubConstInstruction(subtreeRoot);
    if (mvec[0] != NULL)
      break;
    // ELSE FALL THROUGH
    
  case 34:	// reg:   Sub(reg, reg)
    mvec[0] = new MachineInstr(ChooseSubInstruction(subtreeRoot));
    Set3OperandsFromInstr(mvec[0], subtreeRoot, target);
    break;
    
  case 135:	// reg:   Mul(todouble, todouble)
    checkCast = true;
    // FALL THROUGH 
    
  case 35:	// reg:   Mul(reg, reg)
    mvec[0] = new MachineInstr(ChooseMulInstruction(subtreeRoot, checkCast));
    Set3OperandsFromInstr(mvec[0], subtreeRoot, target);
    break;

  case 335:	// reg:   Mul(todouble, todoubleConst)
    checkCast = true;
    // FALL THROUGH 
   
  case 235:	// reg:   Mul(reg, Constant)
    mvec[0] = CreateMulConstInstruction(subtreeRoot, mvec[1]);
    if (mvec[0] == NULL)
      {
	mvec[0]=new MachineInstr(ChooseMulInstruction(subtreeRoot, checkCast));
	Set3OperandsFromInstr(mvec[0], subtreeRoot, target);
      }
    else
      if (mvec[1] != NULL)
	++numInstr;
    break;
    
  case 236:	// reg:   Div(reg, Constant)
    mvec[0] = CreateDivConstInstruction(subtreeRoot, mvec[1]);
    if (mvec[0] != NULL)
      {
	if (mvec[1] != NULL)
	  ++numInstr;
      }
    else
    // ELSE FALL THROUGH
    
  case 36:	// reg:   Div(reg, reg)
    mvec[0] = new MachineInstr(ChooseDivInstruction(subtreeRoot));
    Set3OperandsFromInstr(mvec[0], subtreeRoot, target);
    break;
    
  case  37:	// reg:   Rem(reg, reg)
  case 237:	// reg:   Rem(reg, Constant)
    assert(0 && "REM instruction unimplemented for the SPARC.");
    break;
    
  case  38:	// reg:   And(reg, reg)
  case 238:	// reg:   And(reg, Constant)
    mvec[0] = new MachineInstr(AND);
    Set3OperandsFromInstr(mvec[0], subtreeRoot, target);
    break;
    
  case 138:	// reg:   And(reg, not)
    mvec[0] = new MachineInstr(ANDN);
    Set3OperandsFromInstr(mvec[0], subtreeRoot, target);
    break;
    
  case  39:	// reg:   Or(reg, reg)
  case 239:	// reg:   Or(reg, Constant)
    mvec[0] = new MachineInstr(ORN);
    Set3OperandsFromInstr(mvec[0], subtreeRoot, target);
    break;
    
  case 139:	// reg:   Or(reg, not)
    mvec[0] = new MachineInstr(ORN);
    Set3OperandsFromInstr(mvec[0], subtreeRoot, target);
    break;
    
  case  40:	// reg:   Xor(reg, reg)
  case 240:	// reg:   Xor(reg, Constant)
    mvec[0] = new MachineInstr(XOR);
    Set3OperandsFromInstr(mvec[0], subtreeRoot, target);
    break;
    
  case 140:	// reg:   Xor(reg, not)
    mvec[0] = new MachineInstr(XNOR);
    Set3OperandsFromInstr(mvec[0], subtreeRoot, target);
    break;
    
  case 41:	// boolconst:   SetCC(reg, Constant)
    // Check if this is an integer comparison, and
    // there is a parent, and the parent decided to use
    // a branch-on-integer-register instead of branch-on-condition-code.
    // If so, the SUBcc instruction is not required.
    // (However, we must still check for constants to be loaded from
    // the constant pool so that such a load can be associated with
    // this instruction.)
    // 
    // Otherwise this is just the same as case 42, so just fall through.
    // 
    if (subtreeRoot->leftChild()->getValue()->getType()->isIntegral() &&
	subtreeRoot->parent() != NULL)
      {
	InstructionNode* parentNode = (InstructionNode*) subtreeRoot->parent();
	assert(parentNode->getNodeType() == InstrTreeNode::NTInstructionNode);
	const vector<MachineInstr*>&
	  minstrVec = parentNode->getInstruction()->getMachineInstrVec();
	MachineOpCode parentOpCode;
	if (parentNode->getInstruction()->getOpcode() == Instruction::Br &&
	    (parentOpCode = minstrVec[0]->getOpCode()) >= BRZ &&
	    parentOpCode <= BRGEZ)
	  {
	    numInstr = 0;		// don't forward the operand!
	    break;
	  }
      }
    // ELSE FALL THROUGH
    
  case 42:	// bool:   SetCC(reg, reg):
  {
    // If result of the SetCC is only used for a branch, we can
    // discard the result. otherwise, it must go into an integer register.
    // Note that the user may or may not be in the same tree, so we have
    // to follow SSA def-use edges here, not BURG tree edges.
    // 
    Instruction* result = subtreeRoot->getInstruction();
    Value* firstUse = (Value*) * result->use_begin();
    bool discardResult =
      (result->use_size() == 1
       && firstUse->isInstruction()
       && ((Instruction*) firstUse)->getOpcode() == Instruction::Br);
    
    bool mustClearReg;
    int valueToMove;
    MachineOpCode movOpCode;
    
    if (subtreeRoot->leftChild()->getValue()->getType()->isIntegral())
      {
	// integer condition: destination should be %g0 or integer register
	// if result must be saved but condition is not SetEQ then we need
	// a separate instruction to compute the bool result, so discard
	// result of SUBcc instruction anyway.
	// 
	mvec[0] = new MachineInstr(SUBcc);
	Set3OperandsFromInstr(mvec[0], subtreeRoot, target, discardResult);
	
	// mark the 4th operand as being a CC register, and a "result"
	mvec[0]->SetMachineOperand(3, MachineOperand::MO_CCRegister,
				      subtreeRoot->getValue(), /*def*/ true);
	
	if (!discardResult) 
	  { // recompute bool if needed, using the integer condition codes
	    if (result->getOpcode() == Instruction::SetNE)
	      discardResult = true;
	    else
	      movOpCode =
		ChooseMovpccAfterSub(subtreeRoot, mustClearReg, valueToMove);
	  }
      }
    else
      {
	// FP condition: dest of FCMP should be some FCCn register
	mvec[0] = new MachineInstr(ChooseFcmpInstruction(subtreeRoot));
	mvec[0]->SetMachineOperand(0, MachineOperand::MO_CCRegister,
				      subtreeRoot->getValue());
	mvec[0]->SetMachineOperand(1, MachineOperand::MO_VirtualRegister,
				      subtreeRoot->leftChild()->getValue());
	mvec[0]->SetMachineOperand(2, MachineOperand::MO_VirtualRegister,
				      subtreeRoot->rightChild()->getValue());
	
	if (!discardResult)
	  {// recompute bool using the FP condition codes
	    mustClearReg = true;
	    valueToMove = 1;
	    movOpCode = ChooseMovFpccInstruction(subtreeRoot);
	  }
      }
    
    if (!discardResult)
      {
	if (mustClearReg)
	  {// Unconditionally set register to 0
	   int n = numInstr++;
	   mvec[n] = new MachineInstr(SETHI);
	   mvec[n]->SetMachineOperand(0,MachineOperand::MO_UnextendedImmed,s0);
	   mvec[n]->SetMachineOperand(1, MachineOperand::MO_VirtualRegister,
				         subtreeRoot->getValue());
	  }
	
	// Now conditionally move `valueToMove' (0 or 1) into the register
	int n = numInstr++;
	mvec[n] = new MachineInstr(movOpCode);
	mvec[n]->SetMachineOperand(0, MachineOperand::MO_CCRegister,
				      subtreeRoot->getValue());
	mvec[n]->SetMachineOperand(1, MachineOperand::MO_UnextendedImmed,
				      valueToMove);
	mvec[n]->SetMachineOperand(2, MachineOperand::MO_VirtualRegister,
				      subtreeRoot->getValue());
      }
    break;
  }    
  
  case 43:	// boolreg: VReg
  case 44:	// boolreg: Constant
    numInstr = 0;
    break;

  case 51:	// reg:   Load(reg)
  case 52:	// reg:   Load(ptrreg)
  case 53:	// reg:   LoadIdx(reg,reg)
  case 54:	// reg:   LoadIdx(ptrreg,reg)
    mvec[0] = new MachineInstr(ChooseLoadInstruction(subtreeRoot->getValue()->getType()));
    SetOperandsForMemInstr(mvec[0], subtreeRoot, target);
    break;
    
  case 55:	// reg:   GetElemPtr(reg)
  case 56:	// reg:   GetElemPtrIdx(reg,reg)
    if (subtreeRoot->parent() != NULL)
      {
	// Check if the parent was an array access.
	// If so, we still need to generate this instruction.
	MemAccessInst* memInst =(MemAccessInst*) subtreeRoot->getInstruction();
	const PointerType* ptrType =
	  (const PointerType*) memInst->getPtrOperand()->getType();
	if (! ptrType->getValueType()->isArrayType())
	  {// we don't need a separate instr
	    numInstr = 0;		// don't forward operand!
	    break;
	  }
      }
    // else in all other cases we need to a separate ADD instruction
    mvec[0] = new MachineInstr(ADD);
    SetOperandsForMemInstr(mvec[0], subtreeRoot, target);
    break;
    
  case 57:	// reg:   Alloca: Implement as 2 instructions:
    		//	sub %sp, tmp -> %sp
    {		//	add %sp, 0   -> result
    Instruction* instr = subtreeRoot->getInstruction();
    const PointerType* instrType = (const PointerType*) instr->getType();
    assert(instrType->isPointerType());
    int tsize = (int) target.findOptimalStorageSize(instrType->getValueType());
    assert(tsize != 0 && "Just to check when this can happen");
    // if (tsize == 0)
    //   {
	// numInstr = 0;
	// break;
    // }
    //else go on to create the instructions needed...
    
    // Create a temporary Value to hold the constant type-size
    ConstPoolSInt* valueForTSize = ConstPoolSInt::get(Type::IntTy, tsize);
    
    // Instruction 1: sub %sp, tsize -> %sp
    // tsize is always constant, but it may have to be put into a
    // register if it doesn't fit in the immediate field.
    // 
    mvec[0] = new MachineInstr(SUB);
    mvec[0]->SetMachineOperand(0, /*regNum %sp = o6 = r[14]*/(unsigned int)14);
    mvec[0]->SetMachineOperand(1, MachineOperand::MO_VirtualRegister, valueForTSize);
    mvec[0]->SetMachineOperand(2, /*regNum %sp = o6 = r[14]*/(unsigned int)14);
    
    // Instruction 2: add %sp, 0 -> result
    numInstr++;
    mvec[1] = new MachineInstr(ADD);
    mvec[1]->SetMachineOperand(0, /*regNum %sp = o6 = r[14]*/(unsigned int)14);
    mvec[1]->SetMachineOperand(1, /*regNum %g0*/ (unsigned int) 0);
    mvec[1]->SetMachineOperand(2, MachineOperand::MO_VirtualRegister, instr);
    break;
    }
  
  case 58:	// reg:   Alloca(reg): Implement as 3 instructions:
    		//	mul num, typeSz -> tmp
    		//	sub %sp, tmp    -> %sp
    {		//	add %sp, 0      -> result
    Instruction* instr = subtreeRoot->getInstruction();
    const PointerType* instrType = (const PointerType*) instr->getType();
    assert(instrType->isPointerType() &&
	   instrType->getValueType()->isArrayType());
    const Type* eltType =
      ((ArrayType*) instrType->getValueType())->getElementType();
    int tsize = (int) target.findOptimalStorageSize(eltType);
    
    assert(tsize != 0 && "Just to check when this can happen");
    // if (tsize == 0)
      // {
	// numInstr = 0;
	// break;
      // }
    //else go on to create the instructions needed...

    // Create a temporary Value to hold the constant type-size
    ConstPoolSInt* valueForTSize = ConstPoolSInt::get(Type::IntTy, tsize);
    
    // Create a temporary value to hold `tmp'
    Instruction* tmpInstr = new TmpInstruction(Instruction::UserOp1,
					  subtreeRoot->leftChild()->getValue(),
					  NULL /*could insert tsize here*/);
    subtreeRoot->getInstruction()->getMachineInstrVec().addTempValue(tmpInstr);

    // Instruction 1: mul numElements, typeSize -> tmp
    mvec[0] = new MachineInstr(MULX);
    mvec[0]->SetMachineOperand(0, MachineOperand::MO_VirtualRegister,
				subtreeRoot->leftChild()->getValue());
    mvec[0]->SetMachineOperand(1, MachineOperand::MO_VirtualRegister, valueForTSize);
    mvec[0]->SetMachineOperand(2, MachineOperand::MO_VirtualRegister,tmpInstr);

    tmpInstr->addMachineInstruction(mvec[0]);
    
    // Instruction 2: sub %sp, tmp -> %sp
    numInstr++;
    mvec[1] = new MachineInstr(SUB);
    mvec[1]->SetMachineOperand(0, /*regNum %sp = o6 = r[14]*/(unsigned int)14);
    mvec[1]->SetMachineOperand(1, MachineOperand::MO_VirtualRegister,tmpInstr);
    mvec[1]->SetMachineOperand(2, /*regNum %sp = o6 = r[14]*/(unsigned int)14);
    
    // Instruction 3: add %sp, 0 -> result
    numInstr++;
    mvec[2] = new MachineInstr(ADD);
    mvec[2]->SetMachineOperand(0, /*regNum %sp = o6 = r[14]*/(unsigned int)14);
    mvec[2]->SetMachineOperand(1, /*regNum %g0*/ (unsigned int) 0);
    mvec[2]->SetMachineOperand(2, MachineOperand::MO_VirtualRegister, instr);
    break;
    }
    
  case 61:	// reg:   Call
		// Generate a call-indirect (i.e., JMPL) for now to expose
		// the potential need for registers.  If an absolute address
		// is available, replace this with a CALL instruction.
		// Mark both the indirection register and the return-address
    {		// register as hidden virtual registers.
      
    Instruction* jmpAddrReg = new TmpInstruction(Instruction::UserOp1,
	((CallInst*) subtreeRoot->getInstruction())->getCalledMethod(), NULL);
    Instruction* retAddrReg = new TmpInstruction(Instruction::UserOp1,
					     subtreeRoot->getValue(), NULL);
    subtreeRoot->getInstruction()->getMachineInstrVec().addTempValue(jmpAddrReg);
    subtreeRoot->getInstruction()->getMachineInstrVec().addTempValue(retAddrReg);
    
    mvec[0] = new MachineInstr(JMPL);
    mvec[0]->SetMachineOperand(0, MachineOperand::MO_VirtualRegister, jmpAddrReg);
    mvec[0]->SetMachineOperand(1, MachineOperand::MO_SignExtendedImmed,
			          (int64_t) 0);
    mvec[0]->SetMachineOperand(2, MachineOperand::MO_VirtualRegister, retAddrReg);

    // NOTE: jmpAddrReg will be loaded by a different instruction generated
    //	     by the final code generator, so we just mark the CALL instruction
    //	     as computing that value.
    //	     The retAddrReg is actually computed by the CALL instruction.
    //
    jmpAddrReg->addMachineInstruction(mvec[0]);
    retAddrReg->addMachineInstruction(mvec[0]);
    
    mvec[numInstr++] = new MachineInstr(NOP); // delay slot
    break;
    }
    
  case 62:	// reg:   Shl(reg, reg)
    opType = subtreeRoot->leftChild()->getValue()->getType();
    assert(opType->isIntegral() || opType == Type::BoolTy); 
    mvec[0] = new MachineInstr((opType == Type::LongTy)? SLLX : SLL);
    Set3OperandsFromInstr(mvec[0], subtreeRoot, target);
    break;
    
  case 63:	// reg:   Shr(reg, reg)
    opType = subtreeRoot->leftChild()->getValue()->getType();
    assert(opType->isIntegral() || opType == Type::BoolTy); 
    mvec[0] = new MachineInstr((opType->isSigned()
				? ((opType == Type::LongTy)? SRAX : SRA)
				: ((opType == Type::LongTy)? SRLX : SRL)));
    Set3OperandsFromInstr(mvec[0], subtreeRoot, target);
    break;
    
  case 64:	// reg:   Phi(reg,reg)
  {		// This instruction has variable #operands, so resultPos is 0.
    Instruction* phi = subtreeRoot->getInstruction();
    mvec[0] = new MachineInstr(PHI, 1 + phi->getNumOperands());
    mvec[0]->SetMachineOperand(0, MachineOperand::MO_VirtualRegister,
			          subtreeRoot->getValue());
    for (unsigned i=0, N=phi->getNumOperands(); i < N; i++)
      mvec[0]->SetMachineOperand(i+1, MachineOperand::MO_VirtualRegister,
				      phi->getOperand(i));
    break;
  }  
  case 71:	// reg:     VReg
  case 72:	// reg:     Constant
    numInstr = 0;			// don't forward the value
    break;

  case 111:	// stmt:  reg
  case 112:	// stmt:  boolconst
  case 113:	// stmt:  bool
  case 121:
  case 122:
  case 123:
  case 124:
  case 125:
  case 126:
  case 127:
  case 128:
  case 129:
  case 130:
  case 131:
  case 132:
  case 153:
  case 155:
    // 
    // These are all chain rules, which have a single nonterminal on the RHS.
    // Get the rule that matches the RHS non-terminal and use that instead.
    // 
    assert(ThisIsAChainRule(ruleForNode));
    assert(nts[0] && ! nts[1]
	   && "A chain rule should have only one RHS non-terminal!");
    nextRule = burm_rule(subtreeRoot->state, nts[0]);
    nts = burm_nts[nextRule];
    numInstr = GetInstructionsByRule(subtreeRoot, nextRule, nts,target,mvec);
    break;
    
  default:
    assert(0 && "Unrecognized BURG rule");
    numInstr = 0;
    break;
  }
  
  if (forwardOperandNum >= 0)
    { // We did not generate a machine instruction but need to use operand.
      // If user is in the same tree, replace Value in its machine operand.
      // If not, insert a copy instruction which should get coalesced away
      // by register allocation.
      if (subtreeRoot->parent() != NULL)
	ForwardOperand(subtreeRoot, (InstructionNode*) subtreeRoot->parent(),
		       forwardOperandNum);
      else
	{
	  int n = numInstr++;
	  mvec[n] = new MachineInstr(ADD);
	  mvec[n]->SetMachineOperand(0, MachineOperand::MO_VirtualRegister,
		subtreeRoot->getInstruction()->getOperand(forwardOperandNum));
	  mvec[n]->SetMachineOperand(1, /*regNum %g0*/ (unsigned int) 0);
	  mvec[n]->SetMachineOperand(2, MachineOperand::MO_VirtualRegister,
				        subtreeRoot->getInstruction());
	}
    }
  
  if (! ThisIsAChainRule(ruleForNode))
    numInstr = FixConstantOperands(subtreeRoot, mvec, numInstr, target);
  
  return numInstr;
}


