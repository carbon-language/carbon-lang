// $Id$ -*-c++-*-
//***************************************************************************
// File:
//	InstrSelectionSupport.h
// 
// Purpose:
//	Target-independent instruction selection code.
//      See SparcInstrSelection.cpp for usage.
//      
// History:
//	10/10/01	 -  Vikram Adve  -  Created
//**************************************************************************/

#include "llvm/CodeGen/InstrSelectionSupport.h"
#include "llvm/CodeGen/InstrSelection.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/MachineRegInfo.h"
#include "llvm/ConstPoolVals.h"
#include "llvm/Instruction.h"
#include "llvm/Type.h"
#include "llvm/iMemory.h"


//*************************** Local Functions ******************************/

inline int64_t
GetSignedIntConstantValue(Value* val, bool& isValidConstant)
{
  int64_t intValue = 0;
  isValidConstant = false;
  
  if (val->getValueType() == Value::ConstantVal)
    {
      switch(val->getType()->getPrimitiveID())
	{
	case Type::BoolTyID:
	  intValue = ((ConstPoolBool*) val)->getValue()? 1 : 0;
	  isValidConstant = true;
	  break;
	case Type::SByteTyID:
	case Type::ShortTyID:
	case Type::IntTyID:
	case Type::LongTyID:
	  intValue = ((ConstPoolSInt*) val)->getValue();
	  isValidConstant = true;
	  break;
	default:
	  break;
	}
    }
  
  return intValue;
}

inline uint64_t
GetUnsignedIntConstantValue(Value* val, bool& isValidConstant)
{
  uint64_t intValue = 0;
  isValidConstant = false;
  
  if (val->getValueType() == Value::ConstantVal)
    {
      switch(val->getType()->getPrimitiveID())
	{
	case Type::BoolTyID:
	  intValue = ((ConstPoolBool*) val)->getValue()? 1 : 0;
	  isValidConstant = true;
	  break;
	case Type::UByteTyID:
	case Type::UShortTyID:
	case Type::UIntTyID:
	case Type::ULongTyID:
	  intValue = ((ConstPoolUInt*) val)->getValue();
	  isValidConstant = true;
	  break;
	default:
	  break;
	}
    }
  
  return intValue;
}


inline int64_t
GetConstantValueAsSignedInt(Value* val, bool& isValidConstant)
{
  int64_t intValue = 0;
  
  if (val->getType()->isSigned())
    {
      intValue = GetSignedIntConstantValue(val, isValidConstant);
    }
  else				// non-numeric types will fall here
    {
      uint64_t uintValue = GetUnsignedIntConstantValue(val, isValidConstant);
      if (isValidConstant && uintValue < INT64_MAX)	// safe to use signed
	intValue = (int64_t) uintValue;
      else 
	isValidConstant = false;
    }
  
  return intValue;
}


//---------------------------------------------------------------------------
// Function: FoldGetElemChain
// 
// Purpose:
//   Fold a chain of GetElementPtr instructions into an equivalent
//   (Pointer, IndexVector) pair.  Returns the pointer Value, and
//   stores the resulting IndexVector in argument chainIdxVec.
//---------------------------------------------------------------------------

Value*
FoldGetElemChain(const InstructionNode* getElemInstrNode,
		 vector<ConstPoolVal*>& chainIdxVec)
{
  MemAccessInst* getElemInst = (MemAccessInst*)
    getElemInstrNode->getInstruction();
  
  // Initialize return values from the incoming instruction
  Value* ptrVal = getElemInst->getPtrOperand();
  chainIdxVec = getElemInst->getIndexVec(); // copies index vector values
  
  // Now chase the chain of getElementInstr instructions, if any
  InstrTreeNode* ptrChild = getElemInstrNode->leftChild();
  while (ptrChild->getOpLabel() == Instruction::GetElementPtr ||
	 ptrChild->getOpLabel() == GetElemPtrIdx)
    {
      // Child is a GetElemPtr instruction
      getElemInst = (MemAccessInst*)
	((InstructionNode*) ptrChild)->getInstruction();
      const vector<ConstPoolVal*>& idxVec = getElemInst->getIndexVec();
      
      // Get the pointer value out of ptrChild and *prepend* its index vector
      ptrVal = getElemInst->getPtrOperand();
      chainIdxVec.insert(chainIdxVec.begin(), idxVec.begin(), idxVec.end());
      
      ptrChild = ptrChild->leftChild();
    }
  
  return ptrVal;
}


//------------------------------------------------------------------------ 
// Function Set2OperandsFromInstr
// Function Set3OperandsFromInstr
// 
// For the common case of 2- and 3-operand arithmetic/logical instructions,
// set the m/c instr. operands directly from the VM instruction's operands.
// Check whether the first or second operand is 0 and can use a dedicated "0"
// register.
// Check whether the second operand should use an immediate field or register.
// (First and third operands are never immediates for such instructions.)
// 
// Arguments:
// canDiscardResult: Specifies that the result operand can be discarded
//		     by using the dedicated "0"
// 
// op1position, op2position and resultPosition: Specify in which position
//		     in the machine instruction the 3 operands (arg1, arg2
//		     and result) should go.
// 
// RETURN VALUE: unsigned int flags, where
//	flags & 0x01	=> operand 1 is constant and needs a register
//	flags & 0x02	=> operand 2 is constant and needs a register
//------------------------------------------------------------------------ 

void
Set2OperandsFromInstr(MachineInstr* minstr,
		      InstructionNode* vmInstrNode,
		      const TargetMachine& target,
		      bool canDiscardResult,
		      int op1Position,
		      int resultPosition)
{
  Set3OperandsFromInstr(minstr, vmInstrNode, target,
			canDiscardResult, op1Position,
			/*op2Position*/ -1, resultPosition);
}

#undef REVERT_TO_EXPLICIT_CONSTANT_CHECKS
#ifdef REVERT_TO_EXPLICIT_CONSTANT_CHECKS
unsigned
Set3OperandsFromInstrJUNK(MachineInstr* minstr,
			  InstructionNode* vmInstrNode,
			  const TargetMachine& target,
			  bool canDiscardResult,
			  int op1Position,
			  int op2Position,
			  int resultPosition)
{
  assert(op1Position >= 0);
  assert(resultPosition >= 0);
  
  unsigned returnFlags = 0x0;
  
  // Check if operand 1 is 0.  If so, try to use a hardwired 0 register.
  Value* op1Value = vmInstrNode->leftChild()->getValue();
  bool isValidConstant;
  int64_t intValue = GetConstantValueAsSignedInt(op1Value, isValidConstant);
  if (isValidConstant && intValue == 0 && target.zeroRegNum >= 0)
    minstr->SetMachineOperand(op1Position, /*regNum*/ target.zeroRegNum);
  else
    {
      if (isa<ConstPoolVal>(op1Value))
	{
	  // value is constant and must be loaded from constant pool
	  returnFlags = returnFlags | (1 << op1Position);
	}
      minstr->SetMachineOperand(op1Position, MachineOperand::MO_VirtualRegister,
				op1Value);
    }
  
  // Check if operand 2 (if any) fits in the immed. field of the instruction,
  // or if it is 0 and can use a dedicated machine register
  if (op2Position >= 0)
    {
      Value* op2Value = vmInstrNode->rightChild()->getValue();
      int64_t immedValue;
      unsigned int machineRegNum;
      
      MachineOperand::MachineOperandType
	op2type = ChooseRegOrImmed(op2Value, minstr->getOpCode(), target,
				   /*canUseImmed*/ true,
				   machineRegNum, immedValue);
      
      if (op2type == MachineOperand::MO_MachineRegister)
	minstr->SetMachineOperand(op2Position, machineRegNum);
      else if (op2type == MachineOperand::MO_VirtualRegister)
	{
	  if (isa<ConstPoolVal>(op2Value))
	    {
	      // value is constant and must be loaded from constant pool
	      returnFlags = returnFlags | (1 << op2Position);
	    }
	  minstr->SetMachineOperand(op2Position, op2type, op2Value);
	}
      else
	{
	  assert(op2type != MO_CCRegister);
	  minstr->SetMachineOperand(op2Position, op2type, immedValue);
	}
    }
  
  // If operand 3 (result) can be discarded, use a dead register if one exists
  if (canDiscardResult && target.zeroRegNum >= 0)
    minstr->SetMachineOperand(resultPosition, target.zeroRegNum);
  else
    minstr->SetMachineOperand(resultPosition,
                  MachineOperand::MO_VirtualRegister, vmInstrNode->getValue());
  
  return returnFlags;
}
#endif


void
Set3OperandsFromInstr(MachineInstr* minstr,
		      InstructionNode* vmInstrNode,
		      const TargetMachine& target,
		      bool canDiscardResult,
		      int op1Position,
		      int op2Position,
		      int resultPosition)
{
  assert(op1Position >= 0);
  assert(resultPosition >= 0);
  
  // operand 1
  minstr->SetMachineOperand(op1Position, MachineOperand::MO_VirtualRegister,
			    vmInstrNode->leftChild()->getValue());   
  
  // operand 2 (if any)
  if (op2Position >= 0)
    minstr->SetMachineOperand(op2Position, MachineOperand::MO_VirtualRegister,
			      vmInstrNode->rightChild()->getValue());   
  
  // result operand: if it can be discarded, use a dead register if one exists
  if (canDiscardResult && target.getRegInfo().getZeroRegNum() >= 0)
    minstr->SetMachineOperand(resultPosition,
			      target.getRegInfo().getZeroRegNum());
  else
    minstr->SetMachineOperand(resultPosition,
			      MachineOperand::MO_VirtualRegister, vmInstrNode->getValue());
}


MachineOperand::MachineOperandType
ChooseRegOrImmed(Value* val,
		 MachineOpCode opCode,
		 const TargetMachine& target,
		 bool canUseImmed,
		 unsigned int& getMachineRegNum,
		 int64_t& getImmedValue)
{
  MachineOperand::MachineOperandType opType =
    MachineOperand::MO_VirtualRegister;
  getMachineRegNum = 0;
  getImmedValue = 0;
  
  // Check for the common case first: argument is not constant
  // 
  ConstPoolVal *CPV = dyn_cast<ConstPoolVal>(val);
  if (!CPV) return opType;

  if (CPV->getType() == Type::BoolTy)
    {
      ConstPoolBool *CPB = (ConstPoolBool*)CPV;
      if (!CPB->getValue() && target.getRegInfo().getZeroRegNum() >= 0)
	{
	  getMachineRegNum = target.getRegInfo().getZeroRegNum();
	  return MachineOperand::MO_MachineRegister;
	}

      getImmedValue = 1;
      return MachineOperand::MO_SignExtendedImmed;
    }
  
  if (!CPV->getType()->isIntegral()) return opType;

  // Now get the constant value and check if it fits in the IMMED field.
  // Take advantage of the fact that the max unsigned value will rarely
  // fit into any IMMED field and ignore that case (i.e., cast smaller
  // unsigned constants to signed).
  // 
  int64_t intValue;
  if (CPV->getType()->isSigned())
    {
      intValue = ((ConstPoolSInt*)CPV)->getValue();
    }
  else
    {
      uint64_t V = ((ConstPoolUInt*)CPV)->getValue();
      if (V >= INT64_MAX) return opType;
      intValue = (int64_t)V;
    }

  if (intValue == 0 && target.getRegInfo().getZeroRegNum() >= 0)
    {
      opType = MachineOperand::MO_MachineRegister;
      getMachineRegNum = target.getRegInfo().getZeroRegNum();
    }
  else if (canUseImmed &&
	   target.getInstrInfo().constantFitsInImmedField(opCode, intValue))
    {
      opType = MachineOperand::MO_SignExtendedImmed;
      getImmedValue = intValue;
    }
  
  return opType;
}

