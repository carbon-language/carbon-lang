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


static TmpInstruction*
InsertCodeToLoadConstant(Value* opValue,
                         Instruction* vmInstr,
                         vector<MachineInstr*>& loadConstVec,
                         TargetMachine& target)
{
  vector<TmpInstruction*> tempVec;
  
  // Create a tmp virtual register to hold the constant.
  TmpInstruction* tmpReg =
    new TmpInstruction(TMP_INSTRUCTION_OPCODE, opValue, NULL);
  vmInstr->getMachineInstrVec().addTempValue(tmpReg);
  
  target.getInstrInfo().CreateCodeToLoadConst(opValue, tmpReg,
                                              loadConstVec, tempVec);
  
  // Register the new tmp values created for this m/c instruction sequence
  for (unsigned i=0; i < tempVec.size(); i++)
    vmInstr->getMachineInstrVec().addTempValue(tempVec[i]);
  
  // Record the mapping from the tmp VM instruction to machine instruction.
  // Do this for all machine instructions that were not mapped to any
  // other temp values created by 
  // tmpReg->addMachineInstruction(loadConstVec.back());
  
  return tmpReg;
}


//---------------------------------------------------------------------------
// Function GetConstantValueAsSignedInt
// 
// Convenience function to get the value of an integer constant, for an
// appropriate integer or non-integer type that can be held in an integer.
// The type of the argument must be the following:
//      Signed or unsigned integer
//      Boolean
//      Pointer
// 
// isValidConstant is set to true if a valid constant was found.
//---------------------------------------------------------------------------

int64_t
GetConstantValueAsSignedInt(const Value *V,
                            bool &isValidConstant)
{
  if (!isa<ConstPoolVal>(V))
    {
      isValidConstant = false;
      return 0;
    }
  
  isValidConstant = true;
  
  if (V->getType() == Type::BoolTy)
    return (int64_t) ((ConstPoolBool*)V)->getValue();
  
  if (V->getType()->isIntegral())
    {
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


//---------------------------------------------------------------------------
// Function: FixConstantOperandsForInstr
// 
// Purpose:
// Special handling for constant operands of a machine instruction
// -- if the constant is 0, use the hardwired 0 register, if any;
// -- if the constant fits in the IMMEDIATE field, use that field;
// -- else create instructions to put the constant into a register, either
//    directly or by loading explicitly from the constant pool.
// 
// In the first 2 cases, the operand of `minstr' is modified in place.
// Returns a vector of machine instructions generated for operands that
// fall under case 3; these must be inserted before `minstr'.
//---------------------------------------------------------------------------

vector<MachineInstr*>
FixConstantOperandsForInstr(Instruction* vmInstr,
                            MachineInstr* minstr,
                            TargetMachine& target)
{
  vector<MachineInstr*> loadConstVec;
  
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
          continue;
        }
          
      Value* opValue = mop.getVRegValue();
      bool constantThatMustBeLoaded = false;
      
      if (isa<ConstPoolVal>(opValue))
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
            constantThatMustBeLoaded = true; // load is generated below
          else
            minstr->SetMachineOperand(op, opType, immedValue);
        }

      if (constantThatMustBeLoaded || isa<GlobalValue>(opValue))
        { // opValue is a constant that must be explicitly loaded into a reg.
          TmpInstruction* tmpReg = InsertCodeToLoadConstant(opValue, vmInstr,
                                                        loadConstVec, target);
          minstr->SetMachineOperand(op, MachineOperand::MO_VirtualRegister,
                                        tmpReg);
        }
    }
  
  // 
  // Also, check for implicit operands used (not those defined) by the
  // machine instruction.  These include:
  // -- arguments to a Call
  // -- return value of a Return
  // Any such operand that is a constant value needs to be fixed also.
  // The current instructions with implicit refs (viz., Call and Return)
  // have no immediate fields, so the constant always needs to be loaded
  // into a register.
  // 
  for (unsigned i=0, N=minstr->getNumImplicitRefs(); i < N; ++i)
    if (isa<ConstPoolVal>(minstr->getImplicitRef(i)) ||
        isa<GlobalValue>(minstr->getImplicitRef(i)))
      {
        TmpInstruction* tmpReg =
          InsertCodeToLoadConstant(minstr->getImplicitRef(i), vmInstr,
                                   loadConstVec, target);
        minstr->setImplicitRef(i, tmpReg);
      }
  
  return loadConstVec;
}


#undef SAVE_TO_MOVE_BACK_TO_SPARCISSCPP
#ifdef SAVE_TO_MOVE_BACK_TO_SPARCISSCPP
unsigned
FixConstantOperands(const InstructionNode* vmInstrNode,
                    TargetMachine& target)
{
  Instruction* vmInstr = vmInstrNode->getInstruction();
  MachineCodeForVMInstr& mvec = vmInstr->getMachineInstrVec();
  
  for (unsigned i=0; i < mvec.size(); i++)
    {
      vector<MachineInsr*> loadConstVec =
        FixConstantOperandsForInstr(mvec[i], target);
    }
  
  // 
  // Finally, inserted the generated instructions in the vector
  // to be returned.
  // 
  unsigned numNew = loadConstVec.size();
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
#endif SAVE_TO_MOVE_BACK_TO_SPARCISSCPP


