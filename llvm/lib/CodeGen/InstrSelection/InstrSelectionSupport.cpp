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
#include "llvm/CodeGen/MachineCodeForInstruction.h"
#include "llvm/CodeGen/MachineCodeForMethod.h"
#include "llvm/CodeGen/InstrForest.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/MachineRegInfo.h"
#include "llvm/ConstantVals.h"
#include "llvm/Method.h"
#include "llvm/BasicBlock.h"
#include "llvm/Type.h"
#include "llvm/iMemory.h"
using std::vector;

//*************************** Local Functions ******************************/


static TmpInstruction*
InsertCodeToLoadConstant(Method* method,
                         Value* opValue,
                         Instruction* vmInstr,
                         vector<MachineInstr*>& loadConstVec,
                         TargetMachine& target)
{
  vector<TmpInstruction*> tempVec;
  
  // Create a tmp virtual register to hold the constant.
  TmpInstruction* tmpReg = new TmpInstruction(opValue);
  MachineCodeForInstruction &MCFI = MachineCodeForInstruction::get(vmInstr);
  MCFI.addTemp(tmpReg);
  
  target.getInstrInfo().CreateCodeToLoadConst(method, opValue, tmpReg,
                                              loadConstVec, tempVec);
  
  // Register the new tmp values created for this m/c instruction sequence
  for (unsigned i=0; i < tempVec.size(); i++)
    MCFI.addTemp(tempVec[i]);
  
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
  if (!isa<Constant>(V))
    {
      isValidConstant = false;
      return 0;
    }
  
  isValidConstant = true;
  
  if (V->getType() == Type::BoolTy)
    return (int64_t) cast<ConstantBool>(V)->getValue();
  
  if (V->getType()->isIntegral())
    {
      if (V->getType()->isSigned())
        return cast<ConstantSInt>(V)->getValue();
      
      assert(V->getType()->isUnsigned());
      uint64_t Val = cast<ConstantUInt>(V)->getValue();
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
//   Fold a chain of GetElementPtr instructions containing only
//   structure offsets into an equivalent (Pointer, IndexVector) pair.
//   Returns the pointer Value, and stores the resulting IndexVector
//   in argument chainIdxVec.
//---------------------------------------------------------------------------

Value*
FoldGetElemChain(const InstructionNode* getElemInstrNode,
		 vector<Value*>& chainIdxVec)
{
  MemAccessInst* getElemInst = (MemAccessInst*)
    getElemInstrNode->getInstruction();
  
  // Initialize return values from the incoming instruction
  Value* ptrVal = NULL;
  assert(chainIdxVec.size() == 0);
  
  // Now chase the chain of getElementInstr instructions, if any.
  // Check for any array indices and stop there.
  // 
  const InstrTreeNode* ptrChild = getElemInstrNode;
  while (ptrChild->getOpLabel() == Instruction::GetElementPtr ||
	 ptrChild->getOpLabel() == GetElemPtrIdx)
    {
      // Child is a GetElemPtr instruction
      getElemInst = (MemAccessInst*)
	((InstructionNode*) ptrChild)->getInstruction();
      const vector<Value*>& idxVec = getElemInst->copyIndices();
      bool allStructureOffsets = true;
      
      // If it is a struct* access, the first offset must be array index [0],
      // and all other offsets must be structure (not array) offsets
      if (!isa<ConstantUInt>(idxVec.front()) ||
          cast<ConstantUInt>(idxVec.front())->getValue() != 0)
        allStructureOffsets = false;
      
      if (allStructureOffsets)
        for (unsigned int i=1; i < idxVec.size(); i++)
          if (idxVec[i]->getType() == Type::UIntTy)
            {
              allStructureOffsets = false; 
              break;
            }
      
      if (allStructureOffsets)
        { // Get pointer value out of ptrChild and *prepend* its index vector
          ptrVal = getElemInst->getPointerOperand();
          chainIdxVec.insert(chainIdxVec.begin(),
                             idxVec.begin()+1, idxVec.end());
          ((InstructionNode*) ptrChild)->markFoldedIntoParent();
                                        // mark so no code is generated
        }
      else // cannot fold this getElementPtr instr. or any further ones
        break;
      
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
  minstr->SetMachineOperandVal(op1Position, MachineOperand::MO_VirtualRegister,
			    vmInstrNode->leftChild()->getValue());   
  
  // operand 2 (if any)
  if (op2Position >= 0)
    minstr->SetMachineOperandVal(op2Position, MachineOperand::MO_VirtualRegister,
			      vmInstrNode->rightChild()->getValue());   
  
  // result operand: if it can be discarded, use a dead register if one exists
  if (canDiscardResult && target.getRegInfo().getZeroRegNum() >= 0)
    minstr->SetMachineOperandReg(resultPosition,
			      target.getRegInfo().getZeroRegNum());
  else
    minstr->SetMachineOperandVal(resultPosition,
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
  Constant *CPV = dyn_cast<Constant>(val);
  if (!CPV) return opType;

  if (ConstantBool *CPB = dyn_cast<ConstantBool>(CPV))
    {
      if (!CPB->getValue() && target.getRegInfo().getZeroRegNum() >= 0)
	{
	  getMachineRegNum = target.getRegInfo().getZeroRegNum();
	  return MachineOperand::MO_MachineRegister;
	}

      getImmedValue = 1;
      return MachineOperand::MO_SignExtendedImmed;
    }
  
  // Otherwise it needs to be an integer or a NULL pointer
  if (! CPV->getType()->isIntegral() &&
      ! (CPV->getType()->isPointerType() &&
         CPV->isNullValue()))
    return opType;
  
  // Now get the constant value and check if it fits in the IMMED field.
  // Take advantage of the fact that the max unsigned value will rarely
  // fit into any IMMED field and ignore that case (i.e., cast smaller
  // unsigned constants to signed).
  // 
  int64_t intValue;
  if (CPV->getType()->isPointerType())
    {
      intValue = 0;
    }
  else if (CPV->getType()->isSigned())
    {
      intValue = cast<ConstantSInt>(CPV)->getValue();
    }
  else
    {
      uint64_t V = cast<ConstantUInt>(CPV)->getValue();
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
  
  Method* method = vmInstr->getParent()->getParent();
  
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
      
      if (Constant *opConst = dyn_cast<Constant>(opValue))
        {
          unsigned int machineRegNum;
          int64_t immedValue;
          MachineOperand::MachineOperandType opType =
            ChooseRegOrImmed(opValue, minstr->getOpCode(), target,
                             (target.getInstrInfo().getImmedConstantPos(minstr->getOpCode()) == (int) op),
                             machineRegNum, immedValue);
          
          if (opType == MachineOperand::MO_MachineRegister)
            minstr->SetMachineOperandReg(op, machineRegNum);
          else if (opType == MachineOperand::MO_VirtualRegister)
            constantThatMustBeLoaded = true; // load is generated below
          else
            minstr->SetMachineOperandConst(op, opType, immedValue);
        }
      
      if (constantThatMustBeLoaded || isa<GlobalValue>(opValue))
        { // opValue is a constant that must be explicitly loaded into a reg.
          TmpInstruction* tmpReg = InsertCodeToLoadConstant(method, opValue, vmInstr,
                                                            loadConstVec, target);
          minstr->SetMachineOperandVal(op, MachineOperand::MO_VirtualRegister,
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
    if (isa<Constant>(minstr->getImplicitRef(i)) ||
        isa<GlobalValue>(minstr->getImplicitRef(i)))
      {
        Value* oldVal = minstr->getImplicitRef(i);
        TmpInstruction* tmpReg =
          InsertCodeToLoadConstant(method, oldVal, vmInstr, loadConstVec, target);
        minstr->setImplicitRef(i, tmpReg);
      }
  
  return loadConstVec;
}


