//===-- InstrSelectionSupport.cpp -----------------------------------------===//
//
// Target-independent instruction selection code.  See SparcInstrSelection.cpp
// for usage.
// 
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/InstrSelectionSupport.h"
#include "llvm/CodeGen/InstrSelection.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrAnnot.h"
#include "llvm/CodeGen/MachineCodeForInstruction.h"
#include "llvm/CodeGen/MachineCodeForMethod.h"
#include "llvm/CodeGen/InstrForest.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/MachineRegInfo.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Type.h"
#include "llvm/iMemory.h"
using std::vector;

//*************************** Local Functions ******************************/


// Generate code to load the constant into a TmpInstruction (virtual reg) and
// returns the virtual register.
// 
static TmpInstruction*
InsertCodeToLoadConstant(Function *F,
                         Value* opValue,
                         Instruction* vmInstr,
                         vector<MachineInstr*>& loadConstVec,
                         TargetMachine& target)
{
  // Create a tmp virtual register to hold the constant.
  TmpInstruction* tmpReg = new TmpInstruction(opValue);
  MachineCodeForInstruction &mcfi = MachineCodeForInstruction::get(vmInstr);
  mcfi.addTemp(tmpReg);
  
  target.getInstrInfo().CreateCodeToLoadConst(target, F, opValue, tmpReg,
                                              loadConstVec, mcfi);
  
  // Record the mapping from the tmp VM instruction to machine instruction.
  // Do this for all machine instructions that were not mapped to any
  // other temp values created by 
  // tmpReg->addMachineInstruction(loadConstVec.back());
  
  return tmpReg;
}


//---------------------------------------------------------------------------
// Function GetConstantValueAsUnsignedInt
// Function GetConstantValueAsSignedInt
// 
// Convenience functions to get the value of an integral constant, for an
// appropriate integer or non-integer type that can be held in a signed
// or unsigned integer respectively.  The type of the argument must be
// the following:
//      Signed or unsigned integer
//      Boolean
//      Pointer
// 
// isValidConstant is set to true if a valid constant was found.
//---------------------------------------------------------------------------

uint64_t
GetConstantValueAsUnsignedInt(const Value *V,
                              bool &isValidConstant)
{
  isValidConstant = true;

  if (isa<Constant>(V))
    if (V->getType() == Type::BoolTy)
      return (int64_t) cast<ConstantBool>(V)->getValue();
    else if (V->getType()->isIntegral())
      return (V->getType()->isUnsigned()
              ? cast<ConstantUInt>(V)->getValue()
              : (uint64_t) cast<ConstantSInt>(V)->getValue());

  isValidConstant = false;
  return 0;
}

int64_t
GetConstantValueAsSignedInt(const Value *V,
                            bool &isValidConstant)
{
  uint64_t C = GetConstantValueAsUnsignedInt(V, isValidConstant);
  if (isValidConstant) {
    if (V->getType()->isSigned() || C < INT64_MAX) // safe to cast to signed
      return (int64_t) C;
    else
      isValidConstant = false;
  }
  return 0;
}

//---------------------------------------------------------------------------
// Function: FoldGetElemChain
// 
// Purpose:
//   Fold a chain of GetElementPtr instructions containing only
//   constant offsets into an equivalent (Pointer, IndexVector) pair.
//   Returns the pointer Value, and stores the resulting IndexVector
//   in argument chainIdxVec.
//---------------------------------------------------------------------------

Value*
FoldGetElemChain(const InstructionNode* getElemInstrNode,
		 vector<Value*>& chainIdxVec)
{
  MemAccessInst* getElemInst = (MemAccessInst*)
    getElemInstrNode->getInstruction();
  
  // Return NULL if we don't fold any instructions in.
  Value* ptrVal = NULL;
  
  // Remember if the last instruction had a leading [0] index.
  bool hasLeadingZero = false;
  
  // Now chase the chain of getElementInstr instructions, if any.
  // Check for any non-constant indices and stop there.
  // 
  const InstrTreeNode* ptrChild = getElemInstrNode;
  while (ptrChild->getOpLabel() == Instruction::GetElementPtr ||
	 ptrChild->getOpLabel() == GetElemPtrIdx)
    {
      // Child is a GetElemPtr instruction
      getElemInst = cast<MemAccessInst>(ptrChild->getValue());
      MemAccessInst::op_iterator OI, firstIdx = getElemInst->idx_begin();
      MemAccessInst::op_iterator lastIdx = getElemInst->idx_end();
      bool allConstantOffsets = true;

      // Check that all offsets are constant for this instruction
      for (OI = firstIdx; allConstantOffsets && OI != lastIdx; ++OI)
        allConstantOffsets = isa<ConstantInt>(*OI);

      if (allConstantOffsets)
        { // Get pointer value out of ptrChild.
          ptrVal = getElemInst->getPointerOperand();

          // Check for a leading [0] index, if any.  It will be discarded later.
          ConstantUInt* CV = dyn_cast<ConstantUInt>((Value*) *firstIdx);
          hasLeadingZero = bool(CV && CV->getValue() == 0);

          // Insert its index vector at the start, skipping any leading [0]
          chainIdxVec.insert(chainIdxVec.begin(),
                             firstIdx + hasLeadingZero, lastIdx);

          // Mark the folded node so no code is generated for it.
          ((InstructionNode*) ptrChild)->markFoldedIntoParent();
        }
      else // cannot fold this getElementPtr instr. or any further ones
        break;

      ptrChild = ptrChild->leftChild();
    }

  // If the first getElementPtr instruction had a leading [0], add it back.
  // Note that this instruction is the *last* one successfully folded above.
  if (ptrVal && hasLeadingZero) 
    chainIdxVec.insert(chainIdxVec.begin(), ConstantUInt::get(Type::UIntTy,0));

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
      ! (isa<PointerType>(CPV->getType()) &&
         CPV->isNullValue()))
    return opType;
  
  // Now get the constant value and check if it fits in the IMMED field.
  // Take advantage of the fact that the max unsigned value will rarely
  // fit into any IMMED field and ignore that case (i.e., cast smaller
  // unsigned constants to signed).
  // 
  int64_t intValue;
  if (isa<PointerType>(CPV->getType()))
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
      opType = CPV->getType()->isSigned()
        ? MachineOperand::MO_SignExtendedImmed
        : MachineOperand::MO_UnextendedImmed;
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
  
  Function *F = vmInstr->getParent()->getParent();
  
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
          TmpInstruction* tmpReg = InsertCodeToLoadConstant(F, opValue,vmInstr,
                                                            loadConstVec,
                                                            target);
          minstr->SetMachineOperandVal(op, MachineOperand::MO_VirtualRegister,
                                       tmpReg);
        }
    }
  
  // 
  // Also, check for implicit operands used by the machine instruction
  // (no need to check those defined since they cannot be constants).
  // These include:
  // -- arguments to a Call
  // -- return value of a Return
  // Any such operand that is a constant value needs to be fixed also.
  // The current instructions with implicit refs (viz., Call and Return)
  // have no immediate fields, so the constant always needs to be loaded
  // into a register.
  // 
  bool isCall = target.getInstrInfo().isCall(minstr->getOpCode());
  unsigned lastCallArgNum = 0;          // unused if not a call
  CallArgsDescriptor* argDesc = NULL;   // unused if not a call
  if (isCall)
    argDesc = CallArgsDescriptor::get(minstr);
  
  for (unsigned i=0, N=minstr->getNumImplicitRefs(); i < N; ++i)
    if (isa<Constant>(minstr->getImplicitRef(i)) ||
        isa<GlobalValue>(minstr->getImplicitRef(i)))
      {
        Value* oldVal = minstr->getImplicitRef(i);
        TmpInstruction* tmpReg =
          InsertCodeToLoadConstant(F, oldVal, vmInstr, loadConstVec, target);
        minstr->setImplicitRef(i, tmpReg);
        
        if (isCall)
          { // find and replace the argument in the CallArgsDescriptor
            unsigned i=lastCallArgNum;
            while (argDesc->getArgInfo(i).getArgVal() != oldVal)
              ++i;
            assert(i < argDesc->getNumArgs() &&
                   "Constant operands to a call *must* be in the arg list");
            lastCallArgNum = i;
            argDesc->getArgInfo(i).replaceArgVal(tmpReg);
          }
      }
  
  return loadConstVec;
}


