// $Id$
//***************************************************************************
// File:
//	MachineInstr.cpp
// 
// Purpose:
//	
// 
// Strategy:
// 
// History:
//	7/2/01	 -  Vikram Adve  -  Created
//**************************************************************************/


#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Target/MachineFrameInfo.h"
#include "llvm/Target/MachineRegInfo.h"
#include "llvm/Method.h"
#include "llvm/iOther.h"
#include "llvm/Instruction.h"

AnnotationID MachineCodeForMethod::AID(
                 AnnotationManager::getID("MachineCodeForMethodAnnotation"));


//************************ Class Implementations **************************/

// Constructor for instructions with fixed #operands (nearly all)
MachineInstr::MachineInstr(MachineOpCode _opCode,
			   OpCodeMask    _opCodeMask)
  : opCode(_opCode),
    opCodeMask(_opCodeMask),
    operands(TargetInstrDescriptors[_opCode].numOperands)
{
  assert(TargetInstrDescriptors[_opCode].numOperands >= 0);
}

// Constructor for instructions with variable #operands
MachineInstr::MachineInstr(MachineOpCode _opCode,
			   unsigned	 numOperands,
			   OpCodeMask    _opCodeMask)
  : opCode(_opCode),
    opCodeMask(_opCodeMask),
    operands(numOperands)
{
}

void
MachineInstr::SetMachineOperand(unsigned int i,
				MachineOperand::MachineOperandType operandType,
				Value* _val, bool isdef=false)
{
  assert(i < operands.size());
  operands[i].Initialize(operandType, _val);
  operands[i].isDef = isdef ||
    TargetInstrDescriptors[opCode].resultPos == (int) i;
}

void
MachineInstr::SetMachineOperand(unsigned int i,
				MachineOperand::MachineOperandType operandType,
				int64_t intValue, bool isdef=false)
{
  assert(i < operands.size());
  operands[i].InitializeConst(operandType, intValue);
  operands[i].isDef = isdef ||
    TargetInstrDescriptors[opCode].resultPos == (int) i;
}

void
MachineInstr::SetMachineOperand(unsigned int i,
				int regNum, bool isdef=false)
{
  assert(i < operands.size());
  operands[i].InitializeReg(regNum);
  operands[i].isDef = isdef ||
    TargetInstrDescriptors[opCode].resultPos == (int) i;
}

void
MachineInstr::dump(unsigned int indent) const 
{
  for (unsigned i=0; i < indent; i++)
    cout << "    ";
  
  cout << *this;
}

ostream&
operator<< (ostream& os, const MachineInstr& minstr)
{
  os << TargetInstrDescriptors[minstr.opCode].opCodeString;
  
  for (unsigned i=0, N=minstr.getNumOperands(); i < N; i++)
    os << "\t" << minstr.getOperand(i);
  
#undef DEBUG_VAL_OP_ITERATOR
#ifdef DEBUG_VAL_OP_ITERATOR
  os << endl << "\tValue operands are: ";
  for (MachineInstr::val_op_const_iterator vo(&minstr); ! vo.done(); ++vo)
    {
      const Value* val = *vo;
      os << val << (vo.isDef()? "(def), " : ", ");
    }
#endif
  
 

#if 1
  // code for printing implict references

  unsigned NumOfImpRefs =  minstr.getNumImplicitRefs();
  if(  NumOfImpRefs > 0 ) {
	
    os << "\tImplicit:";

    for(unsigned z=0; z < NumOfImpRefs; z++) {
      os << minstr.getImplicitRef(z);
	  cout << "\t";
    }
  }

#endif


  os << endl;
  
  return os;
}

static inline ostream&
OutputOperand(ostream &os, const MachineOperand &mop)
{
  switch (mop.getOperandType())
    {
    case MachineOperand::MO_CCRegister:
    case MachineOperand::MO_VirtualRegister:
      return os << "(val " << mop.getVRegValue() << ")";
    case MachineOperand::MO_MachineRegister:
      return os << "("     << mop.getMachineRegNum() << ")";
    default:
      assert(0 && "Unknown operand type");
      return os;
    }
}


ostream&
operator<<(ostream &os, const MachineOperand &mop)
{
  switch(mop.opType)
    {
    case MachineOperand::MO_VirtualRegister:
    case MachineOperand::MO_MachineRegister:
      os << "%reg";
      return OutputOperand(os, mop);
    case MachineOperand::MO_CCRegister:
      os << "%ccreg";
      return OutputOperand(os, mop);
    case MachineOperand::MO_SignExtendedImmed:
      return os << mop.immedVal;
    case MachineOperand::MO_UnextendedImmed:
      return os << mop.immedVal;
    case MachineOperand::MO_PCRelativeDisp:
      {
        const Value* opVal = mop.getVRegValue();
        bool isLabel = isa<Method>(opVal) || isa<BasicBlock>(opVal);
        return os << "%disp("
                  << (isLabel? "label " : "addr-of-val ")
                  << opVal << ")";
      }
    default:
      assert(0 && "Unrecognized operand type");
      break;
    }
  
  return os;
}

static unsigned int
ComputeMaxOptionalArgsSize(const TargetMachine& target, const Method* method)
{
  const MachineFrameInfo& frameInfo = target.getFrameInfo();
  
  unsigned int maxSize = 0;
  
  for (Method::inst_const_iterator I=method->inst_begin(),E=method->inst_end();
       I != E; ++I)
    if ((*I)->getOpcode() == Instruction::Call)
      {
        CallInst* callInst = cast<CallInst>(*I);
        unsigned int numOperands = callInst->getNumOperands() - 1;
        int numExtra = (int) numOperands - frameInfo.getNumFixedOutgoingArgs();
        if (numExtra <= 0)
          continue;
        
        unsigned int sizeForThisCall;
        if (frameInfo.argsOnStackHaveFixedSize())
          {
            int argSize = frameInfo.getSizeOfEachArgOnStack(); 
            sizeForThisCall = numExtra * (unsigned) argSize;
          }
        else
          {
            assert(0 && "UNTESTED CODE: Size per stack argument is not fixed on this architecture: use actual arg sizes to compute MaxOptionalArgsSize");
            sizeForThisCall = 0;
            for (unsigned i=0; i < numOperands; ++i)
              sizeForThisCall += target.findOptimalStorageSize(callInst->
                                                    getOperand(i)->getType());
          }
        
        if (maxSize < sizeForThisCall)
          maxSize = sizeForThisCall;
      }
  
  return maxSize;
}


/*ctor*/
MachineCodeForMethod::MachineCodeForMethod(const Method* _M,
                                           const TargetMachine& target)
  : Annotation(AID),
    method(_M), compiledAsLeaf(false), staticStackSize(0),
    automaticVarsSize(0), regSpillsSize(0),
    currentOptionalArgsSize(0), maxOptionalArgsSize(0),
    currentTmpValuesSize(0)
{
  maxOptionalArgsSize = ComputeMaxOptionalArgsSize(target, method);
  staticStackSize = maxOptionalArgsSize +
                    target.getFrameInfo().getMinStackFrameSize();
}

int
MachineCodeForMethod::allocateLocalVar(const TargetMachine& target,
                                       const Value* val)
{
  // Check if we've allocated a stack slot for this value already
  // 
  int offset = getOffset(val);
  if (offset == INVALID_FRAME_OFFSET)
    {
      bool growUp;
      int firstOffset =target.getFrameInfo().getFirstAutomaticVarOffset(*this,
                                                                       growUp);
      unsigned int size = target.findOptimalStorageSize(val->getType());
      
      offset = growUp? firstOffset + getAutomaticVarsSize()
                     : firstOffset - getAutomaticVarsSize() - size;
      offsets[val] = offset;
      
      incrementAutomaticVarsSize(size);
    }
  return offset;
}

int
MachineCodeForMethod::allocateSpilledValue(const TargetMachine& target,
                                           const Type* type)
{
  bool growUp;
  int firstOffset = target.getFrameInfo().getRegSpillAreaOffset(*this, growUp);
  unsigned int size = target.findOptimalStorageSize(type);
  
  int offset = growUp? firstOffset + getRegSpillsSize()
                     : firstOffset - getRegSpillsSize() - size;
  
  incrementRegSpillsSize(size);
  
  return offset;
}
  
int
MachineCodeForMethod::allocateOptionalArg(const TargetMachine& target,
                                          const Type* type)
{
  const MachineFrameInfo& frameInfo = target.getFrameInfo();
  bool growUp;
  int firstOffset = frameInfo.getFirstOptionalOutgoingArgOffset(*this, growUp);

  int size = MAXINT;
  if (frameInfo.argsOnStackHaveFixedSize())
    size = frameInfo.getSizeOfEachArgOnStack(); 
  else
    {
      size = target.findOptimalStorageSize(type);
      assert(0 && "UNTESTED CODE: Size per stack argument is not fixed on this architecture: use actual argument sizes for computing optional arg offsets");
    }
  
  int offset = growUp? firstOffset + getCurrentOptionalArgsSize()
                     : firstOffset - getCurrentOptionalArgsSize() - size;
  incrementCurrentOptionalArgsSize(size);
  
  return offset;
}

void
MachineCodeForMethod::resetOptionalArgs(const TargetMachine& target)
{
  currentOptionalArgsSize = 0;
}

int
MachineCodeForMethod::pushTempValue(const TargetMachine& target,
                                    unsigned int size)
{
  bool growUp;
  int firstTmpOffset = target.getFrameInfo().getTmpAreaOffset(*this, growUp);
  int offset = growUp? firstTmpOffset + currentTmpValuesSize
                     : firstTmpOffset - currentTmpValuesSize - size;
  currentTmpValuesSize += size;
  return offset;
}

void
MachineCodeForMethod::popAllTempValues(const TargetMachine& target)
{
  currentTmpValuesSize = 0;
}


// void
// MachineCodeForMethod::putLocalVarAtOffsetFromSP(const Value* local,
//                                                 int offset,
//                                                 unsigned int size)
// {
//   offsetsFromSP[local] = offset;
//   incrementAutomaticVarsSize(size);
// }
// 

int
MachineCodeForMethod::getOffset(const Value* val) const
{
  hash_map<const Value*, int>::const_iterator pair = offsets.find(val);
  return (pair == offsets.end())? INVALID_FRAME_OFFSET : (*pair).second;
}


// int
// MachineCodeForMethod::getOffsetFromSP(const Value* local) const
// {
//   hash_map<const Value*, int>::const_iterator pair = offsetsFromSP.find(local);
//   return (pair == offsetsFromSP.end())? INVALID_FRAME_OFFSET : (*pair).second;
// }


void
MachineCodeForMethod::dump() const
{
  cout << "\n" << method->getReturnType()
       << " \"" << method->getName() << "\"" << endl;
  
  for (Method::const_iterator BI = method->begin(); BI != method->end(); ++BI)
    {
      BasicBlock* bb = *BI;
      cout << "\n"
	   << (bb->hasName()? bb->getName() : "Label")
	   << " (" << bb << ")" << ":"
	   << endl;
      
      MachineCodeForBasicBlock& mvec = bb->getMachineInstrVec();
      for (unsigned i=0; i < mvec.size(); i++)
	cout << "\t" << *mvec[i];
    } 
  cout << endl << "End method \"" << method->getName() << "\""
       << endl << endl;
}
