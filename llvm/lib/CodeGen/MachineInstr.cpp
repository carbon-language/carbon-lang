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
#include "llvm/Target/MachineRegInfo.h"
#include "llvm/Method.h"
#include "llvm/Instruction.h"



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
				unsigned int regNum, bool isdef=false)
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


void
MachineCodeForMethod::putLocalVarAtOffsetFromFP(const Value* local,
                                                int offset,
                                                unsigned int size)
{
  offsetsFromFP[local] = offset;
  incrementAutomaticVarsSize(size);
}


void
MachineCodeForMethod::putLocalVarAtOffsetFromSP(const Value* local,
                                                int offset,
                                                unsigned int size)
{
  offsetsFromSP[local] = offset;
  incrementAutomaticVarsSize(size);
}


int
MachineCodeForMethod::getOffsetFromFP(const Value* local) const
{
  hash_map<const Value*, int>::const_iterator pair = offsetsFromFP.find(local);
  assert(pair != offsetsFromFP.end() && "Offset from FP unknown for Value");
  return (*pair).second;
}


int
MachineCodeForMethod::getOffsetFromSP(const Value* local) const
{
  hash_map<const Value*, int>::const_iterator pair = offsetsFromSP.find(local);
  assert(pair != offsetsFromSP.end() && "Offset from SP unknown for Value");
  return (*pair).second;
}


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
