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
#include "llvm/Value.h"
#include <iostream>
using std::cerr;


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
MachineInstr::SetMachineOperandVal(unsigned int i,
				MachineOperand::MachineOperandType operandType,
				Value* _val, bool isdef=false)
{
  assert(i < operands.size());
  operands[i].Initialize(operandType, _val);
  operands[i].isDef = isdef ||
    TargetInstrDescriptors[opCode].resultPos == (int) i;
}

void
MachineInstr::SetMachineOperandConst(unsigned int i,
				MachineOperand::MachineOperandType operandType,
                                     int64_t intValue)
{
  assert(i < operands.size());
  assert(TargetInstrDescriptors[opCode].resultPos != (int) i &&
         "immed. constant cannot be defined");
  operands[i].InitializeConst(operandType, intValue);
  operands[i].isDef = false;
}

void
MachineInstr::SetMachineOperandReg(unsigned int i,
                                   int regNum,
                                   bool isdef=false,
                                   bool isCCReg=false)
{
  assert(i < operands.size());
  operands[i].InitializeReg(regNum, isCCReg);
  operands[i].isDef = isdef ||
    TargetInstrDescriptors[opCode].resultPos == (int) i;
}

void
MachineInstr::dump(unsigned int indent) const 
{
  for (unsigned i=0; i < indent; i++)
    cerr << "    ";
  
  cerr << *this;
}

std::ostream &operator<<(std::ostream& os, const MachineInstr& minstr)
{
  os << TargetInstrDescriptors[minstr.opCode].opCodeString;
  
  for (unsigned i=0, N=minstr.getNumOperands(); i < N; i++) {
    os << "\t" << minstr.getOperand(i);
    if( minstr.getOperand(i).opIsDef() ) 
      os << "*";
  }
  
#undef DEBUG_VAL_OP_ITERATOR
#ifdef DEBUG_VAL_OP_ITERATOR
  os << "\n\tValue operands are: ";
  for (MachineInstr::val_const_op_iterator vo(&minstr); ! vo.done(); ++vo)
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
      if( minstr.implicitRefIsDefined(z)) os << "*";
      os << "\t";
    }
  }

#endif
  return os << "\n";
}

static inline std::ostream &OutputOperand(std::ostream &os,
                                          const MachineOperand &mop)
{
  Value* val;
  switch (mop.getOperandType())
    {
    case MachineOperand::MO_CCRegister:
    case MachineOperand::MO_VirtualRegister:
      val = mop.getVRegValue();
      os << "(val ";
      if (val && val->hasName())
        os << val->getName();
      else
        os << val;
      return os << ")";
    case MachineOperand::MO_MachineRegister:
      return os << "("     << mop.getMachineRegNum() << ")";
    default:
      assert(0 && "Unknown operand type");
      return os;
    }
}


std::ostream &operator<<(std::ostream &os, const MachineOperand &mop)
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
      return os << (long)mop.immedVal;
    case MachineOperand::MO_UnextendedImmed:
      return os << (long)mop.immedVal;
    case MachineOperand::MO_PCRelativeDisp:
      {
        const Value* opVal = mop.getVRegValue();
        bool isLabel = isa<Function>(opVal) || isa<BasicBlock>(opVal);
        os << "%disp(" << (isLabel? "label " : "addr-of-val ");
        if (opVal->hasName())
          os << opVal->getName();
        else
          os << opVal;
        return os << ")";
      }
    default:
      assert(0 && "Unrecognized operand type");
      break;
    }
  
  return os;
}
