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
                                   MachineOperand::MachineOperandType opType,
                                   Value* _val,
                                   bool isdef=false,
                                   bool isDefAndUse=false)
{
  assert(i < operands.size());
  operands[i].Initialize(opType, _val);
  operands[i].isDef = isdef ||
    TargetInstrDescriptors[opCode].resultPos == (int) i;
  operands[i].isDefAndUse = isDefAndUse;
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
  operands[i].isDefAndUse = false;
}

void
MachineInstr::SetMachineOperandReg(unsigned int i,
                                   int regNum,
                                   bool isdef=false,
                                   bool isDefAndUse=false,
                                   bool isCCReg=false)
{
  assert(i < operands.size());
  operands[i].InitializeReg(regNum, isCCReg);
  operands[i].isDef = isdef ||
    TargetInstrDescriptors[opCode].resultPos == (int) i;
  operands[i].isDefAndUse = isDefAndUse;
  regsUsed.insert(regNum);
}

void
MachineInstr::SetRegForOperand(unsigned i, int regNum)
{
  operands[i].setRegForValue(regNum);
  regsUsed.insert(regNum);
}


void
MachineInstr::dump() const 
{
  cerr << "  " << *this;
}

static inline std::ostream &OutputValue(std::ostream &os,
                                        const Value* val)
{
  os << "(val ";
  if (val && val->hasName())
    return os << val->getName();
  else
    return os << (void*) val;              // print address only
  os << ")";
}

std::ostream &operator<<(std::ostream& os, const MachineInstr& minstr)
{
  os << TargetInstrDescriptors[minstr.opCode].opCodeString;
  
  for (unsigned i=0, N=minstr.getNumOperands(); i < N; i++) {
    os << "\t" << minstr.getOperand(i);
    if( minstr.operandIsDefined(i) ) 
      os << "*";
    if( minstr.operandIsDefinedAndUsed(i) ) 
      os << "*";
  }
  
  // code for printing implict references
  unsigned NumOfImpRefs =  minstr.getNumImplicitRefs();
  if(  NumOfImpRefs > 0 ) {
    os << "\tImplicit: ";
    for(unsigned z=0; z < NumOfImpRefs; z++) {
      OutputValue(os, minstr.getImplicitRef(z)); 
      if( minstr.implicitRefIsDefined(z)) os << "*";
      if( minstr.implicitRefIsDefinedAndUsed(z)) os << "*";
      os << "\t";
    }
  }
  
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
      return OutputValue(os, mop.getVRegValue());
    case MachineOperand::MO_MachineRegister:
      return os << "(" << mop.getMachineRegNum() << ")";
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
          os << (const void*) opVal;
        return os << ")";
      }
    default:
      assert(0 && "Unrecognized operand type");
      break;
    }
  
  return os;
}
