// $Id$
//***************************************************************************
// File:
//	SparcInstrInfo.cpp
// 
// Purpose:
//	
// History:
//	10/15/01	 -  Vikram Adve  -  Created
//**************************************************************************/


#include "SparcInternals.h"
#include "SparcInstrSelectionSupport.h"
#include "llvm/Target/Sparc.h"
#include "llvm/CodeGen/InstrSelection.h"
#include "llvm/CodeGen/InstrSelectionSupport.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/ConstPoolVals.h"
#include "llvm/Type.h"


//************************ Internal Functions ******************************/


static inline MachineInstr*
CreateIntSetInstruction(int64_t C, bool isSigned, Value* dest)
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

//************************* External Classes *******************************/

//---------------------------------------------------------------------------
// class UltraSparcInstrInfo 
// 
// Purpose:
//   Information about individual instructions.
//   Most information is stored in the SparcMachineInstrDesc array above.
//   Other information is computed on demand, and most such functions
//   default to member functions in base class MachineInstrInfo. 
//---------------------------------------------------------------------------

/*ctor*/
UltraSparcInstrInfo::UltraSparcInstrInfo()
  : MachineInstrInfo(SparcMachineInstrDesc,
		     /*descSize = */ NUM_TOTAL_OPCODES,
		     /*numRealOpCodes = */ NUM_REAL_OPCODES)
{
}


// Create an instruction sequence to put the constant `val' into
// the virtual register `dest'.  `val' may be a ConstPoolVal or a
// GlobalValue, viz., the constant address of a global variable or function.
// The generated instructions are returned in `minstrVec'.
// Any temp. registers (TmpInstruction) created are returned in `tempVec'.
// 
void
UltraSparcInstrInfo::CreateCodeToLoadConst(Value* val,
                                       Instruction* dest,
                                       vector<MachineInstr*>& minstrVec,
                                       vector<TmpInstruction*>& tempVec) const
{
  MachineInstr* minstr;
  
  assert(isa<ConstPoolVal>(val) || isa<GlobalValue>(val) &&
         "I only know about constant values and global addresses");
  
  // Use a "set" instruction for known constants that can go in an integer reg.
  // Use a "load" instruction for all other constants, in particular,
  // floating point constants and addresses of globals.
  // 
  const Type* valType = val->getType();
  
  if (valType->isIntegral() || valType == Type::BoolTy)
    {
      bool isValidConstant;
      int64_t C = GetConstantValueAsSignedInt(val, isValidConstant);
      assert(isValidConstant && "Unrecognized constant");
      minstr = CreateIntSetInstruction(C, valType->isSigned(), dest);
      minstrVec.push_back(minstr);
    }
  else
    {
      // Make an instruction sequence to load the constant, viz:
      //            SETSW <addr-of-constant>, addrReg
      //            LOAD  /*addr*/ addrReg, /*offset*/ 0, dest
      // Only the SETSW is needed if `val' is a GlobalValue, i.e,. it is
      // itself a constant address.  Otherwise, both are needed.
      
      Value* addrVal;
      int64_t zeroOffset = 0; // to avoid ambiguity with (Value*) 0
      
      if (isa<ConstPoolVal>(val))
        {
          // Create another TmpInstruction for the hidden integer register
          TmpInstruction* addrReg =
            new TmpInstruction(Instruction::UserOp1, val, NULL);
          tempVec.push_back(addrReg);
          addrVal = addrReg;
        }
      else
        addrVal = dest;
      
      minstr = new MachineInstr(SETUW);
      minstr->SetMachineOperand(0, MachineOperand::MO_PCRelativeDisp, val);
      minstr->SetMachineOperand(1, MachineOperand::MO_VirtualRegister,addrVal);
      minstrVec.push_back(minstr);
      
      if (isa<ConstPoolVal>(val))
        {
          // addrVal->addMachineInstruction(minstr);
      
          minstr = new MachineInstr(ChooseLoadInstruction(val->getType()));
          minstr->SetMachineOperand(0, MachineOperand::MO_VirtualRegister,
                                       addrVal);
          minstr->SetMachineOperand(1, MachineOperand::MO_SignExtendedImmed,
                                       zeroOffset);
          minstr->SetMachineOperand(2, MachineOperand::MO_VirtualRegister,
                                       dest);
          minstrVec.push_back(minstr);
        }
    }
}


//************************ External Functions ******************************/

