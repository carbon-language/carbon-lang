// $Id$
//***************************************************************************
// File:
//	Sparc.cpp
// 
// Purpose:
//	
// History:
//	7/15/01	 -  Vikram Adve  -  Created
//**************************************************************************/

#include "llvm/CodeGen/Sparc.h"

//************************ Exported Constants ******************************/


// Set external object describing the machine instructions
// 
const MachineInstrInfo* TargetMachineInstrInfo = SparcMachineInstrInfo; 


//************************ Class Implementations **************************/


//---------------------------------------------------------------------------
// class UltraSparcMachine 
// 
// Purpose:
//   Machine description.
// 
//---------------------------------------------------------------------------

UltraSparc::UltraSparc()
  : TargetMachine()
{
  optSizeForSubWordData = 4;
  intSize = 4; 
  floatSize = 4; 
  longSize = 8; 
  doubleSize = 8; 
  longDoubleSize = 16; 
  pointerSize = 8;
  minMemOpWordSize = 8; 
  maxAtomicMemOpWordSize = 8;
  machineInstrInfo = SparcMachineInstrInfo;
  zeroRegNum = 0;			// %g0 always gives 0 on Sparc
}

//**************************************************************************/
