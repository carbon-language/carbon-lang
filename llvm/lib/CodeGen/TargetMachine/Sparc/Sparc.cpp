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

//************************ Class Implementations **************************/



//---------------------------------------------------------------------------
// class UltraSparcMachine 
// 
// Purpose:
//   Primary interface to machine description for the UltraSPARC.
//   Primarily just initializes machine-dependent parameters in
//   class TargetMachine, and creates machine-dependent subclasses
//   for classes such as MachineInstrInfo. 
// 
//---------------------------------------------------------------------------

UltraSparc::UltraSparc()
  : TargetMachine(new UltraSparcInstrInfo)
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
  zeroRegNum = 0;			// %g0 always gives 0 on Sparc
}

UltraSparc::~UltraSparc()
{
  delete (UltraSparcInstrInfo*) machineInstrInfo;
}

//**************************************************************************/
