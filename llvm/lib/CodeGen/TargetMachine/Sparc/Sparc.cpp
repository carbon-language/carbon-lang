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
  : TargetMachine("UltraSparc-Native", new UltraSparcInstrInfo()) {
  optSizeForSubWordData = 4;
  minMemOpWordSize = 8; 
  maxAtomicMemOpWordSize = 8;
  zeroRegNum = 0;			// %g0 always gives 0 on Sparc
}

UltraSparc::~UltraSparc() {
}

//**************************************************************************/
