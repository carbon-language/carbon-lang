// $Id$
//***************************************************************************
// File:
//	TargetMachine.cpp
// 
// Purpose:
//	
// History:
//	7/12/01	 -  Vikram Adve  -  Created
//**************************************************************************/


//*************************** User Include Files ***************************/

#include "llvm/CodeGen/TargetMachine.h"
#include "llvm/DerivedTypes.h"

//************************ Exported Constants ******************************/


// External object describing the machine instructions
// Initialized only when the TargetMachine class is created
// and reset when that class is destroyed.
// 
const MachineInstrDescriptor* TargetInstrDescriptors = NULL;


//************************ Class Implementations **************************/

//---------------------------------------------------------------------------
// class TargetMachine
// 
// Purpose:
//   Machine description.
// 
//---------------------------------------------------------------------------


// function TargetMachine::findOptimalStorageSize 
// 
// Purpose:
//   This default implementation assumes that all sub-word data items use
//   space equal to optSizeForSubWordData, and all other primitive data
//   items use space according to the type.
//   
unsigned int TargetMachine::findOptimalStorageSize(const Type* ty) const {
  switch(ty->getPrimitiveID()) {
  case Type::BoolTyID:
  case Type::UByteTyID:
  case Type::SByteTyID:     
  case Type::UShortTyID:
  case Type::ShortTyID:     
    return optSizeForSubWordData;
    
  default:
    return DataLayout.getTypeSize(ty);
  }
}


//---------------------------------------------------------------------------
// class MachineInstructionInfo
//	Interface to description of machine instructions
//---------------------------------------------------------------------------


/*ctor*/
MachineInstrInfo::MachineInstrInfo(const MachineInstrDescriptor* _desc,
				   unsigned int _descSize)
  : desc(_desc), descSize(_descSize)
{
  assert(TargetInstrDescriptors == NULL && desc != NULL);
  TargetInstrDescriptors = desc;	// initialize global variable
}  


/*dtor*/
MachineInstrInfo::~MachineInstrInfo()
{
  TargetInstrDescriptors = NULL;	// reset global variable
}


bool
MachineInstrInfo::constantFitsInImmedField(MachineOpCode opCode,
					   int64_t intValue) const
{
  // First, check if opCode has an immed field.
  bool isSignExtended;
  uint64_t maxImmedValue = this->maxImmedConstant(opCode, isSignExtended);
  if (maxImmedValue != 0)
    {
      // Now check if the constant fits
      if (intValue <= (int64_t) maxImmedValue &&
	  intValue >= -((int64_t) maxImmedValue+1))
	return true;
    }
  
  return false;
}

//---------------------------------------------------------------------------
