//===-- TargetMachine.cpp - General Target Information ---------------------==//
//
// This file describes the general parts of a Target machine.
// This file also implements the InstInfo interface as well...
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/InstInfo.h"
#include "llvm/DerivedTypes.h"

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
				   unsigned int _descSize,
				   unsigned int _numRealOpCodes)
  : desc(_desc), descSize(_descSize), numRealOpCodes(_numRealOpCodes)
{
  // FIXME: TargetInstrDescriptors should not be global
  assert(TargetInstrDescriptors == NULL && desc != NULL);
  TargetInstrDescriptors = desc;	// initialize global variable
}  


MachineInstrInfo::~MachineInstrInfo() {
  TargetInstrDescriptors = NULL;	// reset global variable
}


bool
MachineInstrInfo::constantFitsInImmedField(MachineOpCode opCode,
					   int64_t intValue) const
{
  // First, check if opCode has an immed field.
  bool isSignExtended;
  uint64_t maxImmedValue = maxImmedConstant(opCode, isSignExtended);
  if (maxImmedValue != 0) {
    // Now check if the constant fits
    if (intValue <= (int64_t) maxImmedValue &&
	intValue >= -((int64_t) maxImmedValue+1))
      return true;
  }
  
  return false;
}
