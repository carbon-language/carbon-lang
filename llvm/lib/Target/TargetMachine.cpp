//===-- TargetMachine.cpp - General Target Information ---------------------==//
//
// This file describes the general parts of a Target machine.
// This file also implements MachineInstrInfo and MachineCacheInfo.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/MachineInstrInfo.h"
#include "llvm/Target/MachineCacheInfo.h"
#include "llvm/Function.h"

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
unsigned int
TargetMachine::findOptimalStorageSize(const Type* ty) const
{
  switch(ty->getPrimitiveID())
    {
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
MachineInstrInfo::MachineInstrInfo(const TargetMachine& tgt,
                                   const MachineInstrDescriptor* _desc,
				   unsigned int _descSize,
				   unsigned int _numRealOpCodes)
  : target(tgt),
    desc(_desc), descSize(_descSize), numRealOpCodes(_numRealOpCodes)
{
  // FIXME: TargetInstrDescriptors should not be global
  assert(TargetInstrDescriptors == NULL && desc != NULL);
  TargetInstrDescriptors = desc;	// initialize global variable
}  


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
  uint64_t maxImmedValue = maxImmedConstant(opCode, isSignExtended);
  if (maxImmedValue != 0)
    {
      // NEED TO HANDLE UNSIGNED VALUES SINCE THEY MAY BECOME MUCH
      // SMALLER AFTER CASTING TO SIGN-EXTENDED int, short, or char.
      // See CreateUIntSetInstruction in SparcInstrInfo.cpp.
      
      // Now check if the constant fits
      if (intValue <= (int64_t) maxImmedValue &&
	  intValue >= -((int64_t) maxImmedValue+1))
	return true;
    }
  
  return false;
}


//---------------------------------------------------------------------------
// class MachineCacheInfo 
// 
// Purpose:
//   Describes properties of the target cache architecture.
//---------------------------------------------------------------------------

/*ctor*/
MachineCacheInfo::MachineCacheInfo(const TargetMachine& tgt)
  : target(tgt)
{
  Initialize();
}

void
MachineCacheInfo::Initialize()
{
  numLevels = 2;
  cacheLineSizes.push_back(16);  cacheLineSizes.push_back(32); 
  cacheSizes.push_back(1 << 15); cacheSizes.push_back(1 << 20);
  cacheAssoc.push_back(1);       cacheAssoc.push_back(4);
}
