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
//   Compute optimal storage size for a structure, based on
//   the optimal member offsets.
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
    case Type::UByteTyID:	case Type::SByteTyID:     
    case Type::UShortTyID:	case Type::ShortTyID:     
      return optSizeForSubWordData;
      break;
      
    case Type::UIntTyID:	case Type::IntTyID:     
      return intSize;
      break;
      
    case Type::FloatTyID:
      return floatSize;
      break;
       
    case Type::ULongTyID:	case Type::LongTyID:
      return longSize;
      break;
      
    case Type::DoubleTyID:
      return doubleSize;
      break;
	
    case Type::PointerTyID:     
    case Type::LabelTyID:	case Type::MethodTyID:     
      return pointerSize;
      break;
    
    case Type::ArrayTyID:
      {
      ArrayType* aty = (ArrayType*) ty;
      assert(aty->getNumElements() >= 0 &&
	     "Attempting to compute size for unknown-size array");
      return (unsigned) aty->getNumElements() *
	this->findOptimalStorageSize(aty->getElementType());
      break;
      }
      
    case Type::StructTyID:     
      {// This code should be invoked only from StructType::getStorageSize().
      StructType* sty = (StructType*) ty;
      unsigned lastMemberIdx = sty->getElementTypes().size() - 1;
      unsigned lastMemberOffset = sty->getElementOffset(lastMemberIdx, *this);
      unsigned storageSize = lastMemberOffset
	+ this->findOptimalStorageSize(sty->getElementTypes()[lastMemberIdx]);
      return storageSize;
      break;
      }
      
    default:
      assert(0 && "Unexpected type in `findOptimalStorageSize'");
      return 0;
      break;
    }
}


// function TargetMachine::findOptimalMemberOffsets 
// 
// Purpose:
//   Compute optimal offsets for the members of a structure.
//   Returns a vector of unsigned ints, one per member.
//   Caller is responsible for freeing the vector.

unsigned int*
TargetMachine::findOptimalMemberOffsets(const StructType* stype) const
{
  int numMembers = stype->getElementTypes().size();
  unsigned int* offsetVec = new unsigned int[numMembers];
  unsigned int netOffset = 0;
  for (int i = 0; i < numMembers; i++)
    {
      offsetVec[i] = netOffset;
      const Type* memberType = stype->getElementTypes()[i];
      netOffset += this->findOptimalStorageSize(memberType);
    }
  return offsetVec;
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
