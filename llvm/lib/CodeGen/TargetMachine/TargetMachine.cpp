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


//************************ Class Implementations **************************/

//---------------------------------------------------------------------------
// function TargetMachine::findOptimalMemberOffsets 
// 
// Purpose:
//   Compute optimal offsets for the members of a structure.
//   Returns a vector of unsigned ints, one per member.
//   Caller is responsible for freeing the vector.
//---------------------------------------------------------------------------

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
