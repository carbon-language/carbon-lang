//===-- TargetMachine.cpp - General Target Information ---------------------==//
//
// This file describes the general parts of a Target machine.
// This file also implements MachineCacheInfo.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/MachineCacheInfo.h"
#include "llvm/Type.h"

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
// class MachineCacheInfo 
// 
// Purpose:
//   Describes properties of the target cache architecture.
//---------------------------------------------------------------------------

void MachineCacheInfo::Initialize() {
  numLevels = 2;
  cacheLineSizes.push_back(16);  cacheLineSizes.push_back(32); 
  cacheSizes.push_back(1 << 15); cacheSizes.push_back(1 << 20);
  cacheAssoc.push_back(1);       cacheAssoc.push_back(4);
}
