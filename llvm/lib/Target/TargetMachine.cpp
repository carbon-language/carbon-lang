//===-- TargetMachine.cpp - General Target Information ---------------------==//
//
// This file describes the general parts of a Target machine.
// This file also implements TargetCacheInfo.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetCacheInfo.h"
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
unsigned TargetMachine::findOptimalStorageSize(const Type *Ty) const {
  // Round integral values smaller than SubWordDataSize up to SubWordDataSize
  if (Ty->isIntegral() &&
      Ty->getPrimitiveSize() < DataLayout.getSubWordDataSize())
    return DataLayout.getSubWordDataSize();

  return DataLayout.getTypeSize(Ty);
}


//---------------------------------------------------------------------------
// class TargetCacheInfo 
// 
// Purpose:
//   Describes properties of the target cache architecture.
//---------------------------------------------------------------------------

void TargetCacheInfo::Initialize() {
  numLevels = 2;
  cacheLineSizes.push_back(16);  cacheLineSizes.push_back(32); 
  cacheSizes.push_back(1 << 15); cacheSizes.push_back(1 << 20);
  cacheAssoc.push_back(1);       cacheAssoc.push_back(4);
}
