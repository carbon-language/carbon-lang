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
unsigned TargetMachine::findOptimalStorageSize(const Type *Ty) const {
  // All integer types smaller than ints promote to 4 byte integers.
  if (Ty->isIntegral() && Ty->getPrimitiveSize() < 4)
    return 4;

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
