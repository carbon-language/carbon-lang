//===- llvm/ADT/SmallVector.cpp - 'Normally small' vectors ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SmallVector class.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
using namespace llvm;

/// grow_pod - This is an implementation of the grow() method which only works
/// on POD-like datatypes and is out of line to reduce code duplication.
void SmallVectorBase::grow_pod(size_t MinSizeInBytes, size_t TSize) {
  size_t CurSizeBytes = size_in_bytes();
  size_t NewCapacityInBytes = 2 * capacity_in_bytes();
  if (NewCapacityInBytes < MinSizeInBytes)
    NewCapacityInBytes = MinSizeInBytes;
  void *NewElts = operator new(NewCapacityInBytes);
  
  // Copy the elements over.  No need to run dtors on PODs.
  memcpy(NewElts, this->BeginX, CurSizeBytes);
  
  // If this wasn't grown from the inline copy, deallocate the old space.
  if (!this->isSmall())
    operator delete(this->BeginX);
  
  this->EndX = (char*)NewElts+CurSizeBytes;
  this->BeginX = NewElts;
  this->CapacityX = (char*)this->BeginX + NewCapacityInBytes;
}

