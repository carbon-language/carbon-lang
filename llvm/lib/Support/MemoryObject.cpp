//===- MemoryObject.cpp - Abstract memory interface -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/MemoryObject.h"
using namespace llvm;
  
MemoryObject::~MemoryObject() {
}

int MemoryObject::readBytes(uint64_t address,
                            uint64_t size,
                            uint8_t* buf) const {
  uint64_t current = address;
  uint64_t limit = getBase() + getExtent();

  if (current + size > limit)
    return -1;

  while (current - address < size) {
    if (readByte(current, &buf[(current - address)]))
      return -1;
    
    current++;
  }
  
  return 0;
}
