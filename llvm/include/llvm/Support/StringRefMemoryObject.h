//===- llvm/Support/StringRefMemoryObject.h ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the StringRefMemObject class, a simple
// wrapper around StringRef implementing the MemoryObject interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_STRINGREFMEMORYOBJECT_H
#define LLVM_SUPPORT_STRINGREFMEMORYOBJECT_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryObject.h"

namespace llvm {

/// StringRefMemoryObject - Simple StringRef-backed MemoryObject
class StringRefMemoryObject : public MemoryObject {
  StringRef Bytes;
  uint64_t Base;
public:
  StringRefMemoryObject(StringRef Bytes, uint64_t Base = 0)
    : Bytes(Bytes), Base(Base) {}

  uint64_t getBase() const { return Base; }
  uint64_t getExtent() const { return Bytes.size(); }

  int readByte(uint64_t Addr, uint8_t *Byte) const;
  int readBytes(uint64_t Addr, uint64_t Size,
                uint8_t *Buf, uint64_t *Copied) const;

};

}

#endif
