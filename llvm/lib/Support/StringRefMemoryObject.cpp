//===- lib/Support/StringRefMemoryObject.cpp --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/StringRefMemoryObject.h"

using namespace llvm;

int StringRefMemoryObject::readByte(uint64_t Addr, uint8_t *Byte) const {
  if (Addr >= Base + getExtent() || Addr < Base)
    return -1;
  *Byte = Bytes[Addr - Base];
  return 0;
}

int StringRefMemoryObject::readBytes(uint64_t Addr,
                                     uint64_t Size,
                                     uint8_t *Buf) const {
  uint64_t Offset = Addr - Base;
  if (Addr >= Base + getExtent() || Offset + Size > getExtent() || Addr < Base)
    return -1;
  memcpy(Buf, Bytes.data() + Offset, Size);
  return 0;
}
