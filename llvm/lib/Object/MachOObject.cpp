//===- MachOObject.cpp - Mach-O Object File Wrapper -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/MachOObject.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace object;

MachOObject::MachOObject(MemoryBuffer *Buffer_) : Buffer(Buffer_) {
}

MachOObject *MachOObject::LoadFromBuffer(MemoryBuffer *Buffer,
                                         std::string *ErrorStr) {
  if (ErrorStr) *ErrorStr = "";
  return new MachOObject(Buffer);
}
