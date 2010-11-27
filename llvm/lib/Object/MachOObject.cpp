//===- MachOObject.cpp - Mach-O Object File Wrapper -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/MachOObject.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace object;

MachOObject::MachOObject(MemoryBuffer *Buffer_, bool IsLittleEndian_,
                         bool Is64Bit_)
  : Buffer(Buffer_), IsLittleEndian(IsLittleEndian_), Is64Bit(Is64Bit_) {
}

MachOObject *MachOObject::LoadFromBuffer(MemoryBuffer *Buffer,
                                         std::string *ErrorStr) {
  // First, check the magic value and initialize the basic object info.
  bool IsLittleEndian = false, Is64Bit = false;
  StringRef Magic = Buffer->getBuffer().slice(0, 4);
  if (Magic == "\xFE\xED\xFA\xCE") {
  }  else if (Magic == "\xCE\xFA\xED\xFE") {
    IsLittleEndian = true;
  } else if (Magic == "\xFE\xED\xFA\xCF") {
    Is64Bit = true;
  } else if (Magic == "\xCF\xFA\xED\xFE") {
    IsLittleEndian = true;
    Is64Bit = true;
  } else {
    *ErrorStr = "not a Mach object file";
    return 0;
  }

  OwningPtr<MachOObject> Object(new MachOObject(Buffer, IsLittleEndian,
                                                Is64Bit));

  if (ErrorStr) *ErrorStr = "";
  return Object.take();
}
