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
#include "llvm/System/Host.h"
#include "llvm/System/SwapByteOrder.h"

using namespace llvm;
using namespace llvm::object;

template<typename T>
static void SwapValue(T &Value) {
  Value = sys::SwapByteOrder(Value);
}

MachOObject::MachOObject(MemoryBuffer *Buffer_, bool IsLittleEndian_,
                         bool Is64Bit_)
  : Buffer(Buffer_), IsLittleEndian(IsLittleEndian_), Is64Bit(Is64Bit_),
    IsSwappedEndian(IsLittleEndian != sys::isLittleEndianHost()),
    LoadCommands(0), NumLoadedCommands(0) {
  // Load the common header.
  memcpy(&Header, Buffer->getBuffer().data(), sizeof(Header));
  if (IsSwappedEndian) {
    SwapValue(Header.Magic);
    SwapValue(Header.CPUType);
    SwapValue(Header.CPUSubtype);
    SwapValue(Header.FileType);
    SwapValue(Header.NumLoadCommands);
    SwapValue(Header.SizeOfLoadCommands);
    SwapValue(Header.Flags);
  }

  if (is64Bit()) {
    memcpy(&Header64Ext, Buffer->getBuffer().data() + sizeof(Header),
           sizeof(Header64Ext));
    if (IsSwappedEndian) {
      SwapValue(Header64Ext.Reserved);
    }
  }

  // Create the load command array if sane.
  if (getHeader().NumLoadCommands < (1 << 20))
    LoadCommands = new LoadCommandInfo[getHeader().NumLoadCommands];
}

MachOObject::~MachOObject() {
  delete LoadCommands;
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
    if (ErrorStr) *ErrorStr = "not a Mach object file (invalid magic)";
    return 0;
  }

  // Ensure that the at least the full header is present.
  unsigned HeaderSize = Is64Bit ? macho::Header64Size : macho::Header32Size;
  if (Buffer->getBufferSize() < HeaderSize) {
    if (ErrorStr) *ErrorStr = "not a Mach object file (invalid header)";
    return 0;
  }

  OwningPtr<MachOObject> Object(new MachOObject(Buffer, IsLittleEndian,
                                                Is64Bit));

  // Check for bogus number of load commands.
  if (Object->getHeader().NumLoadCommands >= (1 << 20)) {
    if (ErrorStr) *ErrorStr = "not a Mach object file (unreasonable header)";
    return 0;
  }

  if (ErrorStr) *ErrorStr = "";
  return Object.take();
}

const MachOObject::LoadCommandInfo &
MachOObject::getLoadCommandInfo(unsigned Index) const {
  assert(Index < getHeader().NumLoadCommands && "Invalid index!");

  // Load the command, if necessary.
  if (Index >= NumLoadedCommands) {
    uint64_t Offset;
    if (Index == 0) {
      Offset = getHeaderSize();
    } else {
      const LoadCommandInfo &Prev = getLoadCommandInfo(Index - 1);
      Offset = Prev.Offset + Prev.Command.Size;
    }

    LoadCommandInfo &Info = LoadCommands[Index];
    memcpy(&Info.Command, Buffer->getBuffer().data() + Offset,
           sizeof(macho::LoadCommand));
    if (IsSwappedEndian) {
      SwapValue(Info.Command.Type);
      SwapValue(Info.Command.Size);
    }
    Info.Offset = Offset;
    NumLoadedCommands = Index + 1;
  }

  return LoadCommands[Index];
}
