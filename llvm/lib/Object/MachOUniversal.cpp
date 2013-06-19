//===- MachOUniversal.cpp - Mach-O universal binary -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MachOUniversalBinary class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/MachOUniversal.h"

#include "llvm/Object/MachO.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace object;

template<typename T>
static void SwapValue(T &Value) {
  Value = sys::SwapByteOrder(Value);
}

template<typename T>
static void SwapStruct(T &Value);

template<>
void SwapStruct(macho::FatHeader &H) {
  SwapValue(H.Magic);
  SwapValue(H.NumFatArch);
}

template<>
void SwapStruct(macho::FatArchHeader &H) {
  SwapValue(H.CPUType);
  SwapValue(H.CPUSubtype);
  SwapValue(H.Offset);
  SwapValue(H.Size);
  SwapValue(H.Align);
}

template<typename T>
static T getUniversalBinaryStruct(const char *Ptr) {
  T Res;
  memcpy(&Res, Ptr, sizeof(T));
  // Universal binary headers have big-endian byte order.
  if (sys::IsLittleEndianHost)
    SwapStruct(Res);
  return Res;
}

MachOUniversalBinary::ObjectForArch::ObjectForArch(
    const MachOUniversalBinary *Parent, uint32_t Index)
    : Parent(Parent), Index(Index) {
  if (Parent == 0 || Index > Parent->getNumberOfObjects()) {
    clear();
  } else {
    // Parse object header.
    StringRef ParentData = Parent->getData();
    const char *HeaderPos = ParentData.begin() + macho::FatHeaderSize +
                            Index * macho::FatArchHeaderSize;
    Header = getUniversalBinaryStruct<macho::FatArchHeader>(HeaderPos);
    if (ParentData.size() < Header.Offset + Header.Size) {
      clear();
    }
  }
}

error_code MachOUniversalBinary::ObjectForArch::getAsObjectFile(
    OwningPtr<ObjectFile> &Result) const {
  if (Parent) {
    StringRef ParentData = Parent->getData();
    StringRef ObjectData = ParentData.substr(Header.Offset, Header.Size);
    std::string ObjectName =
        Parent->getFileName().str() + ":" +
        Triple::getArchTypeName(MachOObjectFile::getArch(Header.CPUType));
    MemoryBuffer *ObjBuffer = MemoryBuffer::getMemBuffer(
        ObjectData, ObjectName, false);
    if (ObjectFile *Obj = ObjectFile::createMachOObjectFile(ObjBuffer)) {
      Result.reset(Obj);
      return object_error::success;
    }
  }
  return object_error::parse_failed;
}

void MachOUniversalBinary::anchor() { }

MachOUniversalBinary::MachOUniversalBinary(MemoryBuffer *Source,
                                           error_code &ec)
  : Binary(Binary::ID_MachOUniversalBinary, Source),
    NumberOfObjects(0) {
  if (Source->getBufferSize() < macho::FatHeaderSize) {
    ec = object_error::invalid_file_type;
    return;
  }
  // Check for magic value and sufficient header size.
  StringRef Buf = getData();
  macho::FatHeader H = getUniversalBinaryStruct<macho::FatHeader>(Buf.begin());
  NumberOfObjects = H.NumFatArch;
  uint32_t MinSize = macho::FatHeaderSize +
                     macho::FatArchHeaderSize * NumberOfObjects;
  if (H.Magic != macho::HM_Universal || Buf.size() < MinSize) {
    ec = object_error::parse_failed;
    return;
  }
  ec = object_error::success;
}

static bool getCTMForArch(Triple::ArchType Arch, mach::CPUTypeMachine &CTM) {
  switch (Arch) {
    case Triple::x86:    CTM = mach::CTM_i386; return true;
    case Triple::x86_64: CTM = mach::CTM_x86_64; return true;
    case Triple::arm:    CTM = mach::CTM_ARM; return true;
    case Triple::sparc:  CTM = mach::CTM_SPARC; return true;
    case Triple::ppc:    CTM = mach::CTM_PowerPC; return true;
    case Triple::ppc64:  CTM = mach::CTM_PowerPC64; return true;
    default: return false;
  }
}

error_code
MachOUniversalBinary::getObjectForArch(Triple::ArchType Arch,
                                       OwningPtr<ObjectFile> &Result) const {
  mach::CPUTypeMachine CTM;
  if (!getCTMForArch(Arch, CTM))
    return object_error::arch_not_found;
  for (object_iterator I = begin_objects(), E = end_objects(); I != E; ++I) {
    if (I->getCPUType() == static_cast<uint32_t>(CTM))
      return I->getAsObjectFile(Result);
  }
  return object_error::arch_not_found;
}
