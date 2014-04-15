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
void SwapStruct(MachO::fat_header &H) {
  SwapValue(H.magic);
  SwapValue(H.nfat_arch);
}

template<>
void SwapStruct(MachO::fat_arch &H) {
  SwapValue(H.cputype);
  SwapValue(H.cpusubtype);
  SwapValue(H.offset);
  SwapValue(H.size);
  SwapValue(H.align);
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
  if (!Parent || Index > Parent->getNumberOfObjects()) {
    clear();
  } else {
    // Parse object header.
    StringRef ParentData = Parent->getData();
    const char *HeaderPos = ParentData.begin() + sizeof(MachO::fat_header) +
                            Index * sizeof(MachO::fat_arch);
    Header = getUniversalBinaryStruct<MachO::fat_arch>(HeaderPos);
    if (ParentData.size() < Header.offset + Header.size) {
      clear();
    }
  }
}

error_code MachOUniversalBinary::ObjectForArch::getAsObjectFile(
    std::unique_ptr<ObjectFile> &Result) const {
  if (Parent) {
    StringRef ParentData = Parent->getData();
    StringRef ObjectData = ParentData.substr(Header.offset, Header.size);
    std::string ObjectName =
        Parent->getFileName().str() + ":" +
        Triple::getArchTypeName(MachOObjectFile::getArch(Header.cputype));
    MemoryBuffer *ObjBuffer = MemoryBuffer::getMemBuffer(
        ObjectData, ObjectName, false);
    ErrorOr<ObjectFile *> Obj = ObjectFile::createMachOObjectFile(ObjBuffer);
    if (error_code EC = Obj.getError())
      return EC;
    Result.reset(Obj.get());
    return object_error::success;
  }
  return object_error::parse_failed;
}

void MachOUniversalBinary::anchor() { }

ErrorOr<MachOUniversalBinary *>
MachOUniversalBinary::create(MemoryBuffer *Source) {
  error_code EC;
  std::unique_ptr<MachOUniversalBinary> Ret(
      new MachOUniversalBinary(Source, EC));
  if (EC)
    return EC;
  return Ret.release();
}

MachOUniversalBinary::MachOUniversalBinary(MemoryBuffer *Source,
                                           error_code &ec)
  : Binary(Binary::ID_MachOUniversalBinary, Source),
    NumberOfObjects(0) {
  if (Source->getBufferSize() < sizeof(MachO::fat_header)) {
    ec = object_error::invalid_file_type;
    return;
  }
  // Check for magic value and sufficient header size.
  StringRef Buf = getData();
  MachO::fat_header H= getUniversalBinaryStruct<MachO::fat_header>(Buf.begin());
  NumberOfObjects = H.nfat_arch;
  uint32_t MinSize = sizeof(MachO::fat_header) +
                     sizeof(MachO::fat_arch) * NumberOfObjects;
  if (H.magic != MachO::FAT_MAGIC || Buf.size() < MinSize) {
    ec = object_error::parse_failed;
    return;
  }
  ec = object_error::success;
}

static bool getCTMForArch(Triple::ArchType Arch, MachO::CPUType &CTM) {
  switch (Arch) {
    case Triple::x86:    CTM = MachO::CPU_TYPE_I386; return true;
    case Triple::x86_64: CTM = MachO::CPU_TYPE_X86_64; return true;
    case Triple::arm:    CTM = MachO::CPU_TYPE_ARM; return true;
    case Triple::sparc:  CTM = MachO::CPU_TYPE_SPARC; return true;
    case Triple::ppc:    CTM = MachO::CPU_TYPE_POWERPC; return true;
    case Triple::ppc64:  CTM = MachO::CPU_TYPE_POWERPC64; return true;
    default: return false;
  }
}

error_code MachOUniversalBinary::getObjectForArch(
    Triple::ArchType Arch, std::unique_ptr<ObjectFile> &Result) const {
  MachO::CPUType CTM;
  if (!getCTMForArch(Arch, CTM))
    return object_error::arch_not_found;
  for (object_iterator I = begin_objects(), E = end_objects(); I != E; ++I) {
    if (I->getCPUType() == static_cast<uint32_t>(CTM))
      return I->getAsObjectFile(Result);
  }
  return object_error::arch_not_found;
}
