//===- MachOObjectFile.cpp - Mach-O object file binding ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MachOObjectFile class, which binds the MachOObject
// class to the generic ObjectFile wrapper.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/MachO.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Object/MachOFormat.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cctype>
#include <cstring>
#include <limits>

using namespace llvm;
using namespace object;

namespace llvm {
namespace object {

MachOObjectFileBase::MachOObjectFileBase(MemoryBuffer *Object,
                                         bool IsLittleEndian, bool Is64bits,
                                         error_code &ec)
    : ObjectFile(getMachOType(IsLittleEndian, Is64bits), Object) {
}

bool MachOObjectFileBase::is64Bit() const {
  return isa<MachOObjectFileLE64>(this) || isa<MachOObjectFileBE64>(this);
}

void MachOObjectFileBase::ReadULEB128s(uint64_t Index,
                                       SmallVectorImpl<uint64_t> &Out) const {
  DataExtractor extractor(ObjectFile::getData(), true, 0);

  uint32_t offset = Index;
  uint64_t data = 0;
  while (uint64_t delta = extractor.getULEB128(&offset)) {
    data += delta;
    Out.push_back(data);
  }
}

unsigned MachOObjectFileBase::getHeaderSize() const {
  return is64Bit() ? macho::Header64Size : macho::Header32Size;
}

StringRef MachOObjectFileBase::getData(size_t Offset, size_t Size) const {
  return ObjectFile::getData().substr(Offset, Size);
}

ObjectFile *ObjectFile::createMachOObjectFile(MemoryBuffer *Buffer) {
  StringRef Magic = Buffer->getBuffer().slice(0, 4);
  error_code ec;
  ObjectFile *Ret;
  if (Magic == "\xFE\xED\xFA\xCE")
    Ret = new MachOObjectFileBE32(Buffer, ec);
  else if (Magic == "\xCE\xFA\xED\xFE")
    Ret = new MachOObjectFileLE32(Buffer, ec);
  else if (Magic == "\xFE\xED\xFA\xCF")
    Ret = new MachOObjectFileBE64(Buffer, ec);
  else if (Magic == "\xCF\xFA\xED\xFE")
    Ret = new MachOObjectFileLE64(Buffer, ec);
  else
    return NULL;

  if (ec)
    return NULL;
  return Ret;
}

/*===-- Symbols -----------------------------------------------------------===*/

error_code MachOObjectFileBase::getSymbolValue(DataRefImpl Symb,
                                               uint64_t &Val) const {
  report_fatal_error("getSymbolValue unimplemented in MachOObjectFileBase");
}

symbol_iterator MachOObjectFileBase::begin_dynamic_symbols() const {
  // TODO: implement
  report_fatal_error("Dynamic symbols unimplemented in MachOObjectFileBase");
}

symbol_iterator MachOObjectFileBase::end_dynamic_symbols() const {
  // TODO: implement
  report_fatal_error("Dynamic symbols unimplemented in MachOObjectFileBase");
}

library_iterator MachOObjectFileBase::begin_libraries_needed() const {
  // TODO: implement
  report_fatal_error("Needed libraries unimplemented in MachOObjectFileBase");
}

library_iterator MachOObjectFileBase::end_libraries_needed() const {
  // TODO: implement
  report_fatal_error("Needed libraries unimplemented in MachOObjectFileBase");
}

StringRef MachOObjectFileBase::getLoadName() const {
  // TODO: Implement
  report_fatal_error("get_load_name() unimplemented in MachOObjectFileBase");
}

/*===-- Sections ----------------------------------------------------------===*/

std::size_t MachOObjectFileBase::getSectionIndex(DataRefImpl Sec) const {
  SectionList::const_iterator loc =
    std::find(Sections.begin(), Sections.end(), Sec);
  assert(loc != Sections.end() && "Sec is not a valid section!");
  return std::distance(Sections.begin(), loc);
}

StringRef MachOObjectFileBase::parseSegmentOrSectionName(const char *P) const {
  if (P[15] == 0)
    // Null terminated.
    return P;
  // Not null terminated, so this is a 16 char string.
  return StringRef(P, 16);
}

error_code MachOObjectFileBase::isSectionData(DataRefImpl DRI,
                                              bool &Result) const {
  // FIXME: Unimplemented.
  Result = false;
  return object_error::success;
}

error_code MachOObjectFileBase::isSectionBSS(DataRefImpl DRI,
                                             bool &Result) const {
  // FIXME: Unimplemented.
  Result = false;
  return object_error::success;
}

error_code
MachOObjectFileBase::isSectionRequiredForExecution(DataRefImpl Sec,
                                                   bool &Result) const {
  // FIXME: Unimplemented.
  Result = true;
  return object_error::success;
}

error_code MachOObjectFileBase::isSectionVirtual(DataRefImpl Sec,
                                                 bool &Result) const {
  // FIXME: Unimplemented.
  Result = false;
  return object_error::success;
}

error_code MachOObjectFileBase::isSectionReadOnlyData(DataRefImpl Sec,
                                                      bool &Result) const {
  // Consider using the code from isSectionText to look for __const sections.
  // Alternately, emit S_ATTR_PURE_INSTRUCTIONS and/or S_ATTR_SOME_INSTRUCTIONS
  // to use section attributes to distinguish code from data.

  // FIXME: Unimplemented.
  Result = false;
  return object_error::success;
}

relocation_iterator MachOObjectFileBase::getSectionRelBegin(DataRefImpl Sec) const {
  DataRefImpl ret;
  ret.d.b = getSectionIndex(Sec);
  return relocation_iterator(RelocationRef(ret, this));
}


/*===-- Relocations -------------------------------------------------------===*/

error_code MachOObjectFileBase::getRelocationNext(DataRefImpl Rel,
                                                  RelocationRef &Res) const {
  ++Rel.d.a;
  Res = RelocationRef(Rel, this);
  return object_error::success;
}

error_code MachOObjectFileBase::getLibraryNext(DataRefImpl LibData,
                                               LibraryRef &Res) const {
  report_fatal_error("Needed libraries unimplemented in MachOObjectFileBase");
}

error_code MachOObjectFileBase::getLibraryPath(DataRefImpl LibData,
                                               StringRef &Res) const {
  report_fatal_error("Needed libraries unimplemented in MachOObjectFileBase");
}

error_code MachOObjectFileBase::getRelocationAdditionalInfo(DataRefImpl Rel,
                                                           int64_t &Res) const {
  Res = 0;
  return object_error::success;
}


/*===-- Miscellaneous -----------------------------------------------------===*/

uint8_t MachOObjectFileBase::getBytesInAddress() const {
  return is64Bit() ? 8 : 4;
}

} // end namespace object
} // end namespace llvm
