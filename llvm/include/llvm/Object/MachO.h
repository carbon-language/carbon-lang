//===- MachO.h - MachO object file implementation ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MachOObjectFile class, which binds the MachOObject
// class to the generic ObjectFile wrapper.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_MACHO_H
#define LLVM_OBJECT_MACHO_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Object/MachOObject.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MachO.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace object {

namespace MachOFormat {
  struct Section {
    char Name[16];
    char SegmentName[16];
    support::ulittle32_t Address;
    support::ulittle32_t Size;
    support::ulittle32_t Offset;
    support::ulittle32_t Align;
    support::ulittle32_t RelocationTableOffset;
    support::ulittle32_t NumRelocationTableEntries;
    support::ulittle32_t Flags;
    support::ulittle32_t Reserved1;
    support::ulittle32_t Reserved2;
  };

  struct Section64 {
    char Name[16];
    char SegmentName[16];
    support::ulittle64_t Address;
    support::ulittle64_t Size;
    support::ulittle32_t Offset;
    support::ulittle32_t Align;
    support::ulittle32_t RelocationTableOffset;
    support::ulittle32_t NumRelocationTableEntries;
    support::ulittle32_t Flags;
    support::ulittle32_t Reserved1;
    support::ulittle32_t Reserved2;
    support::ulittle32_t Reserved3;
  };

  struct RelocationEntry {
    support::ulittle32_t Word0;
    support::ulittle32_t Word1;
  };

  struct SymbolTableEntry {
    support::ulittle32_t StringIndex;
    uint8_t Type;
    uint8_t SectionIndex;
    support::ulittle16_t Flags;
    support::ulittle32_t Value;
  };

  struct Symbol64TableEntry {
    support::ulittle32_t StringIndex;
    uint8_t Type;
    uint8_t SectionIndex;
    support::ulittle16_t Flags;
    support::ulittle64_t Value;
  };

  struct SymtabLoadCommand {
    support::ulittle32_t Type;
    support::ulittle32_t Size;
    support::ulittle32_t SymbolTableOffset;
    support::ulittle32_t NumSymbolTableEntries;
    support::ulittle32_t StringTableOffset;
    support::ulittle32_t StringTableSize;
  };

  struct SegmentLoadCommand {
    support::ulittle32_t Type;
    support::ulittle32_t Size;
    char Name[16];
    support::ulittle32_t VMAddress;
    support::ulittle32_t VMSize;
    support::ulittle32_t FileOffset;
    support::ulittle32_t FileSize;
    support::ulittle32_t MaxVMProtection;
    support::ulittle32_t InitialVMProtection;
    support::ulittle32_t NumSections;
    support::ulittle32_t Flags;
  };

  struct Segment64LoadCommand {
    support::ulittle32_t Type;
    support::ulittle32_t Size;
    char Name[16];
    support::ulittle64_t VMAddress;
    support::ulittle64_t VMSize;
    support::ulittle64_t FileOffset;
    support::ulittle64_t FileSize;
    support::ulittle32_t MaxVMProtection;
    support::ulittle32_t InitialVMProtection;
    support::ulittle32_t NumSections;
    support::ulittle32_t Flags;
  };
}

typedef MachOObject::LoadCommandInfo LoadCommandInfo;

class MachOObjectFile : public ObjectFile {
public:
  MachOObjectFile(MemoryBuffer *Object, MachOObject *MOO, error_code &ec);

  virtual symbol_iterator begin_symbols() const;
  virtual symbol_iterator end_symbols() const;
  virtual symbol_iterator begin_dynamic_symbols() const;
  virtual symbol_iterator end_dynamic_symbols() const;
  virtual library_iterator begin_libraries_needed() const;
  virtual library_iterator end_libraries_needed() const;
  virtual section_iterator begin_sections() const;
  virtual section_iterator end_sections() const;

  virtual uint8_t getBytesInAddress() const;
  virtual StringRef getFileFormatName() const;
  virtual unsigned getArch() const;
  virtual StringRef getLoadName() const;

  // In a MachO file, sections have a segment name. This is used in the .o
  // files. They have a single segment, but this field specifies which segment
  // a section should be put in in the final object.
  StringRef getSectionFinalSegmentName(DataRefImpl Sec) const;

  // Names are stored as 16 bytes. These returns the raw 16 bytes without
  // interpreting them as a C string.
  ArrayRef<char> getSectionRawName(DataRefImpl Sec) const;
  ArrayRef<char>getSectionRawFinalSegmentName(DataRefImpl Sec) const;

  MachOObject *getObject() { return MachOObj.get(); }

  static inline bool classof(const Binary *v) {
    return v->isMachO();
  }

protected:
  virtual error_code getSymbolNext(DataRefImpl Symb, SymbolRef &Res) const;
  virtual error_code getSymbolName(DataRefImpl Symb, StringRef &Res) const;
  virtual error_code getSymbolFileOffset(DataRefImpl Symb, uint64_t &Res) const;
  virtual error_code getSymbolAddress(DataRefImpl Symb, uint64_t &Res) const;
  virtual error_code getSymbolSize(DataRefImpl Symb, uint64_t &Res) const;
  virtual error_code getSymbolNMTypeChar(DataRefImpl Symb, char &Res) const;
  virtual error_code getSymbolFlags(DataRefImpl Symb, uint32_t &Res) const;
  virtual error_code getSymbolType(DataRefImpl Symb, SymbolRef::Type &Res) const;
  virtual error_code getSymbolSection(DataRefImpl Symb,
                                      section_iterator &Res) const;
  virtual error_code getSymbolValue(DataRefImpl Symb, uint64_t &Val) const;

  virtual error_code getSectionNext(DataRefImpl Sec, SectionRef &Res) const;
  virtual error_code getSectionName(DataRefImpl Sec, StringRef &Res) const;
  virtual error_code getSectionAddress(DataRefImpl Sec, uint64_t &Res) const;
  virtual error_code getSectionSize(DataRefImpl Sec, uint64_t &Res) const;
  virtual error_code getSectionContents(DataRefImpl Sec, StringRef &Res) const;
  virtual error_code getSectionAlignment(DataRefImpl Sec, uint64_t &Res) const;
  virtual error_code isSectionText(DataRefImpl Sec, bool &Res) const;
  virtual error_code isSectionData(DataRefImpl Sec, bool &Res) const;
  virtual error_code isSectionBSS(DataRefImpl Sec, bool &Res) const;
  virtual error_code isSectionRequiredForExecution(DataRefImpl Sec,
                                                   bool &Res) const;
  virtual error_code isSectionVirtual(DataRefImpl Sec, bool &Res) const;
  virtual error_code isSectionZeroInit(DataRefImpl Sec, bool &Res) const;
  virtual error_code isSectionReadOnlyData(DataRefImpl Sec, bool &Res) const;
  virtual error_code sectionContainsSymbol(DataRefImpl DRI, DataRefImpl S,
                                           bool &Result) const;
  virtual relocation_iterator getSectionRelBegin(DataRefImpl Sec) const;
  virtual relocation_iterator getSectionRelEnd(DataRefImpl Sec) const;

  virtual error_code getRelocationNext(DataRefImpl Rel,
                                       RelocationRef &Res) const;
  virtual error_code getRelocationAddress(DataRefImpl Rel,
                                          uint64_t &Res) const;
  virtual error_code getRelocationOffset(DataRefImpl Rel,
                                         uint64_t &Res) const;
  virtual error_code getRelocationSymbol(DataRefImpl Rel,
                                         SymbolRef &Res) const;
  virtual error_code getRelocationType(DataRefImpl Rel,
                                       uint64_t &Res) const;
  virtual error_code getRelocationTypeName(DataRefImpl Rel,
                                           SmallVectorImpl<char> &Result) const;
  virtual error_code getRelocationAdditionalInfo(DataRefImpl Rel,
                                                 int64_t &Res) const;
  virtual error_code getRelocationValueString(DataRefImpl Rel,
                                           SmallVectorImpl<char> &Result) const;
  virtual error_code getRelocationHidden(DataRefImpl Rel, bool &Result) const;

  virtual error_code getLibraryNext(DataRefImpl LibData, LibraryRef &Res) const;
  virtual error_code getLibraryPath(DataRefImpl LibData, StringRef &Res) const;

private:
  OwningPtr<MachOObject> MachOObj;
  typedef SmallVector<DataRefImpl, 1> SectionList;
  SectionList Sections;


  void moveToNextSection(DataRefImpl &DRI) const;

  const MachOFormat::SymbolTableEntry *
  getSymbolTableEntry(DataRefImpl DRI) const;

  const MachOFormat::SymbolTableEntry *
  getSymbolTableEntry(DataRefImpl DRI,
                     const MachOFormat::SymtabLoadCommand *SymtabLoadCmd) const;

  const MachOFormat::Symbol64TableEntry *
  getSymbol64TableEntry(DataRefImpl DRI) const;

  const MachOFormat::Symbol64TableEntry *
  getSymbol64TableEntry(DataRefImpl DRI,
                     const MachOFormat::SymtabLoadCommand *SymtabLoadCmd) const;

  void moveToNextSymbol(DataRefImpl &DRI) const;
  const MachOFormat::Section *getSection(DataRefImpl DRI) const;
  const MachOFormat::Section64 *getSection64(DataRefImpl DRI) const;
  const MachOFormat::RelocationEntry *getRelocation(DataRefImpl Rel) const;
  const MachOFormat::SymtabLoadCommand *
    getSymtabLoadCommand(LoadCommandInfo LCI) const;
  const MachOFormat::SegmentLoadCommand *
    getSegmentLoadCommand(LoadCommandInfo LCI) const;
  const MachOFormat::Segment64LoadCommand *
    getSegment64LoadCommand(LoadCommandInfo LCI) const;
  std::size_t getSectionIndex(DataRefImpl Sec) const;

  void printRelocationTargetName(const MachOFormat::RelocationEntry *RE,
                                 raw_string_ostream &fmt) const;
};

}
}

#endif

