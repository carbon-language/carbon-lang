//===- MachO.h - MachO object file implementation ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MachOObjectFile class, which implement the ObjectFile
// interface for MachO files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_MACHO_H
#define LLVM_OBJECT_MACHO_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Object/MachOFormat.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/MachO.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace object {

class MachOObjectFile : public ObjectFile {
public:
  struct LoadCommandInfo {
    const char *Ptr;      // Where in memory the load command is.
    macho::LoadCommand C; // The command itself.
  };

  MachOObjectFile(MemoryBuffer *Object, bool IsLittleEndian, bool Is64Bits,
                  error_code &ec);

  virtual error_code getSymbolNext(DataRefImpl Symb, SymbolRef &Res) const;
  virtual error_code getSymbolName(DataRefImpl Symb, StringRef &Res) const;
  virtual error_code getSymbolAddress(DataRefImpl Symb, uint64_t &Res) const;
  virtual error_code getSymbolFileOffset(DataRefImpl Symb, uint64_t &Res) const;
  virtual error_code getSymbolAlignment(DataRefImpl Symb, uint32_t &Res) const;
  virtual error_code getSymbolSize(DataRefImpl Symb, uint64_t &Res) const;
  virtual error_code getSymbolType(DataRefImpl Symb,
                                   SymbolRef::Type &Res) const;
  virtual error_code getSymbolNMTypeChar(DataRefImpl Symb, char &Res) const;
  virtual error_code getSymbolFlags(DataRefImpl Symb, uint32_t &Res) const;
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
  virtual error_code sectionContainsSymbol(DataRefImpl Sec, DataRefImpl Symb,
                                           bool &Result) const;
  virtual relocation_iterator getSectionRelBegin(DataRefImpl Sec) const;
  virtual relocation_iterator getSectionRelEnd(DataRefImpl Sec) const;

  virtual error_code getRelocationNext(DataRefImpl Rel,
                                       RelocationRef &Res) const;
  virtual error_code getRelocationAddress(DataRefImpl Rel, uint64_t &Res) const;
  virtual error_code getRelocationOffset(DataRefImpl Rel, uint64_t &Res) const;
  virtual error_code getRelocationSymbol(DataRefImpl Rel, SymbolRef &Res) const;
  virtual error_code getRelocationType(DataRefImpl Rel, uint64_t &Res) const;
  virtual error_code getRelocationTypeName(DataRefImpl Rel,
                                           SmallVectorImpl<char> &Result) const;
  virtual error_code getRelocationAdditionalInfo(DataRefImpl Rel,
                                                 int64_t &Res) const;
  virtual error_code getRelocationValueString(DataRefImpl Rel,
                                           SmallVectorImpl<char> &Result) const;
  virtual error_code getRelocationHidden(DataRefImpl Rel, bool &Result) const;

  virtual error_code getLibraryNext(DataRefImpl LibData, LibraryRef &Res) const;
  virtual error_code getLibraryPath(DataRefImpl LibData, StringRef &Res) const;

  // TODO: Would be useful to have an iterator based version
  // of the load command interface too.

  virtual symbol_iterator begin_symbols() const;
  virtual symbol_iterator end_symbols() const;

  virtual symbol_iterator begin_dynamic_symbols() const;
  virtual symbol_iterator end_dynamic_symbols() const;

  virtual section_iterator begin_sections() const;
  virtual section_iterator end_sections() const;

  virtual library_iterator begin_libraries_needed() const;
  virtual library_iterator end_libraries_needed() const;

  virtual uint8_t getBytesInAddress() const;

  virtual StringRef getFileFormatName() const;
  virtual unsigned getArch() const;

  virtual StringRef getLoadName() const;

  relocation_iterator getSectionRelBegin(unsigned Index) const;
  relocation_iterator getSectionRelEnd(unsigned Index) const;

  // In a MachO file, sections have a segment name. This is used in the .o
  // files. They have a single segment, but this field specifies which segment
  // a section should be put in in the final object.
  StringRef getSectionFinalSegmentName(DataRefImpl Sec) const;

  // Names are stored as 16 bytes. These returns the raw 16 bytes without
  // interpreting them as a C string.
  ArrayRef<char> getSectionRawName(DataRefImpl Sec) const;
  ArrayRef<char> getSectionRawFinalSegmentName(DataRefImpl Sec) const;

  // MachO specific Info about relocations.
  bool isRelocationScattered(const macho::RelocationEntry &RE) const;
  unsigned getPlainRelocationSymbolNum(const macho::RelocationEntry &RE) const;
  bool getPlainRelocationExternal(const macho::RelocationEntry &RE) const;
  bool getScatteredRelocationScattered(const macho::RelocationEntry &RE) const;
  uint32_t getScatteredRelocationValue(const macho::RelocationEntry &RE) const;
  unsigned getAnyRelocationAddress(const macho::RelocationEntry &RE) const;
  unsigned getAnyRelocationPCRel(const macho::RelocationEntry &RE) const;
  unsigned getAnyRelocationLength(const macho::RelocationEntry &RE) const;
  unsigned getAnyRelocationType(const macho::RelocationEntry &RE) const;
  SectionRef getRelocationSection(const macho::RelocationEntry &RE) const;

  // Walk load commands.
  LoadCommandInfo getFirstLoadCommandInfo() const;
  LoadCommandInfo getNextLoadCommandInfo(const LoadCommandInfo &L) const;

  // MachO specific structures.
  macho::Section getSection(DataRefImpl DRI) const;
  macho::Section64 getSection64(DataRefImpl DRI) const;
  macho::Section getSection(const LoadCommandInfo &L, unsigned Index) const;
  macho::Section64 getSection64(const LoadCommandInfo &L, unsigned Index) const;
  macho::SymbolTableEntry getSymbolTableEntry(DataRefImpl DRI) const;
  macho::Symbol64TableEntry getSymbol64TableEntry(DataRefImpl DRI) const;

  macho::LinkeditDataLoadCommand
  getLinkeditDataLoadCommand(const LoadCommandInfo &L) const;
  macho::SegmentLoadCommand
  getSegmentLoadCommand(const LoadCommandInfo &L) const;
  macho::Segment64LoadCommand
  getSegment64LoadCommand(const LoadCommandInfo &L) const;
  macho::LinkerOptionsLoadCommand
  getLinkerOptionsLoadCommand(const LoadCommandInfo &L) const;

  macho::RelocationEntry getRelocation(DataRefImpl Rel) const;
  macho::Header getHeader() const;
  macho::Header64Ext getHeader64Ext() const;
  macho::IndirectSymbolTableEntry
  getIndirectSymbolTableEntry(const macho::DysymtabLoadCommand &DLC,
                              unsigned Index) const;
  macho::DataInCodeTableEntry getDataInCodeTableEntry(uint32_t DataOffset,
                                                      unsigned Index) const;
  macho::SymtabLoadCommand getSymtabLoadCommand() const;
  macho::DysymtabLoadCommand getDysymtabLoadCommand() const;

  StringRef getStringTableData() const;
  bool is64Bit() const;
  void ReadULEB128s(uint64_t Index, SmallVectorImpl<uint64_t> &Out) const;

  static bool classof(const Binary *v) {
    return v->isMachO();
  }

private:
  typedef SmallVector<const char*, 1> SectionList;
  SectionList Sections;
  const char *SymtabLoadCmd;
  const char *DysymtabLoadCmd;
};

}
}

#endif

