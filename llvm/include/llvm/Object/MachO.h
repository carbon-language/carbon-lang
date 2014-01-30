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
#include "llvm/ADT/Triple.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/MachO.h"

namespace llvm {
namespace object {

/// DiceRef - This is a value type class that represents a single
/// data in code entry in the table in a Mach-O object file.
class DiceRef {
  DataRefImpl DicePimpl;
  const ObjectFile *OwningObject;

public:
  DiceRef() : OwningObject(NULL) { }

  DiceRef(DataRefImpl DiceP, const ObjectFile *Owner);

  bool operator==(const DiceRef &Other) const;
  bool operator<(const DiceRef &Other) const;

  void moveNext();

  error_code getOffset(uint32_t &Result) const;
  error_code getLength(uint16_t &Result) const;
  error_code getKind(uint16_t &Result) const;

  DataRefImpl getRawDataRefImpl() const;
  const ObjectFile *getObjectFile() const;
};
typedef content_iterator<DiceRef> dice_iterator;

class MachOObjectFile : public ObjectFile {
public:
  struct LoadCommandInfo {
    const char *Ptr;      // Where in memory the load command is.
    MachO::load_command C; // The command itself.
  };

  MachOObjectFile(MemoryBuffer *Object, bool IsLittleEndian, bool Is64Bits,
                  error_code &EC, bool BufferOwned = true);

  void moveSymbolNext(DataRefImpl &Symb) const LLVM_OVERRIDE;
  error_code getSymbolName(DataRefImpl Symb, StringRef &Res) const
      LLVM_OVERRIDE;
  error_code getSymbolAddress(DataRefImpl Symb, uint64_t &Res) const
      LLVM_OVERRIDE;
  error_code getSymbolFileOffset(DataRefImpl Symb, uint64_t &Res) const
      LLVM_OVERRIDE;
  error_code getSymbolAlignment(DataRefImpl Symb, uint32_t &Res) const
      LLVM_OVERRIDE;
  error_code getSymbolSize(DataRefImpl Symb, uint64_t &Res) const LLVM_OVERRIDE;
  error_code getSymbolType(DataRefImpl Symb, SymbolRef::Type &Res) const
      LLVM_OVERRIDE;
  error_code getSymbolFlags(DataRefImpl Symb, uint32_t &Res) const
      LLVM_OVERRIDE;
  error_code getSymbolSection(DataRefImpl Symb, section_iterator &Res) const
      LLVM_OVERRIDE;
  error_code getSymbolValue(DataRefImpl Symb, uint64_t &Val) const
      LLVM_OVERRIDE;

  void moveSectionNext(DataRefImpl &Sec) const LLVM_OVERRIDE;
  error_code getSectionName(DataRefImpl Sec, StringRef &Res) const
      LLVM_OVERRIDE;
  error_code getSectionAddress(DataRefImpl Sec, uint64_t &Res) const
      LLVM_OVERRIDE;
  error_code getSectionSize(DataRefImpl Sec, uint64_t &Res) const LLVM_OVERRIDE;
  error_code getSectionContents(DataRefImpl Sec, StringRef &Res) const
      LLVM_OVERRIDE;
  error_code getSectionAlignment(DataRefImpl Sec, uint64_t &Res) const
      LLVM_OVERRIDE;
  error_code isSectionText(DataRefImpl Sec, bool &Res) const LLVM_OVERRIDE;
  error_code isSectionData(DataRefImpl Sec, bool &Res) const LLVM_OVERRIDE;
  error_code isSectionBSS(DataRefImpl Sec, bool &Res) const LLVM_OVERRIDE;
  error_code isSectionRequiredForExecution(DataRefImpl Sec, bool &Res) const
      LLVM_OVERRIDE;
  error_code isSectionVirtual(DataRefImpl Sec, bool &Res) const LLVM_OVERRIDE;
  error_code isSectionZeroInit(DataRefImpl Sec, bool &Res) const LLVM_OVERRIDE;
  error_code isSectionReadOnlyData(DataRefImpl Sec, bool &Res) const
      LLVM_OVERRIDE;
  error_code sectionContainsSymbol(DataRefImpl Sec, DataRefImpl Symb,
                                   bool &Result) const LLVM_OVERRIDE;
  relocation_iterator section_rel_begin(DataRefImpl Sec) const LLVM_OVERRIDE;
  relocation_iterator section_rel_end(DataRefImpl Sec) const LLVM_OVERRIDE;

  void moveRelocationNext(DataRefImpl &Rel) const LLVM_OVERRIDE;
  error_code getRelocationAddress(DataRefImpl Rel, uint64_t &Res) const
      LLVM_OVERRIDE;
  error_code getRelocationOffset(DataRefImpl Rel, uint64_t &Res) const
      LLVM_OVERRIDE;
  symbol_iterator getRelocationSymbol(DataRefImpl Rel) const LLVM_OVERRIDE;
  error_code getRelocationType(DataRefImpl Rel, uint64_t &Res) const
      LLVM_OVERRIDE;
  error_code getRelocationTypeName(DataRefImpl Rel,
                                   SmallVectorImpl<char> &Result) const
      LLVM_OVERRIDE;
  error_code getRelocationValueString(DataRefImpl Rel,
                                      SmallVectorImpl<char> &Result) const
      LLVM_OVERRIDE;
  error_code getRelocationHidden(DataRefImpl Rel, bool &Result) const
      LLVM_OVERRIDE;

  error_code getLibraryNext(DataRefImpl LibData, LibraryRef &Res) const
      LLVM_OVERRIDE;
  error_code getLibraryPath(DataRefImpl LibData, StringRef &Res) const
      LLVM_OVERRIDE;

  // TODO: Would be useful to have an iterator based version
  // of the load command interface too.

  symbol_iterator begin_symbols() const LLVM_OVERRIDE;
  symbol_iterator end_symbols() const LLVM_OVERRIDE;

  section_iterator begin_sections() const LLVM_OVERRIDE;
  section_iterator end_sections() const LLVM_OVERRIDE;

  library_iterator begin_libraries_needed() const LLVM_OVERRIDE;
  library_iterator end_libraries_needed() const LLVM_OVERRIDE;

  uint8_t getBytesInAddress() const LLVM_OVERRIDE;

  StringRef getFileFormatName() const LLVM_OVERRIDE;
  unsigned getArch() const LLVM_OVERRIDE;

  StringRef getLoadName() const LLVM_OVERRIDE;

  relocation_iterator section_rel_begin(unsigned Index) const;
  relocation_iterator section_rel_end(unsigned Index) const;

  dice_iterator begin_dices() const;
  dice_iterator end_dices() const;

  // In a MachO file, sections have a segment name. This is used in the .o
  // files. They have a single segment, but this field specifies which segment
  // a section should be put in in the final object.
  StringRef getSectionFinalSegmentName(DataRefImpl Sec) const;

  // Names are stored as 16 bytes. These returns the raw 16 bytes without
  // interpreting them as a C string.
  ArrayRef<char> getSectionRawName(DataRefImpl Sec) const;
  ArrayRef<char> getSectionRawFinalSegmentName(DataRefImpl Sec) const;

  // MachO specific Info about relocations.
  bool isRelocationScattered(const MachO::any_relocation_info &RE) const;
  unsigned getPlainRelocationSymbolNum(
                                    const MachO::any_relocation_info &RE) const;
  bool getPlainRelocationExternal(const MachO::any_relocation_info &RE) const;
  bool getScatteredRelocationScattered(
                                    const MachO::any_relocation_info &RE) const;
  uint32_t getScatteredRelocationValue(
                                    const MachO::any_relocation_info &RE) const;
  unsigned getAnyRelocationAddress(const MachO::any_relocation_info &RE) const;
  unsigned getAnyRelocationPCRel(const MachO::any_relocation_info &RE) const;
  unsigned getAnyRelocationLength(const MachO::any_relocation_info &RE) const;
  unsigned getAnyRelocationType(const MachO::any_relocation_info &RE) const;
  SectionRef getRelocationSection(const MachO::any_relocation_info &RE) const;

  // Walk load commands.
  LoadCommandInfo getFirstLoadCommandInfo() const;
  LoadCommandInfo getNextLoadCommandInfo(const LoadCommandInfo &L) const;

  // MachO specific structures.
  MachO::section getSection(DataRefImpl DRI) const;
  MachO::section_64 getSection64(DataRefImpl DRI) const;
  MachO::section getSection(const LoadCommandInfo &L, unsigned Index) const;
  MachO::section_64 getSection64(const LoadCommandInfo &L,unsigned Index) const;
  MachO::nlist getSymbolTableEntry(DataRefImpl DRI) const;
  MachO::nlist_64 getSymbol64TableEntry(DataRefImpl DRI) const;

  MachO::linkedit_data_command
  getLinkeditDataLoadCommand(const LoadCommandInfo &L) const;
  MachO::segment_command
  getSegmentLoadCommand(const LoadCommandInfo &L) const;
  MachO::segment_command_64
  getSegment64LoadCommand(const LoadCommandInfo &L) const;
  MachO::linker_options_command
  getLinkerOptionsLoadCommand(const LoadCommandInfo &L) const;

  MachO::any_relocation_info getRelocation(DataRefImpl Rel) const;
  MachO::data_in_code_entry getDice(DataRefImpl Rel) const;
  MachO::mach_header getHeader() const;
  MachO::mach_header_64 getHeader64() const;
  uint32_t
  getIndirectSymbolTableEntry(const MachO::dysymtab_command &DLC,
                              unsigned Index) const;
  MachO::data_in_code_entry getDataInCodeTableEntry(uint32_t DataOffset,
                                                    unsigned Index) const;
  MachO::symtab_command getSymtabLoadCommand() const;
  MachO::dysymtab_command getDysymtabLoadCommand() const;
  MachO::linkedit_data_command getDataInCodeLoadCommand() const;

  StringRef getStringTableData() const;
  bool is64Bit() const;
  void ReadULEB128s(uint64_t Index, SmallVectorImpl<uint64_t> &Out) const;

  static Triple::ArchType getArch(uint32_t CPUType);

  static bool classof(const Binary *v) {
    return v->isMachO();
  }

private:
  typedef SmallVector<const char*, 1> SectionList;
  SectionList Sections;
  const char *SymtabLoadCmd;
  const char *DysymtabLoadCmd;
  const char *DataInCodeLoadCmd;
};

/// DiceRef
inline DiceRef::DiceRef(DataRefImpl DiceP, const ObjectFile *Owner)
  : DicePimpl(DiceP) , OwningObject(Owner) {}

inline bool DiceRef::operator==(const DiceRef &Other) const {
  return DicePimpl == Other.DicePimpl;
}

inline bool DiceRef::operator<(const DiceRef &Other) const {
  return DicePimpl < Other.DicePimpl;
}

inline void DiceRef::moveNext() {
  const MachO::data_in_code_entry *P =
    reinterpret_cast<const MachO::data_in_code_entry *>(DicePimpl.p);
  DicePimpl.p = reinterpret_cast<uintptr_t>(P + 1);
}

// Since a Mach-O data in code reference, a DiceRef, can only be created when
// the OwningObject ObjectFile is a MachOObjectFile a static_cast<> is used for
// the methods that get the values of the fields of the reference.

inline error_code DiceRef::getOffset(uint32_t &Result) const {
  const MachOObjectFile *MachOOF =
    static_cast<const MachOObjectFile *>(OwningObject);
  MachO::data_in_code_entry Dice = MachOOF->getDice(DicePimpl);
  Result = Dice.offset;
  return object_error::success;
}

inline error_code DiceRef::getLength(uint16_t &Result) const {
  const MachOObjectFile *MachOOF =
    static_cast<const MachOObjectFile *>(OwningObject);
  MachO::data_in_code_entry Dice = MachOOF->getDice(DicePimpl);
  Result = Dice.length;
  return object_error::success;
}

inline error_code DiceRef::getKind(uint16_t &Result) const {
  const MachOObjectFile *MachOOF =
    static_cast<const MachOObjectFile *>(OwningObject);
  MachO::data_in_code_entry Dice = MachOOF->getDice(DicePimpl);
  Result = Dice.kind;
  return object_error::success;
}

inline DataRefImpl DiceRef::getRawDataRefImpl() const {
  return DicePimpl;
}

inline const ObjectFile *DiceRef::getObjectFile() const {
  return OwningObject;
}

}
}

#endif

