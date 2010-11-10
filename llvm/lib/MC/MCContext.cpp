//===- lib/MC/MCContext.cpp - Machine Code Context ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSectionCOFF.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCLabel.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
using namespace llvm;

typedef StringMap<const MCSectionMachO*> MachOUniqueMapTy;
typedef StringMap<const MCSectionELF*> ELFUniqueMapTy;
typedef StringMap<const MCSectionCOFF*> COFFUniqueMapTy;


MCContext::MCContext(const MCAsmInfo &mai) : MAI(mai), NextUniqueID(0),
                     CurrentDwarfLoc(0,0,0,0,0) {
  MachOUniquingMap = 0;
  ELFUniquingMap = 0;
  COFFUniquingMap = 0;

  SecureLogFile = getenv("AS_SECURE_LOG_FILE");
  SecureLog = 0;
  SecureLogUsed = false;

  DwarfLocSeen = false;
}

MCContext::~MCContext() {
  // NOTE: The symbols are all allocated out of a bump pointer allocator,
  // we don't need to free them here.
  
  // If we have the MachO uniquing map, free it.
  delete (MachOUniqueMapTy*)MachOUniquingMap;
  delete (ELFUniqueMapTy*)ELFUniquingMap;
  delete (COFFUniqueMapTy*)COFFUniquingMap;

  // If the stream for the .secure_log_unique directive was created free it.
  delete (raw_ostream*)SecureLog;
}

//===----------------------------------------------------------------------===//
// Symbol Manipulation
//===----------------------------------------------------------------------===//

MCSymbol *MCContext::GetOrCreateSymbol(StringRef Name) {
  assert(!Name.empty() && "Normal symbols cannot be unnamed!");
  
  // Determine whether this is an assembler temporary or normal label.
  bool isTemporary = Name.startswith(MAI.getPrivateGlobalPrefix());
  
  // Do the lookup and get the entire StringMapEntry.  We want access to the
  // key if we are creating the entry.
  StringMapEntry<MCSymbol*> &Entry = Symbols.GetOrCreateValue(Name);
  if (Entry.getValue()) return Entry.getValue();

  // Ok, the entry doesn't already exist.  Have the MCSymbol object itself refer
  // to the copy of the string that is embedded in the StringMapEntry.
  MCSymbol *Result = new (*this) MCSymbol(Entry.getKey(), isTemporary);
  Entry.setValue(Result);
  return Result; 
}

MCSymbol *MCContext::GetOrCreateSymbol(const Twine &Name) {
  SmallString<128> NameSV;
  Name.toVector(NameSV);
  return GetOrCreateSymbol(NameSV.str());
}

MCSymbol *MCContext::CreateTempSymbol() {
  return GetOrCreateSymbol(Twine(MAI.getPrivateGlobalPrefix()) +
                           "tmp" + Twine(NextUniqueID++));
}

unsigned MCContext::NextInstance(int64_t LocalLabelVal) {
  MCLabel *&Label = Instances[LocalLabelVal];
  if (!Label)
    Label = new (*this) MCLabel(0);
  return Label->incInstance();
}

unsigned MCContext::GetInstance(int64_t LocalLabelVal) {
  MCLabel *&Label = Instances[LocalLabelVal];
  if (!Label)
    Label = new (*this) MCLabel(0);
  return Label->getInstance();
}

MCSymbol *MCContext::CreateDirectionalLocalSymbol(int64_t LocalLabelVal) {
  return GetOrCreateSymbol(Twine(MAI.getPrivateGlobalPrefix()) +
                           Twine(LocalLabelVal) +
                           "\2" +
                           Twine(NextInstance(LocalLabelVal)));
}
MCSymbol *MCContext::GetDirectionalLocalSymbol(int64_t LocalLabelVal,
                                               int bORf) {
  return GetOrCreateSymbol(Twine(MAI.getPrivateGlobalPrefix()) +
                           Twine(LocalLabelVal) +
                           "\2" +
                           Twine(GetInstance(LocalLabelVal) + bORf));
}

MCSymbol *MCContext::LookupSymbol(StringRef Name) const {
  return Symbols.lookup(Name);
}

//===----------------------------------------------------------------------===//
// Section Management
//===----------------------------------------------------------------------===//

const MCSectionMachO *MCContext::
getMachOSection(StringRef Segment, StringRef Section,
                unsigned TypeAndAttributes,
                unsigned Reserved2, SectionKind Kind) {
  
  // We unique sections by their segment/section pair.  The returned section
  // may not have the same flags as the requested section, if so this should be
  // diagnosed by the client as an error.
  
  // Create the map if it doesn't already exist.
  if (MachOUniquingMap == 0)
    MachOUniquingMap = new MachOUniqueMapTy();
  MachOUniqueMapTy &Map = *(MachOUniqueMapTy*)MachOUniquingMap;
  
  // Form the name to look up.
  SmallString<64> Name;
  Name += Segment;
  Name.push_back(',');
  Name += Section;
  
  // Do the lookup, if we have a hit, return it.
  const MCSectionMachO *&Entry = Map[Name.str()];
  if (Entry) return Entry;
  
  // Otherwise, return a new section.
  return Entry = new (*this) MCSectionMachO(Segment, Section, TypeAndAttributes,
                                            Reserved2, Kind);
}

const MCSectionELF *MCContext::
getELFSection(StringRef Section, unsigned Type, unsigned Flags,
              SectionKind Kind, unsigned EntrySize) {
  if (ELFUniquingMap == 0)
    ELFUniquingMap = new ELFUniqueMapTy();
  ELFUniqueMapTy &Map = *(ELFUniqueMapTy*)ELFUniquingMap;
  
  // Do the lookup, if we have a hit, return it.
  StringMapEntry<const MCSectionELF*> &Entry = Map.GetOrCreateValue(Section);
  if (Entry.getValue()) return Entry.getValue();
  
  // Possibly refine the entry size first.
  if (!EntrySize) {
    EntrySize = MCSectionELF::DetermineEntrySize(Kind);
  }
  MCSectionELF *Result = new (*this) MCSectionELF(Entry.getKey(), Type, Flags,
                                                  Kind, EntrySize);
  Entry.setValue(Result);
  return Result;
}

const MCSection *MCContext::getCOFFSection(StringRef Section,
                                           unsigned Characteristics,
                                           int Selection,
                                           SectionKind Kind) {
  if (COFFUniquingMap == 0)
    COFFUniquingMap = new COFFUniqueMapTy();
  COFFUniqueMapTy &Map = *(COFFUniqueMapTy*)COFFUniquingMap;
  
  // Do the lookup, if we have a hit, return it.
  StringMapEntry<const MCSectionCOFF*> &Entry = Map.GetOrCreateValue(Section);
  if (Entry.getValue()) return Entry.getValue();
  
  MCSectionCOFF *Result = new (*this) MCSectionCOFF(Entry.getKey(),
                                                    Characteristics,
                                                    Selection, Kind);
  
  Entry.setValue(Result);
  return Result;
}

//===----------------------------------------------------------------------===//
// Dwarf Management
//===----------------------------------------------------------------------===//

/// GetDwarfFile - takes a file name an number to place in the dwarf file and
/// directory tables.  If the file number has already been allocated it is an
/// error and zero is returned and the client reports the error, else the
/// allocated file number is returned.  The file numbers may be in any order.
unsigned MCContext::GetDwarfFile(StringRef FileName, unsigned FileNumber) {
  // TODO: a FileNumber of zero says to use the next available file number.
  // Note: in GenericAsmParser::ParseDirectiveFile() FileNumber was checked
  // to not be less than one.  This needs to be change to be not less than zero.

  // Make space for this FileNumber in the MCDwarfFiles vector if needed.
  if (FileNumber >= MCDwarfFiles.size()) {
    MCDwarfFiles.resize(FileNumber + 1);
  } else {
    MCDwarfFile *&ExistingFile = MCDwarfFiles[FileNumber];
    if (ExistingFile)
      // It is an error to use see the same number more than once.
      return 0;
  }

  // Get the new MCDwarfFile slot for this FileNumber.
  MCDwarfFile *&File = MCDwarfFiles[FileNumber];

  // Separate the directory part from the basename of the FileName.
  std::pair<StringRef, StringRef> Slash = FileName.rsplit('/');

  // Find or make a entry in the MCDwarfDirs vector for this Directory.
  StringRef Name;
  unsigned DirIndex;
  // Capture directory name.
  if (Slash.second.empty()) {
    Name = Slash.first;
    DirIndex = 0; // For FileNames with no directories a DirIndex of 0 is used.
  } else {
    StringRef Directory = Slash.first;
    Name = Slash.second;
    for (DirIndex = 0; DirIndex < MCDwarfDirs.size(); DirIndex++) {
      if (Directory == MCDwarfDirs[DirIndex])
        break;
    }
    if (DirIndex >= MCDwarfDirs.size()) {
      char *Buf = static_cast<char *>(Allocate(Directory.size()));
      memcpy(Buf, Directory.data(), Directory.size());
      MCDwarfDirs.push_back(StringRef(Buf, Directory.size()));
    }
    // The DirIndex is one based, as DirIndex of 0 is used for FileNames with
    // no directories.  MCDwarfDirs[] is unlike MCDwarfFiles[] in that the
    // directory names are stored at MCDwarfDirs[DirIndex-1] where FileNames are
    // stored at MCDwarfFiles[FileNumber].Name .
    DirIndex++;
  }
  
  // Now make the MCDwarfFile entry and place it in the slot in the MCDwarfFiles
  // vector.
  char *Buf = static_cast<char *>(Allocate(Name.size()));
  memcpy(Buf, Name.data(), Name.size());
  File = new (*this) MCDwarfFile(StringRef(Buf, Name.size()), DirIndex);

  // return the allocated FileNumber.
  return FileNumber;
}

/// isValidDwarfFileNumber - takes a dwarf file number and returns true if it
/// currently is assigned and false otherwise.
bool MCContext::isValidDwarfFileNumber(unsigned FileNumber) {
  if(FileNumber == 0 || FileNumber >= MCDwarfFiles.size())
    return false;

  return MCDwarfFiles[FileNumber] != 0;
}
