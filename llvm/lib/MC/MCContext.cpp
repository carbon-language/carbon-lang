//===- lib/MC/MCContext.cpp - Machine Code Context ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCContext.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCLabel.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSectionCOFF.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include <map>

using namespace llvm;

typedef std::pair<std::string, std::string> SectionGroupPair;

typedef StringMap<const MCSectionMachO*> MachOUniqueMapTy;
typedef std::map<SectionGroupPair, const MCSectionELF *> ELFUniqueMapTy;
typedef std::map<SectionGroupPair, const MCSectionCOFF *> COFFUniqueMapTy;

MCContext::MCContext(const MCAsmInfo *mai, const MCRegisterInfo *mri,
                     const MCObjectFileInfo *mofi, const SourceMgr *mgr,
                     bool DoAutoReset)
    : SrcMgr(mgr), MAI(mai), MRI(mri), MOFI(mofi), Allocator(),
      Symbols(Allocator), UsedNames(Allocator), NextUniqueID(0),
      CurrentDwarfLoc(0, 0, 0, DWARF2_FLAG_IS_STMT, 0, 0), DwarfLocSeen(false),
      GenDwarfForAssembly(false), GenDwarfFileNumber(0),
      AllowTemporaryLabels(true), DwarfCompileUnitID(0),
      AutoReset(DoAutoReset) {

  error_code EC = llvm::sys::fs::current_path(CompilationDir);
  if (EC)
    CompilationDir.clear();

  MachOUniquingMap = 0;
  ELFUniquingMap = 0;
  COFFUniquingMap = 0;

  SecureLogFile = getenv("AS_SECURE_LOG_FILE");
  SecureLog = 0;
  SecureLogUsed = false;

  if (SrcMgr && SrcMgr->getNumBuffers() > 0)
    MainFileName = SrcMgr->getMemoryBuffer(0)->getBufferIdentifier();
}

MCContext::~MCContext() {

  if (AutoReset)
    reset();

  // NOTE: The symbols are all allocated out of a bump pointer allocator,
  // we don't need to free them here.

  // If the stream for the .secure_log_unique directive was created free it.
  delete (raw_ostream*)SecureLog;
}

//===----------------------------------------------------------------------===//
// Module Lifetime Management
//===----------------------------------------------------------------------===//

void MCContext::reset() {
  UsedNames.clear();
  Symbols.clear();
  Allocator.Reset();
  Instances.clear();
  MCDwarfLineTablesCUMap.clear();
  MCGenDwarfLabelEntries.clear();
  DwarfDebugFlags = StringRef();
  DwarfCompileUnitID = 0;
  CurrentDwarfLoc = MCDwarfLoc(0,0,0,DWARF2_FLAG_IS_STMT,0,0);

  // If we have the MachO uniquing map, free it.
  delete (MachOUniqueMapTy*)MachOUniquingMap;
  delete (ELFUniqueMapTy*)ELFUniquingMap;
  delete (COFFUniqueMapTy*)COFFUniquingMap;
  MachOUniquingMap = 0;
  ELFUniquingMap = 0;
  COFFUniquingMap = 0;

  NextUniqueID = 0;
  AllowTemporaryLabels = true;
  DwarfLocSeen = false;
  GenDwarfForAssembly = false;
  GenDwarfFileNumber = 0;
}

//===----------------------------------------------------------------------===//
// Symbol Manipulation
//===----------------------------------------------------------------------===//

MCSymbol *MCContext::GetOrCreateSymbol(StringRef Name) {
  assert(!Name.empty() && "Normal symbols cannot be unnamed!");

  // Do the lookup and get the entire StringMapEntry.  We want access to the
  // key if we are creating the entry.
  StringMapEntry<MCSymbol*> &Entry = Symbols.GetOrCreateValue(Name);
  MCSymbol *Sym = Entry.getValue();

  if (Sym)
    return Sym;

  Sym = CreateSymbol(Name);
  Entry.setValue(Sym);
  return Sym;
}

MCSymbol *MCContext::CreateSymbol(StringRef Name) {
  // Determine whether this is an assembler temporary or normal label, if used.
  bool isTemporary = false;
  if (AllowTemporaryLabels)
    isTemporary = Name.startswith(MAI->getPrivateGlobalPrefix());

  StringMapEntry<bool> *NameEntry = &UsedNames.GetOrCreateValue(Name);
  if (NameEntry->getValue()) {
    assert(isTemporary && "Cannot rename non-temporary symbols");
    SmallString<128> NewName = Name;
    do {
      NewName.resize(Name.size());
      raw_svector_ostream(NewName) << NextUniqueID++;
      NameEntry = &UsedNames.GetOrCreateValue(NewName);
    } while (NameEntry->getValue());
  }
  NameEntry->setValue(true);

  // Ok, the entry doesn't already exist.  Have the MCSymbol object itself refer
  // to the copy of the string that is embedded in the UsedNames entry.
  MCSymbol *Result = new (*this) MCSymbol(NameEntry->getKey(), isTemporary);

  return Result;
}

MCSymbol *MCContext::GetOrCreateSymbol(const Twine &Name) {
  SmallString<128> NameSV;
  return GetOrCreateSymbol(Name.toStringRef(NameSV));
}

MCSymbol *MCContext::CreateLinkerPrivateTempSymbol() {
  SmallString<128> NameSV;
  raw_svector_ostream(NameSV)
    << MAI->getLinkerPrivateGlobalPrefix() << "tmp" << NextUniqueID++;
  return CreateSymbol(NameSV);
}

MCSymbol *MCContext::CreateTempSymbol() {
  SmallString<128> NameSV;
  raw_svector_ostream(NameSV)
    << MAI->getPrivateGlobalPrefix() << "tmp" << NextUniqueID++;
  return CreateSymbol(NameSV);
}

unsigned MCContext::NextInstance(unsigned LocalLabelVal) {
  MCLabel *&Label = Instances[LocalLabelVal];
  if (!Label)
    Label = new (*this) MCLabel(0);
  return Label->incInstance();
}

unsigned MCContext::GetInstance(unsigned LocalLabelVal) {
  MCLabel *&Label = Instances[LocalLabelVal];
  if (!Label)
    Label = new (*this) MCLabel(0);
  return Label->getInstance();
}

MCSymbol *MCContext::getOrCreateDirectionalLocalSymbol(unsigned LocalLabelVal,
                                                       unsigned Instance) {
  MCSymbol *&Sym = LocalSymbols[std::make_pair(LocalLabelVal, Instance)];
  if (!Sym)
    Sym = CreateTempSymbol();
  return Sym;
}

MCSymbol *MCContext::CreateDirectionalLocalSymbol(unsigned LocalLabelVal) {
  unsigned Instance = NextInstance(LocalLabelVal);
  return getOrCreateDirectionalLocalSymbol(LocalLabelVal, Instance);
}

MCSymbol *MCContext::GetDirectionalLocalSymbol(unsigned LocalLabelVal,
                                               bool Before) {
  unsigned Instance = GetInstance(LocalLabelVal);
  if (!Before)
    ++Instance;
  return getOrCreateDirectionalLocalSymbol(LocalLabelVal, Instance);
}

MCSymbol *MCContext::LookupSymbol(StringRef Name) const {
  return Symbols.lookup(Name);
}

MCSymbol *MCContext::LookupSymbol(const Twine &Name) const {
  SmallString<128> NameSV;
  Name.toVector(NameSV);
  return LookupSymbol(NameSV.str());
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
              SectionKind Kind) {
  return getELFSection(Section, Type, Flags, Kind, 0, "");
}

const MCSectionELF *MCContext::
getELFSection(StringRef Section, unsigned Type, unsigned Flags,
              SectionKind Kind, unsigned EntrySize, StringRef Group) {
  if (ELFUniquingMap == 0)
    ELFUniquingMap = new ELFUniqueMapTy();
  ELFUniqueMapTy &Map = *(ELFUniqueMapTy*)ELFUniquingMap;

  SmallString<32> ZDebugName;
  if (MAI->compressDebugSections() && Section.startswith(".debug_") &&
      Section != ".debug_frame" && Section != ".debug_line")
    Section = (".z" + Section.drop_front(1)).toStringRef(ZDebugName);

  // Do the lookup, if we have a hit, return it.
  std::pair<ELFUniqueMapTy::iterator, bool> Entry = Map.insert(
      std::make_pair(SectionGroupPair(Section, Group), (MCSectionELF *)0));
  if (!Entry.second) return Entry.first->second;

  // Possibly refine the entry size first.
  if (!EntrySize) {
    EntrySize = MCSectionELF::DetermineEntrySize(Kind);
  }

  MCSymbol *GroupSym = NULL;
  if (!Group.empty())
    GroupSym = GetOrCreateSymbol(Group);

  MCSectionELF *Result = new (*this) MCSectionELF(
      Entry.first->first.first, Type, Flags, Kind, EntrySize, GroupSym);
  Entry.first->second = Result;
  return Result;
}

const MCSectionELF *MCContext::CreateELFGroupSection() {
  MCSectionELF *Result =
    new (*this) MCSectionELF(".group", ELF::SHT_GROUP, 0,
                             SectionKind::getReadOnly(), 4, NULL);
  return Result;
}

const MCSectionCOFF *
MCContext::getCOFFSection(StringRef Section, unsigned Characteristics,
                          SectionKind Kind, StringRef COMDATSymName,
                          int Selection, const MCSectionCOFF *Assoc) {
  if (COFFUniquingMap == 0)
    COFFUniquingMap = new COFFUniqueMapTy();
  COFFUniqueMapTy &Map = *(COFFUniqueMapTy*)COFFUniquingMap;

  // Do the lookup, if we have a hit, return it.

  SectionGroupPair P(Section, COMDATSymName);
  std::pair<COFFUniqueMapTy::iterator, bool> Entry =
      Map.insert(std::make_pair(P, (MCSectionCOFF *)0));
  COFFUniqueMapTy::iterator Iter = Entry.first;
  if (!Entry.second)
    return Iter->second;

  const MCSymbol *COMDATSymbol = NULL;
  if (!COMDATSymName.empty())
    COMDATSymbol = GetOrCreateSymbol(COMDATSymName);

  MCSectionCOFF *Result =
      new (*this) MCSectionCOFF(Iter->first.first, Characteristics,
                                COMDATSymbol, Selection, Assoc, Kind);

  Iter->second = Result;
  return Result;
}

const MCSectionCOFF *
MCContext::getCOFFSection(StringRef Section, unsigned Characteristics,
                          SectionKind Kind) {
  return getCOFFSection(Section, Characteristics, Kind, "", 0);
}

const MCSectionCOFF *MCContext::getCOFFSection(StringRef Section) {
  if (COFFUniquingMap == 0)
    COFFUniquingMap = new COFFUniqueMapTy();
  COFFUniqueMapTy &Map = *(COFFUniqueMapTy*)COFFUniquingMap;

  SectionGroupPair P(Section, "");
  COFFUniqueMapTy::iterator Iter = Map.find(P);
  if (Iter == Map.end())
    return 0;
  return Iter->second;
}

//===----------------------------------------------------------------------===//
// Dwarf Management
//===----------------------------------------------------------------------===//

/// GetDwarfFile - takes a file name an number to place in the dwarf file and
/// directory tables.  If the file number has already been allocated it is an
/// error and zero is returned and the client reports the error, else the
/// allocated file number is returned.  The file numbers may be in any order.
unsigned MCContext::GetDwarfFile(StringRef Directory, StringRef FileName,
                                 unsigned FileNumber, unsigned CUID) {
  MCDwarfLineTable &Table = MCDwarfLineTablesCUMap[CUID];
  return Table.getFile(Directory, FileName, FileNumber);
}

/// isValidDwarfFileNumber - takes a dwarf file number and returns true if it
/// currently is assigned and false otherwise.
bool MCContext::isValidDwarfFileNumber(unsigned FileNumber, unsigned CUID) {
  const SmallVectorImpl<MCDwarfFile>& MCDwarfFiles = getMCDwarfFiles(CUID);
  if(FileNumber == 0 || FileNumber >= MCDwarfFiles.size())
    return false;

  return !MCDwarfFiles[FileNumber].Name.empty();
}

void MCContext::FatalError(SMLoc Loc, const Twine &Msg) {
  // If we have a source manager and a location, use it. Otherwise just
  // use the generic report_fatal_error().
  if (!SrcMgr || Loc == SMLoc())
    report_fatal_error(Msg, false);

  // Use the source manager to print the message.
  SrcMgr->PrintMessage(Loc, SourceMgr::DK_Error, Msg);

  // If we reached here, we are failing ungracefully. Run the interrupt handlers
  // to make sure any special cleanups get done, in particular that we remove
  // files registered with RemoveFileOnSignal.
  sys::RunInterruptHandlers();
  exit(1);
}
