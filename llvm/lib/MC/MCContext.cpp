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

MCContext::MCContext(const MCAsmInfo *mai, const MCRegisterInfo *mri,
                     const MCObjectFileInfo *mofi, const SourceMgr *mgr,
                     bool DoAutoReset)
    : SrcMgr(mgr), MAI(mai), MRI(mri), MOFI(mofi), Allocator(),
      Symbols(Allocator), UsedNames(Allocator), NextUniqueID(0),
      CurrentDwarfLoc(0, 0, 0, DWARF2_FLAG_IS_STMT, 0, 0), DwarfLocSeen(false),
      GenDwarfForAssembly(false), GenDwarfFileNumber(0), DwarfVersion(4),
      AllowTemporaryLabels(true), DwarfCompileUnitID(0),
      AutoReset(DoAutoReset) {

  std::error_code EC = llvm::sys::fs::current_path(CompilationDir);
  if (EC)
    CompilationDir.clear();

  SecureLogFile = getenv("AS_SECURE_LOG_FILE");
  SecureLog = nullptr;
  SecureLogUsed = false;

  if (SrcMgr && SrcMgr->getNumBuffers())
    MainFileName =
        SrcMgr->getMemoryBuffer(SrcMgr->getMainFileID())->getBufferIdentifier();
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
  CompilationDir.clear();
  MainFileName.clear();
  MCDwarfLineTablesCUMap.clear();
  SectionStartEndSyms.clear();
  MCGenDwarfLabelEntries.clear();
  DwarfDebugFlags = StringRef();
  DwarfCompileUnitID = 0;
  CurrentDwarfLoc = MCDwarfLoc(0,0,0,DWARF2_FLAG_IS_STMT,0,0);

  MachOUniquingMap.clear();
  ELFUniquingMap.clear();
  COFFUniquingMap.clear();

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

  // Form the name to look up.
  SmallString<64> Name;
  Name += Segment;
  Name.push_back(',');
  Name += Section;

  // Do the lookup, if we have a hit, return it.
  const MCSectionMachO *&Entry = MachOUniquingMap[Name.str()];
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

void MCContext::renameELFSection(const MCSectionELF *Section, StringRef Name) {
  StringRef GroupName;
  if (const MCSymbol *Group = Section->getGroup())
    GroupName = Group->getName();

  ELFUniquingMap.erase(SectionGroupPair(Section->getSectionName(), GroupName));
  auto I =
      ELFUniquingMap.insert(std::make_pair(SectionGroupPair(Name, GroupName),
                                           Section)).first;
  StringRef CachedName = I->first.first;
  const_cast<MCSectionELF*>(Section)->setSectionName(CachedName);
}

const MCSectionELF *MCContext::
getELFSection(StringRef Section, unsigned Type, unsigned Flags,
              SectionKind Kind, unsigned EntrySize, StringRef Group) {
  // Do the lookup, if we have a hit, return it.
  auto IterBool = ELFUniquingMap.insert(
      std::make_pair(SectionGroupPair(Section, Group), nullptr));
  auto &Entry = *IterBool.first;
  if (!IterBool.second) return Entry.second;

  // Possibly refine the entry size first.
  if (!EntrySize) {
    EntrySize = MCSectionELF::DetermineEntrySize(Kind);
  }

  MCSymbol *GroupSym = nullptr;
  if (!Group.empty())
    GroupSym = GetOrCreateSymbol(Group);

  StringRef CachedName = Entry.first.first;
  MCSectionELF *Result = new (*this)
      MCSectionELF(CachedName, Type, Flags, Kind, EntrySize, GroupSym);
  Entry.second = Result;
  return Result;
}

const MCSectionELF *MCContext::CreateELFGroupSection() {
  MCSectionELF *Result =
    new (*this) MCSectionELF(".group", ELF::SHT_GROUP, 0,
                             SectionKind::getReadOnly(), 4, nullptr);
  return Result;
}

const MCSectionCOFF *MCContext::getCOFFSection(StringRef Section,
                                               unsigned Characteristics,
                                               SectionKind Kind,
                                               StringRef COMDATSymName,
                                               int Selection) {
  // Do the lookup, if we have a hit, return it.

  SectionGroupTriple T(Section, COMDATSymName, Selection);
  auto IterBool = COFFUniquingMap.insert(std::make_pair(T, nullptr));
  auto Iter = IterBool.first;
  if (!IterBool.second)
    return Iter->second;

  MCSymbol *COMDATSymbol = nullptr;
  if (!COMDATSymName.empty())
    COMDATSymbol = GetOrCreateSymbol(COMDATSymName);

  StringRef CachedName = std::get<0>(Iter->first);
  MCSectionCOFF *Result = new (*this)
      MCSectionCOFF(CachedName, Characteristics, COMDATSymbol, Selection, Kind);

  Iter->second = Result;
  return Result;
}

const MCSectionCOFF *
MCContext::getCOFFSection(StringRef Section, unsigned Characteristics,
                          SectionKind Kind) {
  return getCOFFSection(Section, Characteristics, Kind, "", 0);
}

const MCSectionCOFF *MCContext::getCOFFSection(StringRef Section) {
  SectionGroupTriple T(Section, "", 0);
  auto Iter = COFFUniquingMap.find(T);
  if (Iter == COFFUniquingMap.end())
    return nullptr;
  return Iter->second;
}

const MCSectionCOFF *
MCContext::getAssociativeCOFFSection(const MCSectionCOFF *Sec,
                                     const MCSymbol *KeySym) {
  // Return the normal section if we don't have to be associative.
  if (!KeySym)
    return Sec;

  // Make an associative section with the same name and kind as the normal
  // section.
  unsigned Characteristics =
      Sec->getCharacteristics() | COFF::IMAGE_SCN_LNK_COMDAT;
  return getCOFFSection(Sec->getSectionName(), Characteristics, Sec->getKind(),
                        KeySym->getName(),
                        COFF::IMAGE_COMDAT_SELECT_ASSOCIATIVE);
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

/// finalizeDwarfSections - Emit end symbols for each non-empty code section.
/// Also remove empty sections from SectionStartEndSyms, to avoid generating
/// useless debug info for them.
void MCContext::finalizeDwarfSections(MCStreamer &MCOS) {
  MCContext &context = MCOS.getContext();

  auto sec = SectionStartEndSyms.begin();
  while (sec != SectionStartEndSyms.end()) {
    assert(sec->second.first && "Start symbol must be set by now");
    MCOS.SwitchSection(sec->first);
    if (MCOS.mayHaveInstructions()) {
      MCSymbol *SectionEndSym = context.CreateTempSymbol();
      MCOS.EmitLabel(SectionEndSym);
      sec->second.second = SectionEndSym;
      ++sec;
    } else {
      MapVector<const MCSection *, std::pair<MCSymbol *, MCSymbol *> >::iterator
        to_erase = sec;
      sec = SectionStartEndSyms.erase(to_erase);
    }
  }
}

void MCContext::FatalError(SMLoc Loc, const Twine &Msg) const {
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
