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
#include "llvm/MC/MCSymbol.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
using namespace llvm;

typedef StringMap<const MCSectionMachO*> MachOUniqueMapTy;
typedef StringMap<const MCSectionELF*> ELFUniqueMapTy;


MCContext::MCContext(const MCAsmInfo &mai) : MAI(mai), NextUniqueID(0) {
  MachOUniquingMap = 0;
  ELFUniquingMap = 0;
}

MCContext::~MCContext() {
  // NOTE: The symbols are all allocated out of a bump pointer allocator,
  // we don't need to free them here.
  
  // If we have the MachO uniquing map, free it.
  delete (MachOUniqueMapTy*)MachOUniquingMap;
  delete (ELFUniqueMapTy*)ELFUniquingMap;
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


const MCSection *MCContext::
getELFSection(StringRef Section, unsigned Type, unsigned Flags,
              SectionKind Kind, bool IsExplicit) {
  if (ELFUniquingMap == 0)
    ELFUniquingMap = new ELFUniqueMapTy();
  ELFUniqueMapTy &Map = *(ELFUniqueMapTy*)ELFUniquingMap;
  
  // Do the lookup, if we have a hit, return it.
  StringMapEntry<const MCSectionELF*> &Entry = Map.GetOrCreateValue(Section);
  if (Entry.getValue()) return Entry.getValue();
  
  MCSectionELF *Result = new (*this) MCSectionELF(Entry.getKey(), Type, Flags,
                                                  Kind, IsExplicit);
  Entry.setValue(Result);
  return Result;
}


