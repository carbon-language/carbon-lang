//===-- RuntimeDyldMachO.cpp - Run-time dynamic linker for MC-JIT -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of the MC-JIT runtime dynamic linker.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "dyld"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/STLExtras.h"
#include "RuntimeDyldMachO.h"
using namespace llvm;
using namespace llvm::object;

namespace llvm {

bool RuntimeDyldMachO::
resolveRelocation(uint8_t *LocalAddress,
                  uint64_t FinalAddress,
                  uint64_t Value,
                  bool isPCRel,
                  unsigned Type,
                  unsigned Size,
                  int64_t Addend) {
  // This just dispatches to the proper target specific routine.
  switch (CPUType) {
  default: llvm_unreachable("Unsupported CPU type!");
  case mach::CTM_x86_64:
    return resolveX86_64Relocation(LocalAddress,
                                   FinalAddress,
                                   (uintptr_t)Value,
                                   isPCRel,
                                   Type,
                                   Size,
                                   Addend);
  case mach::CTM_ARM:
    return resolveARMRelocation(LocalAddress,
                                FinalAddress,
                                (uintptr_t)Value,
                                isPCRel,
                                Type,
                                Size,
                                Addend);
  }
}

bool RuntimeDyldMachO::
resolveX86_64Relocation(uint8_t *LocalAddress,
                        uint64_t FinalAddress,
                        uint64_t Value,
                        bool isPCRel,
                        unsigned Type,
                        unsigned Size,
                        int64_t Addend) {
  // If the relocation is PC-relative, the value to be encoded is the
  // pointer difference.
  if (isPCRel)
    // FIXME: It seems this value needs to be adjusted by 4 for an effective PC
    // address. Is that expected? Only for branches, perhaps?
    Value -= FinalAddress + 4;

  switch(Type) {
  default:
    llvm_unreachable("Invalid relocation type!");
  case macho::RIT_X86_64_Signed1:
  case macho::RIT_X86_64_Signed2:
  case macho::RIT_X86_64_Signed4:
  case macho::RIT_X86_64_Signed:
  case macho::RIT_X86_64_Unsigned:
  case macho::RIT_X86_64_Branch: {
    Value += Addend;
    // Mask in the target value a byte at a time (we don't have an alignment
    // guarantee for the target address, so this is safest).
    uint8_t *p = (uint8_t*)LocalAddress;
    for (unsigned i = 0; i < Size; ++i) {
      *p++ = (uint8_t)Value;
      Value >>= 8;
    }
    return false;
  }
  case macho::RIT_X86_64_GOTLoad:
  case macho::RIT_X86_64_GOT:
  case macho::RIT_X86_64_Subtractor:
  case macho::RIT_X86_64_TLV:
    return Error("Relocation type not implemented yet!");
  }
}

bool RuntimeDyldMachO::
resolveARMRelocation(uint8_t *LocalAddress,
                     uint64_t FinalAddress,
                     uint64_t Value,
                     bool isPCRel,
                     unsigned Type,
                     unsigned Size,
                     int64_t Addend) {
  // If the relocation is PC-relative, the value to be encoded is the
  // pointer difference.
  if (isPCRel) {
    Value -= FinalAddress;
    // ARM PCRel relocations have an effective-PC offset of two instructions
    // (four bytes in Thumb mode, 8 bytes in ARM mode).
    // FIXME: For now, assume ARM mode.
    Value -= 8;
  }

  switch(Type) {
  default:
    llvm_unreachable("Invalid relocation type!");
  case macho::RIT_Vanilla: {
    // Mask in the target value a byte at a time (we don't have an alignment
    // guarantee for the target address, so this is safest).
    uint8_t *p = (uint8_t*)LocalAddress;
    for (unsigned i = 0; i < Size; ++i) {
      *p++ = (uint8_t)Value;
      Value >>= 8;
    }
    break;
  }
  case macho::RIT_ARM_Branch24Bit: {
    // Mask the value into the target address. We know instructions are
    // 32-bit aligned, so we can do it all at once.
    uint32_t *p = (uint32_t*)LocalAddress;
    // The low two bits of the value are not encoded.
    Value >>= 2;
    // Mask the value to 24 bits.
    Value &= 0xffffff;
    // FIXME: If the destination is a Thumb function (and the instruction
    // is a non-predicated BL instruction), we need to change it to a BLX
    // instruction instead.

    // Insert the value into the instruction.
    *p = (*p & ~0xffffff) | Value;
    break;
  }
  case macho::RIT_ARM_ThumbBranch22Bit:
  case macho::RIT_ARM_ThumbBranch32Bit:
  case macho::RIT_ARM_Half:
  case macho::RIT_ARM_HalfDifference:
  case macho::RIT_Pair:
  case macho::RIT_Difference:
  case macho::RIT_ARM_LocalDifference:
  case macho::RIT_ARM_PreboundLazyPointer:
    return Error("Relocation type not implemented yet!");
  }
  return false;
}

bool RuntimeDyldMachO::
loadSegment32(const MachOObject *Obj,
              const MachOObject::LoadCommandInfo *SegmentLCI,
              const InMemoryStruct<macho::SymtabLoadCommand> &SymtabLC) {
  // FIXME: This should really be combined w/ loadSegment64. Templatized
  // function on the 32/64 datatypes maybe?
  InMemoryStruct<macho::SegmentLoadCommand> SegmentLC;
  Obj->ReadSegmentLoadCommand(*SegmentLCI, SegmentLC);
  if (!SegmentLC)
    return Error("unable to load segment load command");


  SmallVector<unsigned, 16> SectionMap;
  for (unsigned SectNum = 0; SectNum != SegmentLC->NumSections; ++SectNum) {
    InMemoryStruct<macho::Section> Sect;
    Obj->ReadSection(*SegmentLCI, SectNum, Sect);
    if (!Sect)
      return Error("unable to load section: '" + Twine(SectNum) + "'");

    // Allocate memory via the MM for the section.
    uint8_t *Buffer;
    uint32_t SectionID = Sections.size();
    if (Sect->Flags == 0x80000400)
      Buffer = MemMgr->allocateCodeSection(Sect->Size, Sect->Align, SectionID);
    else
      Buffer = MemMgr->allocateDataSection(Sect->Size, Sect->Align, SectionID);

    DEBUG(dbgs() << "Loading "
                 << ((Sect->Flags == 0x80000400) ? "text" : "data")
                 << " (ID #" << SectionID << ")"
                 << " '" << Sect->SegmentName << ","
                 << Sect->Name << "' of size " << Sect->Size
                 << " to address " << Buffer << ".\n");

    // Copy the payload from the object file into the allocated buffer.
    uint8_t *Base = (uint8_t*)Obj->getData(SegmentLC->FileOffset,
                                           SegmentLC->FileSize).data();
    memcpy(Buffer, Base + Sect->Address, Sect->Size);

    // Remember what got allocated for this SectionID.
    Sections.push_back(sys::MemoryBlock(Buffer, Sect->Size));
    SectionLocalMemToID[Buffer] = SectionID;

    // By default, the load address of a section is its memory buffer.
    SectionLoadAddress.push_back((uint64_t)Buffer);

    // Keep a map of object file section numbers to corresponding SectionIDs
    // while processing the file.
    SectionMap.push_back(SectionID);
  }

  // Process the symbol table.
  SmallVector<StringRef, 64> SymbolNames;
  processSymbols32(Obj, SectionMap, SymbolNames, SymtabLC);

  // Process the relocations for each section we're loading.
  Relocations.grow(Relocations.size() + SegmentLC->NumSections);
  Referrers.grow(Referrers.size() + SegmentLC->NumSections);
  for (unsigned SectNum = 0; SectNum != SegmentLC->NumSections; ++SectNum) {
    InMemoryStruct<macho::Section> Sect;
    Obj->ReadSection(*SegmentLCI, SectNum, Sect);
    if (!Sect)
      return Error("unable to load section: '" + Twine(SectNum) + "'");
    for (unsigned j = 0; j != Sect->NumRelocationTableEntries; ++j) {
      InMemoryStruct<macho::RelocationEntry> RE;
      Obj->ReadRelocationEntry(Sect->RelocationTableOffset, j, RE);
      if (RE->Word0 & macho::RF_Scattered)
        return Error("NOT YET IMPLEMENTED: scattered relocations.");
      // Word0 of the relocation is the offset into the section where the
      // relocation should be applied. We need to translate that into an
      // offset into a function since that's our atom.
      uint32_t Offset = RE->Word0;
      bool isExtern = (RE->Word1 >> 27) & 1;

      // FIXME: Get the relocation addend from the target address.
      // FIXME: VERY imporant for internal relocations.

      // Figure out the source symbol of the relocation. If isExtern is true,
      // this relocation references the symbol table, otherwise it references
      // a section in the same object, numbered from 1 through NumSections
      // (SectionBases is [0, NumSections-1]).
      uint32_t SourceNum = RE->Word1 & 0xffffff; // 24-bit value
      if (!isExtern) {
        assert(SourceNum > 0 && "Invalid relocation section number!");
        unsigned SectionID = SectionMap[SourceNum - 1];
        unsigned TargetID = SectionMap[SectNum];
        DEBUG(dbgs() << "Internal relocation at Section #"
                     << TargetID << " + " << Offset
                     << " from Section #"
                     << SectionID << " (Word1: "
                     << format("0x%x", RE->Word1) << ")\n");

        // Store the relocation information. It will get resolved when
        // the section addresses are assigned.
        uint32_t RelocationIndex = Relocations[SectionID].size();
        Relocations[SectionID].push_back(RelocationEntry(TargetID,
                                                         Offset,
                                                         RE->Word1,
                                                         0 /*Addend*/));
        Referrers[TargetID].push_back(Referrer(SectionID, RelocationIndex));
      } else {
        StringRef SourceName = SymbolNames[SourceNum];

        // Now store the relocation information. Associate it with the source
        // symbol. Just add it to the unresolved list and let the general
        // path post-load resolve it if we know where the symbol is.
        UnresolvedRelocations[SourceName].push_back(RelocationEntry(SectNum,
                                                                    Offset,
                                                                    RE->Word1,
                                                                 0 /*Addend*/));
        DEBUG(dbgs() << "Relocation at Section #" << SectNum << " + " << Offset
              << " from '" << SourceName << "(Word1: "
              << format("0x%x", RE->Word1) << ")\n");
      }
    }
  }

  // Resolve the addresses of any symbols that were defined in this segment.
  for (int i = 0, e = SymbolNames.size(); i != e; ++i)
    resolveSymbol(SymbolNames[i]);

  return false;
}


bool RuntimeDyldMachO::
loadSegment64(const MachOObject *Obj,
              const MachOObject::LoadCommandInfo *SegmentLCI,
              const InMemoryStruct<macho::SymtabLoadCommand> &SymtabLC) {
  InMemoryStruct<macho::Segment64LoadCommand> Segment64LC;
  Obj->ReadSegment64LoadCommand(*SegmentLCI, Segment64LC);
  if (!Segment64LC)
    return Error("unable to load segment load command");


  SmallVector<unsigned, 16> SectionMap;
  for (unsigned SectNum = 0; SectNum != Segment64LC->NumSections; ++SectNum) {
    InMemoryStruct<macho::Section64> Sect;
    Obj->ReadSection64(*SegmentLCI, SectNum, Sect);
    if (!Sect)
      return Error("unable to load section: '" + Twine(SectNum) + "'");

    // Allocate memory via the MM for the section.
    uint8_t *Buffer;
    uint32_t SectionID = Sections.size();
    unsigned Align = 1 << Sect->Align; // .o file has log2 alignment.
    if (Sect->Flags == 0x80000400)
      Buffer = MemMgr->allocateCodeSection(Sect->Size, Align, SectionID);
    else
      Buffer = MemMgr->allocateDataSection(Sect->Size, Align, SectionID);

    DEBUG(dbgs() << "Loading "
                 << ((Sect->Flags == 0x80000400) ? "text" : "data")
                 << " (ID #" << SectionID << ")"
                 << " '" << Sect->SegmentName << ","
                 << Sect->Name << "' of size " << Sect->Size
                 << " (align " << Align << ")"
                 << " to address " << Buffer << ".\n");

    // Copy the payload from the object file into the allocated buffer.
    uint8_t *Base = (uint8_t*)Obj->getData(Segment64LC->FileOffset,
                                           Segment64LC->FileSize).data();
    memcpy(Buffer, Base + Sect->Address, Sect->Size);

    // Remember what got allocated for this SectionID.
    Sections.push_back(sys::MemoryBlock(Buffer, Sect->Size));
    SectionLocalMemToID[Buffer] = SectionID;

    // By default, the load address of a section is its memory buffer.
    SectionLoadAddress.push_back((uint64_t)Buffer);

    // Keep a map of object file section numbers to corresponding SectionIDs
    // while processing the file.
    SectionMap.push_back(SectionID);
  }

  // Process the symbol table.
  SmallVector<StringRef, 64> SymbolNames;
  processSymbols64(Obj, SectionMap, SymbolNames, SymtabLC);

  // Process the relocations for each section we're loading.
  Relocations.grow(Relocations.size() + Segment64LC->NumSections);
  Referrers.grow(Referrers.size() + Segment64LC->NumSections);
  for (unsigned SectNum = 0; SectNum != Segment64LC->NumSections; ++SectNum) {
    InMemoryStruct<macho::Section64> Sect;
    Obj->ReadSection64(*SegmentLCI, SectNum, Sect);
    if (!Sect)
      return Error("unable to load section: '" + Twine(SectNum) + "'");
    for (unsigned j = 0; j != Sect->NumRelocationTableEntries; ++j) {
      InMemoryStruct<macho::RelocationEntry> RE;
      Obj->ReadRelocationEntry(Sect->RelocationTableOffset, j, RE);
      if (RE->Word0 & macho::RF_Scattered)
        return Error("NOT YET IMPLEMENTED: scattered relocations.");
      // Word0 of the relocation is the offset into the section where the
      // relocation should be applied. We need to translate that into an
      // offset into a function since that's our atom.
      uint32_t Offset = RE->Word0;
      bool isExtern = (RE->Word1 >> 27) & 1;

      // FIXME: Get the relocation addend from the target address.
      // FIXME: VERY imporant for internal relocations.

      // Figure out the source symbol of the relocation. If isExtern is true,
      // this relocation references the symbol table, otherwise it references
      // a section in the same object, numbered from 1 through NumSections
      // (SectionBases is [0, NumSections-1]).
      uint32_t SourceNum = RE->Word1 & 0xffffff; // 24-bit value
      if (!isExtern) {
        assert(SourceNum > 0 && "Invalid relocation section number!");
        unsigned SectionID = SectionMap[SourceNum - 1];
        unsigned TargetID = SectionMap[SectNum];
        DEBUG(dbgs() << "Internal relocation at Section #"
                     << TargetID << " + " << Offset
                     << " from Section #"
                     << SectionID << " (Word1: "
                     << format("0x%x", RE->Word1) << ")\n");

        // Store the relocation information. It will get resolved when
        // the section addresses are assigned.
        uint32_t RelocationIndex = Relocations[SectionID].size();
        Relocations[SectionID].push_back(RelocationEntry(TargetID,
                                                         Offset,
                                                         RE->Word1,
                                                         0 /*Addend*/));
        Referrers[TargetID].push_back(Referrer(SectionID, RelocationIndex));
      } else {
        StringRef SourceName = SymbolNames[SourceNum];

        // Now store the relocation information. Associate it with the source
        // symbol. Just add it to the unresolved list and let the general
        // path post-load resolve it if we know where the symbol is.
        UnresolvedRelocations[SourceName].push_back(RelocationEntry(SectNum,
                                                                    Offset,
                                                                    RE->Word1,
                                                                 0 /*Addend*/));
        DEBUG(dbgs() << "Relocation at Section #" << SectNum << " + " << Offset
              << " from '" << SourceName << "(Word1: "
              << format("0x%x", RE->Word1) << ")\n");
      }
    }
  }

  // Resolve the addresses of any symbols that were defined in this segment.
  for (int i = 0, e = SymbolNames.size(); i != e; ++i)
    resolveSymbol(SymbolNames[i]);

  return false;
}

bool RuntimeDyldMachO::
processSymbols32(const MachOObject *Obj,
                 SmallVectorImpl<unsigned> &SectionMap,
                 SmallVectorImpl<StringRef> &SymbolNames,
                 const InMemoryStruct<macho::SymtabLoadCommand> &SymtabLC) {
  // FIXME: Combine w/ processSymbols64. Factor 64/32 datatype and such.
  for (unsigned i = 0; i != SymtabLC->NumSymbolTableEntries; ++i) {
    InMemoryStruct<macho::SymbolTableEntry> STE;
    Obj->ReadSymbolTableEntry(SymtabLC->SymbolTableOffset, i, STE);
    if (!STE)
      return Error("unable to read symbol: '" + Twine(i) + "'");
    // Get the symbol name.
    StringRef Name = Obj->getStringAtIndex(STE->StringIndex);
    SymbolNames.push_back(Name);

    // FIXME: Check the symbol type and flags.
    if (STE->Type != 0xF)  // external, defined in this segment.
      continue;
    // Flags in the upper nibble we don't care about.
    if ((STE->Flags & 0xf) != 0x0)
      continue;

    // Remember the symbol.
    uint32_t SectionID = SectionMap[STE->SectionIndex - 1];
    SymbolTable[Name] = SymbolLoc(SectionID, STE->Value);

    DEBUG(dbgs() << "Symbol: '" << Name << "' @ "
                 << (getSectionAddress(SectionID) + STE->Value)
                 << "\n");
  }
  return false;
}

bool RuntimeDyldMachO::
processSymbols64(const MachOObject *Obj,
                 SmallVectorImpl<unsigned> &SectionMap,
                 SmallVectorImpl<StringRef> &SymbolNames,
                 const InMemoryStruct<macho::SymtabLoadCommand> &SymtabLC) {
  for (unsigned i = 0; i != SymtabLC->NumSymbolTableEntries; ++i) {
    InMemoryStruct<macho::Symbol64TableEntry> STE;
    Obj->ReadSymbol64TableEntry(SymtabLC->SymbolTableOffset, i, STE);
    if (!STE)
      return Error("unable to read symbol: '" + Twine(i) + "'");
    // Get the symbol name.
    StringRef Name = Obj->getStringAtIndex(STE->StringIndex);
    SymbolNames.push_back(Name);

    // FIXME: Check the symbol type and flags.
    if (STE->Type != 0xF)  // external, defined in this segment.
      continue;
    // Flags in the upper nibble we don't care about.
    if ((STE->Flags & 0xf) != 0x0)
      continue;

    // Remember the symbol.
    uint32_t SectionID = SectionMap[STE->SectionIndex - 1];
    SymbolTable[Name] = SymbolLoc(SectionID, STE->Value);

    DEBUG(dbgs() << "Symbol: '" << Name << "' @ "
                 << (getSectionAddress(SectionID) + STE->Value)
                 << "\n");
  }
  return false;
}

// resolveSymbol - Resolve any relocations to the specified symbol if
// we know where it lives.
void RuntimeDyldMachO::resolveSymbol(StringRef Name) {
  StringMap<SymbolLoc>::const_iterator Loc = SymbolTable.find(Name);
  if (Loc == SymbolTable.end())
    return;

  RelocationList &Relocs = UnresolvedRelocations[Name];
  DEBUG(dbgs() << "Resolving symbol '" << Name << "'\n");
  for (int i = 0, e = Relocs.size(); i != e; ++i) {
    // Change the relocation to be section relative rather than symbol
    // relative and move it to the resolved relocation list.
    RelocationEntry Entry = Relocs[i];
    Entry.Addend += Loc->second.second;
    uint32_t RelocationIndex = Relocations[Loc->second.first].size();
    Relocations[Loc->second.first].push_back(Entry);
    Referrers[Entry.SectionID].push_back(Referrer(Loc->second.first, RelocationIndex));
  }
  // FIXME: Keep a worklist of the relocations we've added so that we can
  // resolve more selectively later.
  Relocs.clear();
}

bool RuntimeDyldMachO::loadObject(MemoryBuffer *InputBuffer) {
  // If the linker is in an error state, don't do anything.
  if (hasError())
    return true;
  // Load the Mach-O wrapper object.
  std::string ErrorStr;
  OwningPtr<MachOObject> Obj(
    MachOObject::LoadFromBuffer(InputBuffer, &ErrorStr));
  if (!Obj)
    return Error("unable to load object: '" + ErrorStr + "'");

  // Get the CPU type information from the header.
  const macho::Header &Header = Obj->getHeader();

  // FIXME: Error checking that the loaded object is compatible with
  //        the system we're running on.
  CPUType = Header.CPUType;
  CPUSubtype = Header.CPUSubtype;

  // Validate that the load commands match what we expect.
  const MachOObject::LoadCommandInfo *SegmentLCI = 0, *SymtabLCI = 0,
    *DysymtabLCI = 0;
  for (unsigned i = 0; i != Header.NumLoadCommands; ++i) {
    const MachOObject::LoadCommandInfo &LCI = Obj->getLoadCommandInfo(i);
    switch (LCI.Command.Type) {
    case macho::LCT_Segment:
    case macho::LCT_Segment64:
      if (SegmentLCI)
        return Error("unexpected input object (multiple segments)");
      SegmentLCI = &LCI;
      break;
    case macho::LCT_Symtab:
      if (SymtabLCI)
        return Error("unexpected input object (multiple symbol tables)");
      SymtabLCI = &LCI;
      break;
    case macho::LCT_Dysymtab:
      if (DysymtabLCI)
        return Error("unexpected input object (multiple symbol tables)");
      DysymtabLCI = &LCI;
      break;
    default:
      return Error("unexpected input object (unexpected load command");
    }
  }

  if (!SymtabLCI)
    return Error("no symbol table found in object");
  if (!SegmentLCI)
    return Error("no segments found in object");

  // Read and register the symbol table data.
  InMemoryStruct<macho::SymtabLoadCommand> SymtabLC;
  Obj->ReadSymtabLoadCommand(*SymtabLCI, SymtabLC);
  if (!SymtabLC)
    return Error("unable to load symbol table load command");
  Obj->RegisterStringTable(*SymtabLC);

  // Read the dynamic link-edit information, if present (not present in static
  // objects).
  if (DysymtabLCI) {
    InMemoryStruct<macho::DysymtabLoadCommand> DysymtabLC;
    Obj->ReadDysymtabLoadCommand(*DysymtabLCI, DysymtabLC);
    if (!DysymtabLC)
      return Error("unable to load dynamic link-exit load command");

    // FIXME: We don't support anything interesting yet.
//    if (DysymtabLC->LocalSymbolsIndex != 0)
//      return Error("NOT YET IMPLEMENTED: local symbol entries");
//    if (DysymtabLC->ExternalSymbolsIndex != 0)
//      return Error("NOT YET IMPLEMENTED: non-external symbol entries");
//    if (DysymtabLC->UndefinedSymbolsIndex != SymtabLC->NumSymbolTableEntries)
//      return Error("NOT YET IMPLEMENTED: undefined symbol entries");
  }

  // Load the segment load command.
  if (SegmentLCI->Command.Type == macho::LCT_Segment) {
    if (loadSegment32(Obj.get(), SegmentLCI, SymtabLC))
      return true;
  } else {
    if (loadSegment64(Obj.get(), SegmentLCI, SymtabLC))
      return true;
  }

  // Assign the addresses of the sections from the object so that any
  // relocations to them get set properly.
  // FIXME: This is done directly from the client at the moment. We should
  // default the values to the local storage, at least when the target arch
  // is the same as the host arch.

  return false;
}

// Assign an address to a symbol name and resolve all the relocations
// associated with it.
void RuntimeDyldMachO::reassignSectionAddress(unsigned SectionID,
                                              uint64_t Addr) {
  // The address to use for relocation resolution is not
  // the address of the local section buffer. We must be doing
  // a remote execution environment of some sort. Re-apply any
  // relocations referencing this section with the given address.
  //
  // Addr is a uint64_t because we can't assume the pointer width
  // of the target is the same as that of the host. Just use a generic
  // "big enough" type.

  SectionLoadAddress[SectionID] = Addr;

  RelocationList &Relocs = Relocations[SectionID];
  for (unsigned i = 0, e = Relocs.size(); i != e; ++i) {
    RelocationEntry &RE = Relocs[i];
    uint8_t *Target = (uint8_t*)Sections[RE.SectionID].base() + RE.Offset;
    uint64_t FinalTarget = (uint64_t)SectionLoadAddress[RE.SectionID] + RE.Offset;
    bool isPCRel = (RE.Data >> 24) & 1;
    unsigned Type = (RE.Data >> 28) & 0xf;
    unsigned Size = 1 << ((RE.Data >> 25) & 3);

    DEBUG(dbgs() << "Resolving relocation at Section #" << RE.SectionID
          << " + " << RE.Offset << " (" << format("%p", Target) << ")"
          << " from Section #" << SectionID << " (" << format("%p", Addr) << ")"
          << "(" << (isPCRel ? "pcrel" : "absolute")
          << ", type: " << Type << ", Size: " << Size << ", Addend: "
          << RE.Addend << ").\n");

    resolveRelocation(Target,
                      FinalTarget,
                      Addr,
                      isPCRel,
                      Type,
                      Size,
                      RE.Addend);
  }
  ReferrerList &Refers = Referrers[SectionID];
  for (unsigned i = 0, e = Refers.size(); i != e; ++i) {
    Referrer &R = Refers[i];
    RelocationEntry &RE = Relocations[R.SectionID][R.Index];
    uint8_t *Target = (uint8_t*)Sections[RE.SectionID].base() + RE.Offset;
    uint64_t FinalTarget = (uint64_t)SectionLoadAddress[RE.SectionID] + RE.Offset;
    bool isPCRel = (RE.Data >> 24) & 1;
    unsigned Type = (RE.Data >> 28) & 0xf;
    unsigned Size = 1 << ((RE.Data >> 25) & 3);

    DEBUG(dbgs() << "Resolving relocation at Section #" << RE.SectionID
          << " + " << RE.Offset << " (" << format("%p", Target) << ")"
          << " from Section #" << SectionID << " (" << format("%p", Addr) << ")"
          << "(" << (isPCRel ? "pcrel" : "absolute")
          << ", type: " << Type << ", Size: " << Size << ", Addend: "
          << RE.Addend << ").\n");

    resolveRelocation(Target,
                      FinalTarget,
                      Addr,
                      isPCRel,
                      Type,
                      Size,
                      RE.Addend);
  }
}

bool RuntimeDyldMachO::isKnownFormat(const MemoryBuffer *InputBuffer) {
  StringRef Magic = InputBuffer->getBuffer().slice(0, 4);
  if (Magic == "\xFE\xED\xFA\xCE") return true;
  if (Magic == "\xCE\xFA\xED\xFE") return true;
  if (Magic == "\xFE\xED\xFA\xCF") return true;
  if (Magic == "\xCF\xFA\xED\xFE") return true;
  return false;
}

} // end namespace llvm
