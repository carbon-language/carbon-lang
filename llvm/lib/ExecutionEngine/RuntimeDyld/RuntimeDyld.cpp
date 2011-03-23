//===-- RuntimeDyld.h - Run-time dynamic linker for MC-JIT ------*- C++ -*-===//
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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/Object/MachOObject.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/system_error.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;
using namespace llvm::object;

namespace llvm {
class RuntimeDyldImpl {
  unsigned CPUType;
  unsigned CPUSubtype;

  // Master symbol table. As modules are loaded and external symbols are
  // resolved, their addresses are stored here.
  StringMap<void*> SymbolTable;

  // FIXME: Should have multiple data blocks, one for each loaded chunk of
  //        compiled code.
  sys::MemoryBlock Data;

  bool HasError;
  std::string ErrorStr;

  // Set the error state and record an error string.
  bool Error(const Twine &Msg) {
    ErrorStr = Msg.str();
    HasError = true;
    return true;
  }

  bool resolveRelocation(uint32_t BaseSection, macho::RelocationEntry RE,
                         SmallVectorImpl<void *> &SectionBases,
                         SmallVectorImpl<StringRef> &SymbolNames);
  bool resolveX86_64Relocation(intptr_t Address, intptr_t Value, bool isPCRel,
                               unsigned Type, unsigned Size);
  bool resolveARMRelocation(intptr_t Address, intptr_t Value, bool isPCRel,
                            unsigned Type, unsigned Size);

  bool loadSegment32(const MachOObject *Obj,
                     const MachOObject::LoadCommandInfo *SegmentLCI,
                     const InMemoryStruct<macho::SymtabLoadCommand> &SymtabLC);
  bool loadSegment64(const MachOObject *Obj,
                     const MachOObject::LoadCommandInfo *SegmentLCI,
                     const InMemoryStruct<macho::SymtabLoadCommand> &SymtabLC);

public:
  RuntimeDyldImpl() : HasError(false) {}

  bool loadObject(MemoryBuffer *InputBuffer);

  void *getSymbolAddress(StringRef Name) {
    // Use lookup() rather than [] because we don't want to add an entry
    // if there isn't one already, which the [] operator does.
    return SymbolTable.lookup(Name);
  }

  sys::MemoryBlock getMemoryBlock() { return Data; }

  // Is the linker in an error state?
  bool hasError() { return HasError; }

  // Mark the error condition as handled and continue.
  void clearError() { HasError = false; }

  // Get the error message.
  StringRef getErrorString() { return ErrorStr; }
};

// FIXME: Relocations for targets other than x86_64.
bool RuntimeDyldImpl::
resolveRelocation(uint32_t BaseSection, macho::RelocationEntry RE,
                  SmallVectorImpl<void *> &SectionBases,
                  SmallVectorImpl<StringRef> &SymbolNames) {
  // struct relocation_info {
  //   int32_t r_address;
  //   uint32_t r_symbolnum:24,
  //            r_pcrel:1,
  //            r_length:2,
  //            r_extern:1,
  //            r_type:4;
  // };
  uint32_t SymbolNum = RE.Word1 & 0xffffff; // 24-bit value
  bool isPCRel = (RE.Word1 >> 24) & 1;
  unsigned Log2Size = (RE.Word1 >> 25) & 3;
  bool isExtern = (RE.Word1 >> 27) & 1;
  unsigned Type = (RE.Word1 >> 28) & 0xf;
  if (RE.Word0 & macho::RF_Scattered)
    return Error("NOT YET IMPLEMENTED: scattered relocations.");

  // The address requiring a relocation.
  intptr_t Address = (intptr_t)SectionBases[BaseSection] + RE.Word0;

  // Figure out the target address of the relocation. If isExtern is true,
  // this relocation references the symbol table, otherwise it references
  // a section in the same object, numbered from 1 through NumSections
  // (SectionBases is [0, NumSections-1]).
  intptr_t Value;
  if (isExtern) {
    StringRef Name = SymbolNames[SymbolNum];
    if (SymbolTable.lookup(Name)) {
      // The symbol is in our symbol table, so we can resolve it directly.
      Value = (intptr_t)SymbolTable[Name];
    } else {
      return Error("NOT YET IMPLEMENTED: relocations to pre-compiled code.");
    }
    DEBUG(dbgs() << "Resolve relocation(" << Type << ") from '" << Name
                 << "' to " << format("0x%x", Address) << ".\n");
  } else {
    // For non-external relocations, the SymbolNum is actual a section number
    // as described above.
    Value = (intptr_t)SectionBases[SymbolNum - 1];
  }

  unsigned Size = 1 << Log2Size;
  switch (CPUType) {
  default: assert(0 && "Unsupported CPU type!");
  case mach::CTM_x86_64:
    return resolveX86_64Relocation(Address, Value, isPCRel, Type, Size);
  case mach::CTM_ARM:
    return resolveARMRelocation(Address, Value, isPCRel, Type, Size);
  }
  llvm_unreachable("");
}

bool RuntimeDyldImpl::resolveX86_64Relocation(intptr_t Address, intptr_t Value,
                                              bool isPCRel, unsigned Type,
                                              unsigned Size) {
  // If the relocation is PC-relative, the value to be encoded is the
  // pointer difference.
  if (isPCRel)
    // FIXME: It seems this value needs to be adjusted by 4 for an effective PC
    // address. Is that expected? Only for branches, perhaps?
    Value -= Address + 4;

  switch(Type) {
  default:
    llvm_unreachable("Invalid relocation type!");
  case macho::RIT_X86_64_Unsigned:
  case macho::RIT_X86_64_Branch: {
    // Mask in the target value a byte at a time (we don't have an alignment
    // guarantee for the target address, so this is safest).
    uint8_t *p = (uint8_t*)Address;
    for (unsigned i = 0; i < Size; ++i) {
      *p++ = (uint8_t)Value;
      Value >>= 8;
    }
    return false;
  }
  case macho::RIT_X86_64_Signed:
  case macho::RIT_X86_64_GOTLoad:
  case macho::RIT_X86_64_GOT:
  case macho::RIT_X86_64_Subtractor:
  case macho::RIT_X86_64_Signed1:
  case macho::RIT_X86_64_Signed2:
  case macho::RIT_X86_64_Signed4:
  case macho::RIT_X86_64_TLV:
    return Error("Relocation type not implemented yet!");
  }
  return false;
}

bool RuntimeDyldImpl::resolveARMRelocation(intptr_t Address, intptr_t Value,
                                           bool isPCRel, unsigned Type,
                                           unsigned Size) {
  // If the relocation is PC-relative, the value to be encoded is the
  // pointer difference.
  if (isPCRel) {
    Value -= Address;
    // ARM PCRel relocations have an effective-PC offset of two instructions
    // (four bytes in Thumb mode, 8 bytes in ARM mode).
    // FIXME: For now, assume ARM mode.
    Value -= 8;
  }

  switch(Type) {
  default:
  case macho::RIT_Vanilla: {
    llvm_unreachable("Invalid relocation type!");
    // Mask in the target value a byte at a time (we don't have an alignment
    // guarantee for the target address, so this is safest).
    uint8_t *p = (uint8_t*)Address;
    for (unsigned i = 0; i < Size; ++i) {
      *p++ = (uint8_t)Value;
      Value >>= 8;
    }
    return false;
  }
  case macho::RIT_Pair:
  case macho::RIT_Difference:
  case macho::RIT_ARM_LocalDifference:
  case macho::RIT_ARM_PreboundLazyPointer:
  case macho::RIT_ARM_Branch24Bit:
  case macho::RIT_ARM_ThumbBranch22Bit:
  case macho::RIT_ARM_ThumbBranch32Bit:
  case macho::RIT_ARM_Half:
  case macho::RIT_ARM_HalfDifference:
    return Error("Relocation type not implemented yet!");
  }
  return false;
}

bool RuntimeDyldImpl::
loadSegment32(const MachOObject *Obj,
              const MachOObject::LoadCommandInfo *SegmentLCI,
              const InMemoryStruct<macho::SymtabLoadCommand> &SymtabLC) {
  InMemoryStruct<macho::SegmentLoadCommand> Segment32LC;
  Obj->ReadSegmentLoadCommand(*SegmentLCI, Segment32LC);
  if (!Segment32LC)
    return Error("unable to load segment load command");

  // Map the segment into memory.
  std::string ErrorStr;
  Data = sys::Memory::AllocateRWX(Segment32LC->VMSize, 0, &ErrorStr);
  if (!Data.base())
    return Error("unable to allocate memory block: '" + ErrorStr + "'");
  memcpy(Data.base(), Obj->getData(Segment32LC->FileOffset,
                                   Segment32LC->FileSize).data(),
         Segment32LC->FileSize);
  memset((char*)Data.base() + Segment32LC->FileSize, 0,
         Segment32LC->VMSize - Segment32LC->FileSize);

  // Bind the section indices to address.
  SmallVector<void *, 16> SectionBases;
  for (unsigned i = 0; i != Segment32LC->NumSections; ++i) {
    InMemoryStruct<macho::Section> Sect;
    Obj->ReadSection(*SegmentLCI, i, Sect);
    if (!Sect)
      return Error("unable to load section: '" + Twine(i) + "'");

    // FIXME: We don't support relocations yet.
    if (Sect->NumRelocationTableEntries != 0)
      return Error("not yet implemented: relocations!");

    // FIXME: Improve check.
    if (Sect->Flags != 0x80000400)
      return Error("unsupported section type!");

    SectionBases.push_back((char*) Data.base() + Sect->Address);
  }

  // Bind all the symbols to address.
  for (unsigned i = 0; i != SymtabLC->NumSymbolTableEntries; ++i) {
    InMemoryStruct<macho::SymbolTableEntry> STE;
    Obj->ReadSymbolTableEntry(SymtabLC->SymbolTableOffset, i, STE);
    if (!STE)
      return Error("unable to read symbol: '" + Twine(i) + "'");
    if (STE->SectionIndex == 0)
      return Error("unexpected undefined symbol!");

    unsigned Index = STE->SectionIndex - 1;
    if (Index >= Segment32LC->NumSections)
      return Error("invalid section index for symbol: '" + Twine() + "'");

    // Get the symbol name.
    StringRef Name = Obj->getStringAtIndex(STE->StringIndex);

    // Get the section base address.
    void *SectionBase = SectionBases[Index];

    // Get the symbol address.
    void *Address = (char*) SectionBase + STE->Value;

    // FIXME: Check the symbol type and flags.
    if (STE->Type != 0xF)
      return Error("unexpected symbol type!");
    if (STE->Flags != 0x0)
      return Error("unexpected symbol type!");

    DEBUG(dbgs() << "Symbol: '" << Name << "' @ " << Address << "\n");

    SymbolTable[Name] = Address;
  }

  // We've loaded the section; now mark the functions in it as executable.
  // FIXME: We really should use the JITMemoryManager for this.
  sys::Memory::setRangeExecutable(Data.base(), Data.size());

  return false;
}


bool RuntimeDyldImpl::
loadSegment64(const MachOObject *Obj,
              const MachOObject::LoadCommandInfo *SegmentLCI,
              const InMemoryStruct<macho::SymtabLoadCommand> &SymtabLC) {
  InMemoryStruct<macho::Segment64LoadCommand> Segment64LC;
  Obj->ReadSegment64LoadCommand(*SegmentLCI, Segment64LC);
  if (!Segment64LC)
    return Error("unable to load segment load command");

  // Map the segment into memory.
  std::string ErrorStr;
  Data = sys::Memory::AllocateRWX(Segment64LC->VMSize, 0, &ErrorStr);
  if (!Data.base())
    return Error("unable to allocate memory block: '" + ErrorStr + "'");
  memcpy(Data.base(), Obj->getData(Segment64LC->FileOffset,
                                   Segment64LC->FileSize).data(),
         Segment64LC->FileSize);
  memset((char*)Data.base() + Segment64LC->FileSize, 0,
         Segment64LC->VMSize - Segment64LC->FileSize);

  // Bind the section indices to addresses and record the relocations we
  // need to resolve.
  typedef std::pair<uint32_t, macho::RelocationEntry> RelocationMap;
  SmallVector<RelocationMap, 64> Relocations;

  SmallVector<void *, 16> SectionBases;
  for (unsigned i = 0; i != Segment64LC->NumSections; ++i) {
    InMemoryStruct<macho::Section64> Sect;
    Obj->ReadSection64(*SegmentLCI, i, Sect);
    if (!Sect)
      return Error("unable to load section: '" + Twine(i) + "'");

    // Resolve any relocations the section has.
    for (unsigned j = 0; j != Sect->NumRelocationTableEntries; ++j) {
      InMemoryStruct<macho::RelocationEntry> RE;
      Obj->ReadRelocationEntry(Sect->RelocationTableOffset, j, RE);
      Relocations.push_back(RelocationMap(j, *RE));
    }

    // FIXME: Improve check.
    if (Sect->Flags != 0x80000400)
      return Error("unsupported section type!");

    SectionBases.push_back((char*) Data.base() + Sect->Address);
  }

  // Bind all the symbols to address. Keep a record of the names for use
  // by relocation resolution.
  SmallVector<StringRef, 64> SymbolNames;
  for (unsigned i = 0; i != SymtabLC->NumSymbolTableEntries; ++i) {
    InMemoryStruct<macho::Symbol64TableEntry> STE;
    Obj->ReadSymbol64TableEntry(SymtabLC->SymbolTableOffset, i, STE);
    if (!STE)
      return Error("unable to read symbol: '" + Twine(i) + "'");
    // Get the symbol name.
    StringRef Name = Obj->getStringAtIndex(STE->StringIndex);
    SymbolNames.push_back(Name);

    // Just skip undefined symbols. They'll be loaded from whatever
    // module they come from (or system dylib) when we resolve relocations
    // involving them.
    if (STE->SectionIndex == 0)
      continue;

    unsigned Index = STE->SectionIndex - 1;
    if (Index >= Segment64LC->NumSections)
      return Error("invalid section index for symbol: '" + Twine() + "'");

    // Get the section base address.
    void *SectionBase = SectionBases[Index];

    // Get the symbol address.
    void *Address = (char*) SectionBase + STE->Value;

    // FIXME: Check the symbol type and flags.
    if (STE->Type != 0xF)
      return Error("unexpected symbol type!");
    if (STE->Flags != 0x0)
      return Error("unexpected symbol type!");

    DEBUG(dbgs() << "Symbol: '" << Name << "' @ " << Address << "\n");
    SymbolTable[Name] = Address;
  }

  // Now resolve any relocations.
  for (unsigned i = 0, e = Relocations.size(); i != e; ++i) {
    if (resolveRelocation(Relocations[i].first, Relocations[i].second,
                          SectionBases, SymbolNames))
      return true;
  }

  // We've loaded the section; now mark the functions in it as executable.
  // FIXME: We really should use the JITMemoryManager for this.
  sys::Memory::setRangeExecutable(Data.base(), Data.size());

  return false;
}

bool RuntimeDyldImpl::loadObject(MemoryBuffer *InputBuffer) {
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
    return Error("no symbol table found in object");

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

  return false;
}


//===----------------------------------------------------------------------===//
// RuntimeDyld class implementation
RuntimeDyld::RuntimeDyld() {
  Dyld = new RuntimeDyldImpl;
}

RuntimeDyld::~RuntimeDyld() {
  delete Dyld;
}

bool RuntimeDyld::loadObject(MemoryBuffer *InputBuffer) {
  return Dyld->loadObject(InputBuffer);
}

void *RuntimeDyld::getSymbolAddress(StringRef Name) {
  return Dyld->getSymbolAddress(Name);
}

sys::MemoryBlock RuntimeDyld::getMemoryBlock() {
  return Dyld->getMemoryBlock();
}

StringRef RuntimeDyld::getErrorString() {
  return Dyld->getErrorString();
}

} // end namespace llvm
