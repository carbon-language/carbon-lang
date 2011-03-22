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

#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/Object/MachOObject.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/system_error.h"
using namespace llvm;
using namespace llvm::object;

namespace llvm {
class RuntimeDyldImpl {
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
  void **SectionBases = new void*[Segment32LC->NumSections];
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

    SectionBases[i] = (char*) Data.base() + Sect->Address;
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

    SymbolTable[Name] = Address;
  }

  // We've loaded the section; now mark the functions in it as executable.
  // FIXME: We really should use the JITMemoryManager for this.
  sys::Memory::setRangeExecutable(Data.base(), Data.size());

  delete SectionBases;
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

  // Bind the section indices to address.
  void **SectionBases = new void*[Segment64LC->NumSections];
  for (unsigned i = 0; i != Segment64LC->NumSections; ++i) {
    InMemoryStruct<macho::Section64> Sect;
    Obj->ReadSection64(*SegmentLCI, i, Sect);
    if (!Sect)
      return Error("unable to load section: '" + Twine(i) + "'");

    // FIXME: We don't support relocations yet.
    if (Sect->NumRelocationTableEntries != 0)
      return Error("not yet implemented: relocations!");

    // FIXME: Improve check.
    if (Sect->Flags != 0x80000400)
      return Error("unsupported section type!");

    SectionBases[i] = (char*) Data.base() + Sect->Address;
  }

  // Bind all the symbols to address.
  for (unsigned i = 0; i != SymtabLC->NumSymbolTableEntries; ++i) {
    InMemoryStruct<macho::Symbol64TableEntry> STE;
    Obj->ReadSymbol64TableEntry(SymtabLC->SymbolTableOffset, i, STE);
    if (!STE)
      return Error("unable to read symbol: '" + Twine(i) + "'");
    if (STE->SectionIndex == 0)
      return Error("unexpected undefined symbol!");

    unsigned Index = STE->SectionIndex - 1;
    if (Index >= Segment64LC->NumSections)
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

    SymbolTable[Name] = Address;
  }

  // We've loaded the section; now mark the functions in it as executable.
  // FIXME: We really should use the JITMemoryManager for this.
  sys::Memory::setRangeExecutable(Data.base(), Data.size());

  delete SectionBases;
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

  // Validate that the load commands match what we expect.
  const MachOObject::LoadCommandInfo *SegmentLCI = 0, *SymtabLCI = 0,
    *DysymtabLCI = 0;
  for (unsigned i = 0; i != Obj->getHeader().NumLoadCommands; ++i) {
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
    if (DysymtabLC->LocalSymbolsIndex != 0)
      return Error("NOT YET IMPLEMENTED: local symbol entries");
    if (DysymtabLC->ExternalSymbolsIndex != 0)
      return Error("NOT YET IMPLEMENTED: non-external symbol entries");
    if (DysymtabLC->UndefinedSymbolsIndex != SymtabLC->NumSymbolTableEntries)
      return Error("NOT YET IMPLEMENTED: undefined symbol entries");
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

} // end namespace llvm
