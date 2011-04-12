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
#include "llvm/ADT/STLExtras.h"
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

// Empty out-of-line virtual destructor as the key function.
RTDyldMemoryManager::~RTDyldMemoryManager() {}

namespace llvm {
class RuntimeDyldImpl {
  unsigned CPUType;
  unsigned CPUSubtype;

  // The MemoryManager to load objects into.
  RTDyldMemoryManager *MemMgr;

  // FIXME: This all assumes we're dealing with external symbols for anything
  //        explicitly referenced. I.e., we can index by name and things
  //        will work out. In practice, this may not be the case, so we
  //        should find a way to effectively generalize.

  // For each function, we have a MemoryBlock of it's instruction data.
  StringMap<sys::MemoryBlock> Functions;

  // Master symbol table. As modules are loaded and external symbols are
  // resolved, their addresses are stored here.
  StringMap<uint8_t*> SymbolTable;

  // For each symbol, keep a list of relocations based on it. Anytime
  // its address is reassigned (the JIT re-compiled the function, e.g.),
  // the relocations get re-resolved.
  struct RelocationEntry {
    std::string Target;     // Object this relocation is contained in.
    uint64_t    Offset;     // Offset into the object for the relocation.
    uint32_t    Data;       // Second word of the raw macho relocation entry.
    int64_t     Addend;     // Addend encoded in the instruction itself, if any.
    bool        isResolved; // Has this relocation been resolved previously?

    RelocationEntry(StringRef t, uint64_t offset, uint32_t data, int64_t addend)
      : Target(t), Offset(offset), Data(data), Addend(addend),
        isResolved(false) {}
  };
  typedef SmallVector<RelocationEntry, 4> RelocationList;
  StringMap<RelocationList> Relocations;

  // FIXME: Also keep a map of all the relocations contained in an object. Use
  // this to dynamically answer whether all of the relocations in it have
  // been resolved or not.

  bool HasError;
  std::string ErrorStr;

  // Set the error state and record an error string.
  bool Error(const Twine &Msg) {
    ErrorStr = Msg.str();
    HasError = true;
    return true;
  }

  void extractFunction(StringRef Name, uint8_t *StartAddress,
                       uint8_t *EndAddress);
  bool resolveRelocation(uint8_t *Address, uint8_t *Value, bool isPCRel,
                         unsigned Type, unsigned Size);
  bool resolveX86_64Relocation(uintptr_t Address, uintptr_t Value, bool isPCRel,
                               unsigned Type, unsigned Size);
  bool resolveARMRelocation(uintptr_t Address, uintptr_t Value, bool isPCRel,
                            unsigned Type, unsigned Size);

  bool loadSegment32(const MachOObject *Obj,
                     const MachOObject::LoadCommandInfo *SegmentLCI,
                     const InMemoryStruct<macho::SymtabLoadCommand> &SymtabLC);
  bool loadSegment64(const MachOObject *Obj,
                     const MachOObject::LoadCommandInfo *SegmentLCI,
                     const InMemoryStruct<macho::SymtabLoadCommand> &SymtabLC);

public:
  RuntimeDyldImpl(RTDyldMemoryManager *mm) : MemMgr(mm), HasError(false) {}

  bool loadObject(MemoryBuffer *InputBuffer);

  void *getSymbolAddress(StringRef Name) {
    // FIXME: Just look up as a function for now. Overly simple of course.
    // Work in progress.
    return SymbolTable.lookup(Name);
  }

  void resolveRelocations();

  void reassignSymbolAddress(StringRef Name, uint8_t *Addr);

  // Is the linker in an error state?
  bool hasError() { return HasError; }

  // Mark the error condition as handled and continue.
  void clearError() { HasError = false; }

  // Get the error message.
  StringRef getErrorString() { return ErrorStr; }
};

void RuntimeDyldImpl::extractFunction(StringRef Name, uint8_t *StartAddress,
                                      uint8_t *EndAddress) {
  // Allocate memory for the function via the memory manager.
  uintptr_t Size = EndAddress - StartAddress + 1;
  uint8_t *Mem = MemMgr->startFunctionBody(Name.data(), Size);
  assert(Size >= (uint64_t)(EndAddress - StartAddress + 1) &&
         "Memory manager failed to allocate enough memory!");
  // Copy the function payload into the memory block.
  memcpy(Mem, StartAddress, EndAddress - StartAddress + 1);
  MemMgr->endFunctionBody(Name.data(), Mem, Mem + Size);
  // Remember where we put it.
  Functions[Name] = sys::MemoryBlock(Mem, Size);
  // Default the assigned address for this symbol to wherever this
  // allocated it.
  SymbolTable[Name] = Mem;
  DEBUG(dbgs() << "    allocated to " << Mem << "\n");
}

bool RuntimeDyldImpl::
resolveRelocation(uint8_t *Address, uint8_t *Value, bool isPCRel,
                  unsigned Type, unsigned Size) {
  // This just dispatches to the proper target specific routine.
  switch (CPUType) {
  default: assert(0 && "Unsupported CPU type!");
  case mach::CTM_x86_64:
    return resolveX86_64Relocation((uintptr_t)Address, (uintptr_t)Value,
                                   isPCRel, Type, Size);
  case mach::CTM_ARM:
    return resolveARMRelocation((uintptr_t)Address, (uintptr_t)Value,
                                isPCRel, Type, Size);
  }
  llvm_unreachable("");
}

bool RuntimeDyldImpl::
resolveX86_64Relocation(uintptr_t Address, uintptr_t Value,
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

bool RuntimeDyldImpl::resolveARMRelocation(uintptr_t Address, uintptr_t Value,
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
    llvm_unreachable("Invalid relocation type!");
  case macho::RIT_Vanilla: {
    llvm_unreachable("Invalid relocation type!");
    // Mask in the target value a byte at a time (we don't have an alignment
    // guarantee for the target address, so this is safest).
    uint8_t *p = (uint8_t*)Address;
    for (unsigned i = 0; i < Size; ++i) {
      *p++ = (uint8_t)Value;
      Value >>= 8;
    }
    break;
  }
  case macho::RIT_ARM_Branch24Bit: {
    // Mask the value into the target address. We know instructions are
    // 32-bit aligned, so we can do it all at once.
    uint32_t *p = (uint32_t*)Address;
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

bool RuntimeDyldImpl::
loadSegment32(const MachOObject *Obj,
              const MachOObject::LoadCommandInfo *SegmentLCI,
              const InMemoryStruct<macho::SymtabLoadCommand> &SymtabLC) {
  InMemoryStruct<macho::SegmentLoadCommand> SegmentLC;
  Obj->ReadSegmentLoadCommand(*SegmentLCI, SegmentLC);
  if (!SegmentLC)
    return Error("unable to load segment load command");

  for (unsigned SectNum = 0; SectNum != SegmentLC->NumSections; ++SectNum) {
    InMemoryStruct<macho::Section> Sect;
    Obj->ReadSection(*SegmentLCI, SectNum, Sect);
    if (!Sect)
      return Error("unable to load section: '" + Twine(SectNum) + "'");

    // FIXME: Improve check.
    if (Sect->Flags != 0x80000400)
      return Error("unsupported section type!");

    // Address and names of symbols in the section.
    typedef std::pair<uint64_t, StringRef> SymbolEntry;
    SmallVector<SymbolEntry, 64> Symbols;
    // Index of all the names, in this section or not. Used when we're
    // dealing with relocation entries.
    SmallVector<StringRef, 64> SymbolNames;
    for (unsigned i = 0; i != SymtabLC->NumSymbolTableEntries; ++i) {
      InMemoryStruct<macho::SymbolTableEntry> STE;
      Obj->ReadSymbolTableEntry(SymtabLC->SymbolTableOffset, i, STE);
      if (!STE)
        return Error("unable to read symbol: '" + Twine(i) + "'");
      if (STE->SectionIndex > SegmentLC->NumSections)
        return Error("invalid section index for symbol: '" + Twine(i) + "'");
      // Get the symbol name.
      StringRef Name = Obj->getStringAtIndex(STE->StringIndex);
      SymbolNames.push_back(Name);

      // Just skip symbols not defined in this section.
      if ((unsigned)STE->SectionIndex - 1 != SectNum)
        continue;

      // FIXME: Check the symbol type and flags.
      if (STE->Type != 0xF)  // external, defined in this section.
        return Error("unexpected symbol type!");
      // Flags == 0x8 marks a thumb function for ARM, which is fine as it
      // doesn't require any special handling here.
      if (STE->Flags != 0x0 && STE->Flags != 0x8)
        return Error("unexpected symbol type!");

      // Remember the symbol.
      Symbols.push_back(SymbolEntry(STE->Value, Name));

      DEBUG(dbgs() << "Function sym: '" << Name << "' @ " <<
            (Sect->Address + STE->Value) << "\n");
    }
    // Sort the symbols by address, just in case they didn't come in that way.
    array_pod_sort(Symbols.begin(), Symbols.end());

    // Extract the function data.
    uint8_t *Base = (uint8_t*)Obj->getData(SegmentLC->FileOffset,
                                           SegmentLC->FileSize).data();
    for (unsigned i = 0, e = Symbols.size() - 1; i != e; ++i) {
      uint64_t StartOffset = Sect->Address + Symbols[i].first;
      uint64_t EndOffset = Symbols[i + 1].first - 1;
      DEBUG(dbgs() << "Extracting function: " << Symbols[i].second
                   << " from [" << StartOffset << ", " << EndOffset << "]\n");
      extractFunction(Symbols[i].second, Base + StartOffset, Base + EndOffset);
    }
    // The last symbol we do after since the end address is calculated
    // differently because there is no next symbol to reference.
    uint64_t StartOffset = Symbols[Symbols.size() - 1].first;
    uint64_t EndOffset = Sect->Size - 1;
    DEBUG(dbgs() << "Extracting function: " << Symbols[Symbols.size()-1].second
                 << " from [" << StartOffset << ", " << EndOffset << "]\n");
    extractFunction(Symbols[Symbols.size()-1].second,
                    Base + StartOffset, Base + EndOffset);

    // Now extract the relocation information for each function and process it.
    for (unsigned j = 0; j != Sect->NumRelocationTableEntries; ++j) {
      InMemoryStruct<macho::RelocationEntry> RE;
      Obj->ReadRelocationEntry(Sect->RelocationTableOffset, j, RE);
      if (RE->Word0 & macho::RF_Scattered)
        return Error("NOT YET IMPLEMENTED: scattered relocations.");
      // Word0 of the relocation is the offset into the section where the
      // relocation should be applied. We need to translate that into an
      // offset into a function since that's our atom.
      uint32_t Offset = RE->Word0;
      // Look for the function containing the address. This is used for JIT
      // code, so the number of functions in section is almost always going
      // to be very small (usually just one), so until we have use cases
      // where that's not true, just use a trivial linear search.
      unsigned SymbolNum;
      unsigned NumSymbols = Symbols.size();
      assert(NumSymbols > 0 && Symbols[0].first <= Offset &&
             "No symbol containing relocation!");
      for (SymbolNum = 0; SymbolNum < NumSymbols - 1; ++SymbolNum)
        if (Symbols[SymbolNum + 1].first > Offset)
          break;
      // Adjust the offset to be relative to the symbol.
      Offset -= Symbols[SymbolNum].first;
      // Get the name of the symbol containing the relocation.
      StringRef TargetName = SymbolNames[SymbolNum];

      bool isExtern = (RE->Word1 >> 27) & 1;
      // Figure out the source symbol of the relocation. If isExtern is true,
      // this relocation references the symbol table, otherwise it references
      // a section in the same object, numbered from 1 through NumSections
      // (SectionBases is [0, NumSections-1]).
      // FIXME: Some targets (ARM) use internal relocations even for
      // externally visible symbols, if the definition is in the same
      // file as the reference. We need to convert those back to by-name
      // references. We can resolve the address based on the section
      // offset and see if we have a symbol at that address. If we do,
      // use that; otherwise, puke.
      if (!isExtern)
        return Error("Internal relocations not supported.");
      uint32_t SourceNum = RE->Word1 & 0xffffff; // 24-bit value
      StringRef SourceName = SymbolNames[SourceNum];

      // FIXME: Get the relocation addend from the target address.

      // Now store the relocation information. Associate it with the source
      // symbol.
      Relocations[SourceName].push_back(RelocationEntry(TargetName,
                                                        Offset,
                                                        RE->Word1,
                                                        0 /*Addend*/));
      DEBUG(dbgs() << "Relocation at '" << TargetName << "' + " << Offset
                   << " from '" << SourceName << "(Word1: "
                   << format("0x%x", RE->Word1) << ")\n");
    }
  }
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

  for (unsigned SectNum = 0; SectNum != Segment64LC->NumSections; ++SectNum) {
    InMemoryStruct<macho::Section64> Sect;
    Obj->ReadSection64(*SegmentLCI, SectNum, Sect);
    if (!Sect)
      return Error("unable to load section: '" + Twine(SectNum) + "'");

    // FIXME: Improve check.
    if (Sect->Flags != 0x80000400)
      return Error("unsupported section type!");

    // Address and names of symbols in the section.
    typedef std::pair<uint64_t, StringRef> SymbolEntry;
    SmallVector<SymbolEntry, 64> Symbols;
    // Index of all the names, in this section or not. Used when we're
    // dealing with relocation entries.
    SmallVector<StringRef, 64> SymbolNames;
    for (unsigned i = 0; i != SymtabLC->NumSymbolTableEntries; ++i) {
      InMemoryStruct<macho::Symbol64TableEntry> STE;
      Obj->ReadSymbol64TableEntry(SymtabLC->SymbolTableOffset, i, STE);
      if (!STE)
        return Error("unable to read symbol: '" + Twine(i) + "'");
      if (STE->SectionIndex > Segment64LC->NumSections)
        return Error("invalid section index for symbol: '" + Twine(i) + "'");
      // Get the symbol name.
      StringRef Name = Obj->getStringAtIndex(STE->StringIndex);
      SymbolNames.push_back(Name);

      // Just skip symbols not defined in this section.
      if ((unsigned)STE->SectionIndex - 1 != SectNum)
        continue;

      // FIXME: Check the symbol type and flags.
      if (STE->Type != 0xF)  // external, defined in this section.
        return Error("unexpected symbol type!");
      if (STE->Flags != 0x0)
        return Error("unexpected symbol type!");

      // Remember the symbol.
      Symbols.push_back(SymbolEntry(STE->Value, Name));

      DEBUG(dbgs() << "Function sym: '" << Name << "' @ " <<
            (Sect->Address + STE->Value) << "\n");
    }
    // Sort the symbols by address, just in case they didn't come in that way.
    array_pod_sort(Symbols.begin(), Symbols.end());

    // Extract the function data.
    uint8_t *Base = (uint8_t*)Obj->getData(Segment64LC->FileOffset,
                                           Segment64LC->FileSize).data();
    for (unsigned i = 0, e = Symbols.size() - 1; i != e; ++i) {
      uint64_t StartOffset = Sect->Address + Symbols[i].first;
      uint64_t EndOffset = Symbols[i + 1].first - 1;
      DEBUG(dbgs() << "Extracting function: " << Symbols[i].second
                   << " from [" << StartOffset << ", " << EndOffset << "]\n");
      extractFunction(Symbols[i].second, Base + StartOffset, Base + EndOffset);
    }
    // The last symbol we do after since the end address is calculated
    // differently because there is no next symbol to reference.
    uint64_t StartOffset = Symbols[Symbols.size() - 1].first;
    uint64_t EndOffset = Sect->Size - 1;
    DEBUG(dbgs() << "Extracting function: " << Symbols[Symbols.size()-1].second
                 << " from [" << StartOffset << ", " << EndOffset << "]\n");
    extractFunction(Symbols[Symbols.size()-1].second,
                    Base + StartOffset, Base + EndOffset);

    // Now extract the relocation information for each function and process it.
    for (unsigned j = 0; j != Sect->NumRelocationTableEntries; ++j) {
      InMemoryStruct<macho::RelocationEntry> RE;
      Obj->ReadRelocationEntry(Sect->RelocationTableOffset, j, RE);
      if (RE->Word0 & macho::RF_Scattered)
        return Error("NOT YET IMPLEMENTED: scattered relocations.");
      // Word0 of the relocation is the offset into the section where the
      // relocation should be applied. We need to translate that into an
      // offset into a function since that's our atom.
      uint32_t Offset = RE->Word0;
      // Look for the function containing the address. This is used for JIT
      // code, so the number of functions in section is almost always going
      // to be very small (usually just one), so until we have use cases
      // where that's not true, just use a trivial linear search.
      unsigned SymbolNum;
      unsigned NumSymbols = Symbols.size();
      assert(NumSymbols > 0 && Symbols[0].first <= Offset &&
             "No symbol containing relocation!");
      for (SymbolNum = 0; SymbolNum < NumSymbols - 1; ++SymbolNum)
        if (Symbols[SymbolNum + 1].first > Offset)
          break;
      // Adjust the offset to be relative to the symbol.
      Offset -= Symbols[SymbolNum].first;
      // Get the name of the symbol containing the relocation.
      StringRef TargetName = SymbolNames[SymbolNum];

      bool isExtern = (RE->Word1 >> 27) & 1;
      // Figure out the source symbol of the relocation. If isExtern is true,
      // this relocation references the symbol table, otherwise it references
      // a section in the same object, numbered from 1 through NumSections
      // (SectionBases is [0, NumSections-1]).
      if (!isExtern)
        return Error("Internal relocations not supported.");
      uint32_t SourceNum = RE->Word1 & 0xffffff; // 24-bit value
      StringRef SourceName = SymbolNames[SourceNum];

      // FIXME: Get the relocation addend from the target address.

      // Now store the relocation information. Associate it with the source
      // symbol.
      Relocations[SourceName].push_back(RelocationEntry(TargetName,
                                                        Offset,
                                                        RE->Word1,
                                                        0 /*Addend*/));
      DEBUG(dbgs() << "Relocation at '" << TargetName << "' + " << Offset
                   << " from '" << SourceName << "(Word1: "
                   << format("0x%x", RE->Word1) << ")\n");
    }
  }
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

// Resolve the relocations for all symbols we currently know about.
void RuntimeDyldImpl::resolveRelocations() {
  // Just iterate over the symbols in our symbol table and assign their
  // addresses.
  StringMap<uint8_t*>::iterator i = SymbolTable.begin();
  StringMap<uint8_t*>::iterator e = SymbolTable.end();
  for (;i != e; ++i)
    reassignSymbolAddress(i->getKey(), i->getValue());
}

// Assign an address to a symbol name and resolve all the relocations
// associated with it.
void RuntimeDyldImpl::reassignSymbolAddress(StringRef Name, uint8_t *Addr) {
  // Assign the address in our symbol table.
  SymbolTable[Name] = Addr;

  RelocationList &Relocs = Relocations[Name];
  for (unsigned i = 0, e = Relocs.size(); i != e; ++i) {
    RelocationEntry &RE = Relocs[i];
    uint8_t *Target = SymbolTable[RE.Target] + RE.Offset;
    bool isPCRel = (RE.Data >> 24) & 1;
    unsigned Type = (RE.Data >> 28) & 0xf;
    unsigned Size = 1 << ((RE.Data >> 25) & 3);

    DEBUG(dbgs() << "Resolving relocation at '" << RE.Target
          << "' + " << RE.Offset << " (" << format("%p", Target) << ")"
          << " from '" << Name << " (" << format("%p", Addr) << ")"
          << "(" << (isPCRel ? "pcrel" : "absolute")
          << ", type: " << Type << ", Size: " << Size << ").\n");

    resolveRelocation(Target, Addr, isPCRel, Type, Size);
    RE.isResolved = true;
  }
}

//===----------------------------------------------------------------------===//
// RuntimeDyld class implementation
RuntimeDyld::RuntimeDyld(RTDyldMemoryManager *MM) {
  Dyld = new RuntimeDyldImpl(MM);
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

void RuntimeDyld::resolveRelocations() {
  Dyld->resolveRelocations();
}

void RuntimeDyld::reassignSymbolAddress(StringRef Name, uint8_t *Addr) {
  Dyld->reassignSymbolAddress(Name, Addr);
}

StringRef RuntimeDyld::getErrorString() {
  return Dyld->getErrorString();
}

} // end namespace llvm
