//===-- RuntimeDyldCOFFX86_64.cpp - COFF/X86_64 specific code ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// COFF x86_x64 support for MC-JIT runtime dynamic linker.
//
//===----------------------------------------------------------------------===//

#include "RuntimeDyldCOFFX86_64.h"

#define DEBUG_TYPE "dyld"

namespace llvm {

void RuntimeDyldCOFFX86_64::registerEHFrames() {
  if (!MemMgr)
    return;
  for (auto const &EHFrameSID : UnregisteredEHFrameSections) {
    uint8_t *EHFrameAddr = Sections[EHFrameSID].Address;
    uint64_t EHFrameLoadAddr = Sections[EHFrameSID].LoadAddress;
    size_t EHFrameSize = Sections[EHFrameSID].Size;
    MemMgr->registerEHFrames(EHFrameAddr, EHFrameLoadAddr, EHFrameSize);
    RegisteredEHFrameSections.push_back(EHFrameSID);
  }
  UnregisteredEHFrameSections.clear();
}

void RuntimeDyldCOFFX86_64::deregisterEHFrames() {
  // Stub
}

// The target location for the relocation is described by RE.SectionID and
// RE.Offset.  RE.SectionID can be used to find the SectionEntry.  Each
// SectionEntry has three members describing its location.
// SectionEntry::Address is the address at which the section has been loaded
// into memory in the current (host) process.  SectionEntry::LoadAddress is the
// address that the section will have in the target process.
// SectionEntry::ObjAddress is the address of the bits for this section in the
// original emitted object image (also in the current address space).
//
// Relocations will be applied as if the section were loaded at
// SectionEntry::LoadAddress, but they will be applied at an address based
// on SectionEntry::Address.  SectionEntry::ObjAddress will be used to refer to
// Target memory contents if they are required for value calculations.
//
// The Value parameter here is the load address of the symbol for the
// relocation to be applied.  For relocations which refer to symbols in the
// current object Value will be the LoadAddress of the section in which
// the symbol resides (RE.Addend provides additional information about the
// symbol location).  For external symbols, Value will be the address of the
// symbol in the target address space.
void RuntimeDyldCOFFX86_64::resolveRelocation(const RelocationEntry &RE,
                                              uint64_t Value) {
  const SectionEntry &Section = Sections[RE.SectionID];
  uint8_t *Target = Section.Address + RE.Offset;

  switch (RE.RelType) {

  case COFF::IMAGE_REL_AMD64_REL32:
  case COFF::IMAGE_REL_AMD64_REL32_1:
  case COFF::IMAGE_REL_AMD64_REL32_2:
  case COFF::IMAGE_REL_AMD64_REL32_3:
  case COFF::IMAGE_REL_AMD64_REL32_4:
  case COFF::IMAGE_REL_AMD64_REL32_5: {
    uint32_t *TargetAddress = (uint32_t *)Target;
    uint64_t FinalAddress = Section.LoadAddress + RE.Offset;
    // Delta is the distance from the start of the reloc to the end of the
    // instruction with the reloc.
    uint64_t Delta = 4 + (RE.RelType - COFF::IMAGE_REL_AMD64_REL32);
    Value -= FinalAddress + Delta;
    uint64_t Result = Value + RE.Addend;
    assert(((int64_t)Result <= INT32_MAX) && "Relocation overflow");
    assert(((int64_t)Result >= INT32_MIN) && "Relocation underflow");
    *TargetAddress = Result;
    break;
  }

  case COFF::IMAGE_REL_AMD64_ADDR32NB: {
    // Note ADDR32NB requires a well-established notion of
    // image base. This address must be less than or equal
    // to every section's load address, and all sections must be
    // within a 32 bit offset from the base.
    //
    // For now we just set these to zero.
    uint32_t *TargetAddress = (uint32_t *)Target;
    *TargetAddress = 0;
    break;
  }

  case COFF::IMAGE_REL_AMD64_ADDR64: {
    uint64_t *TargetAddress = (uint64_t *)Target;
    *TargetAddress = Value + RE.Addend;
    break;
  }

  default:
    llvm_unreachable("Relocation type not implemented yet!");
    break;
  }
}

relocation_iterator RuntimeDyldCOFFX86_64::processRelocationRef(
    unsigned SectionID, relocation_iterator RelI, const ObjectFile &Obj,
    ObjSectionToIDMap &ObjSectionToID, StubMap &Stubs) {

  // Find the symbol referred to in the relocation, and
  // get its section and offset.
  //
  // Insist for now that all symbols be resolvable within
  // the scope of this object file.
  symbol_iterator Symbol = RelI->getSymbol();
  if (Symbol == Obj.symbol_end())
    report_fatal_error("Unknown symbol in relocation");
  unsigned TargetSectionID = 0;
  uint64_t TargetOffset = UnknownAddressOrSize;
  section_iterator SecI(Obj.section_end());
  Symbol->getSection(SecI);
  if (SecI == Obj.section_end())
    report_fatal_error("Unknown section in relocation");
  bool IsCode = SecI->isText();
  TargetSectionID = findOrEmitSection(Obj, *SecI, IsCode, ObjSectionToID);
  TargetOffset = getSymbolOffset(*Symbol);

  // Determine the Addend used to adjust the relocation value.
  uint64_t RelType;
  Check(RelI->getType(RelType));
  uint64_t Offset;
  Check(RelI->getOffset(Offset));
  uint64_t Addend = 0;
  SectionEntry &Section = Sections[SectionID];
  uintptr_t ObjTarget = Section.ObjAddress + Offset;

  switch (RelType) {

  case COFF::IMAGE_REL_AMD64_REL32:
  case COFF::IMAGE_REL_AMD64_REL32_1:
  case COFF::IMAGE_REL_AMD64_REL32_2:
  case COFF::IMAGE_REL_AMD64_REL32_3:
  case COFF::IMAGE_REL_AMD64_REL32_4:
  case COFF::IMAGE_REL_AMD64_REL32_5:
  case COFF::IMAGE_REL_AMD64_ADDR32NB: {
    uint32_t *Displacement = (uint32_t *)ObjTarget;
    Addend = *Displacement;
    break;
  }

  case COFF::IMAGE_REL_AMD64_ADDR64: {
    uint64_t *Displacement = (uint64_t *)ObjTarget;
    Addend = *Displacement;
    break;
  }

  default:
    break;
  }

  StringRef TargetName;
  Symbol->getName(TargetName);
  DEBUG(dbgs() << "\t\tIn Section " << SectionID << " Offset " << Offset
               << " RelType: " << RelType << " TargetName: " << TargetName
               << " Addend " << Addend << "\n");

  RelocationEntry RE(SectionID, Offset, RelType, TargetOffset + Addend);
  addRelocationForSection(RE, TargetSectionID);

  return ++RelI;
}

void RuntimeDyldCOFFX86_64::finalizeLoad(const ObjectFile &Obj,
                                         ObjSectionToIDMap &SectionMap) {
  // Look for and record the EH frame section IDs.
  for (const auto &SectionPair : SectionMap) {
    const SectionRef &Section = SectionPair.first;
    StringRef Name;
    Check(Section.getName(Name));
    // Note unwind info is split across .pdata and .xdata, so this
    // may not be sufficiently general for all users.
    if (Name == ".xdata") {
      UnregisteredEHFrameSections.push_back(SectionPair.second);
    }
  }
}
}
