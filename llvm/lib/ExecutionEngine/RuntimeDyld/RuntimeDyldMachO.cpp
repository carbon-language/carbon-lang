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
#include "RuntimeDyldMachO.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
using namespace llvm;
using namespace llvm::object;

namespace llvm {

static unsigned char *processFDE(unsigned char *P, intptr_t DeltaForText, intptr_t DeltaForEH) {
  uint32_t Length = *((uint32_t*)P);
  P += 4;
  unsigned char *Ret = P + Length;
  uint32_t Offset = *((uint32_t*)P);
  if (Offset == 0) // is a CIE
    return Ret;

  P += 4;
  intptr_t FDELocation = *((intptr_t*)P);
  intptr_t NewLocation = FDELocation - DeltaForText;
  *((intptr_t*)P) = NewLocation;
  P += sizeof(intptr_t);

  // Skip the FDE address range
  P += sizeof(intptr_t);

  uint8_t Augmentationsize = *P;
  P += 1;
  if (Augmentationsize != 0) {
    intptr_t LSDA = *((intptr_t*)P);
    intptr_t NewLSDA = LSDA - DeltaForEH;
    *((intptr_t*)P) = NewLSDA;
  }

  return Ret;
}

static intptr_t computeDelta(SectionEntry *A, SectionEntry *B) {
  intptr_t ObjDistance = A->ObjAddress  - B->ObjAddress;
  intptr_t MemDistance = A->LoadAddress - B->LoadAddress;
  return ObjDistance - MemDistance;
}

void RuntimeDyldMachO::registerEHFrames() {

  if (!MemMgr)
    return;
  for (int i = 0, e = UnregisteredEHFrameSections.size(); i != e; ++i) {
    EHFrameRelatedSections &SectionInfo = UnregisteredEHFrameSections[i];
    if (SectionInfo.EHFrameSID == RTDYLD_INVALID_SECTION_ID ||
        SectionInfo.TextSID == RTDYLD_INVALID_SECTION_ID)
      continue;
    SectionEntry *Text = &Sections[SectionInfo.TextSID];
    SectionEntry *EHFrame = &Sections[SectionInfo.EHFrameSID];
    SectionEntry *ExceptTab = NULL;
    if (SectionInfo.ExceptTabSID != RTDYLD_INVALID_SECTION_ID)
      ExceptTab = &Sections[SectionInfo.ExceptTabSID];

    intptr_t DeltaForText = computeDelta(Text, EHFrame);
    intptr_t DeltaForEH = 0;
    if (ExceptTab)
      DeltaForEH = computeDelta(ExceptTab, EHFrame);

    unsigned char *P = EHFrame->Address;
    unsigned char *End = P + EHFrame->Size;
    do  {
      P = processFDE(P, DeltaForText, DeltaForEH);
    } while(P != End);

    MemMgr->registerEHFrames(EHFrame->Address,
                             EHFrame->LoadAddress,
                             EHFrame->Size);
  }
  UnregisteredEHFrameSections.clear();
}

void RuntimeDyldMachO::finalizeLoad(ObjSectionToIDMap &SectionMap) {
  unsigned EHFrameSID = RTDYLD_INVALID_SECTION_ID;
  unsigned TextSID = RTDYLD_INVALID_SECTION_ID;
  unsigned ExceptTabSID = RTDYLD_INVALID_SECTION_ID;
  ObjSectionToIDMap::iterator i, e;
  for (i = SectionMap.begin(), e = SectionMap.end(); i != e; ++i) {
    const SectionRef &Section = i->first;
    StringRef Name;
    Section.getName(Name);
    if (Name == "__eh_frame")
      EHFrameSID = i->second;
    else if (Name == "__text")
      TextSID = i->second;
    else if (Name == "__gcc_except_tab")
      ExceptTabSID = i->second;
  }
  UnregisteredEHFrameSections.push_back(EHFrameRelatedSections(EHFrameSID,
                                                               TextSID,
                                                               ExceptTabSID));
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
void RuntimeDyldMachO::resolveRelocation(const RelocationEntry &RE,
                                         uint64_t Value) {
  const SectionEntry &Section = Sections[RE.SectionID];
  return resolveRelocation(Section, RE.Offset, Value, RE.RelType, RE.Addend,
                           RE.IsPCRel, RE.Size);
}

void RuntimeDyldMachO::resolveRelocation(const SectionEntry &Section,
                                         uint64_t Offset,
                                         uint64_t Value,
                                         uint32_t Type,
                                         int64_t Addend,
                                         bool isPCRel,
                                         unsigned LogSize) {
  uint8_t *LocalAddress = Section.Address + Offset;
  uint64_t FinalAddress = Section.LoadAddress + Offset;
  unsigned MachoType = Type;
  unsigned Size = 1 << LogSize;

  DEBUG(dbgs() << "resolveRelocation LocalAddress: "
        << format("%p", LocalAddress)
        << " FinalAddress: " << format("%p", FinalAddress)
        << " Value: " << format("%p", Value)
        << " Addend: " << Addend
        << " isPCRel: " << isPCRel
        << " MachoType: " << MachoType
        << " Size: " << Size
        << "\n");

  // This just dispatches to the proper target specific routine.
  switch (Arch) {
  default: llvm_unreachable("Unsupported CPU type!");
  case Triple::x86_64:
    resolveX86_64Relocation(LocalAddress,
                            FinalAddress,
                            (uintptr_t)Value,
                            isPCRel,
                            MachoType,
                            Size,
                            Addend);
    break;
  case Triple::x86:
    resolveI386Relocation(LocalAddress,
                          FinalAddress,
                          (uintptr_t)Value,
                          isPCRel,
                          MachoType,
                          Size,
                          Addend);
    break;
  case Triple::arm:    // Fall through.
  case Triple::thumb:
    resolveARMRelocation(LocalAddress,
                         FinalAddress,
                         (uintptr_t)Value,
                         isPCRel,
                         MachoType,
                         Size,
                         Addend);
    break;
  }
}

bool RuntimeDyldMachO::resolveI386Relocation(uint8_t *LocalAddress,
                                             uint64_t FinalAddress,
                                             uint64_t Value,
                                             bool isPCRel,
                                             unsigned Type,
                                             unsigned Size,
                                             int64_t Addend) {
  if (isPCRel)
    Value -= FinalAddress + 4; // see resolveX86_64Relocation

  switch (Type) {
  default:
    llvm_unreachable("Invalid relocation type!");
  case MachO::GENERIC_RELOC_VANILLA: {
    uint8_t *p = LocalAddress;
    uint64_t ValueToWrite = Value + Addend;
    for (unsigned i = 0; i < Size; ++i) {
      *p++ = (uint8_t)(ValueToWrite & 0xff);
      ValueToWrite >>= 8;
    }
    return false;
  }
  case MachO::GENERIC_RELOC_SECTDIFF:
  case MachO::GENERIC_RELOC_LOCAL_SECTDIFF:
  case MachO::GENERIC_RELOC_PB_LA_PTR:
    return Error("Relocation type not implemented yet!");
  }
}

bool RuntimeDyldMachO::resolveX86_64Relocation(uint8_t *LocalAddress,
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
  case MachO::X86_64_RELOC_SIGNED_1:
  case MachO::X86_64_RELOC_SIGNED_2:
  case MachO::X86_64_RELOC_SIGNED_4:
  case MachO::X86_64_RELOC_SIGNED:
  case MachO::X86_64_RELOC_UNSIGNED:
  case MachO::X86_64_RELOC_BRANCH: {
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
  case MachO::X86_64_RELOC_GOT_LOAD:
  case MachO::X86_64_RELOC_GOT:
  case MachO::X86_64_RELOC_SUBTRACTOR:
  case MachO::X86_64_RELOC_TLV:
    return Error("Relocation type not implemented yet!");
  }
}

bool RuntimeDyldMachO::resolveARMRelocation(uint8_t *LocalAddress,
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
  case MachO::ARM_RELOC_VANILLA: {
    // Mask in the target value a byte at a time (we don't have an alignment
    // guarantee for the target address, so this is safest).
    uint8_t *p = (uint8_t*)LocalAddress;
    for (unsigned i = 0; i < Size; ++i) {
      *p++ = (uint8_t)Value;
      Value >>= 8;
    }
    break;
  }
  case MachO::ARM_RELOC_BR24: {
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
  case MachO::ARM_THUMB_RELOC_BR22:
  case MachO::ARM_THUMB_32BIT_BRANCH:
  case MachO::ARM_RELOC_HALF:
  case MachO::ARM_RELOC_HALF_SECTDIFF:
  case MachO::ARM_RELOC_PAIR:
  case MachO::ARM_RELOC_SECTDIFF:
  case MachO::ARM_RELOC_LOCAL_SECTDIFF:
  case MachO::ARM_RELOC_PB_LA_PTR:
    return Error("Relocation type not implemented yet!");
  }
  return false;
}

void RuntimeDyldMachO::processRelocationRef(unsigned SectionID,
                                            RelocationRef RelI,
                                            ObjectImage &Obj,
                                            ObjSectionToIDMap &ObjSectionToID,
                                            const SymbolTableMap &Symbols,
                                            StubMap &Stubs) {
  const ObjectFile *OF = Obj.getObjectFile();
  const MachOObjectFile *MachO = static_cast<const MachOObjectFile*>(OF);
  MachO::any_relocation_info RE= MachO->getRelocation(RelI.getRawDataRefImpl());

  uint32_t RelType = MachO->getAnyRelocationType(RE);

  // FIXME: Properly handle scattered relocations.
  //        For now, optimistically skip these: they can often be ignored, as
  //        the static linker will already have applied the relocation, and it
  //        only needs to be reapplied if symbols move relative to one another.
  //        Note: This will fail horribly where the relocations *do* need to be
  //        applied, but that was already the case.
  if (MachO->isRelocationScattered(RE))
    return;

  RelocationValueRef Value;
  SectionEntry &Section = Sections[SectionID];

  bool isExtern = MachO->getPlainRelocationExternal(RE);
  bool IsPCRel = MachO->getAnyRelocationPCRel(RE);
  unsigned Size = MachO->getAnyRelocationLength(RE);
  uint64_t Offset;
  RelI.getOffset(Offset);
  uint8_t *LocalAddress = Section.Address + Offset;
  unsigned NumBytes = 1 << Size;
  uint64_t Addend = 0;
  memcpy(&Addend, LocalAddress, NumBytes);

  if (isExtern) {
    // Obtain the symbol name which is referenced in the relocation
    symbol_iterator Symbol = RelI.getSymbol();
    StringRef TargetName;
    Symbol->getName(TargetName);
    // First search for the symbol in the local symbol table
    SymbolTableMap::const_iterator lsi = Symbols.find(TargetName.data());
    if (lsi != Symbols.end()) {
      Value.SectionID = lsi->second.first;
      Value.Addend = lsi->second.second + Addend;
    } else {
      // Search for the symbol in the global symbol table
      SymbolTableMap::const_iterator gsi = GlobalSymbolTable.find(TargetName.data());
      if (gsi != GlobalSymbolTable.end()) {
        Value.SectionID = gsi->second.first;
        Value.Addend = gsi->second.second + Addend;
      } else {
        Value.SymbolName = TargetName.data();
        Value.Addend = Addend;
      }
    }
  } else {
    SectionRef Sec = MachO->getRelocationSection(RE);
    bool IsCode = false;
    Sec.isText(IsCode);
    Value.SectionID = findOrEmitSection(Obj, Sec, IsCode, ObjSectionToID);
    uint64_t Addr;
    Sec.getAddress(Addr);
    Value.Addend = Addend - Addr;
    if (IsPCRel)
      Value.Addend += Offset + NumBytes;
  }

  if (Arch == Triple::x86_64 && (RelType == MachO::X86_64_RELOC_GOT ||
                                 RelType == MachO::X86_64_RELOC_GOT_LOAD)) {
    assert(IsPCRel);
    assert(Size == 2);
    StubMap::const_iterator i = Stubs.find(Value);
    uint8_t *Addr;
    if (i != Stubs.end()) {
      Addr = Section.Address + i->second;
    } else {
      Stubs[Value] = Section.StubOffset;
      uint8_t *GOTEntry = Section.Address + Section.StubOffset;
      RelocationEntry RE(SectionID, Section.StubOffset,
                         MachO::X86_64_RELOC_UNSIGNED, 0, false, 3);
      if (Value.SymbolName)
        addRelocationForSymbol(RE, Value.SymbolName);
      else
        addRelocationForSection(RE, Value.SectionID);
      Section.StubOffset += 8;
      Addr = GOTEntry;
    }
    resolveRelocation(Section, Offset, (uint64_t)Addr,
                      MachO::X86_64_RELOC_UNSIGNED, Value.Addend, true, 2);
  } else if (Arch == Triple::arm &&
             (RelType & 0xf) == MachO::ARM_RELOC_BR24) {
    // This is an ARM branch relocation, need to use a stub function.

    //  Look up for existing stub.
    StubMap::const_iterator i = Stubs.find(Value);
    if (i != Stubs.end())
      resolveRelocation(Section, Offset,
                        (uint64_t)Section.Address + i->second,
                        RelType, 0, IsPCRel, Size);
    else {
      // Create a new stub function.
      Stubs[Value] = Section.StubOffset;
      uint8_t *StubTargetAddr = createStubFunction(Section.Address +
                                                   Section.StubOffset);
      RelocationEntry RE(SectionID, StubTargetAddr - Section.Address,
                         MachO::GENERIC_RELOC_VANILLA, Value.Addend);
      if (Value.SymbolName)
        addRelocationForSymbol(RE, Value.SymbolName);
      else
        addRelocationForSection(RE, Value.SectionID);
      resolveRelocation(Section, Offset,
                        (uint64_t)Section.Address + Section.StubOffset,
                        RelType, 0, IsPCRel, Size);
      Section.StubOffset += getMaxStubSize();
    }
  } else {
    RelocationEntry RE(SectionID, Offset, RelType, Value.Addend,
                       IsPCRel, Size);
    if (Value.SymbolName)
      addRelocationForSymbol(RE, Value.SymbolName);
    else
      addRelocationForSection(RE, Value.SectionID);
  }
}


bool RuntimeDyldMachO::isCompatibleFormat(
        const ObjectBuffer *InputBuffer) const {
  if (InputBuffer->getBufferSize() < 4)
    return false;
  StringRef Magic(InputBuffer->getBufferStart(), 4);
  if (Magic == "\xFE\xED\xFA\xCE") return true;
  if (Magic == "\xCE\xFA\xED\xFE") return true;
  if (Magic == "\xFE\xED\xFA\xCF") return true;
  if (Magic == "\xCF\xFA\xED\xFE") return true;
  return false;
}

bool RuntimeDyldMachO::isCompatibleFile(
        const object::ObjectFile *Obj) const {
  return Obj->isMachO();
}

} // end namespace llvm
