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

#include "RuntimeDyldMachO.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include "Targets/RuntimeDyldMachOARM.h"
#include "Targets/RuntimeDyldMachOAArch64.h"
#include "Targets/RuntimeDyldMachOI386.h"
#include "Targets/RuntimeDyldMachOX86_64.h"

using namespace llvm;
using namespace llvm::object;

#define DEBUG_TYPE "dyld"

namespace llvm {

int64_t RuntimeDyldMachO::memcpyAddend(const RelocationEntry &RE) const {
  const SectionEntry &Section = Sections[RE.SectionID];
  uint8_t *LocalAddress = Section.Address + RE.Offset;
  unsigned NumBytes = 1 << RE.Size;
  int64_t Addend = 0;
  memcpy(&Addend, LocalAddress, NumBytes);
  return Addend;
}

RelocationValueRef RuntimeDyldMachO::getRelocationValueRef(
    ObjectImage &ObjImg, const relocation_iterator &RI,
    const RelocationEntry &RE, ObjSectionToIDMap &ObjSectionToID,
    const SymbolTableMap &Symbols) {

  const MachOObjectFile &Obj =
      static_cast<const MachOObjectFile &>(*ObjImg.getObjectFile());
  MachO::any_relocation_info RelInfo =
      Obj.getRelocation(RI->getRawDataRefImpl());
  RelocationValueRef Value;

  bool IsExternal = Obj.getPlainRelocationExternal(RelInfo);
  if (IsExternal) {
    symbol_iterator Symbol = RI->getSymbol();
    StringRef TargetName;
    Symbol->getName(TargetName);
    SymbolTableMap::const_iterator SI = Symbols.find(TargetName.data());
    if (SI != Symbols.end()) {
      Value.SectionID = SI->second.first;
      Value.Addend = SI->second.second + RE.Addend;
    } else {
      SI = GlobalSymbolTable.find(TargetName.data());
      if (SI != GlobalSymbolTable.end()) {
        Value.SectionID = SI->second.first;
        Value.Addend = SI->second.second + RE.Addend;
      } else {
        Value.SymbolName = TargetName.data();
        Value.Addend = RE.Addend;
      }
    }
  } else {
    SectionRef Sec = Obj.getRelocationSection(RelInfo);
    bool IsCode = false;
    Sec.isText(IsCode);
    Value.SectionID = findOrEmitSection(ObjImg, Sec, IsCode, ObjSectionToID);
    uint64_t Addr;
    Sec.getAddress(Addr);
    Value.Addend = RE.Addend - Addr;
  }

  return Value;
}

void RuntimeDyldMachO::makeValueAddendPCRel(RelocationValueRef &Value,
                                            ObjectImage &ObjImg,
                                            const relocation_iterator &RI,
                                            unsigned OffsetToNextPC) {
  const MachOObjectFile &Obj =
      static_cast<const MachOObjectFile &>(*ObjImg.getObjectFile());
  MachO::any_relocation_info RelInfo =
      Obj.getRelocation(RI->getRawDataRefImpl());

  bool IsPCRel = Obj.getAnyRelocationPCRel(RelInfo);
  if (IsPCRel) {
    uint64_t RelocAddr = 0;
    RI->getAddress(RelocAddr);
    Value.Addend += RelocAddr + OffsetToNextPC;
  }
}

void RuntimeDyldMachO::dumpRelocationToResolve(const RelocationEntry &RE,
                                               uint64_t Value) const {
  const SectionEntry &Section = Sections[RE.SectionID];
  uint8_t *LocalAddress = Section.Address + RE.Offset;
  uint64_t FinalAddress = Section.LoadAddress + RE.Offset;

  dbgs() << "resolveRelocation Section: " << RE.SectionID
         << " LocalAddress: " << format("%p", LocalAddress)
         << " FinalAddress: " << format("%p", FinalAddress)
         << " Value: " << format("%p", Value) << " Addend: " << RE.Addend
         << " isPCRel: " << RE.IsPCRel << " MachoType: " << RE.RelType
         << " Size: " << (1 << RE.Size) << "\n";
}

bool RuntimeDyldMachO::writeBytesUnaligned(uint8_t *Addr, uint64_t Value,
                                           unsigned Size) {
  for (unsigned i = 0; i < Size; ++i) {
    *Addr++ = (uint8_t)Value;
    Value >>= 8;
  }

  return false;
}

bool
RuntimeDyldMachO::isCompatibleFormat(const ObjectBuffer *InputBuffer) const {
  if (InputBuffer->getBufferSize() < 4)
    return false;
  StringRef Magic(InputBuffer->getBufferStart(), 4);
  if (Magic == "\xFE\xED\xFA\xCE")
    return true;
  if (Magic == "\xCE\xFA\xED\xFE")
    return true;
  if (Magic == "\xFE\xED\xFA\xCF")
    return true;
  if (Magic == "\xCF\xFA\xED\xFE")
    return true;
  return false;
}

bool RuntimeDyldMachO::isCompatibleFile(const object::ObjectFile *Obj) const {
  return Obj->isMachO();
}

static unsigned char *processFDE(unsigned char *P, intptr_t DeltaForText,
                                 intptr_t DeltaForEH) {
  DEBUG(dbgs() << "Processing FDE: Delta for text: " << DeltaForText
               << ", Delta for EH: " << DeltaForEH << "\n");
  uint32_t Length = *((uint32_t *)P);
  P += 4;
  unsigned char *Ret = P + Length;
  uint32_t Offset = *((uint32_t *)P);
  if (Offset == 0) // is a CIE
    return Ret;

  P += 4;
  intptr_t FDELocation = *((intptr_t *)P);
  intptr_t NewLocation = FDELocation - DeltaForText;
  *((intptr_t *)P) = NewLocation;
  P += sizeof(intptr_t);

  // Skip the FDE address range
  P += sizeof(intptr_t);

  uint8_t Augmentationsize = *P;
  P += 1;
  if (Augmentationsize != 0) {
    intptr_t LSDA = *((intptr_t *)P);
    intptr_t NewLSDA = LSDA - DeltaForEH;
    *((intptr_t *)P) = NewLSDA;
  }

  return Ret;
}

static intptr_t computeDelta(SectionEntry *A, SectionEntry *B) {
  intptr_t ObjDistance = A->ObjAddress - B->ObjAddress;
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
    SectionEntry *ExceptTab = nullptr;
    if (SectionInfo.ExceptTabSID != RTDYLD_INVALID_SECTION_ID)
      ExceptTab = &Sections[SectionInfo.ExceptTabSID];

    intptr_t DeltaForText = computeDelta(Text, EHFrame);
    intptr_t DeltaForEH = 0;
    if (ExceptTab)
      DeltaForEH = computeDelta(ExceptTab, EHFrame);

    unsigned char *P = EHFrame->Address;
    unsigned char *End = P + EHFrame->Size;
    do {
      P = processFDE(P, DeltaForText, DeltaForEH);
    } while (P != End);

    MemMgr->registerEHFrames(EHFrame->Address, EHFrame->LoadAddress,
                             EHFrame->Size);
  }
  UnregisteredEHFrameSections.clear();
}

std::unique_ptr<RuntimeDyldMachO>
llvm::RuntimeDyldMachO::create(Triple::ArchType Arch, RTDyldMemoryManager *MM) {
  switch (Arch) {
  default:
    llvm_unreachable("Unsupported target for RuntimeDyldMachO.");
    break;
  case Triple::arm: return make_unique<RuntimeDyldMachOARM>(MM);
  case Triple::aarch64: return make_unique<RuntimeDyldMachOAArch64>(MM);
  case Triple::x86: return make_unique<RuntimeDyldMachOI386>(MM);
  case Triple::x86_64: return make_unique<RuntimeDyldMachOX86_64>(MM);
  }
}

} // end namespace llvm
