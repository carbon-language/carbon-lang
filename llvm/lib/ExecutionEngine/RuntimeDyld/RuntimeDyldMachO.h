//===-- RuntimeDyldMachO.h - Run-time dynamic linker for MC-JIT ---*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// MachO support for MC-JIT runtime dynamic linker.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_RUNTIME_DYLD_MACHO_H
#define LLVM_RUNTIME_DYLD_MACHO_H

#include "RuntimeDyldImpl.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/Object/MachO.h"
#include "llvm/Support/Format.h"

using namespace llvm;
using namespace llvm::object;


namespace llvm {
class RuntimeDyldMachO : public RuntimeDyldImpl {
  bool resolveI386Relocation(uint8_t *LocalAddress,
                             uint64_t FinalAddress,
                             uint64_t Value,
                             bool isPCRel,
                             unsigned Type,
                             unsigned Size,
                             int64_t Addend);
  bool resolveX86_64Relocation(uint8_t *LocalAddress,
                               uint64_t FinalAddress,
                               uint64_t Value,
                               bool isPCRel,
                               unsigned Type,
                               unsigned Size,
                               int64_t Addend);
  bool resolveARMRelocation(uint8_t *LocalAddress,
                            uint64_t FinalAddress,
                            uint64_t Value,
                            bool isPCRel,
                            unsigned Type,
                            unsigned Size,
                            int64_t Addend);

  void resolveRelocation(const SectionEntry &Section,
                         uint64_t Offset,
                         uint64_t Value,
                         uint32_t Type,
                         int64_t Addend,
                         bool isPCRel,
                         unsigned Size);

  unsigned getMaxStubSize() override {
    if (Arch == Triple::arm || Arch == Triple::thumb)
      return 8; // 32-bit instruction and 32-bit address
    else if (Arch == Triple::x86_64)
      return 8; // GOT entry
    else
      return 0;
  }

  unsigned getStubAlignment() override {
    return 1;
  }

  struct EHFrameRelatedSections {
    EHFrameRelatedSections() : EHFrameSID(RTDYLD_INVALID_SECTION_ID),
                               TextSID(RTDYLD_INVALID_SECTION_ID),
                               ExceptTabSID(RTDYLD_INVALID_SECTION_ID) {}
    EHFrameRelatedSections(SID EH, SID T, SID Ex)
      : EHFrameSID(EH), TextSID(T), ExceptTabSID(Ex) {}
    SID EHFrameSID;
    SID TextSID;
    SID ExceptTabSID;
  };

  // When a module is loaded we save the SectionID of the EH frame section
  // in a table until we receive a request to register all unregistered
  // EH frame sections with the memory manager.
  SmallVector<EHFrameRelatedSections, 2> UnregisteredEHFrameSections;
public:
  RuntimeDyldMachO(RTDyldMemoryManager *mm) : RuntimeDyldImpl(mm) {}

  void resolveRelocation(const RelocationEntry &RE, uint64_t Value) override;
  void processRelocationRef(unsigned SectionID, RelocationRef RelI,
                            ObjectImage &Obj, ObjSectionToIDMap &ObjSectionToID,
                            const SymbolTableMap &Symbols,
                            StubMap &Stubs) override;
  bool isCompatibleFormat(const ObjectBuffer *Buffer) const override;
  bool isCompatibleFile(const object::ObjectFile *Obj) const override;
  void registerEHFrames() override;
  void finalizeLoad(ObjSectionToIDMap &SectionMap) override;
};

} // end namespace llvm

#endif
