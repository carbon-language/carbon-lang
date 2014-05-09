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

#include "ObjectImageCommon.h"
#include "RuntimeDyldImpl.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/Object/MachO.h"
#include "llvm/Support/Format.h"

using namespace llvm;
using namespace llvm::object;

namespace llvm {
class RuntimeDyldMachO : public RuntimeDyldImpl {
private:

  /// Write the least significant 'Size' bytes in 'Value' out at the address
  /// pointed to by Addr. Check for overflow.
  bool applyRelocationValue(uint8_t *Addr, uint64_t Value, unsigned Size) {
    for (unsigned i = 0; i < Size; ++i) {
      *Addr++ = (uint8_t)Value;
      Value >>= 8;
    }

    if (Value) // Catch overflow
      return Error("Relocation out of range.");

    return false;
  }

  bool resolveI386Relocation(const RelocationEntry &RE, uint64_t Value);
  bool resolveX86_64Relocation(const RelocationEntry &RE, uint64_t Value);
  bool resolveARMRelocation(const RelocationEntry &RE, uint64_t Value);
  bool resolveARM64Relocation(const RelocationEntry &RE, uint64_t Value);

  unsigned getMaxStubSize() override {
    if (Arch == Triple::arm || Arch == Triple::thumb)
      return 8; // 32-bit instruction and 32-bit address
    else if (Arch == Triple::x86_64)
      return 8; // GOT entry
    else
      return 0;
  }

  unsigned getStubAlignment() override { return 1; }

  struct EHFrameRelatedSections {
    EHFrameRelatedSections()
        : EHFrameSID(RTDYLD_INVALID_SECTION_ID),
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
  relocation_iterator
  processRelocationRef(unsigned SectionID, relocation_iterator RelI,
                       ObjectImage &Obj, ObjSectionToIDMap &ObjSectionToID,
                       const SymbolTableMap &Symbols, StubMap &Stubs) override;
  bool isCompatibleFormat(const ObjectBuffer *Buffer) const override;
  bool isCompatibleFile(const object::ObjectFile *Obj) const override;
  void registerEHFrames() override;
  void finalizeLoad(ObjSectionToIDMap &SectionMap) override;

  static ObjectImage *createObjectImage(ObjectBuffer *InputBuffer) {
    return new ObjectImageCommon(InputBuffer);
  }

  static ObjectImage *
  createObjectImageFromFile(std::unique_ptr<object::ObjectFile> InputObject) {
    return new ObjectImageCommon(std::move(InputObject));
  }
};

} // end namespace llvm

#endif
