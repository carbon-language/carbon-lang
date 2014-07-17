//===----- RuntimeDyldMachOARM.h ---- MachO/ARM specific code. ----*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_RUNTIMEDYLDMACHOARM_H
#define LLVM_RUNTIMEDYLDMACHOARM_H

#include "../RuntimeDyldMachO.h"

#define DEBUG_TYPE "dyld"

namespace llvm {

class RuntimeDyldMachOARM
    : public RuntimeDyldMachOCRTPBase<RuntimeDyldMachOARM> {
public:
  RuntimeDyldMachOARM(RTDyldMemoryManager *MM) : RuntimeDyldMachOCRTPBase(MM) {}

  unsigned getMaxStubSize() override { return 8; }

  unsigned getStubAlignment() override { return 4; }

  relocation_iterator
  processRelocationRef(unsigned SectionID, relocation_iterator RelI,
                       ObjectImage &ObjImg, ObjSectionToIDMap &ObjSectionToID,
                       const SymbolTableMap &Symbols, StubMap &Stubs) override {
    const MachOObjectFile &Obj =
        static_cast<const MachOObjectFile &>(*ObjImg.getObjectFile());
    MachO::any_relocation_info RelInfo =
        Obj.getRelocation(RelI->getRawDataRefImpl());

    if (Obj.isRelocationScattered(RelInfo))
      return ++++RelI;

    RelocationEntry RE(getBasicRelocationEntry(SectionID, ObjImg, RelI));
    RelocationValueRef Value(
        getRelocationValueRef(ObjImg, RelI, RE, ObjSectionToID, Symbols));

    bool IsExtern = Obj.getPlainRelocationExternal(RelInfo);
    if (!IsExtern && RE.IsPCRel)
      makeValueAddendPCRel(Value, ObjImg, RelI);

    if ((RE.RelType & 0xf) == MachO::ARM_RELOC_BR24)
      processBranchRelocation(RE, Value, Stubs);
    else {
      RE.Addend = Value.Addend;
      if (Value.SymbolName)
        addRelocationForSymbol(RE, Value.SymbolName);
      else
        addRelocationForSection(RE, Value.SectionID);
    }

    return ++RelI;
  }

  void resolveRelocation(const RelocationEntry &RE, uint64_t Value) {
    DEBUG(dumpRelocationToResolve(RE, Value));
    const SectionEntry &Section = Sections[RE.SectionID];
    uint8_t *LocalAddress = Section.Address + RE.Offset;

    // If the relocation is PC-relative, the value to be encoded is the
    // pointer difference.
    if (RE.IsPCRel) {
      uint64_t FinalAddress = Section.LoadAddress + RE.Offset;
      Value -= FinalAddress;
      // ARM PCRel relocations have an effective-PC offset of two instructions
      // (four bytes in Thumb mode, 8 bytes in ARM mode).
      // FIXME: For now, assume ARM mode.
      Value -= 8;
    }

    switch (RE.RelType) {
    default:
      llvm_unreachable("Invalid relocation type!");
    case MachO::ARM_RELOC_VANILLA:
      writeBytesUnaligned(LocalAddress, Value, 1 << RE.Size);
      break;
    case MachO::ARM_RELOC_BR24: {
      // Mask the value into the target address. We know instructions are
      // 32-bit aligned, so we can do it all at once.
      uint32_t *p = (uint32_t *)LocalAddress;
      // The low two bits of the value are not encoded.
      Value >>= 2;
      // Mask the value to 24 bits.
      uint64_t FinalValue = Value & 0xffffff;
      // Check for overflow.
      if (Value != FinalValue) {
        Error("ARM BR24 relocation out of range.");
        return;
      }
      // FIXME: If the destination is a Thumb function (and the instruction
      // is a non-predicated BL instruction), we need to change it to a BLX
      // instruction instead.

      // Insert the value into the instruction.
      *p = (*p & ~0xffffff) | FinalValue;
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
      Error("Relocation type not implemented yet!");
      return;
    }
  }

  void finalizeSection(ObjectImage &ObjImg, unsigned SectionID,
                       const SectionRef &Section) {}

private:
  void processBranchRelocation(const RelocationEntry &RE,
                               const RelocationValueRef &Value,
                               StubMap &Stubs) {
    // This is an ARM branch relocation, need to use a stub function.
    // Look up for existing stub.
    SectionEntry &Section = Sections[RE.SectionID];
    RuntimeDyldMachO::StubMap::const_iterator i = Stubs.find(Value);
    uint8_t *Addr;
    if (i != Stubs.end()) {
      Addr = Section.Address + i->second;
    } else {
      // Create a new stub function.
      Stubs[Value] = Section.StubOffset;
      uint8_t *StubTargetAddr =
          createStubFunction(Section.Address + Section.StubOffset);
      RelocationEntry StubRE(RE.SectionID, StubTargetAddr - Section.Address,
                             MachO::GENERIC_RELOC_VANILLA, Value.Addend);
      if (Value.SymbolName)
        addRelocationForSymbol(StubRE, Value.SymbolName);
      else
        addRelocationForSection(StubRE, Value.SectionID);
      Addr = Section.Address + Section.StubOffset;
      Section.StubOffset += getMaxStubSize();
    }
    RelocationEntry TargetRE(Value.SectionID, RE.Offset, RE.RelType, 0,
                             RE.IsPCRel, RE.Size);
    resolveRelocation(TargetRE, (uint64_t)Addr);
  }
};
}

#undef DEBUG_TYPE

#endif // LLVM_RUNTIMEDYLDMACHOARM_H
