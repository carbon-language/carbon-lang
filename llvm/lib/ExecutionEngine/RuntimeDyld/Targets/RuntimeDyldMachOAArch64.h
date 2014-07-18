//===-- RuntimeDyldMachOAArch64.h -- MachO/AArch64 specific code. -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_RUNTIMEDYLDMACHOAARCH64_H
#define LLVM_RUNTIMEDYLDMACHOAARCH64_H

#include "../RuntimeDyldMachO.h"

#define DEBUG_TYPE "dyld"

namespace llvm {

class RuntimeDyldMachOAArch64
    : public RuntimeDyldMachOCRTPBase<RuntimeDyldMachOAArch64> {
public:
  RuntimeDyldMachOAArch64(RTDyldMemoryManager *MM)
      : RuntimeDyldMachOCRTPBase(MM) {}

  unsigned getMaxStubSize() override { return 8; }

  unsigned getStubAlignment() override { return 8; }

  relocation_iterator
  processRelocationRef(unsigned SectionID, relocation_iterator RelI,
                       ObjectImage &ObjImg, ObjSectionToIDMap &ObjSectionToID,
                       const SymbolTableMap &Symbols, StubMap &Stubs) override {
    const MachOObjectFile &Obj =
        static_cast<const MachOObjectFile &>(*ObjImg.getObjectFile());
    MachO::any_relocation_info RelInfo =
        Obj.getRelocation(RelI->getRawDataRefImpl());

    assert(!Obj.isRelocationScattered(RelInfo) && "");

    // ARM64 has an ARM64_RELOC_ADDEND relocation type that carries an explicit
    // addend for the following relocation. If found: (1) store the associated
    // addend, (2) consume the next relocation, and (3) use the stored addend to
    // override the addend.
    bool HasExplicitAddend = false;
    int64_t ExplicitAddend = 0;
    if (Obj.getAnyRelocationType(RelInfo) == MachO::ARM64_RELOC_ADDEND) {
      assert(!Obj.getPlainRelocationExternal(RelInfo));
      assert(!Obj.getAnyRelocationPCRel(RelInfo));
      assert(Obj.getAnyRelocationLength(RelInfo) == 2);
      HasExplicitAddend = true;
      int64_t RawAddend = Obj.getPlainRelocationSymbolNum(RelInfo);
      // Sign-extend the 24-bit to 64-bit.
      ExplicitAddend = (RawAddend << 40) >> 40;
      ++RelI;
      RelInfo = Obj.getRelocation(RelI->getRawDataRefImpl());
    }

    RelocationEntry RE(getBasicRelocationEntry(SectionID, ObjImg, RelI));
    RelocationValueRef Value(
        getRelocationValueRef(ObjImg, RelI, RE, ObjSectionToID, Symbols));

    if (HasExplicitAddend) {
      RE.Addend = ExplicitAddend;
      Value.Addend = ExplicitAddend;
    }

    bool IsExtern = Obj.getPlainRelocationExternal(RelInfo);
    if (!IsExtern && RE.IsPCRel)
      makeValueAddendPCRel(Value, ObjImg, RelI);

    RE.Addend = Value.Addend;

    if (RE.RelType == MachO::ARM64_RELOC_GOT_LOAD_PAGE21 ||
        RE.RelType == MachO::ARM64_RELOC_GOT_LOAD_PAGEOFF12)
      processGOTRelocation(RE, Value, Stubs);
    else {
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

    switch (RE.RelType) {
    default:
      llvm_unreachable("Invalid relocation type!");
    case MachO::ARM64_RELOC_UNSIGNED: {
      assert(!RE.IsPCRel && "PCRel and ARM64_RELOC_UNSIGNED not supported");
      // Mask in the target value a byte at a time (we don't have an alignment
      // guarantee for the target address, so this is safest).
      if (RE.Size < 2)
        llvm_unreachable("Invalid size for ARM64_RELOC_UNSIGNED");

      writeBytesUnaligned(LocalAddress, Value + RE.Addend, 1 << RE.Size);
      break;
    }
    case MachO::ARM64_RELOC_BRANCH26: {
      assert(RE.IsPCRel && "not PCRel and ARM64_RELOC_BRANCH26 not supported");
      // Mask the value into the target address. We know instructions are
      // 32-bit aligned, so we can do it all at once.
      uint32_t *p = (uint32_t *)LocalAddress;
      // Check if the addend is encoded in the instruction.
      uint32_t EncodedAddend = *p & 0x03FFFFFF;
      if (EncodedAddend != 0) {
        if (RE.Addend == 0)
          llvm_unreachable("branch26 instruction has embedded addend.");
        else
          llvm_unreachable("branch26 instruction has embedded addend and"
                           "ARM64_RELOC_ADDEND.");
      }
      // Check if branch is in range.
      uint64_t FinalAddress = Section.LoadAddress + RE.Offset;
      uint64_t PCRelVal = Value - FinalAddress + RE.Addend;
      assert(isInt<26>(PCRelVal) && "Branch target out of range!");
      // Insert the value into the instruction.
      *p = (*p & 0xFC000000) | ((uint32_t)(PCRelVal >> 2) & 0x03FFFFFF);
      break;
    }
    case MachO::ARM64_RELOC_GOT_LOAD_PAGE21:
    case MachO::ARM64_RELOC_PAGE21: {
      assert(RE.IsPCRel && "not PCRel and ARM64_RELOC_PAGE21 not supported");
      // Mask the value into the target address. We know instructions are
      // 32-bit aligned, so we can do it all at once.
      uint32_t *p = (uint32_t *)LocalAddress;
      // Check if the addend is encoded in the instruction.
      uint32_t EncodedAddend =
          ((*p & 0x60000000) >> 29) | ((*p & 0x01FFFFE0) >> 3);
      if (EncodedAddend != 0) {
        if (RE.Addend == 0)
          llvm_unreachable("adrp instruction has embedded addend.");
        else
          llvm_unreachable("adrp instruction has embedded addend and"
                           "ARM64_RELOC_ADDEND.");
      }
      // Adjust for PC-relative relocation and offset.
      uint64_t FinalAddress = Section.LoadAddress + RE.Offset;
      uint64_t PCRelVal =
          ((Value + RE.Addend) & (-4096)) - (FinalAddress & (-4096));
      // Check that the value fits into 21 bits (+ 12 lower bits).
      assert(isInt<33>(PCRelVal) && "Invalid page reloc value!");
      // Insert the value into the instruction.
      uint32_t ImmLoValue = (uint32_t)(PCRelVal << 17) & 0x60000000;
      uint32_t ImmHiValue = (uint32_t)(PCRelVal >> 9) & 0x00FFFFE0;
      *p = (*p & 0x9F00001F) | ImmHiValue | ImmLoValue;
      break;
    }
    case MachO::ARM64_RELOC_GOT_LOAD_PAGEOFF12:
    case MachO::ARM64_RELOC_PAGEOFF12: {
      assert(!RE.IsPCRel && "PCRel and ARM64_RELOC_PAGEOFF21 not supported");
      // Mask the value into the target address. We know instructions are
      // 32-bit aligned, so we can do it all at once.
      uint32_t *p = (uint32_t *)LocalAddress;
      // Check if the addend is encoded in the instruction.
      uint32_t EncodedAddend = *p & 0x003FFC00;
      if (EncodedAddend != 0) {
        if (RE.Addend == 0)
          llvm_unreachable("adrp instruction has embedded addend.");
        else
          llvm_unreachable("adrp instruction has embedded addend and"
                           "ARM64_RELOC_ADDEND.");
      }
      // Add the offset from the symbol.
      Value += RE.Addend;
      // Mask out the page address and only use the lower 12 bits.
      Value &= 0xFFF;
      // Check which instruction we are updating to obtain the implicit shift
      // factor from LDR/STR instructions.
      if (*p & 0x08000000) {
        uint32_t ImplicitShift = ((*p >> 30) & 0x3);
        switch (ImplicitShift) {
        case 0:
          // Check if this a vector op.
          if ((*p & 0x04800000) == 0x04800000) {
            ImplicitShift = 4;
            assert(((Value & 0xF) == 0) &&
                   "128-bit LDR/STR not 16-byte aligned.");
          }
          break;
        case 1:
          assert(((Value & 0x1) == 0) && "16-bit LDR/STR not 2-byte aligned.");
        case 2:
          assert(((Value & 0x3) == 0) && "32-bit LDR/STR not 4-byte aligned.");
        case 3:
          assert(((Value & 0x7) == 0) && "64-bit LDR/STR not 8-byte aligned.");
        }
        // Compensate for implicit shift.
        Value >>= ImplicitShift;
      }
      // Insert the value into the instruction.
      *p = (*p & 0xFFC003FF) | ((uint32_t)(Value << 10) & 0x003FFC00);
      break;
    }
    case MachO::ARM64_RELOC_SUBTRACTOR:
    case MachO::ARM64_RELOC_POINTER_TO_GOT:
    case MachO::ARM64_RELOC_TLVP_LOAD_PAGE21:
    case MachO::ARM64_RELOC_TLVP_LOAD_PAGEOFF12:
      llvm_unreachable("Relocation type not implemented yet!");
    case MachO::ARM64_RELOC_ADDEND:
      llvm_unreachable("ARM64_RELOC_ADDEND should have been handeled by "
                       "processRelocationRef!");
    }
  }

  void finalizeSection(ObjectImage &ObjImg, unsigned SectionID,
                       const SectionRef &Section) {}

private:
  void processGOTRelocation(const RelocationEntry &RE,
                            RelocationValueRef &Value, StubMap &Stubs) {
    assert(RE.Size == 2);
    SectionEntry &Section = Sections[RE.SectionID];
    StubMap::const_iterator i = Stubs.find(Value);
    uint8_t *Addr;
    if (i != Stubs.end())
      Addr = Section.Address + i->second;
    else {
      // FIXME: There must be a better way to do this then to check and fix the
      // alignment every time!!!
      uintptr_t BaseAddress = uintptr_t(Section.Address);
      uintptr_t StubAlignment = getStubAlignment();
      uintptr_t StubAddress =
          (BaseAddress + Section.StubOffset + StubAlignment - 1) &
          -StubAlignment;
      unsigned StubOffset = StubAddress - BaseAddress;
      Stubs[Value] = StubOffset;
      assert(((StubAddress % getStubAlignment()) == 0) &&
             "GOT entry not aligned");
      RelocationEntry GOTRE(RE.SectionID, StubOffset,
                            MachO::ARM64_RELOC_UNSIGNED, Value.Addend,
                            /*IsPCRel=*/false, /*Size=*/3);
      if (Value.SymbolName)
        addRelocationForSymbol(GOTRE, Value.SymbolName);
      else
        addRelocationForSection(GOTRE, Value.SectionID);
      Section.StubOffset = StubOffset + getMaxStubSize();
      Addr = (uint8_t *)StubAddress;
    }
    RelocationEntry TargetRE(RE.SectionID, RE.Offset, RE.RelType, /*Addend=*/0,
                             RE.IsPCRel, RE.Size);
    resolveRelocation(TargetRE, (uint64_t)Addr);
  }
};
}

#undef DEBUG_TYPE

#endif // LLVM_RUNTIMEDYLDMACHOAARCH64_H
