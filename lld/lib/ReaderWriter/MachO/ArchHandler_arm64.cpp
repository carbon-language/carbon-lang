//===- lib/FileFormat/MachO/ArchHandler_arm64.cpp -------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ArchHandler.h"
#include "Atoms.h"
#include "MachONormalizedFileBinaryUtils.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"

using namespace llvm::MachO;
using namespace lld::mach_o::normalized;

namespace lld {
namespace mach_o {

class ArchHandler_arm64 : public ArchHandler {
public:
  ArchHandler_arm64();
  virtual ~ArchHandler_arm64();

  const Registry::KindStrings *kindStrings() override { return _sKindStrings; }

  Reference::KindArch kindArch() override {
    return Reference::KindArch::AArch64;
  }

  /// Used by GOTPass to locate GOT References
  bool isGOTAccess(const Reference &ref, bool &canBypassGOT) override {
    if (ref.kindNamespace() != Reference::KindNamespace::mach_o)
      return false;
    assert(ref.kindArch() == Reference::KindArch::AArch64);
    switch (ref.kindValue()) {
    case gotPage21:
    case gotOffset12:
      canBypassGOT = true;
      return true;
    default:
      return false;
    }
  }

  /// Used by GOTPass to update GOT References.
  void updateReferenceToGOT(const Reference *ref, bool targetNowGOT) override {
    // If GOT slot was instanciated, transform:
    //   gotPage21/gotOffset12 -> page21/offset12scale8
    // If GOT slot optimized away, transform:
    //   gotPage21/gotOffset12 -> page21/addOffset12
    assert(ref->kindNamespace() == Reference::KindNamespace::mach_o);
    assert(ref->kindArch() == Reference::KindArch::AArch64);
    switch (ref->kindValue()) {
    case gotPage21:
      const_cast<Reference *>(ref)->setKindValue(page21);
      break;
    case gotOffset12:
      const_cast<Reference *>(ref)->setKindValue(targetNowGOT ?
                                                 offset12scale8 : addOffset12);
      break;
    default:
      llvm_unreachable("Not a GOT reference");
    }
  }

  const StubInfo &stubInfo() override { return _sStubInfo; }

  bool isCallSite(const Reference &) override;
  bool isNonCallBranch(const Reference &) override {
    return false;
  }

  bool isPointer(const Reference &) override;
  bool isPairedReloc(const normalized::Relocation &) override;

  bool needsCompactUnwind() override {
    return false;
  }
  Reference::KindValue imageOffsetKind() override {
    return invalid;
  }
  Reference::KindValue imageOffsetKindIndirect() override {
    return invalid;
  }

  std::error_code getReferenceInfo(const normalized::Relocation &reloc,
                                   const DefinedAtom *inAtom,
                                   uint32_t offsetInAtom,
                                   uint64_t fixupAddress, bool swap,
                                   FindAtomBySectionAndAddress atomFromAddress,
                                   FindAtomBySymbolIndex atomFromSymbolIndex,
                                   Reference::KindValue *kind,
                                   const lld::Atom **target,
                                   Reference::Addend *addend) override;
  std::error_code
      getPairReferenceInfo(const normalized::Relocation &reloc1,
                           const normalized::Relocation &reloc2,
                           const DefinedAtom *inAtom,
                           uint32_t offsetInAtom,
                           uint64_t fixupAddress, bool swap,
                           FindAtomBySectionAndAddress atomFromAddress,
                           FindAtomBySymbolIndex atomFromSymbolIndex,
                           Reference::KindValue *kind,
                           const lld::Atom **target,
                           Reference::Addend *addend) override;

  virtual bool needsLocalSymbolInRelocatableFile(const DefinedAtom *atom) {
    return (atom->contentType() == DefinedAtom::typeCString);
  }

  void generateAtomContent(const DefinedAtom &atom, bool relocatable,
                           FindAddressForAtom findAddress,
                           uint64_t imageBaseAddress,
                           uint8_t *atomContentBuffer) override;

  void appendSectionRelocations(const DefinedAtom &atom,
                                uint64_t atomSectionOffset,
                                const Reference &ref,
                                FindSymbolIndexForAtom symbolIndexForAtom,
                                FindSectionIndexForAtom sectionIndexForAtom,
                                FindAddressForAtom addressForAtom,
                                normalized::Relocations &relocs) override;

private:
  static const Registry::KindStrings _sKindStrings[];
  static const StubInfo _sStubInfo;

  enum Arm64_Kinds : Reference::KindValue {
    invalid,               /// for error condition

    // Kinds found in mach-o .o files:
    branch26,              /// ex: bl   _foo
    page21,                /// ex: adrp x1, _foo@PAGE
    offset12,              /// ex: ldrb w0, [x1, _foo@PAGEOFF]
    offset12scale2,        /// ex: ldrs w0, [x1, _foo@PAGEOFF]
    offset12scale4,        /// ex: ldr  w0, [x1, _foo@PAGEOFF]
    offset12scale8,        /// ex: ldr  x0, [x1, _foo@PAGEOFF]
    offset12scale16,       /// ex: ldr  q0, [x1, _foo@PAGEOFF]
    gotPage21,             /// ex: adrp x1, _foo@GOTPAGE
    gotOffset12,           /// ex: ldr  w0, [x1, _foo@GOTPAGEOFF]
    tlvPage21,             /// ex: adrp x1, _foo@TLVPAGE
    tlvOffset12,           /// ex: ldr  w0, [x1, _foo@TLVPAGEOFF]

    pointer64,             /// ex: .quad _foo
    delta64,               /// ex: .quad _foo - .
    delta32,               /// ex: .long _foo - .
    pointer64ToGOT,        /// ex: .quad _foo@GOT
    delta32ToGOT,          /// ex: .long _foo@GOT - .

    // Kinds introduced by Passes:
    addOffset12,           /// Location contains LDR to change into ADD.
    lazyPointer,           /// Location contains a lazy pointer.
    lazyImmediateLocation, /// Location contains immediate value used in stub.
  };

  void applyFixupFinal(const Reference &ref, uint8_t *location,
                       uint64_t fixupAddress, uint64_t targetAddress,
                       uint64_t inAtomAddress);

  void applyFixupRelocatable(const Reference &ref, uint8_t *location,
                             uint64_t fixupAddress, uint64_t targetAddress,
                             uint64_t inAtomAddress);

  // Utility functions for inspecting/updating instructions.
  static uint32_t setDisplacementInBranch26(uint32_t instr, int32_t disp);
  static uint32_t setDisplacementInADRP(uint32_t instr, int64_t disp);
  static Arm64_Kinds offset12KindFromInstruction(uint32_t instr);
  static uint32_t setImm12(uint32_t instr, uint32_t offset);

  const bool _swap;
};

ArchHandler_arm64::ArchHandler_arm64()
  : _swap(!MachOLinkingContext::isHostEndian(MachOLinkingContext::arch_arm64)) {
}

ArchHandler_arm64::~ArchHandler_arm64() {}

const Registry::KindStrings ArchHandler_arm64::_sKindStrings[] = {
  LLD_KIND_STRING_ENTRY(invalid),
  LLD_KIND_STRING_ENTRY(branch26),
  LLD_KIND_STRING_ENTRY(page21),
  LLD_KIND_STRING_ENTRY(offset12),
  LLD_KIND_STRING_ENTRY(offset12scale2),
  LLD_KIND_STRING_ENTRY(offset12scale4),
  LLD_KIND_STRING_ENTRY(offset12scale8),
  LLD_KIND_STRING_ENTRY(offset12scale16),
  LLD_KIND_STRING_ENTRY(gotPage21),
  LLD_KIND_STRING_ENTRY(gotOffset12),
  LLD_KIND_STRING_ENTRY(tlvPage21),
  LLD_KIND_STRING_ENTRY(tlvOffset12),
  LLD_KIND_STRING_ENTRY(pointer64),
  LLD_KIND_STRING_ENTRY(delta64),
  LLD_KIND_STRING_ENTRY(delta32),
  LLD_KIND_STRING_ENTRY(pointer64ToGOT),
  LLD_KIND_STRING_ENTRY(delta32ToGOT),

  LLD_KIND_STRING_ENTRY(addOffset12),
  LLD_KIND_STRING_ENTRY(lazyPointer),
  LLD_KIND_STRING_ENTRY(lazyImmediateLocation),
  LLD_KIND_STRING_ENTRY(pointer64),

  LLD_KIND_STRING_END
};

const ArchHandler::StubInfo ArchHandler_arm64::_sStubInfo = {
  "dyld_stub_binder",

  // Lazy pointer references
  { Reference::KindArch::AArch64, pointer64, 0, 0 },
  { Reference::KindArch::AArch64, lazyPointer, 0, 0 },

  // GOT pointer to dyld_stub_binder
  { Reference::KindArch::AArch64, pointer64, 0, 0 },

  // arm64 code alignment 2^2
  2,

  // Stub size and code
  12,
  { 0x10, 0x00, 0x00, 0x90,   // ADRP  X16, lazy_pointer@page
    0x10, 0x02, 0x40, 0xF9,   // LDR   X16, [X16, lazy_pointer@pageoff]
    0x00, 0x02, 0x1F, 0xD6 }, // BR    X16
  { Reference::KindArch::AArch64, page21, 0, 0 },
  { true,                         offset12scale8, 4, 0 },

  // Stub Helper size and code
  12,
  { 0x50, 0x00, 0x00, 0x18,   //      LDR   W16, L0
    0x00, 0x00, 0x00, 0x14,   //      LDR   B  helperhelper
    0x00, 0x00, 0x00, 0x00 }, // L0: .long 0
  { Reference::KindArch::AArch64, lazyImmediateLocation, 8, 0 },
  { Reference::KindArch::AArch64, branch26, 4, 0 },

  // Stub Helper-Common size and code
  24,
  { 0x11, 0x00, 0x00, 0x90,   //  ADRP  X17, dyld_ImageLoaderCache@page
    0x31, 0x02, 0x00, 0x91,   //  ADD   X17, X17, dyld_ImageLoaderCache@pageoff
    0xF0, 0x47, 0xBF, 0xA9,   //  STP   X16/X17, [SP, #-16]!
    0x10, 0x00, 0x00, 0x90,   //  ADRP  X16, _fast_lazy_bind@page
    0x10, 0x02, 0x40, 0xF9,   //  LDR   X16, [X16,_fast_lazy_bind@pageoff]
    0x00, 0x02, 0x1F, 0xD6 }, //  BR    X16
  { Reference::KindArch::AArch64, page21,   0, 0 },
  { true,                         offset12, 4, 0 },
  { Reference::KindArch::AArch64, page21,   12, 0 },
  { true,                         offset12scale8, 16, 0 }
};

bool ArchHandler_arm64::isCallSite(const Reference &ref) {
  if (ref.kindNamespace() != Reference::KindNamespace::mach_o)
    return false;
  assert(ref.kindArch() == Reference::KindArch::AArch64);
  return (ref.kindValue() == branch26);
}

bool ArchHandler_arm64::isPointer(const Reference &ref) {
  if (ref.kindNamespace() != Reference::KindNamespace::mach_o)
    return false;
  assert(ref.kindArch() == Reference::KindArch::AArch64);
  Reference::KindValue kind = ref.kindValue();
  return (kind == pointer64);
}

bool ArchHandler_arm64::isPairedReloc(const Relocation &r) {
  return ((r.type == ARM64_RELOC_ADDEND) || (r.type == ARM64_RELOC_SUBTRACTOR));
}

uint32_t ArchHandler_arm64::setDisplacementInBranch26(uint32_t instr,
                                                      int32_t displacement) {
  assert((displacement <= 134217727) && (displacement > (-134217728)) &&
         "arm64 branch out of range");
  return (instr & 0xFC000000) | ((uint32_t)(displacement >> 2) & 0x03FFFFFF);
}

uint32_t ArchHandler_arm64::setDisplacementInADRP(uint32_t instruction,
                                                  int64_t displacement) {
  assert((displacement <= 0x100000000LL) && (displacement > (-0x100000000LL)) &&
         "arm64 ADRP out of range");
  assert(((instruction & 0x9F000000) == 0x90000000) &&
         "reloc not on ADRP instruction");
  uint32_t immhi = (displacement >> 9) & (0x00FFFFE0);
  uint32_t immlo = (displacement << 17) & (0x60000000);
  return (instruction & 0x9F00001F) | immlo | immhi;
}

ArchHandler_arm64::Arm64_Kinds
ArchHandler_arm64::offset12KindFromInstruction(uint32_t instruction) {
  if (instruction & 0x08000000) {
    switch ((instruction >> 30) & 0x3) {
    case 0:
      if ((instruction & 0x04800000) == 0x04800000)
        return offset12scale16;
      return offset12;
    case 1:
      return offset12scale2;
    case 2:
      return offset12scale4;
    case 3:
      return offset12scale8;
    }
  }
  return offset12;
}

uint32_t ArchHandler_arm64::setImm12(uint32_t instruction, uint32_t offset) {
  assert(((offset & 0xFFFFF000) == 0) && "imm12 offset out of range");
  uint32_t imm12 = offset << 10;
  return (instruction & 0xFFC003FF) | imm12;
}

std::error_code ArchHandler_arm64::getReferenceInfo(
    const Relocation &reloc, const DefinedAtom *inAtom, uint32_t offsetInAtom,
    uint64_t fixupAddress, bool swap,
    FindAtomBySectionAndAddress atomFromAddress,
    FindAtomBySymbolIndex atomFromSymbolIndex, Reference::KindValue *kind,
    const lld::Atom **target, Reference::Addend *addend) {
  const uint8_t *fixupContent = &inAtom->rawContent()[offsetInAtom];
  switch (relocPattern(reloc)) {
  case ARM64_RELOC_BRANCH26           | rPcRel | rExtern | rLength4:
    // ex: bl _foo
    *kind = branch26;
    if (auto ec = atomFromSymbolIndex(reloc.symbol, target))
      return ec;
    *addend = 0;
    return std::error_code();
  case ARM64_RELOC_PAGE21             | rPcRel | rExtern | rLength4:
    // ex: adrp x1, _foo@PAGE
    *kind = page21;
    if (auto ec = atomFromSymbolIndex(reloc.symbol, target))
      return ec;
    *addend = 0;
    return std::error_code();
  case ARM64_RELOC_PAGEOFF12                   | rExtern | rLength4:
    // ex: ldr x0, [x1, _foo@PAGEOFF]
    *kind = offset12KindFromInstruction(readS32(swap, fixupContent));
    if (auto ec = atomFromSymbolIndex(reloc.symbol, target))
      return ec;
    *addend = 0;
    return std::error_code();
  case ARM64_RELOC_GOT_LOAD_PAGE21    | rPcRel | rExtern | rLength4:
    // ex: adrp x1, _foo@GOTPAGE
    *kind = gotPage21;
    if (auto ec = atomFromSymbolIndex(reloc.symbol, target))
      return ec;
    *addend = 0;
    return std::error_code();
  case ARM64_RELOC_GOT_LOAD_PAGEOFF12          | rExtern | rLength4:
    // ex: ldr x0, [x1, _foo@GOTPAGEOFF]
    *kind = gotOffset12;
    if (auto ec = atomFromSymbolIndex(reloc.symbol, target))
      return ec;
    *addend = 0;
    return std::error_code();
  case ARM64_RELOC_TLVP_LOAD_PAGE21   | rPcRel | rExtern | rLength4:
    // ex: adrp x1, _foo@TLVPAGE
    *kind = tlvPage21;
    if (auto ec = atomFromSymbolIndex(reloc.symbol, target))
      return ec;
    *addend = 0;
    return std::error_code();
  case ARM64_RELOC_TLVP_LOAD_PAGEOFF12         | rExtern | rLength4:
    // ex: ldr x0, [x1, _foo@TLVPAGEOFF]
    *kind = tlvOffset12;
    if (auto ec = atomFromSymbolIndex(reloc.symbol, target))
      return ec;
    *addend = 0;
    return std::error_code();
  case X86_64_RELOC_UNSIGNED                   | rExtern | rLength8:
    // ex: .quad _foo + N
    *kind = pointer64;
    if (auto ec = atomFromSymbolIndex(reloc.symbol, target))
      return ec;
    *addend = readS64(swap, fixupContent);
    return std::error_code();
  case ARM64_RELOC_POINTER_TO_GOT              | rExtern | rLength8:
    // ex: .quad _foo@GOT
    *kind = pointer64ToGOT;
    if (auto ec = atomFromSymbolIndex(reloc.symbol, target))
      return ec;
    *addend = 0;
    return std::error_code();
  case ARM64_RELOC_POINTER_TO_GOT     | rPcRel | rExtern | rLength4:
    // ex: .long _foo@GOT - .
    *kind = delta32ToGOT;
    if (auto ec = atomFromSymbolIndex(reloc.symbol, target))
      return ec;
    *addend = 0;
    return std::error_code();
  default:
    return make_dynamic_error_code(Twine("unsupported arm relocation type"));
  }
}

std::error_code ArchHandler_arm64::getPairReferenceInfo(
    const normalized::Relocation &reloc1, const normalized::Relocation &reloc2,
    const DefinedAtom *inAtom, uint32_t offsetInAtom, uint64_t fixupAddress,
    bool swap, FindAtomBySectionAndAddress atomFromAddress,
    FindAtomBySymbolIndex atomFromSymbolIndex, Reference::KindValue *kind,
    const lld::Atom **target, Reference::Addend *addend) {
  const uint8_t *fixupContent = &inAtom->rawContent()[offsetInAtom];
  const uint32_t *cont32 = reinterpret_cast<const uint32_t *>(fixupContent);
  switch (relocPattern(reloc1) << 16 | relocPattern(reloc2)) {
  case ((ARM64_RELOC_ADDEND                                | rLength4) << 16 |
         ARM64_RELOC_BRANCH26           | rPcRel | rExtern | rLength4):
    // ex: bl _foo+8
    *kind = branch26;
    if (auto ec = atomFromSymbolIndex(reloc2.symbol, target))
      return ec;
    *addend = reloc1.symbol;
    return std::error_code();
  case ((ARM64_RELOC_ADDEND                                | rLength4) << 16 |
         ARM64_RELOC_PAGE21             | rPcRel | rExtern | rLength4):
    // ex: adrp x1, _foo@PAGE
    *kind = page21;
    if (auto ec = atomFromSymbolIndex(reloc2.symbol, target))
      return ec;
    *addend = reloc1.symbol;
    return std::error_code();
  case ((ARM64_RELOC_ADDEND                                | rLength4) << 16 |
         ARM64_RELOC_PAGEOFF12                   | rExtern | rLength4):
    // ex: ldr w0, [x1, _foo@PAGEOFF]
    *kind = offset12KindFromInstruction(*cont32);
    if (auto ec = atomFromSymbolIndex(reloc2.symbol, target))
      return ec;
    *addend = reloc1.symbol;
    return std::error_code();
  case ((ARM64_RELOC_SUBTRACTOR                  | rExtern | rLength8) << 16 |
         ARM64_RELOC_UNSIGNED                    | rExtern | rLength8):
    // ex: .quad _foo - .
    *kind = delta64;
    if (auto ec = atomFromSymbolIndex(reloc2.symbol, target))
      return ec;
    *addend = readS64(swap, fixupContent) + offsetInAtom;
    return std::error_code();
  case ((ARM64_RELOC_SUBTRACTOR                  | rExtern | rLength4) << 16 |
         ARM64_RELOC_UNSIGNED                    | rExtern | rLength4):
    // ex: .quad _foo - .
    *kind = delta32;
    if (auto ec = atomFromSymbolIndex(reloc2.symbol, target))
      return ec;
    *addend = readS32(swap, fixupContent) + offsetInAtom;
    return std::error_code();
  default:
    return make_dynamic_error_code(Twine("unsupported arm64 relocation pair"));
  }
}

void ArchHandler_arm64::generateAtomContent(const DefinedAtom &atom,
                                            bool relocatable,
                                            FindAddressForAtom findAddress,
                                            uint64_t imageBaseAddress,
                                            uint8_t *atomContentBuffer) {
  // Copy raw bytes.
  memcpy(atomContentBuffer, atom.rawContent().data(), atom.size());
  // Apply fix-ups.
  for (const Reference *ref : atom) {
    uint32_t offset = ref->offsetInAtom();
    const Atom *target = ref->target();
    uint64_t targetAddress = 0;
    if (isa<DefinedAtom>(target))
      targetAddress = findAddress(*target);
    uint64_t atomAddress = findAddress(atom);
    uint64_t fixupAddress = atomAddress + offset;
    if (relocatable) {
      applyFixupRelocatable(*ref, &atomContentBuffer[offset], fixupAddress,
                            targetAddress, atomAddress);
    } else {
      applyFixupFinal(*ref, &atomContentBuffer[offset], fixupAddress,
                      targetAddress, atomAddress);
    }
  }
}

void ArchHandler_arm64::applyFixupFinal(const Reference &ref, uint8_t *location,
                                        uint64_t fixupAddress,
                                        uint64_t targetAddress,
                                        uint64_t inAtomAddress) {
  if (ref.kindNamespace() != Reference::KindNamespace::mach_o)
    return;
  assert(ref.kindArch() == Reference::KindArch::AArch64);
  int32_t *loc32 = reinterpret_cast<int32_t *>(location);
  uint64_t *loc64 = reinterpret_cast<uint64_t *>(location);
  int32_t displacement;
  uint32_t instruction;
  uint32_t value32;
  switch (static_cast<Arm64_Kinds>(ref.kindValue())) {
  case branch26:
    displacement = (targetAddress - fixupAddress) + ref.addend();
    value32 = setDisplacementInBranch26(*loc32, displacement);
    write32(*loc32, _swap, value32);
    return;
  case page21:
  case gotPage21:
  case tlvPage21:
    displacement =
        ((targetAddress + ref.addend()) & (-4096)) - (fixupAddress & (-4096));
    value32 = setDisplacementInADRP(*loc32, displacement);
    write32(*loc32, _swap, value32);
    return;
  case offset12:
  case gotOffset12:
  case tlvOffset12:
    displacement = (targetAddress + ref.addend()) & 0x00000FFF;
    value32 = setImm12(*loc32, displacement);
    write32(*loc32, _swap, value32);
    return;
  case offset12scale2:
    displacement = (targetAddress + ref.addend()) & 0x00000FFF;
    assert(((displacement & 0x1) == 0) &&
           "scaled imm12 not accessing 2-byte aligneds");
    value32 = setImm12(*loc32, displacement >> 1);
    write32(*loc32, _swap, value32);
    return;
  case offset12scale4:
    displacement = (targetAddress + ref.addend()) & 0x00000FFF;
    assert(((displacement & 0x3) == 0) &&
           "scaled imm12 not accessing 4-byte aligned");
    value32 = setImm12(*loc32, displacement >> 2);
    write32(*loc32, _swap, value32);
    return;
  case offset12scale8:
    displacement = (targetAddress + ref.addend()) & 0x00000FFF;
    assert(((displacement & 0x7) == 0) &&
           "scaled imm12 not accessing 8-byte aligned");
    value32 = setImm12(*loc32, displacement >> 3);
    write32(*loc32, _swap, value32);
    return;
  case offset12scale16:
    displacement = (targetAddress + ref.addend()) & 0x00000FFF;
    assert(((displacement & 0xF) == 0) &&
           "scaled imm12 not accessing 16-byte aligned");
    value32 = setImm12(*loc32, displacement >> 4);
    write32(*loc32, _swap, value32);
    return;
  case addOffset12:
    instruction = read32(_swap, *loc32);
    assert(((instruction & 0xFFC00000) == 0xF9400000) &&
           "GOT reloc is not an LDR instruction");
    displacement = (targetAddress + ref.addend()) & 0x00000FFF;
    value32 = 0x91000000 | (instruction & 0x000003FF);
    instruction = setImm12(value32, displacement);
    write32(*loc32, _swap, instruction);
    return;
  case pointer64:
  case pointer64ToGOT:
    write64(*loc64, _swap, targetAddress + ref.addend());
    return;
  case delta64:
    write64(*loc64, _swap, (targetAddress - fixupAddress) + ref.addend());
    return;
  case delta32:
  case delta32ToGOT:
    write32(*loc32, _swap, (targetAddress - fixupAddress) + ref.addend());
    return;
  case lazyPointer:
  case lazyImmediateLocation:
    // Do nothing
    return;
  case invalid:
    // Fall into llvm_unreachable().
    break;
  }
  llvm_unreachable("invalid arm64 Reference Kind");
}

void ArchHandler_arm64::applyFixupRelocatable(const Reference &ref,
                                              uint8_t *location,
                                              uint64_t fixupAddress,
                                              uint64_t targetAddress,
                                              uint64_t inAtomAddress) {
  if (ref.kindNamespace() != Reference::KindNamespace::mach_o)
    return;
  assert(ref.kindArch() == Reference::KindArch::AArch64);
  int32_t *loc32 = reinterpret_cast<int32_t *>(location);
  uint64_t *loc64 = reinterpret_cast<uint64_t *>(location);
  uint32_t value32;
  switch (static_cast<Arm64_Kinds>(ref.kindValue())) {
  case branch26:
    value32 = setDisplacementInBranch26(*loc32, 0);
    write32(*loc32, _swap, value32);
    return;
  case page21:
  case gotPage21:
  case tlvPage21:
    value32 = setDisplacementInADRP(*loc32, 0);
    write32(*loc32, _swap, value32);
    return;
  case offset12:
  case offset12scale2:
  case offset12scale4:
  case offset12scale8:
  case offset12scale16:
  case gotOffset12:
  case tlvOffset12:
    value32 = setImm12(*loc32, 0);
    write32(*loc32, _swap, value32);
    return;
  case pointer64:
    write64(*loc64, _swap, ref.addend());
    return;
  case delta64:
    write64(*loc64, _swap, ref.addend() + inAtomAddress - fixupAddress);
    return;
  case delta32:
    write32(*loc32, _swap, ref.addend() + inAtomAddress - fixupAddress);
    return;
  case pointer64ToGOT:
    write64(*loc64, _swap, 0);
    return;
  case delta32ToGOT:
    write32(*loc32, _swap, -fixupAddress);
    return;
  case addOffset12:
    llvm_unreachable("lazy reference kind implies GOT pass was run");
  case lazyPointer:
  case lazyImmediateLocation:
    llvm_unreachable("lazy reference kind implies Stubs pass was run");
  case invalid:
    // Fall into llvm_unreachable().
    break;
  }
  llvm_unreachable("unknown arm64 Reference Kind");
}

void ArchHandler_arm64::appendSectionRelocations(
    const DefinedAtom &atom, uint64_t atomSectionOffset, const Reference &ref,
    FindSymbolIndexForAtom symbolIndexForAtom,
    FindSectionIndexForAtom sectionIndexForAtom,
    FindAddressForAtom addressForAtom, normalized::Relocations &relocs) {
  if (ref.kindNamespace() != Reference::KindNamespace::mach_o)
    return;
  assert(ref.kindArch() == Reference::KindArch::AArch64);
  uint32_t sectionOffset = atomSectionOffset + ref.offsetInAtom();
  switch (static_cast<Arm64_Kinds>(ref.kindValue())) {
  case branch26:
    if (ref.addend()) {
      appendReloc(relocs, sectionOffset, ref.addend(), 0,
                  ARM64_RELOC_ADDEND | rLength4);
      appendReloc(relocs, sectionOffset, symbolIndexForAtom(*ref.target()), 0,
                  ARM64_RELOC_BRANCH26 | rPcRel | rExtern | rLength4);
     } else {
      appendReloc(relocs, sectionOffset, symbolIndexForAtom(*ref.target()), 0,
                  ARM64_RELOC_BRANCH26 | rPcRel | rExtern | rLength4);
    }
    return;
  case page21:
    if (ref.addend()) {
      appendReloc(relocs, sectionOffset, ref.addend(), 0,
                  ARM64_RELOC_ADDEND | rLength4);
      appendReloc(relocs, sectionOffset, symbolIndexForAtom(*ref.target()), 0,
                  ARM64_RELOC_PAGE21 | rPcRel | rExtern | rLength4);
     } else {
      appendReloc(relocs, sectionOffset, symbolIndexForAtom(*ref.target()), 0,
                  ARM64_RELOC_PAGE21 | rPcRel | rExtern | rLength4);
    }
    return;
  case offset12:
  case offset12scale2:
  case offset12scale4:
  case offset12scale8:
  case offset12scale16:
    if (ref.addend()) {
      appendReloc(relocs, sectionOffset, ref.addend(), 0,
                  ARM64_RELOC_ADDEND | rLength4);
      appendReloc(relocs, sectionOffset, symbolIndexForAtom(*ref.target()), 0,
                  ARM64_RELOC_PAGEOFF12  | rExtern | rLength4);
     } else {
      appendReloc(relocs, sectionOffset, symbolIndexForAtom(*ref.target()), 0,
                  ARM64_RELOC_PAGEOFF12 | rExtern | rLength4);
    }
    return;
  case gotPage21:
    assert(ref.addend() == 0);
    appendReloc(relocs, sectionOffset, symbolIndexForAtom(*ref.target()), 0,
                  ARM64_RELOC_GOT_LOAD_PAGE21 | rPcRel | rExtern | rLength4);
    return;
  case gotOffset12:
    assert(ref.addend() == 0);
    appendReloc(relocs, sectionOffset, symbolIndexForAtom(*ref.target()), 0,
                  ARM64_RELOC_GOT_LOAD_PAGEOFF12 | rExtern | rLength4);
    return;
  case tlvPage21:
    assert(ref.addend() == 0);
    appendReloc(relocs, sectionOffset, symbolIndexForAtom(*ref.target()), 0,
                  ARM64_RELOC_TLVP_LOAD_PAGE21 | rPcRel | rExtern | rLength4);
    return;
  case tlvOffset12:
    assert(ref.addend() == 0);
    appendReloc(relocs, sectionOffset, symbolIndexForAtom(*ref.target()), 0,
                  ARM64_RELOC_TLVP_LOAD_PAGEOFF12 | rExtern | rLength4);
    return;
  case pointer64:
    appendReloc(relocs, sectionOffset, symbolIndexForAtom(*ref.target()), 0,
                  X86_64_RELOC_UNSIGNED | rExtern | rLength8);
    return;
  case delta64:
    appendReloc(relocs, sectionOffset, symbolIndexForAtom(atom), 0,
                ARM64_RELOC_SUBTRACTOR | rExtern | rLength8);
    appendReloc(relocs, sectionOffset, symbolIndexForAtom(*ref.target()), 0,
                ARM64_RELOC_UNSIGNED  | rExtern | rLength8);
    return;
  case delta32:
    appendReloc(relocs, sectionOffset, symbolIndexForAtom(atom), 0,
                ARM64_RELOC_SUBTRACTOR | rExtern | rLength4 );
    appendReloc(relocs, sectionOffset, symbolIndexForAtom(*ref.target()), 0,
                ARM64_RELOC_UNSIGNED   | rExtern | rLength4 );
    return;
  case pointer64ToGOT:
    assert(ref.addend() == 0);
    appendReloc(relocs, sectionOffset, symbolIndexForAtom(*ref.target()), 0,
                  ARM64_RELOC_POINTER_TO_GOT | rExtern | rLength8);
    return;
  case delta32ToGOT:
    assert(ref.addend() == 0);
    appendReloc(relocs, sectionOffset, symbolIndexForAtom(*ref.target()), 0,
                  ARM64_RELOC_POINTER_TO_GOT | rPcRel | rExtern | rLength4);
    return;
  case addOffset12:
    llvm_unreachable("lazy reference kind implies GOT pass was run");
  case lazyPointer:
  case lazyImmediateLocation:
    llvm_unreachable("lazy reference kind implies Stubs pass was run");
  case invalid:
    // Fall into llvm_unreachable().
    break;
  }
  llvm_unreachable("unknown arm64 Reference Kind");
}

std::unique_ptr<mach_o::ArchHandler> ArchHandler::create_arm64() {
  return std::unique_ptr<mach_o::ArchHandler>(new ArchHandler_arm64());
}

} // namespace mach_o
} // namespace lld
