//===- lib/FileFormat/MachO/ArchHandler_x86_64.cpp ------------------------===//
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

using namespace llvm::MachO;
using namespace lld::mach_o::normalized;

namespace lld {
namespace mach_o {

class ArchHandler_x86_64 : public ArchHandler {
public:
           ArchHandler_x86_64();
  virtual ~ArchHandler_x86_64();

  const Registry::KindStrings *kindStrings() override { return _sKindStrings; }

  Reference::KindArch kindArch() override {
    return Reference::KindArch::x86_64;
  }

  /// Used by GOTPass to locate GOT References
  bool isGOTAccess(const Reference &ref, bool &canBypassGOT) override {
    if (ref.kindNamespace() != Reference::KindNamespace::mach_o)
      return false;
    assert(ref.kindArch() == Reference::KindArch::x86_64);
    switch (ref.kindValue()) {
    case ripRel32GotLoad:
      canBypassGOT = true;
      return true;
    case ripRel32Got:
      canBypassGOT = false;
      return true;
    default:
      return false;
    }
  }

  /// Used by GOTPass to update GOT References
  void updateReferenceToGOT(const Reference *ref, bool targetNowGOT) override {
    assert(ref->kindNamespace() == Reference::KindNamespace::mach_o);
    assert(ref->kindArch() == Reference::KindArch::x86_64);
    const_cast<Reference *>(ref)
        ->setKindValue(targetNowGOT ? ripRel32 : ripRel32GotLoadNowLea);
  }

  const StubInfo &stubInfo() override { return _sStubInfo; }

  bool isCallSite(const Reference &) override;
  bool isPointer(const Reference &) override;
  bool isPairedReloc(const normalized::Relocation &) override;
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
  static const StubInfo              _sStubInfo;

  enum X86_64_Kinds: Reference::KindValue {
    invalid,               /// for error condition
    
    // Kinds found in mach-o .o files:
    branch32,              /// ex: call _foo
    ripRel32,              /// ex: movq _foo(%rip), %rax
    ripRel32Minus1,        /// ex: movb $0x12, _foo(%rip)
    ripRel32Minus2,        /// ex: movw $0x1234, _foo(%rip)
    ripRel32Minus4,        /// ex: movl $0x12345678, _foo(%rip)
    ripRel32Anon,          /// ex: movq L1(%rip), %rax
    ripRel32GotLoad,       /// ex: movq  _foo@GOTPCREL(%rip), %rax
    ripRel32Got,           /// ex: pushq _foo@GOTPCREL(%rip)
    pointer64,             /// ex: .quad _foo
    pointer64Anon,         /// ex: .quad L1
    delta64,               /// ex: .quad _foo - .
    delta32,               /// ex: .long _foo - .
    delta64Anon,           /// ex: .quad L1 - .
    delta32Anon,           /// ex: .long L1 - .
    
    // Kinds introduced by Passes:
    ripRel32GotLoadNowLea, /// Target of GOT load is in linkage unit so 
                           ///  "movq  _foo@GOTPCREL(%rip), %rax" can be changed
                           /// to "leaq _foo(%rip), %rax
    lazyPointer,           /// Location contains a lazy pointer.
    lazyImmediateLocation, /// Location contains immediate value used in stub.
  };

  Reference::KindValue kindFromReloc(const normalized::Relocation &reloc);
  Reference::KindValue kindFromRelocPair(const normalized::Relocation &reloc1,
                                         const normalized::Relocation &reloc2);

  void applyFixupFinal(const Reference &ref, uint8_t *location,
                       uint64_t fixupAddress, uint64_t targetAddress,
                       uint64_t inAtomAddress);

  void applyFixupRelocatable(const Reference &ref, uint8_t *location,
                             uint64_t fixupAddress,
                             uint64_t targetAddress,
                             uint64_t inAtomAddress);

  const bool _swap;
};


ArchHandler_x86_64::ArchHandler_x86_64() :
  _swap(!MachOLinkingContext::isHostEndian(MachOLinkingContext::arch_x86_64)) {}

ArchHandler_x86_64::~ArchHandler_x86_64() { }

const Registry::KindStrings ArchHandler_x86_64::_sKindStrings[] = {
  LLD_KIND_STRING_ENTRY(invalid), LLD_KIND_STRING_ENTRY(branch32),
  LLD_KIND_STRING_ENTRY(ripRel32), LLD_KIND_STRING_ENTRY(ripRel32Minus1),
  LLD_KIND_STRING_ENTRY(ripRel32Minus2), LLD_KIND_STRING_ENTRY(ripRel32Minus4),
  LLD_KIND_STRING_ENTRY(ripRel32Anon), LLD_KIND_STRING_ENTRY(ripRel32GotLoad),
  LLD_KIND_STRING_ENTRY(ripRel32GotLoadNowLea),
  LLD_KIND_STRING_ENTRY(ripRel32Got), LLD_KIND_STRING_ENTRY(lazyPointer),
  LLD_KIND_STRING_ENTRY(lazyImmediateLocation),
  LLD_KIND_STRING_ENTRY(pointer64), LLD_KIND_STRING_ENTRY(pointer64Anon),
  LLD_KIND_STRING_ENTRY(delta32), LLD_KIND_STRING_ENTRY(delta64),
  LLD_KIND_STRING_ENTRY(delta32Anon), LLD_KIND_STRING_ENTRY(delta64Anon),
  LLD_KIND_STRING_END
};

const ArchHandler::StubInfo ArchHandler_x86_64::_sStubInfo = {
  "dyld_stub_binder",

  // Lazy pointer references 
  { Reference::KindArch::x86_64, pointer64, 0, 0 },
  { Reference::KindArch::x86_64, lazyPointer, 0, 0 },
  
  // GOT pointer to dyld_stub_binder
  { Reference::KindArch::x86_64, pointer64, 0, 0 },

  // x86_64 code alignment 2^1
  1, 
  
  // Stub size and code
  6, 
  { 0xff, 0x25, 0x00, 0x00, 0x00, 0x00 },       // jmp *lazyPointer
  { Reference::KindArch::x86_64, ripRel32, 2, 0 },
  { false, 0, 0, 0 },
  
  // Stub Helper size and code
  10,
  { 0x68, 0x00, 0x00, 0x00, 0x00,               // pushq $lazy-info-offset
    0xE9, 0x00, 0x00, 0x00, 0x00 },             // jmp helperhelper
  { Reference::KindArch::x86_64, lazyImmediateLocation, 1, 0 },
  { Reference::KindArch::x86_64, branch32, 6, 0 },
  
  // Stub Helper-Common size and code
  16,
  { 0x4C, 0x8D, 0x1D, 0x00, 0x00, 0x00, 0x00,   // leaq cache(%rip),%r11
    0x41, 0x53,                                 // push %r11
    0xFF, 0x25, 0x00, 0x00, 0x00, 0x00,         // jmp *binder(%rip)
    0x90 },                                     // nop
  { Reference::KindArch::x86_64, ripRel32, 3, 0 },
  { false, 0, 0, 0 },
  { Reference::KindArch::x86_64, ripRel32, 11, 0 },
  { false, 0, 0, 0 }
  
};

bool ArchHandler_x86_64::isCallSite(const Reference &ref) {
  if (ref.kindNamespace() != Reference::KindNamespace::mach_o)
    return false;
  assert(ref.kindArch() == Reference::KindArch::x86_64);
  return (ref.kindValue() == branch32);
}

bool ArchHandler_x86_64::isPointer(const Reference &ref) {
  if (ref.kindNamespace() != Reference::KindNamespace::mach_o)
    return false;
  assert(ref.kindArch() == Reference::KindArch::x86_64);
  Reference::KindValue kind = ref.kindValue();
  return (kind == pointer64 || kind == pointer64Anon);
}

bool ArchHandler_x86_64::isPairedReloc(const Relocation &reloc) {
  return (reloc.type == X86_64_RELOC_SUBTRACTOR);
}

Reference::KindValue
ArchHandler_x86_64::kindFromReloc(const Relocation &reloc) {
  switch(relocPattern(reloc)) {
  case X86_64_RELOC_BRANCH   | rPcRel | rExtern | rLength4:
    return branch32;
  case X86_64_RELOC_SIGNED   | rPcRel | rExtern | rLength4:
    return ripRel32;
  case X86_64_RELOC_SIGNED   | rPcRel |           rLength4:
    return ripRel32Anon;
  case X86_64_RELOC_SIGNED_1 | rPcRel | rExtern | rLength4:
    return ripRel32Minus1;
  case X86_64_RELOC_SIGNED_2 | rPcRel | rExtern | rLength4:
    return ripRel32Minus2;
  case X86_64_RELOC_SIGNED_4 | rPcRel | rExtern | rLength4:
    return ripRel32Minus4;
  case X86_64_RELOC_GOT_LOAD | rPcRel | rExtern | rLength4:
    return ripRel32GotLoad;
  case X86_64_RELOC_GOT      | rPcRel | rExtern | rLength4:
    return ripRel32Got;
  case X86_64_RELOC_UNSIGNED          | rExtern | rLength8:
    return pointer64;
  case X86_64_RELOC_UNSIGNED                    | rLength8:
    return pointer64Anon;
  default:
    return invalid;
  }
}

std::error_code 
ArchHandler_x86_64::getReferenceInfo(const Relocation &reloc,
                                    const DefinedAtom *inAtom,
                                    uint32_t offsetInAtom,
                                    uint64_t fixupAddress, bool swap,
                                    FindAtomBySectionAndAddress atomFromAddress,
                                    FindAtomBySymbolIndex atomFromSymbolIndex,
                                    Reference::KindValue *kind, 
                                    const lld::Atom **target, 
                                    Reference::Addend *addend) {
  typedef std::error_code E;
  *kind = kindFromReloc(reloc);
  if (*kind == invalid)
    return make_dynamic_error_code(Twine("unknown type"));
  const uint8_t *fixupContent = &inAtom->rawContent()[offsetInAtom];
  uint64_t targetAddress;
  switch (*kind) {
  case branch32:
  case ripRel32:
    if (E ec = atomFromSymbolIndex(reloc.symbol, target))
      return ec;
    *addend = readS32(swap, fixupContent);
    return std::error_code();
  case ripRel32Minus1:
    if (E ec = atomFromSymbolIndex(reloc.symbol, target))
      return ec;
    *addend = readS32(swap, fixupContent) + 1;
    return std::error_code();
  case ripRel32Minus2:
    if (E ec = atomFromSymbolIndex(reloc.symbol, target))
      return ec;
    *addend = readS32(swap, fixupContent) + 2;
    return std::error_code();
  case ripRel32Minus4:
    if (E ec = atomFromSymbolIndex(reloc.symbol, target))
      return ec;
    *addend = readS32(swap, fixupContent) + 4;
    return std::error_code();
  case ripRel32Anon:
    targetAddress = fixupAddress + 4 + readS32(swap, fixupContent);
    return atomFromAddress(reloc.symbol, targetAddress, target, addend);
  case ripRel32GotLoad:
  case ripRel32Got:
    if (E ec = atomFromSymbolIndex(reloc.symbol, target))
      return ec;
    *addend = 0;
    return std::error_code();
  case pointer64:
    if (E ec = atomFromSymbolIndex(reloc.symbol, target))
      return ec;
    *addend = readS64(swap, fixupContent);
    return std::error_code();
  case pointer64Anon:
    targetAddress = readS64(swap, fixupContent);
    return atomFromAddress(reloc.symbol, targetAddress, target, addend);
  default:
    llvm_unreachable("bad reloc kind");
  }
}

Reference::KindValue
ArchHandler_x86_64::kindFromRelocPair(const normalized::Relocation &reloc1,
                                      const normalized::Relocation &reloc2) {
  switch(relocPattern(reloc1) << 16 | relocPattern(reloc2)) {
  case ((X86_64_RELOC_SUBTRACTOR | rExtern | rLength8) << 16 |
        X86_64_RELOC_UNSIGNED    | rExtern | rLength8):
    return delta64;
  case ((X86_64_RELOC_SUBTRACTOR | rExtern | rLength4) << 16 |
        X86_64_RELOC_UNSIGNED    | rExtern | rLength4):
    return delta32;
  case ((X86_64_RELOC_SUBTRACTOR | rExtern | rLength8) << 16 |
        X86_64_RELOC_UNSIGNED              | rLength8):
    return delta64Anon;
  case ((X86_64_RELOC_SUBTRACTOR | rExtern | rLength4) << 16 |
        X86_64_RELOC_UNSIGNED              | rLength4):
    return delta32Anon;
  default:
    llvm_unreachable("bad reloc pairs");
  }
}

std::error_code
ArchHandler_x86_64::getPairReferenceInfo(const normalized::Relocation &reloc1,
                                   const normalized::Relocation &reloc2,
                                   const DefinedAtom *inAtom,
                                   uint32_t offsetInAtom,
                                   uint64_t fixupAddress, bool swap,
                                   FindAtomBySectionAndAddress atomFromAddress,
                                   FindAtomBySymbolIndex atomFromSymbolIndex,
                                   Reference::KindValue *kind, 
                                   const lld::Atom **target, 
                                   Reference::Addend *addend) {
  *kind = kindFromRelocPair(reloc1, reloc2);
  if (*kind == invalid)
    return make_dynamic_error_code(Twine("unknown pair"));
  const uint8_t *fixupContent = &inAtom->rawContent()[offsetInAtom];
  typedef std::error_code E;
  uint64_t targetAddress;
  const lld::Atom *fromTarget;
  if (E ec = atomFromSymbolIndex(reloc1.symbol, &fromTarget))
    return ec;
  if (fromTarget != inAtom)
    return make_dynamic_error_code(Twine("pointer diff not in base atom"));
  switch (*kind) {
  case delta64:
    if (E ec = atomFromSymbolIndex(reloc2.symbol, target))
      return ec;
    *addend = readS64(swap, fixupContent) + offsetInAtom;
    return std::error_code();
  case delta32:
    if (E ec = atomFromSymbolIndex(reloc2.symbol, target))
      return ec;
    *addend = readS32(swap, fixupContent) + offsetInAtom;
    return std::error_code();
  case delta64Anon:
    targetAddress = offsetInAtom + readS64(swap, fixupContent);
    return atomFromAddress(reloc2.symbol, targetAddress, target, addend);
  case delta32Anon:
    targetAddress = offsetInAtom + readS32(swap, fixupContent);
    return atomFromAddress(reloc2.symbol, targetAddress, target, addend);
  default:
    llvm_unreachable("bad reloc pair kind");
  }
}

void ArchHandler_x86_64::generateAtomContent(const DefinedAtom &atom,
                                             bool relocatable,
                                             FindAddressForAtom findAddress,
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
      applyFixupRelocatable(*ref, &atomContentBuffer[offset],
                                        fixupAddress, targetAddress,
                                        atomAddress);
    } else {
      applyFixupFinal(*ref, &atomContentBuffer[offset],
                                  fixupAddress, targetAddress,
                                  atomAddress);
    }
  }
}

void ArchHandler_x86_64::applyFixupFinal(const Reference &ref,
                                         uint8_t *location,
                                         uint64_t fixupAddress,
                                         uint64_t targetAddress,
                                         uint64_t inAtomAddress) {
  if (ref.kindNamespace() != Reference::KindNamespace::mach_o)
    return;
  assert(ref.kindArch() == Reference::KindArch::x86_64);
  int32_t *loc32 = reinterpret_cast<int32_t *>(location);
  uint64_t *loc64 = reinterpret_cast<uint64_t *>(location);
  switch (static_cast<X86_64_Kinds>(ref.kindValue())) {
  case branch32:
  case ripRel32:
  case ripRel32Anon:
  case ripRel32Got:
  case ripRel32GotLoad:
    write32(*loc32, _swap, (targetAddress - (fixupAddress + 4)) + ref.addend());
    return;
  case pointer64:
  case pointer64Anon:
    write64(*loc64, _swap, targetAddress + ref.addend());
    return;
  case ripRel32Minus1:
    write32(*loc32, _swap, (targetAddress - (fixupAddress + 5)) + ref.addend());
    return;
  case ripRel32Minus2:
    write32(*loc32, _swap, (targetAddress - (fixupAddress + 6)) + ref.addend());
    return;
  case ripRel32Minus4:
    write32(*loc32, _swap, (targetAddress - (fixupAddress + 8)) + ref.addend());
    return;
  case delta32:
  case delta32Anon:
    write32(*loc32, _swap, (targetAddress - fixupAddress) + ref.addend());
    return;
  case delta64:
  case delta64Anon:
    write64(*loc64, _swap, (targetAddress - fixupAddress) + ref.addend());
    return;
  case ripRel32GotLoadNowLea:
    // Change MOVQ to LEA
    assert(location[-2] == 0x8B);
    location[-2] = 0x8D;
    write32(*loc32, _swap, (targetAddress - (fixupAddress + 4)) + ref.addend());
    return;
  case lazyPointer:
  case lazyImmediateLocation:
    // do nothing
    return;
  case invalid:
    // Fall into llvm_unreachable().
    break;
  }
  llvm_unreachable("invalid x86_64 Reference Kind");
}


void ArchHandler_x86_64::applyFixupRelocatable(const Reference &ref,
                                               uint8_t *location,
                                               uint64_t fixupAddress,
                                               uint64_t targetAddress,
                                               uint64_t inAtomAddress)  {
  int32_t *loc32 = reinterpret_cast<int32_t *>(location);
  uint64_t *loc64 = reinterpret_cast<uint64_t *>(location);
  switch (static_cast<X86_64_Kinds>(ref.kindValue())) {
  case branch32:
  case ripRel32:
  case ripRel32Got:
  case ripRel32GotLoad:
    write32(*loc32, _swap, ref.addend());
    return;
  case ripRel32Anon:
    write32(*loc32, _swap, (targetAddress - (fixupAddress + 4)) + ref.addend());
    return;
  case pointer64:
    write64(*loc64, _swap, ref.addend());
    return;
  case pointer64Anon:
    write64(*loc64, _swap, targetAddress + ref.addend());
    return;
  case ripRel32Minus1:
    write32(*loc32, _swap, ref.addend() - 1);
    return;
  case ripRel32Minus2:
    write32(*loc32, _swap, ref.addend() - 2);
    return;
  case ripRel32Minus4:
    write32(*loc32, _swap, ref.addend() - 4);
    return;
  case delta32:
    write32(*loc32, _swap, ref.addend() + inAtomAddress - fixupAddress);
    return;
  case delta32Anon:
    write32(*loc32, _swap, (targetAddress - fixupAddress) + ref.addend());
    return;
  case delta64:
    write64(*loc64, _swap, ref.addend() + inAtomAddress - fixupAddress);
    return;
  case delta64Anon:
    write64(*loc64, _swap, (targetAddress - fixupAddress) + ref.addend());
    return;
  case ripRel32GotLoadNowLea:
    llvm_unreachable("ripRel32GotLoadNowLea implies GOT pass was run");
    return;
  case lazyPointer:
  case lazyImmediateLocation:
    llvm_unreachable("lazy reference kind implies Stubs pass was run");
    return;
  case invalid:
    // Fall into llvm_unreachable().
    break;
  }
  llvm_unreachable("unknown x86_64 Reference Kind");
}

void ArchHandler_x86_64::appendSectionRelocations(
                                   const DefinedAtom &atom,
                                   uint64_t atomSectionOffset,
                                   const Reference &ref,
                                   FindSymbolIndexForAtom symbolIndexForAtom,
                                   FindSectionIndexForAtom sectionIndexForAtom,
                                   FindAddressForAtom addressForAtom,
                                   normalized::Relocations &relocs) {
  if (ref.kindNamespace() != Reference::KindNamespace::mach_o)
    return;
  assert(ref.kindArch() == Reference::KindArch::x86_64);
  uint32_t sectionOffset = atomSectionOffset + ref.offsetInAtom();
  switch (static_cast<X86_64_Kinds>(ref.kindValue())) {
  case branch32:
    appendReloc(relocs, sectionOffset, symbolIndexForAtom(*ref.target()), 0,
                X86_64_RELOC_BRANCH | rPcRel | rExtern | rLength4);
    return;
  case ripRel32:
    appendReloc(relocs, sectionOffset, symbolIndexForAtom(*ref.target()), 0,
                X86_64_RELOC_SIGNED | rPcRel | rExtern | rLength4 );
    return;
  case ripRel32Anon:
    appendReloc(relocs, sectionOffset, sectionIndexForAtom(*ref.target()), 0,
                X86_64_RELOC_SIGNED | rPcRel          | rLength4 );
    return;
  case ripRel32Got:
    appendReloc(relocs, sectionOffset, symbolIndexForAtom(*ref.target()), 0,
                X86_64_RELOC_GOT | rPcRel | rExtern | rLength4 );
    return;
  case ripRel32GotLoad:
    appendReloc(relocs, sectionOffset, symbolIndexForAtom(*ref.target()), 0,
                X86_64_RELOC_GOT_LOAD | rPcRel | rExtern | rLength4 );
    return;
  case pointer64:
    appendReloc(relocs, sectionOffset, symbolIndexForAtom(*ref.target()), 0,
                X86_64_RELOC_UNSIGNED  | rExtern | rLength8);
    return;
  case pointer64Anon:
    appendReloc(relocs, sectionOffset, sectionIndexForAtom(*ref.target()), 0,
                X86_64_RELOC_UNSIGNED | rLength8);
    return;
  case ripRel32Minus1:
    appendReloc(relocs, sectionOffset, symbolIndexForAtom(*ref.target()), 0,
                X86_64_RELOC_SIGNED_1 | rPcRel | rExtern | rLength4 );
    return;
  case ripRel32Minus2:
    appendReloc(relocs, sectionOffset, symbolIndexForAtom(*ref.target()), 0,
                X86_64_RELOC_SIGNED_2 | rPcRel | rExtern | rLength4 );
    return;
  case ripRel32Minus4:
    appendReloc(relocs, sectionOffset, symbolIndexForAtom(*ref.target()), 0,
                X86_64_RELOC_SIGNED_4 | rPcRel | rExtern | rLength4 );
    return;
  case delta32:
    appendReloc(relocs, sectionOffset, symbolIndexForAtom(atom), 0,
                X86_64_RELOC_SUBTRACTOR | rExtern | rLength4 );
    appendReloc(relocs, sectionOffset, symbolIndexForAtom(*ref.target()), 0,
                X86_64_RELOC_UNSIGNED   | rExtern | rLength4 );
    return;
  case delta32Anon:
    appendReloc(relocs, sectionOffset, symbolIndexForAtom(atom), 0,
                X86_64_RELOC_SUBTRACTOR | rExtern | rLength4 );
    appendReloc(relocs, sectionOffset, sectionIndexForAtom(*ref.target()), 0,
                X86_64_RELOC_UNSIGNED             | rLength4 );
    return;
  case delta64:
    appendReloc(relocs, sectionOffset, symbolIndexForAtom(atom), 0,
                X86_64_RELOC_SUBTRACTOR | rExtern | rLength8 );
    appendReloc(relocs, sectionOffset, symbolIndexForAtom(*ref.target()), 0,
                X86_64_RELOC_UNSIGNED   | rExtern | rLength8 );
    return;
  case delta64Anon:
    appendReloc(relocs, sectionOffset, symbolIndexForAtom(atom), 0,
                X86_64_RELOC_SUBTRACTOR | rExtern | rLength8 );
    appendReloc(relocs, sectionOffset, sectionIndexForAtom(*ref.target()), 0,
                X86_64_RELOC_UNSIGNED             | rLength8 );
    return;
  case ripRel32GotLoadNowLea:
    llvm_unreachable("ripRel32GotLoadNowLea implies GOT pass was run");
    return;
  case lazyPointer:
  case lazyImmediateLocation:
    llvm_unreachable("lazy reference kind implies Stubs pass was run");
    return;
  case invalid:
    // Fall into llvm_unreachable().
    break;
  }
  llvm_unreachable("unknown x86_64 Reference Kind");
}


std::unique_ptr<mach_o::ArchHandler> ArchHandler::create_x86_64() {
  return std::unique_ptr<mach_o::ArchHandler>(new ArchHandler_x86_64());
}

} // namespace mach_o
} // namespace lld
