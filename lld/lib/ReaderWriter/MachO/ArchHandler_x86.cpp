//===- lib/FileFormat/MachO/ArchHandler_x86.cpp ---------------------------===//
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

class ArchHandler_x86 : public ArchHandler {
public:
           ArchHandler_x86();
  virtual ~ArchHandler_x86();

  const Registry::KindStrings *kindStrings() override { return _sKindStrings; }

  Reference::KindArch kindArch() override { return Reference::KindArch::x86; }

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

  void applyFixup(Reference::KindNamespace ns, Reference::KindArch arch,
                          Reference::KindValue kindValue, uint64_t addend,
                          uint8_t *location, uint64_t fixupAddress,
                          uint64_t targetAddress, uint64_t inAtomAddress) 
                          override;

private:
  static const Registry::KindStrings _sKindStrings[];
  static const StubInfo              _sStubInfo;

  enum : Reference::KindValue {
    invalid,               /// for error condition

    // Kinds found in mach-o .o files:
    branch32,              /// ex: call _foo
    branch16,              /// ex: callw _foo
    abs32,                 /// ex: movl _foo, %eax
    funcRel32,             /// ex: movl _foo-L1(%eax), %eax
    pointer32,             /// ex: .long _foo
    delta32,               /// ex: .long _foo - .

    // Kinds introduced by Passes:
    lazyPointer,           /// Location contains a lazy pointer.
    lazyImmediateLocation, /// Location contains immediate value used in stub.
  };
  
  const bool _swap;
};

//===----------------------------------------------------------------------===//
//  ArchHandler_x86
//===----------------------------------------------------------------------===//

ArchHandler_x86::ArchHandler_x86() :
  _swap(!MachOLinkingContext::isHostEndian(MachOLinkingContext::arch_x86)) {}
  
ArchHandler_x86::~ArchHandler_x86() { }
  
const Registry::KindStrings ArchHandler_x86::_sKindStrings[] = {
  LLD_KIND_STRING_ENTRY(invalid),
  LLD_KIND_STRING_ENTRY(branch32),
  LLD_KIND_STRING_ENTRY(branch16),
  LLD_KIND_STRING_ENTRY(abs32),
  LLD_KIND_STRING_ENTRY(funcRel32),
  LLD_KIND_STRING_ENTRY(pointer32),
  LLD_KIND_STRING_ENTRY(delta32),
  LLD_KIND_STRING_ENTRY(lazyPointer),
  LLD_KIND_STRING_ENTRY(lazyImmediateLocation),
  LLD_KIND_STRING_END
};

const ArchHandler::StubInfo ArchHandler_x86::_sStubInfo = {
  "dyld_stub_binder",

  // Lazy pointer references 
  { Reference::KindArch::x86, pointer32, 0, 0 },
  { Reference::KindArch::x86, lazyPointer, 0, 0 },
  
  // GOT pointer to dyld_stub_binder
  { Reference::KindArch::x86, pointer32, 0, 0 },

  // x86 code alignment
  1, 
  
  // Stub size and code
  6, 
  { 0xff, 0x25, 0x00, 0x00, 0x00, 0x00 },       // jmp *lazyPointer
  { Reference::KindArch::x86, abs32, 2, 0 },
  
  // Stub Helper size and code
  10,
  { 0x68, 0x00, 0x00, 0x00, 0x00,               // pushl $lazy-info-offset
    0xE9, 0x00, 0x00, 0x00, 0x00 },             // jmp helperhelper
  { Reference::KindArch::x86, lazyImmediateLocation, 1, 0 },
  { Reference::KindArch::x86, branch32, 6, 0 },
  
  // Stub Helper-Common size and code
  12,
  { 0x68, 0x00, 0x00, 0x00, 0x00,               // pushl $dyld_ImageLoaderCache
    0xFF, 0x25, 0x00, 0x00, 0x00, 0x00,         // jmp *_fast_lazy_bind
    0x90 },                                     // nop
  { Reference::KindArch::x86, abs32, 1, 0 },
  { Reference::KindArch::x86, abs32, 7, 0 }
};

bool ArchHandler_x86::isCallSite(const Reference &ref) {
  return (ref.kindValue() == branch32);
}

bool ArchHandler_x86::isPointer(const Reference &ref) {
  return (ref.kindValue() == pointer32);
}

bool ArchHandler_x86::isPairedReloc(const Relocation &reloc) {
  if (!reloc.scattered)
    return false;
  return (reloc.type == GENERIC_RELOC_LOCAL_SECTDIFF) ||
         (reloc.type == GENERIC_RELOC_SECTDIFF);
}

std::error_code
ArchHandler_x86::getReferenceInfo(const Relocation &reloc,
                                  const DefinedAtom *inAtom,
                                  uint32_t offsetInAtom,
                                  uint64_t fixupAddress, bool swap,
                                  FindAtomBySectionAndAddress atomFromAddress,
                                  FindAtomBySymbolIndex atomFromSymbolIndex,
                                  Reference::KindValue *kind,
                                  const lld::Atom **target,
                                  Reference::Addend *addend) {
  typedef std::error_code E;
  DefinedAtom::ContentPermissions perms;
  const uint8_t *fixupContent = &inAtom->rawContent()[offsetInAtom];
  uint64_t targetAddress;
  switch (relocPattern(reloc)) {
  case GENERIC_RELOC_VANILLA | rPcRel | rExtern | rLength4:
    // ex: call _foo (and _foo undefined)
    *kind = branch32;
    if (E ec = atomFromSymbolIndex(reloc.symbol, target))
      return ec;
    *addend = fixupAddress + 4 + readS32(swap, fixupContent);
    break;
  case GENERIC_RELOC_VANILLA | rPcRel | rLength4:
    // ex: call _foo (and _foo defined)
    *kind = branch32;
    targetAddress = fixupAddress + 4 + readS32(swap, fixupContent);
    return atomFromAddress(reloc.symbol, targetAddress, target, addend);
    break;
  case GENERIC_RELOC_VANILLA | rPcRel | rExtern | rLength2:
    // ex: callw _foo (and _foo undefined)
    *kind = branch16;
    if (E ec = atomFromSymbolIndex(reloc.symbol, target))
      return ec;
    *addend = fixupAddress + 2 + readS16(swap, fixupContent);
    break;
  case GENERIC_RELOC_VANILLA | rPcRel | rLength2:
    // ex: callw _foo (and _foo defined)
    *kind = branch16;
    targetAddress = fixupAddress + 2 + readS16(swap, fixupContent);
    return atomFromAddress(reloc.symbol, targetAddress, target, addend);
    break;
  case GENERIC_RELOC_VANILLA | rExtern | rLength4:
    // ex: movl	_foo, %eax   (and _foo undefined)
    // ex: .long _foo        (and _foo undefined)
    perms = inAtom->permissions();
    *kind =
        ((perms & DefinedAtom::permR_X) == DefinedAtom::permR_X) ? abs32
                                                                 : pointer32;
    if (E ec = atomFromSymbolIndex(reloc.symbol, target))
      return ec;
    *addend = readU32(swap, fixupContent);
    break;
  case GENERIC_RELOC_VANILLA | rLength4:
    // ex: movl	_foo, %eax   (and _foo defined)
    // ex: .long _foo        (and _foo defined)
    perms = inAtom->permissions();
    *kind =
        ((perms & DefinedAtom::permR_X) == DefinedAtom::permR_X) ? abs32
                                                                 : pointer32;
    targetAddress = readU32(swap, fixupContent);
    return atomFromAddress(reloc.symbol, targetAddress, target, addend);
    break;
  default:
    return make_dynamic_error_code(Twine("unsupported i386 relocation type"));
  }
  return std::error_code();
}

std::error_code
ArchHandler_x86::getPairReferenceInfo(const normalized::Relocation &reloc1,
                                      const normalized::Relocation &reloc2,
                                      const DefinedAtom *inAtom,
                                      uint32_t offsetInAtom,
                                      uint64_t fixupAddress, bool swap,
                                      FindAtomBySectionAndAddress atomFromAddr,
                                      FindAtomBySymbolIndex atomFromSymbolIndex,
                                      Reference::KindValue *kind,
                                      const lld::Atom **target,
                                      Reference::Addend *addend) {
  const uint8_t *fixupContent = &inAtom->rawContent()[offsetInAtom];
  std::error_code ec;
  DefinedAtom::ContentPermissions perms = inAtom->permissions();
  uint32_t fromAddress;
  uint32_t toAddress;
  uint32_t value;
  const lld::Atom *fromTarget;
  Reference::Addend offsetInTo;
  Reference::Addend offsetInFrom;
  switch (relocPattern(reloc1) << 16 | relocPattern(reloc2)) {
  case((GENERIC_RELOC_SECTDIFF | rScattered | rLength4) << 16 |
       GENERIC_RELOC_PAIR | rScattered | rLength4)
      :
  case((GENERIC_RELOC_LOCAL_SECTDIFF | rScattered | rLength4) << 16 |
       GENERIC_RELOC_PAIR | rScattered | rLength4)
      :
    toAddress = reloc1.value;
    fromAddress = reloc2.value;
    value = readS32(swap, fixupContent);
    ec = atomFromAddr(0, toAddress, target, &offsetInTo);
    if (ec)
      return ec;
    ec = atomFromAddr(0, fromAddress, &fromTarget, &offsetInFrom);
    if (ec)
      return ec;
    if (fromTarget != inAtom)
      return make_dynamic_error_code(Twine("SECTDIFF relocation where "
                                           "subtrahend label is not in atom"));
    *kind = ((perms & DefinedAtom::permR_X) == DefinedAtom::permR_X) ? funcRel32
                                                                     : delta32;
    if (*kind == funcRel32) {
      // SECTDIFF relocations are used in i386 codegen where the function
      // prolog does a CALL to the next instruction which POPs the return
      // address into EBX which becomes the pic-base register.  The POP
      // instruction is label the used for the subtrahend in expressions.
      // The funcRel32 kind represents the 32-bit delta to some symbol from
      // the start of the function (atom) containing the funcRel32.
      uint32_t ta = fromAddress + value - toAddress;
      *addend = ta - offsetInFrom;
    } else {
      *addend = fromAddress + value - toAddress;
    }
    return std::error_code();
    break;
  default:
    return make_dynamic_error_code(Twine("unsupported i386 relocation type"));
  }
}

void ArchHandler_x86::applyFixup(Reference::KindNamespace ns,
                                 Reference::KindArch arch,
                                 Reference::KindValue kindValue,
                                 uint64_t addend, uint8_t *location,
                                 uint64_t fixupAddress, uint64_t targetAddress,
                                 uint64_t inAtomAddress) {
  if (ns != Reference::KindNamespace::mach_o)
    return;
  assert(arch == Reference::KindArch::x86);
  int32_t *loc32 = reinterpret_cast<int32_t *>(location);
  int16_t *loc16 = reinterpret_cast<int16_t *>(location);
  switch (kindValue) {
  case branch32:
    write32(*loc32, _swap, (targetAddress - (fixupAddress + 4)) + addend);
    break;
  case branch16:
    write16(*loc16, _swap, (targetAddress - (fixupAddress + 4)) + addend);
    break;
  case pointer32:
  case abs32:
    write32(*loc32, _swap, targetAddress + addend);
    break;
  case funcRel32:
    write32(*loc32, _swap, targetAddress - inAtomAddress + addend); // FIXME
    break;
  case delta32:
    write32(*loc32, _swap, targetAddress - fixupAddress + addend);
    break;
  case lazyPointer:
  case lazyImmediateLocation:
    // do nothing
    break;
  default:
    llvm_unreachable("invalid x86 Reference Kind");
    break;
  }
}

std::unique_ptr<mach_o::ArchHandler> ArchHandler::create_x86() {
  return std::unique_ptr<mach_o::ArchHandler>(new ArchHandler_x86());
}

} // namespace mach_o
} // namespace lld
