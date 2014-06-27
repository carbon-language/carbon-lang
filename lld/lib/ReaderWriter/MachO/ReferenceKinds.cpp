//===- lib/FileFormat/MachO/ReferenceKinds.cpp ----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "ReferenceKinds.h"
#include "MachONormalizedFileBinaryUtils.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"

#include "llvm/Support/ErrorHandling.h"

using namespace llvm::MachO;
using namespace lld::mach_o::normalized;

namespace lld {
namespace mach_o {

//===----------------------------------------------------------------------===//
//  KindHandler
//===----------------------------------------------------------------------===//

KindHandler::KindHandler() {
}

KindHandler::~KindHandler() {
}

std::unique_ptr<mach_o::KindHandler>
KindHandler::create(MachOLinkingContext::Arch arch) {
  switch (arch) {
  case MachOLinkingContext::arch_x86_64:
    return std::unique_ptr<mach_o::KindHandler>(new KindHandler_x86_64());
  case MachOLinkingContext::arch_x86:
    return std::unique_ptr<mach_o::KindHandler>(new KindHandler_x86());
    case MachOLinkingContext::arch_armv6:
    case MachOLinkingContext::arch_armv7:
    case MachOLinkingContext::arch_armv7s:
      return std::unique_ptr<mach_o::KindHandler>(new KindHandler_arm());
    default:
      llvm_unreachable("Unknown arch");
  }
}

KindHandler::RelocPattern KindHandler::relocPattern(const Relocation &reloc) {
  assert((reloc.type & 0xFFF0) == 0);
  uint16_t result = reloc.type;
  if (reloc.scattered)
    result |= rScattered;
  if (reloc.pcRel)
    result |= rPcRel;
  if (reloc.isExtern)
    result |= rExtern;
  switch(reloc.length) {
  case 0:
    break;
  case 1:
    result |= rLength2;
    break;
  case 2:
    result |= rLength4;
    break;
  case 3:
    result |= rLength8;
    break;
  default:
    llvm_unreachable("bad r_length");
  }
  return result;
}

bool KindHandler::isPairedReloc(const Relocation &reloc) {
  llvm_unreachable("abstract");
}

std::error_code 
KindHandler::getReferenceInfo(const Relocation &reloc,
                                    const DefinedAtom *inAtom,
                                    uint32_t offsetInAtom,
                                    uint64_t fixupAddress, bool swap,
                                    FindAtomBySectionAndAddress atomFromAddress,
                                    FindAtomBySymbolIndex atomFromSymbolIndex,
                                    Reference::KindValue *kind, 
                                    const lld::Atom **target, 
                                    Reference::Addend *addend) {
  llvm_unreachable("abstract");
}

std::error_code 
KindHandler::getPairReferenceInfo(const normalized::Relocation &reloc1,
                           const normalized::Relocation &reloc2,
                           const DefinedAtom *inAtom,
                           uint32_t offsetInAtom,
                           uint64_t fixupAddress, bool swap,
                           FindAtomBySectionAndAddress atomFromAddress,
                           FindAtomBySymbolIndex atomFromSymbolIndex,
                           Reference::KindValue *kind, 
                           const lld::Atom **target, 
                           Reference::Addend *addend) {
  llvm_unreachable("abstract");
}

//===----------------------------------------------------------------------===//
//  KindHandler_x86_64
//===----------------------------------------------------------------------===//

KindHandler_x86_64::~KindHandler_x86_64() {
}

const Registry::KindStrings KindHandler_x86_64::kindStrings[] = {
  LLD_KIND_STRING_ENTRY(invalid),
  LLD_KIND_STRING_ENTRY(branch32),
  LLD_KIND_STRING_ENTRY(ripRel32),
  LLD_KIND_STRING_ENTRY(ripRel32Minus1),
  LLD_KIND_STRING_ENTRY(ripRel32Minus2),
  LLD_KIND_STRING_ENTRY(ripRel32Minus4),
  LLD_KIND_STRING_ENTRY(ripRel32Anon),
  LLD_KIND_STRING_ENTRY(ripRel32GotLoad),
  LLD_KIND_STRING_ENTRY(ripRel32GotLoadNowLea),
  LLD_KIND_STRING_ENTRY(ripRel32Got),
  LLD_KIND_STRING_ENTRY(lazyPointer),
  LLD_KIND_STRING_ENTRY(lazyImmediateLocation),
  LLD_KIND_STRING_ENTRY(pointer64),
  LLD_KIND_STRING_ENTRY(pointer64Anon),
  LLD_KIND_STRING_ENTRY(delta32),
  LLD_KIND_STRING_ENTRY(delta64),
  LLD_KIND_STRING_ENTRY(delta32Anon),
  LLD_KIND_STRING_ENTRY(delta64Anon),
  LLD_KIND_STRING_END
};
 
bool KindHandler_x86_64::isCallSite(const Reference &ref) {
  if (ref.kindNamespace() != Reference::KindNamespace::mach_o)
    return false;
  assert(ref.kindArch() == Reference::KindArch::x86_64);
  return (ref.kindValue() == branch32);
}

bool KindHandler_x86_64::isPointer(const Reference &ref) {
  if (ref.kindNamespace() != Reference::KindNamespace::mach_o)
    return false;
  assert(ref.kindArch() == Reference::KindArch::x86_64);
  return (ref.kindValue() == pointer64);
}

bool KindHandler_x86_64::isLazyImmediate(const Reference &ref) {
  if (ref.kindNamespace() != Reference::KindNamespace::mach_o)
    return false;
  assert(ref.kindArch() == Reference::KindArch::x86_64);
  return (ref.kindValue() == lazyImmediateLocation);
}

bool KindHandler_x86_64::isLazyTarget(const Reference &ref) {
  if (ref.kindNamespace() != Reference::KindNamespace::mach_o)
    return false;
  assert(ref.kindArch() == Reference::KindArch::x86_64);
  return (ref.kindValue() == lazyPointer);
}

bool KindHandler_x86_64::isPairedReloc(const Relocation &reloc) {
  return (reloc.type == X86_64_RELOC_SUBTRACTOR);
}

static int32_t readS32(bool swap, const uint8_t *addr) {
  return read32(swap, *reinterpret_cast<const uint32_t*>(addr));
}

static int64_t readS64(bool swap, const uint8_t *addr) {
  return read64(swap, *reinterpret_cast<const uint64_t*>(addr));
}

Reference::KindValue 
KindHandler_x86_64::kindFromReloc(const Relocation &reloc) {
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
KindHandler_x86_64::getReferenceInfo(const Relocation &reloc,
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
KindHandler_x86_64::kindFromRelocPair(const normalized::Relocation &reloc1,
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
KindHandler_x86_64::getPairReferenceInfo(const normalized::Relocation &reloc1,
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



void KindHandler_x86_64::applyFixup(Reference::KindNamespace ns,
                                    Reference::KindArch arch,
                                    Reference::KindValue kindValue,
                                    uint64_t addend, uint8_t *location,
                                    uint64_t fixupAddress,
                                    uint64_t targetAddress) {
  if (ns != Reference::KindNamespace::mach_o)
    return;
  assert(arch == Reference::KindArch::x86_64);
  int32_t *loc32 = reinterpret_cast<int32_t*>(location);
  uint64_t* loc64 = reinterpret_cast<uint64_t*>(location);
  switch (kindValue) {
  case branch32:
  case ripRel32:
  case ripRel32Got:
  case ripRel32GotLoad:
    *loc32 = (targetAddress - (fixupAddress+4)) + addend;
    break;
  case pointer64:
  case pointer64Anon:
    *loc64 = targetAddress + addend;
    break;
  case ripRel32Minus1:
    *loc32 = (targetAddress - (fixupAddress+5)) + addend;
    break;
  case ripRel32Minus2:
    *loc32 = (targetAddress - (fixupAddress+6)) + addend;
    break;
  case ripRel32Minus4:
    *loc32 = (targetAddress - (fixupAddress+8)) + addend;
    break;
  case delta32:
  case delta32Anon:
   *loc32 = (targetAddress - fixupAddress) + addend;
    break;
  case delta64:
  case delta64Anon:
    *loc64 = (targetAddress - fixupAddress) + addend;
    break;
  case ripRel32GotLoadNowLea:
    // Change MOVQ to LEA
    assert(location[-2] == 0x8B);
    location[-2] = 0x8D;
    *loc32 = (targetAddress - (fixupAddress+4)) + addend;
    break;
  case lazyPointer:
  case lazyImmediateLocation:
    // do nothing
    break;
  default:
    llvm_unreachable("invalid x86_64 Reference Kind");
      break;
  }
}


//===----------------------------------------------------------------------===//
//  KindHandler_x86
//===----------------------------------------------------------------------===//

KindHandler_x86::~KindHandler_x86() {
}

const Registry::KindStrings KindHandler_x86::kindStrings[] = {
  LLD_KIND_STRING_ENTRY(LLD_X86_RELOC_BRANCH32),
  LLD_KIND_STRING_ENTRY(LLD_X86_RELOC_ABS32),
  LLD_KIND_STRING_ENTRY(LLD_X86_RELOC_FUNC_REL32),
  LLD_KIND_STRING_ENTRY(LLD_X86_RELOC_POINTER32),
  LLD_KIND_STRING_ENTRY(LLD_X86_RELOC_LAZY_TARGET),
  LLD_KIND_STRING_ENTRY(LLD_X86_RELOC_LAZY_IMMEDIATE),
  LLD_KIND_STRING_END
};

bool KindHandler_x86::isCallSite(const Reference &ref) {
  return (ref.kindValue() == LLD_X86_RELOC_BRANCH32);
}

bool KindHandler_x86::isPointer(const Reference &ref) {
  return (ref.kindValue() == LLD_X86_RELOC_POINTER32);
}

bool KindHandler_x86::isLazyImmediate(const Reference &ref) {
  return (ref.kindValue() == LLD_X86_RELOC_LAZY_TARGET);
}

bool KindHandler_x86::isLazyTarget(const Reference &ref) {
  return (ref.kindValue() == LLD_X86_RELOC_LAZY_TARGET);
}

void KindHandler_x86::applyFixup(Reference::KindNamespace ns,
                                 Reference::KindArch arch,
                                 Reference::KindValue kindValue,
                                 uint64_t addend, uint8_t *location,
                                 uint64_t fixupAddress,
                                 uint64_t targetAddress) {
  if (ns != Reference::KindNamespace::mach_o)
    return;
  assert(arch == Reference::KindArch::x86);
  int32_t *loc32 = reinterpret_cast<int32_t*>(location);
  switch (kindValue) {
  case LLD_X86_RELOC_BRANCH32:
      *loc32 = (targetAddress - (fixupAddress+4)) + addend;
      break;
  case LLD_X86_RELOC_POINTER32:
  case LLD_X86_RELOC_ABS32:
      *loc32 = targetAddress + addend;
      break;
  case LLD_X86_RELOC_FUNC_REL32:
      *loc32 = targetAddress + addend;
      break;
  case LLD_X86_RELOC_LAZY_TARGET:
  case LLD_X86_RELOC_LAZY_IMMEDIATE:
      // do nothing
      break;
  default:
    llvm_unreachable("invalid x86 Reference Kind");
      break;
  }
}


//===----------------------------------------------------------------------===//
//  KindHandler_arm
//===----------------------------------------------------------------------===//

KindHandler_arm::~KindHandler_arm() {
}

const Registry::KindStrings KindHandler_arm::kindStrings[] = {
  LLD_KIND_STRING_ENTRY(ARM_RELOC_BR24),
  LLD_KIND_STRING_ENTRY(ARM_THUMB_RELOC_BR22),
  LLD_KIND_STRING_ENTRY(LLD_ARM_RELOC_THUMB_ABS_LO16),
  LLD_KIND_STRING_ENTRY(LLD_ARM_RELOC_THUMB_ABS_HI16),
  LLD_KIND_STRING_ENTRY(LLD_ARM_RELOC_THUMB_REL_LO16),
  LLD_KIND_STRING_ENTRY(LLD_ARM_RELOC_THUMB_REL_HI16),
  LLD_KIND_STRING_ENTRY(LLD_ARM_RELOC_ABS32),
  LLD_KIND_STRING_ENTRY(LLD_ARM_RELOC_POINTER32),
  LLD_KIND_STRING_ENTRY(LLD_ARM_RELOC_LAZY_TARGET),
  LLD_KIND_STRING_ENTRY(LLD_ARM_RELOC_LAZY_IMMEDIATE),
  LLD_KIND_STRING_END
};

bool KindHandler_arm::isCallSite(const Reference &ref) {
  return (ref.kindValue() == ARM_THUMB_RELOC_BR22) ||
         (ref.kindValue() == ARM_RELOC_BR24);
}

bool KindHandler_arm::isPointer(const Reference &ref) {
  return (ref.kindValue() == LLD_ARM_RELOC_POINTER32);
}

bool KindHandler_arm::isLazyImmediate(const Reference &ref) {
  return (ref.kindValue() == LLD_ARM_RELOC_LAZY_IMMEDIATE);
}

bool KindHandler_arm::isLazyTarget(const Reference &ref) {
  return (ref.kindValue() == LLD_ARM_RELOC_LAZY_TARGET);
}

void KindHandler_arm::applyFixup(Reference::KindNamespace ns,
                                 Reference::KindArch arch,
                                 Reference::KindValue kindValue,
                                 uint64_t addend, uint8_t *location,
                                 uint64_t fixupAddress,
                                 uint64_t targetAddress) {
  if (ns != Reference::KindNamespace::mach_o)
    return;
  assert(arch == Reference::KindArch::ARM);
  //int32_t *loc32 = reinterpret_cast<int32_t*>(location);
  switch (kindValue) {
  case ARM_THUMB_RELOC_BR22:
      // FIXME
      break;
  case ARM_RELOC_BR24:
      // FIXME
      break;
  case LLD_ARM_RELOC_THUMB_ABS_LO16:
      // FIXME
      break;
  case LLD_ARM_RELOC_THUMB_ABS_HI16:
      // FIXME
      break;
  case LLD_ARM_RELOC_THUMB_REL_LO16:
      // FIXME
      break;
  case LLD_ARM_RELOC_THUMB_REL_HI16:
      // FIXME
      break;
  case LLD_ARM_RELOC_ABS32:
      // FIXME
      break;
  case LLD_ARM_RELOC_POINTER32:
      // FIXME
      break;
  case LLD_ARM_RELOC_LAZY_TARGET:
  case LLD_ARM_RELOC_LAZY_IMMEDIATE:
      // do nothing
      break;
  default:
    llvm_unreachable("invalid ARM Reference Kind");
      break;
  }
}


} // namespace mach_o
} // namespace lld



