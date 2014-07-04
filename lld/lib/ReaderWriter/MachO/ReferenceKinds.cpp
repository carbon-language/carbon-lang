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
#include "Atoms.h"

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

int16_t KindHandler::readS16(bool swap, const uint8_t *addr) {
  return read16(swap, *reinterpret_cast<const uint16_t*>(addr));
}

int32_t KindHandler::readS32(bool swap, const uint8_t *addr) {
  return read32(swap, *reinterpret_cast<const uint32_t*>(addr));
}

uint32_t KindHandler::readU32(bool swap, const uint8_t *addr) {
  return read32(swap, *reinterpret_cast<const uint32_t*>(addr));
}

int64_t KindHandler::readS64(bool swap, const uint8_t *addr) {
  return read64(swap, *reinterpret_cast<const uint64_t*>(addr));
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
  Reference::KindValue kind = ref.kindValue();
  return (kind == pointer64 || kind == pointer64Anon);
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
                                    uint64_t targetAddress,
                                    uint64_t inAtomAddress) {
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

bool KindHandler_x86::isCallSite(const Reference &ref) {
  return (ref.kindValue() == branch32);
}

bool KindHandler_x86::isPointer(const Reference &ref) {
  return (ref.kindValue() == pointer32);
}

bool KindHandler_x86::isLazyImmediate(const Reference &ref) {
  return (ref.kindValue() == lazyImmediateLocation);
}

bool KindHandler_x86::isLazyTarget(const Reference &ref) {
  return (ref.kindValue() == lazyPointer);
}


bool KindHandler_x86::isPairedReloc(const Relocation &reloc) {
  if (!reloc.scattered)
    return false;
  return (reloc.type == GENERIC_RELOC_LOCAL_SECTDIFF) || 
         (reloc.type == GENERIC_RELOC_SECTDIFF);
}


std::error_code
KindHandler_x86::getReferenceInfo(const Relocation &reloc,
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
  case GENERIC_RELOC_VANILLA | rPcRel  | rLength4:
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
  case GENERIC_RELOC_VANILLA | rPcRel  | rLength2:
    // ex: callw _foo (and _foo defined)
    *kind = branch16;
    targetAddress = fixupAddress + 2 + readS16(swap, fixupContent);
    return atomFromAddress(reloc.symbol, targetAddress, target, addend);
    break;
  case GENERIC_RELOC_VANILLA  | rExtern | rLength4:
    // ex: movl	_foo, %eax   (and _foo undefined)
    // ex: .long _foo        (and _foo undefined)
    perms = inAtom->permissions();
    *kind = ((perms & DefinedAtom::permR_X) == DefinedAtom::permR_X)
                                                            ? abs32 : pointer32;
    if (E ec = atomFromSymbolIndex(reloc.symbol, target))
      return ec;
    *addend = readU32(swap, fixupContent);
    break;
  case GENERIC_RELOC_VANILLA  | rLength4:
    // ex: movl	_foo, %eax   (and _foo defined)
    // ex: .long _foo        (and _foo defined)
    perms = inAtom->permissions();
    *kind = ((perms & DefinedAtom::permR_X) == DefinedAtom::permR_X)
                                                            ? abs32 : pointer32;
    targetAddress = readU32(swap, fixupContent);
    return atomFromAddress(reloc.symbol, targetAddress, target, addend);
    break;
  default:
    return make_dynamic_error_code(Twine("unsupported i386 relocation type"));
  }
  return std::error_code();
}


std::error_code
KindHandler_x86::getPairReferenceInfo(const normalized::Relocation &reloc1,
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
  switch(relocPattern(reloc1) << 16 | relocPattern(reloc2)) {
  case ((GENERIC_RELOC_SECTDIFF       | rScattered | rLength4) << 16 |
         GENERIC_RELOC_PAIR           | rScattered | rLength4):
  case ((GENERIC_RELOC_LOCAL_SECTDIFF | rScattered | rLength4) << 16 |
         GENERIC_RELOC_PAIR           | rScattered | rLength4):
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
    *kind = ((perms & DefinedAtom::permR_X) == DefinedAtom::permR_X)
                                                          ? funcRel32 : delta32;
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

void KindHandler_x86::applyFixup(Reference::KindNamespace ns,
                                 Reference::KindArch arch,
                                 Reference::KindValue kindValue,
                                 uint64_t addend, uint8_t *location,
                                 uint64_t fixupAddress,
                                 uint64_t targetAddress,
                                 uint64_t inAtomAddress) {
  if (ns != Reference::KindNamespace::mach_o)
    return;
  assert(arch == Reference::KindArch::x86);
  int32_t *loc32 = reinterpret_cast<int32_t*>(location);
  int16_t *loc16 = reinterpret_cast<int16_t*>(location);
  // FIXME: these writes may need a swap.
  switch (kindValue) {
  case branch32:
      *loc32 = (targetAddress - (fixupAddress+4)) + addend;
      break;
  case branch16:
      *loc16 = (targetAddress - (fixupAddress+4)) + addend;
      break;
  case pointer32:
  case abs32:
      *loc32 = targetAddress + addend;
      break;
  case funcRel32:
      *loc32 = targetAddress - inAtomAddress + addend; // FIXME
      break;
  case delta32:
      *loc32 = targetAddress - fixupAddress + addend;
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


//===----------------------------------------------------------------------===//
//  KindHandler_arm
//===----------------------------------------------------------------------===//

KindHandler_arm::~KindHandler_arm() {
}

const Registry::KindStrings KindHandler_arm::kindStrings[] = {
  LLD_KIND_STRING_ENTRY(thumb_b22),
  LLD_KIND_STRING_ENTRY(thumb_movw),
  LLD_KIND_STRING_ENTRY(thumb_movt),
  LLD_KIND_STRING_ENTRY(thumb_movw_funcRel),
  LLD_KIND_STRING_ENTRY(thumb_movt_funcRel),
  LLD_KIND_STRING_ENTRY(arm_b24),
  LLD_KIND_STRING_ENTRY(arm_movw),
  LLD_KIND_STRING_ENTRY(arm_movt),
  LLD_KIND_STRING_ENTRY(arm_movw_funcRel),
  LLD_KIND_STRING_ENTRY(arm_movt_funcRel),
  LLD_KIND_STRING_ENTRY(pointer32),
  LLD_KIND_STRING_ENTRY(delta32),
  LLD_KIND_STRING_ENTRY(lazyPointer),
  LLD_KIND_STRING_ENTRY(lazyImmediateLocation),
  LLD_KIND_STRING_END
};

bool KindHandler_arm::isCallSite(const Reference &ref) {
  return (ref.kindValue() == thumb_b22) ||
         (ref.kindValue() == arm_b24);
}

bool KindHandler_arm::isPointer(const Reference &ref) {
  return (ref.kindValue() == pointer32);
}

bool KindHandler_arm::isLazyImmediate(const Reference &ref) {
  return (ref.kindValue() == lazyImmediateLocation);
}

bool KindHandler_arm::isLazyTarget(const Reference &ref) {
  return (ref.kindValue() == lazyPointer);
}

bool KindHandler_arm::isPairedReloc(const Relocation &reloc) {
  switch (reloc.type) {
  case ARM_RELOC_SECTDIFF:
  case ARM_RELOC_LOCAL_SECTDIFF:
  case ARM_RELOC_HALF_SECTDIFF:
  case ARM_RELOC_HALF:
    return true;
  default:
    return false;
  }
}


int32_t KindHandler_arm::getDisplacementFromThumbBranch(uint32_t instruction) {
  uint32_t s = (instruction >> 10) & 0x1;
  uint32_t j1 = (instruction >> 29) & 0x1;
  uint32_t j2 = (instruction >> 27) & 0x1;
  uint32_t imm10 = instruction & 0x3FF;
  uint32_t imm11 = (instruction >> 16) & 0x7FF;
  uint32_t i1 = (j1 == s);
  uint32_t i2 = (j2 == s);
  uint32_t dis = (s << 24) | (i1 << 23) | (i2 << 22) 
               | (imm10 << 12) | (imm11 << 1);
  int32_t sdis = dis;
  if (s)
    return (sdis | 0xFE000000);
  else
    return sdis;
}

int32_t KindHandler_arm::getDisplacementFromArmBranch(uint32_t instruction) {
  // Sign-extend imm24
  int32_t displacement = (instruction & 0x00FFFFFF) << 2;
  if ( (displacement & 0x02000000) != 0 )
    displacement |= 0xFC000000;
  // If this is BLX and H bit set, add 2.
  if ((instruction & 0xFF000000) == 0xFB000000)
    displacement += 2;
  return displacement;
}


uint16_t KindHandler_arm::getWordFromThumbMov(uint32_t instruction) {
  uint32_t i =    ((instruction & 0x00000400) >> 10);
  uint32_t imm4 =  (instruction & 0x0000000F);
  uint32_t imm3 = ((instruction & 0x70000000) >> 28);
  uint32_t imm8 = ((instruction & 0x00FF0000) >> 16);
  return (imm4 << 12) | (i << 11) | (imm3 << 8) | imm8;
}

uint16_t KindHandler_arm::getWordFromArmMov(uint32_t instruction) {
  uint32_t imm4 = ((instruction & 0x000F0000) >> 16);
  uint32_t imm12 = (instruction & 0x00000FFF);
  return (imm4 << 12) | imm12;
}

uint32_t KindHandler_arm::clearThumbBit(uint32_t value, const Atom* target) {
  // The assembler often adds one to the address of a thumb function.
  // We need to undo that so it does not look like an addend.
  if (value & 1) {
    if (isa<DefinedAtom>(target)) {
      const MachODefinedAtom *machoTarget = reinterpret_cast<
                                              const MachODefinedAtom*>(target);
      if (machoTarget->isThumb())
        value &= -2;  // mask off thumb-bit
    }
  }
  return value;
}

std::error_code
KindHandler_arm::getReferenceInfo(const Relocation &reloc,
                                  const DefinedAtom *inAtom,
                                  uint32_t offsetInAtom,
                                  uint64_t fixupAddress, bool swap,
                                  FindAtomBySectionAndAddress atomFromAddress,
                                  FindAtomBySymbolIndex atomFromSymbolIndex,
                                  Reference::KindValue *kind,
                                  const lld::Atom **target,
                                  Reference::Addend *addend) {
  typedef std::error_code E;
  const uint8_t *fixupContent = &inAtom->rawContent()[offsetInAtom];
  uint64_t targetAddress;
  uint32_t instruction = readU32(swap, fixupContent);
  int32_t displacement;
  switch (relocPattern(reloc)) {
  case ARM_THUMB_RELOC_BR22 | rPcRel | rExtern | rLength4:
    // ex: bl _foo (and _foo is undefined)
    *kind = thumb_b22;
    if (E ec = atomFromSymbolIndex(reloc.symbol, target))
      return ec;
    // Instruction contains branch to addend.
    displacement = getDisplacementFromThumbBranch(instruction);
    *addend = fixupAddress + 4 + displacement;
    return std::error_code();
  case ARM_THUMB_RELOC_BR22 | rPcRel           | rLength4:
    // ex: bl _foo (and _foo is defined)
    *kind = thumb_b22;
    displacement = getDisplacementFromThumbBranch(instruction);
    targetAddress = fixupAddress + 4 + displacement;
    return atomFromAddress(reloc.symbol, targetAddress, target, addend);
  case ARM_THUMB_RELOC_BR22 | rScattered | rPcRel | rLength4:
    // ex: bl _foo+4 (and _foo is defined)
    *kind = thumb_b22;
    displacement = getDisplacementFromThumbBranch(instruction);
    targetAddress = fixupAddress + 4 + displacement;
    if (E ec = atomFromAddress(0, reloc.value, target, addend))
      return ec;
    // reloc.value is target atom's address.  Instruction contains branch
    // to atom+addend.
    *addend += (targetAddress - reloc.value);
    return std::error_code();
  case ARM_RELOC_BR24 | rPcRel | rExtern | rLength4:
    // ex: bl _foo (and _foo is undefined)
    *kind = arm_b24;
    if (E ec = atomFromSymbolIndex(reloc.symbol, target))
      return ec;
    // Instruction contains branch to addend.
    displacement = getDisplacementFromArmBranch(instruction);
    *addend = fixupAddress + 8 + displacement;
    return std::error_code();
  case ARM_RELOC_BR24 | rPcRel           | rLength4:
    // ex: bl _foo (and _foo is defined)
    *kind = arm_b24;
    displacement = getDisplacementFromArmBranch(instruction);
    targetAddress = fixupAddress + 8 + displacement;
    return atomFromAddress(reloc.symbol, targetAddress, target, addend);
  case ARM_RELOC_BR24 | rScattered | rPcRel | rLength4:
    // ex: bl _foo+4 (and _foo is defined)
    *kind = arm_b24;
    displacement = getDisplacementFromArmBranch(instruction);
    targetAddress = fixupAddress + 8 + displacement;
    if (E ec = atomFromAddress(0, reloc.value, target, addend))
      return ec;
    // reloc.value is target atom's address.  Instruction contains branch
    // to atom+addend.
    *addend += (targetAddress - reloc.value);
    return std::error_code();
  case ARM_RELOC_VANILLA | rExtern   | rLength4:
    // ex: .long _foo (and _foo is undefined)
    *kind = pointer32;
    if (E ec = atomFromSymbolIndex(reloc.symbol, target))
      return ec;
    *addend = instruction;
    return std::error_code();
  case ARM_RELOC_VANILLA             | rLength4:
    // ex: .long _foo (and _foo is defined)
    *kind = pointer32;
    if (E ec = atomFromAddress(reloc.symbol, instruction, target, addend))
      return ec;
    *addend = clearThumbBit((uint32_t)*addend, *target);
    return std::error_code();
  case ARM_RELOC_VANILLA | rScattered | rLength4:
    // ex: .long _foo+a (and _foo is defined)
    *kind = pointer32;
    if (E ec = atomFromAddress(0, reloc.value, target, addend))
      return ec;
   *addend += (clearThumbBit(instruction, *target) - reloc.value);
    return std::error_code();
  default:
    return make_dynamic_error_code(Twine("unsupported arm relocation type"));
  }
  return std::error_code();
}


std::error_code
KindHandler_arm::getPairReferenceInfo(const normalized::Relocation &reloc1,
                                     const normalized::Relocation &reloc2,
                                     const DefinedAtom *inAtom,
                                     uint32_t offsetInAtom,
                                     uint64_t fixupAddress, bool swap,
                                     FindAtomBySectionAndAddress atomFromAddr,
                                     FindAtomBySymbolIndex atomFromSymbolIndex,
                                     Reference::KindValue *kind,
                                     const lld::Atom **target,
                                     Reference::Addend *addend) {
  bool pointerDiff = false;
  bool funcRel;
  bool top;
  bool thumbReloc;
  switch(relocPattern(reloc1) << 16 | relocPattern(reloc2)) {
  case ((ARM_RELOC_HALF_SECTDIFF  | rScattered | rLength4) << 16 |
         ARM_RELOC_PAIR           | rScattered | rLength4):
    // ex: movw	r1, :lower16:(_x-L1) [thumb mode]
    *kind = thumb_movw_funcRel;
    funcRel = true;
    top = false;
    thumbReloc = true;
    break;
  case ((ARM_RELOC_HALF_SECTDIFF  | rScattered | rLength8) << 16 |
         ARM_RELOC_PAIR           | rScattered | rLength8):
    // ex: movt	r1, :upper16:(_x-L1) [thumb mode]
    *kind = thumb_movt_funcRel;
    funcRel = true;
    top = true;
    thumbReloc = true;
    break;
  case ((ARM_RELOC_HALF_SECTDIFF  | rScattered | rLength1) << 16 |
         ARM_RELOC_PAIR           | rScattered | rLength1):
    // ex: movw	r1, :lower16:(_x-L1) [arm mode]
    *kind = arm_movw_funcRel;
    funcRel = true;
    top = false;
    thumbReloc = false;
    break;
  case ((ARM_RELOC_HALF_SECTDIFF  | rScattered | rLength2) << 16 |
         ARM_RELOC_PAIR           | rScattered | rLength2):
    // ex: movt	r1, :upper16:(_x-L1) [arm mode]
    *kind = arm_movt_funcRel;
    funcRel = true;
    top = true;
    thumbReloc = false;
    break;
  case ((ARM_RELOC_HALF     | rLength4) << 16 |
         ARM_RELOC_PAIR     | rLength4):
    // ex: movw	r1, :lower16:_x [thumb mode]
    *kind = thumb_movw;
    funcRel = false;
    top = false;
    thumbReloc = true;
    break;
  case ((ARM_RELOC_HALF     | rLength8) << 16 |
         ARM_RELOC_PAIR     | rLength8):
    // ex: movt	r1, :upper16:_x [thumb mode]
    *kind = thumb_movt;
    funcRel = false;
    top = true;
    thumbReloc = true;
    break;
  case ((ARM_RELOC_HALF     | rLength1) << 16 |
         ARM_RELOC_PAIR     | rLength1):
    // ex: movw	r1, :lower16:_x [arm mode]
    *kind = arm_movw;
    funcRel = false;
    top = false;
    thumbReloc = false;
    break;
  case ((ARM_RELOC_HALF     | rLength2) << 16 |
         ARM_RELOC_PAIR     | rLength2):
    // ex: movt	r1, :upper16:_x [arm mode]
    *kind = arm_movt;
    funcRel = false;
    top = true;
    thumbReloc = false;
    break;
  case ((ARM_RELOC_HALF | rScattered  | rLength4) << 16 |
         ARM_RELOC_PAIR               | rLength4):
    // ex: movw	r1, :lower16:_x+a [thumb mode]
    *kind = thumb_movw;
    funcRel = false;
    top = false;
    thumbReloc = true;
    break;
  case ((ARM_RELOC_HALF | rScattered  | rLength8) << 16 |
         ARM_RELOC_PAIR               | rLength8):
    // ex: movt	r1, :upper16:_x+a [thumb mode]
    *kind = thumb_movt;
    funcRel = false;
    top = true;
    thumbReloc = true;
    break;
  case ((ARM_RELOC_HALF | rScattered  | rLength1) << 16 |
         ARM_RELOC_PAIR               | rLength1):
    // ex: movw	r1, :lower16:_x+a [arm mode]
    *kind = arm_movw;
    funcRel = false;
    top = false;
    thumbReloc = false;
    break;
  case ((ARM_RELOC_HALF | rScattered  | rLength2) << 16 |
         ARM_RELOC_PAIR               | rLength2):
    // ex: movt	r1, :upper16:_x+a [arm mode]
    *kind = arm_movt;
    funcRel = false;
    top = true;
    thumbReloc = false;
    break;
  case ((ARM_RELOC_HALF | rExtern   | rLength4) << 16 |
         ARM_RELOC_PAIR             | rLength4):
    // ex: movw	r1, :lower16:_undef [thumb mode]
    *kind = thumb_movw;
    funcRel = false;
    top = false;
    thumbReloc = true;
    break;
  case ((ARM_RELOC_HALF | rExtern   | rLength8) << 16 |
         ARM_RELOC_PAIR             | rLength8):
    // ex: movt	r1, :upper16:_undef [thumb mode]
    *kind = thumb_movt;
    funcRel = false;
    top = true;
    thumbReloc = true;
    break;
  case ((ARM_RELOC_HALF | rExtern   | rLength1) << 16 |
         ARM_RELOC_PAIR             | rLength1):
    // ex: movw	r1, :lower16:_undef [arm mode]
    *kind = arm_movw;
    funcRel = false;
    top = false;
    thumbReloc = false;
    break;
  case ((ARM_RELOC_HALF | rExtern   | rLength2) << 16 |
         ARM_RELOC_PAIR             | rLength2):
    // ex: movt	r1, :upper16:_undef [arm mode]
    *kind = arm_movt;
    funcRel = false;
    top = true;
    thumbReloc = false;
    break;
  case ((ARM_RELOC_SECTDIFF       | rScattered | rLength4) << 16 |
         ARM_RELOC_PAIR           | rScattered | rLength4):
  case ((ARM_RELOC_LOCAL_SECTDIFF | rScattered | rLength4) << 16 |
         ARM_RELOC_PAIR           | rScattered | rLength4):
    // ex: .long _foo - .
    pointerDiff = true;
    break;
  default:
    return make_dynamic_error_code(Twine("unsupported arm relocation pair"));
  }
  const uint8_t *fixupContent = &inAtom->rawContent()[offsetInAtom];
  std::error_code ec;
  uint32_t instruction = readU32(swap, fixupContent);
  uint32_t value;
  uint32_t fromAddress;
  uint32_t toAddress;
  uint16_t instruction16;
  uint16_t other16;
  const lld::Atom *fromTarget;
  Reference::Addend offsetInTo;
  Reference::Addend offsetInFrom;
  if (pointerDiff) {
    toAddress = reloc1.value;
    fromAddress = reloc2.value;
    ec = atomFromAddr(0, toAddress, target, &offsetInTo);
    if (ec)
      return ec;
    ec = atomFromAddr(0, fromAddress, &fromTarget, &offsetInFrom);
    if (ec)
      return ec;
    if (fromTarget != inAtom)
      return make_dynamic_error_code(Twine("SECTDIFF relocation where "
                                     "subtrahend label is not in atom"));
    *kind = delta32;
    value = clearThumbBit(instruction, *target);
    *addend = value - (toAddress - fromAddress);
  } else if (funcRel) {
    toAddress = reloc1.value;
    fromAddress = reloc2.value;
    ec = atomFromAddr(0, toAddress, target, &offsetInTo);
    if (ec)
      return ec;
    ec = atomFromAddr(0, fromAddress, &fromTarget, &offsetInFrom);
    if (ec)
      return ec;
    if (fromTarget != inAtom)
      return make_dynamic_error_code(Twine("ARM_RELOC_HALF_SECTDIFF relocation "
                                     "where subtrahend label is not in atom"));
    other16 = (reloc2.offset & 0xFFFF);
    if (thumbReloc)
      instruction16 = getWordFromThumbMov(instruction);
    else
      instruction16 = getWordFromArmMov(instruction);
    if (top)
      value = (instruction16 << 16) | other16;
    else
      value = (other16 << 16) | instruction16;
    value = clearThumbBit(value, *target);
    int64_t ta = (int64_t)value - (toAddress - fromAddress);
    *addend = ta - offsetInFrom;
    return std::error_code();
  } else {
    uint32_t sectIndex;
    if (thumbReloc)
      instruction16 = getWordFromThumbMov(instruction);
    else
      instruction16 = getWordFromArmMov(instruction);
    other16 = (reloc2.offset & 0xFFFF);
    if (top)
      value = (instruction16 << 16) | other16;
    else
      value = (other16 << 16) | instruction16;
    if (reloc1.isExtern) {
      ec = atomFromSymbolIndex(reloc1.symbol, target);
      if (ec)
        return ec;
      *addend = value;
    } else {
      if (reloc1.scattered) {
        toAddress = reloc1.value;
        sectIndex = 0;
      } else {
        toAddress = value;
        sectIndex = reloc1.symbol;
      }
      ec = atomFromAddr(sectIndex, toAddress, target, &offsetInTo);
      if (ec)
        return ec;
      *addend = value - toAddress;
    }
  }
  
  return std::error_code();
}



void KindHandler_arm::applyFixup(Reference::KindNamespace ns,
                                 Reference::KindArch arch,
                                 Reference::KindValue kindValue,
                                 uint64_t addend, uint8_t *location,
                                 uint64_t fixupAddress,
                                 uint64_t targetAddress,
                                 uint64_t inAtomAddress) {
  if (ns != Reference::KindNamespace::mach_o)
    return;
  assert(arch == Reference::KindArch::ARM);
  //int32_t *loc32 = reinterpret_cast<int32_t*>(location);
  // FIXME: these writes may need a swap.
  switch (kindValue) {
  case thumb_b22:
      // FIXME
    break;
  case thumb_movw:
      // FIXME
      break;
  case thumb_movt:
      // FIXME
      break;
  case thumb_movw_funcRel:
      // FIXME
      break;
  case thumb_movt_funcRel:
      // FIXME
      break;
  case arm_b24:
      // FIXME
      break;
  case arm_movw:
      // FIXME
      break;
  case arm_movt:
      // FIXME
      break;
  case arm_movw_funcRel:
      // FIXME
      break;
  case arm_movt_funcRel:
      // FIXME
      break;
  case pointer32:
      // FIXME
      break;
  case delta32:
      // FIXME
      break;
  case lazyPointer:
  case lazyImmediateLocation:
      // do nothing
      break;
  case invalid:
    llvm_unreachable("invalid ARM Reference Kind");
      break;
  }
}


} // namespace mach_o
} // namespace lld



