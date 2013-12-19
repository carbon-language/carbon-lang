//===- lib/FileFormat/MachO/ReferenceKinds.cpp ----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "ReferenceKinds.h"


#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"

#include "llvm/Support/ErrorHandling.h"

using namespace llvm::MachO;

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

//===----------------------------------------------------------------------===//
//  KindHandler_x86_64
//===----------------------------------------------------------------------===//

KindHandler_x86_64::~KindHandler_x86_64() {
}


const Registry::KindStrings KindHandler_x86_64::kindStrings[] = {
    LLD_KIND_STRING_ENTRY(X86_64_RELOC_UNSIGNED),
    LLD_KIND_STRING_ENTRY(X86_64_RELOC_BRANCH),
    LLD_KIND_STRING_ENTRY(X86_64_RELOC_SIGNED),
    LLD_KIND_STRING_ENTRY(X86_64_RELOC_SIGNED_1),
    LLD_KIND_STRING_ENTRY(X86_64_RELOC_SIGNED_2),
    LLD_KIND_STRING_ENTRY(X86_64_RELOC_SIGNED_4),
    LLD_KIND_STRING_ENTRY(X86_64_RELOC_GOT_LOAD),
    LLD_KIND_STRING_ENTRY(X86_64_RELOC_GOT),
    LLD_KIND_STRING_ENTRY(X86_64_RELOC_TLV),
    LLD_KIND_STRING_ENTRY(LLD_X86_64_RELOC_GOT_LOAD_NOW_LEA),
    LLD_KIND_STRING_ENTRY(LLD_X86_64_RELOC_TLV_NOW_LEA),
    LLD_KIND_STRING_ENTRY(LLD_X86_64_RELOC_LAZY_TARGET),
    LLD_KIND_STRING_ENTRY(LLD_X86_64_RELOC_LAZY_IMMEDIATE),
    LLD_KIND_STRING_END
};

bool KindHandler_x86_64::isCallSite(const Reference &ref) {
  return (ref.kindValue() == X86_64_RELOC_BRANCH);
}

bool KindHandler_x86_64::isPointer(const Reference &ref) {
  return (ref.kindValue() == X86_64_RELOC_UNSIGNED);
}

bool KindHandler_x86_64::isLazyImmediate(const Reference &ref) {
  return (ref.kindValue() == LLD_X86_64_RELOC_LAZY_IMMEDIATE);
}

bool KindHandler_x86_64::isLazyTarget(const Reference &ref) {
  return (ref.kindValue() == LLD_X86_64_RELOC_LAZY_TARGET);
}


void KindHandler_x86_64::applyFixup(Reference::KindNamespace ns, 
                                    Reference::KindArch arch, 
                                    Reference::KindValue kindValue, 
                                    uint64_t addend,
                                    uint8_t *location, uint64_t fixupAddress,
                                    uint64_t targetAddress) {
  if (ns != Reference::KindNamespace::mach_o)
    return;
  assert(arch == Reference::KindArch::x86_64);
  int32_t *loc32 = reinterpret_cast<int32_t*>(location);
  uint64_t* loc64 = reinterpret_cast<uint64_t*>(location);
  switch ( kindValue ) {
    case X86_64_RELOC_BRANCH:
    case X86_64_RELOC_SIGNED:
    case X86_64_RELOC_GOT_LOAD:
    case X86_64_RELOC_GOT:
    case X86_64_RELOC_TLV:
      *loc32 = (targetAddress - (fixupAddress+4)) + addend;
      break;
    case X86_64_RELOC_UNSIGNED:
      *loc64 = targetAddress + addend;
      break;
    case X86_64_RELOC_SIGNED_1:
      *loc32 = (targetAddress - (fixupAddress+5)) + addend;
      break;
    case X86_64_RELOC_SIGNED_2:
      *loc32 = (targetAddress - (fixupAddress+6)) + addend;
      break;
    case X86_64_RELOC_SIGNED_4:
      *loc32 = (targetAddress - (fixupAddress+8)) + addend;
      break;
    case LLD_X86_64_RELOC_SIGNED_32:
      *loc32 = (targetAddress - fixupAddress) + addend;
      break;
    case LLD_X86_64_RELOC_GOT_LOAD_NOW_LEA:
    case LLD_X86_64_RELOC_TLV_NOW_LEA:
      // Change MOVQ to LEA
      assert(location[-2] == 0x8B);
      location[-2] = 0x8D;
      *loc32 = (targetAddress - (fixupAddress+4)) + addend;
      break;
    case LLD_X86_64_RELOC_LAZY_TARGET:
    case LLD_X86_64_RELOC_LAZY_IMMEDIATE:
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
  switch ( kindValue ) {
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



