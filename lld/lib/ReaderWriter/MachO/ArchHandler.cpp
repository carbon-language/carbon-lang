//===- lib/FileFormat/MachO/ArchHandler.cpp -------------------------------===//
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


ArchHandler::ArchHandler() {
}

ArchHandler::~ArchHandler() {
}

std::unique_ptr<mach_o::ArchHandler> ArchHandler::create(
                                               MachOLinkingContext::Arch arch) {
  switch (arch) {
  case MachOLinkingContext::arch_x86_64:
    return create_x86_64();
  case MachOLinkingContext::arch_x86:
    return create_x86();
  case MachOLinkingContext::arch_armv6:
  case MachOLinkingContext::arch_armv7:
  case MachOLinkingContext::arch_armv7s:
    return create_arm();
  case MachOLinkingContext::arch_arm64:
    return create_arm64();
  default:
    llvm_unreachable("Unknown arch");
  }
}


bool ArchHandler::isLazyPointer(const Reference &ref) {
  // A lazy bind entry is needed for a lazy pointer.
  const StubInfo &info = stubInfo();
  if (ref.kindNamespace() != Reference::KindNamespace::mach_o)
    return false;
  if (ref.kindArch() != info.lazyPointerReferenceToFinal.arch)
    return false;
  return (ref.kindValue() == info.lazyPointerReferenceToFinal.kind);
}


ArchHandler::RelocPattern ArchHandler::relocPattern(const Relocation &reloc) {
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

normalized::Relocation
ArchHandler::relocFromPattern(ArchHandler::RelocPattern pattern) {
  normalized::Relocation result;
  result.offset    = 0;
  result.scattered = (pattern & rScattered);
  result.type     = (RelocationInfoType)(pattern & 0xF);
  result.pcRel    = (pattern & rPcRel);
  result.isExtern = (pattern & rExtern);
  result.value    = 0;
  result.symbol    = 0;
  switch (pattern & 0x300) {
  case rLength1:
    result.length = 0;
    break;
  case rLength2:
    result.length = 1;
    break;
  case rLength4:
    result.length = 2;
    break;
  case rLength8:
    result.length = 3;
    break;
  }
  return result;
}

void ArchHandler::appendReloc(normalized::Relocations &relocs, uint32_t offset,
                              uint32_t symbol, uint32_t value,
                              RelocPattern pattern) {
  normalized::Relocation reloc = relocFromPattern(pattern);
  reloc.offset = offset;
  reloc.symbol = symbol;
  reloc.value  = value;
  relocs.push_back(reloc);
}


int16_t ArchHandler::readS16(bool swap, const uint8_t *addr) {
  return read16(swap, *reinterpret_cast<const uint16_t*>(addr));
}

int32_t ArchHandler::readS32(bool swap, const uint8_t *addr) {
  return read32(swap, *reinterpret_cast<const uint32_t*>(addr));
}

uint32_t ArchHandler::readU32(bool swap, const uint8_t *addr) {
  return read32(swap, *reinterpret_cast<const uint32_t*>(addr));
}

int64_t ArchHandler::readS64(bool swap, const uint8_t *addr) {
  return read64(swap, *reinterpret_cast<const uint64_t*>(addr));
}

} // namespace mach_o
} // namespace lld



