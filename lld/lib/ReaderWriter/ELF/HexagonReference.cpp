//===- lib/ReaderWriter/ELF/HexagonReference.cpp ----------------------------===//
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

#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ELF.h"

namespace lld {
namespace elf {

//===----------------------------------------------------------------------===//
//  KindHandler_hexagon
//  TODO: more to do here
//===----------------------------------------------------------------------===//

KindHandler_hexagon::~KindHandler_hexagon() {
}

/// \brief The following relocation routines are derived from the
/// Hexagon ABI specification, Section 11.6: Relocation
/// Symbols used:
///  A: Added used to compute the value, r_addend
///  P: Place address of the field being relocated, r_offset
///  S: Value of the symbol whose index resides in the relocation entry.

namespace hexagon {
int reloc_NONE(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  return KindHandler_hexagon::NoError;
}

/// \brief Word32_B22: 0x01ff3ffe : (S + A - P) >> 2 : Verify
int reloc_B22_PCREL(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  int32_t result = (uint32_t)(((S + A) - P)>>2);
  if ((result < 0x200000) && (result > -0x200000)) {
    result = ((result<<1) & 0x3ffe) | ((result<<3) & 0x01ff0000);
    *reinterpret_cast<llvm::support::ulittle32_t *>(location) = result |
                     *reinterpret_cast<llvm::support::ulittle32_t *>(location);
    return KindHandler_hexagon::NoError;
  }
  return KindHandler_hexagon::Overflow;
}

/// \brief Word32_B15: 0x00df20fe : (S + A - P) >> 2 : Verify
int reloc_B15_PCREL(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  int32_t result = (uint32_t)(((S + A) - P)>>2);
  if ((result < 0x8000) && (result > -0x8000)) {
    result = ((result<<1) & 0x20fe) | ((result<<7) & 0x00df0000);
    *reinterpret_cast<llvm::support::ulittle32_t *>(location) = result |
                      *reinterpret_cast<llvm::support::ulittle32_t *>(location);
    return KindHandler_hexagon::NoError;
  }
  return KindHandler_hexagon::Overflow;
}

/// \brief Word32_LO: 0x00c03fff : (S + A) : Truncate
int reloc_LO16(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  uint32_t result = (uint32_t)(S + A);
  result = ((result & 0x3fff) | ((result << 2) & 0x00c00000));
  *reinterpret_cast<llvm::support::ulittle32_t *>(location) = result |
                    *reinterpret_cast<llvm::support::ulittle32_t *>(location);
  return KindHandler_hexagon::NoError;
}

/// \brief Word32_LO: 0x00c03fff : (S + A) >> 16 : Truncate
int reloc_HI16(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  uint32_t result = (uint32_t)((S + A)>>16);
  result = ((result & 0x3fff) | ((result << 2) & 0x00c00000));
  *reinterpret_cast<llvm::support::ulittle32_t *>(location) = result |
                    *reinterpret_cast<llvm::support::ulittle32_t *>(location);
  return KindHandler_hexagon::NoError;
}

/// \brief Word32: 0xffffffff : (S + A) : Truncate
int reloc_32(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  uint32_t result = (uint32_t)(S + A);
  *reinterpret_cast<llvm::support::ulittle32_t *>(location) = result |
                    *reinterpret_cast<llvm::support::ulittle32_t *>(location);
  return KindHandler_hexagon::NoError;
}
} // namespace hexagon

KindHandler_hexagon::KindHandler_hexagon(){
  _fixupHandler[llvm::ELF::R_HEX_B22_PCREL] = hexagon::reloc_B22_PCREL;
  _fixupHandler[llvm::ELF::R_HEX_B15_PCREL] = hexagon::reloc_B15_PCREL;
  _fixupHandler[llvm::ELF::R_HEX_LO16]      = hexagon::reloc_LO16;
  _fixupHandler[llvm::ELF::R_HEX_HI16]      = hexagon::reloc_HI16;
  _fixupHandler[llvm::ELF::R_HEX_32]        = hexagon::reloc_32;
}

Reference::Kind KindHandler_hexagon::stringToKind(StringRef str) {
  return llvm::StringSwitch<Reference::Kind>(str)
    .Case("none",                  none)
    .Case("R_HEX_B22_PCREL", llvm::ELF::R_HEX_B22_PCREL)
    .Case("R_HEX_B15_PCREL", llvm::ELF::R_HEX_B15_PCREL)
    .Case("R_HEX_LO16",      llvm::ELF::R_HEX_LO16)
    .Case("R_HEX_HI16",      llvm::ELF::R_HEX_HI16)
    .Case("R_HEX_32",        llvm::ELF::R_HEX_32)
    .Default(invalid);
}

StringRef KindHandler_hexagon::kindToString(Reference::Kind kind) {
  switch (static_cast<int32_t>(kind)) {
  case llvm::ELF::R_HEX_B22_PCREL:
    return "R_HEX_B22_PCREL";
  case llvm::ELF::R_HEX_B15_PCREL:
    return "R_HEX_B15_PCREL";
  case llvm::ELF::R_HEX_LO16:
    return "R_HEX_LO16";
  case llvm::ELF::R_HEX_HI16:
    return "R_HEX_HI16";
  case llvm::ELF::R_HEX_32:
    return "R_HEX_32";
  default:
    return "none";
  }
}

bool KindHandler_hexagon::isCallSite(Kind kind) {
  llvm_unreachable("Unimplemented: KindHandler_hexagon::isCallSite");
  return false;
}

bool KindHandler_hexagon::isPointer(Kind kind) {
  llvm_unreachable("Unimplemented: KindHandler_hexagon::isPointer");
  return false;
}

bool KindHandler_hexagon::isLazyImmediate(Kind kind) {
  llvm_unreachable("Unimplemented: KindHandler_hexagon::isLazyImmediate");
  return false;
}

bool KindHandler_hexagon::isLazyTarget(Kind kind) {
  llvm_unreachable("Unimplemented: KindHandler_hexagon::isLazyTarget");
  return false;
}

void KindHandler_hexagon::applyFixup(int32_t reloc, uint64_t addend,
                                     uint8_t *location, uint64_t fixupAddress,
                                     uint64_t targetAddress) {
  int error;
  if (_fixupHandler[reloc])
  {
    error = (*_fixupHandler[reloc])(location,
                                    fixupAddress, targetAddress, addend);

    switch ((RelocationError)error) {
    case NoError:
      return;
    case Overflow:
      llvm::report_fatal_error("applyFixup relocation overflow");
      return;
    }
  }
}


} // namespace elf
} // namespace lld
