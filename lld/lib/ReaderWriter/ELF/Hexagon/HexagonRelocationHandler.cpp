//===- lib/ReaderWriter/ELF/Hexagon/HexagonRelocationHandler.cpp ---------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "HexagonTargetHandler.h"
#include "HexagonTargetInfo.h"
#include "HexagonRelocationHandler.h"
#include "HexagonRelocationFunctions.h"


using namespace lld;
using namespace elf;

using namespace llvm::ELF;

namespace {

#define APPLY_RELOC(result)                                                    \
  *reinterpret_cast<llvm::support::ulittle32_t *>(location) =                  \
      result |                                                                 \
      (uint32_t) * reinterpret_cast<llvm::support::ulittle32_t *>(location);

int relocBNPCREL(uint8_t *location, uint64_t P, uint64_t S, uint64_t A,
                 int32_t nBits) {
  int32_t result = (uint32_t)(((S + A) - P) >> 2);
  int32_t range = 1 << nBits;
  if (result < range && result > -range) {
    result = lld::scatterBits<int32_t>(result, FINDV4BITMASK(location));
    APPLY_RELOC(result);
    return 0;
  }
  return 1;
}

/// \brief Word32_LO: 0x00c03fff : (S + A) : Truncate
int relocLO16(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  uint32_t result = (uint32_t)(S + A);
  result = lld::scatterBits<int32_t>(result, 0x00c03fff);
  APPLY_RELOC(result);
  return 0;
}

/// \brief Word32_LO: 0x00c03fff : (S + A) >> 16 : Truncate
int relocHI16(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  uint32_t result = (uint32_t)((S + A) >> 16);
  result = lld::scatterBits<int32_t>(result, 0x00c03fff);
  APPLY_RELOC(result);
  return 0;
}

/// \brief Word32: 0xffffffff : (S + A) : Truncate
int reloc32(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  uint32_t result = (uint32_t)(S + A);
  APPLY_RELOC(result);
  return 0;
}

int reloc32_6_X(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  int64_t result = ((S + A) >> 6);
  int64_t range = ((int64_t)1) << 32;
  if (result > range)
    return 1;
  result = lld::scatterBits<int32_t>(result, 0xfff3fff);
  APPLY_RELOC(result);
  return 0;
}

// R_HEX_B32_PCREL_X
int relocHexB32PCRELX(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  int64_t result = ((S + A - P) >> 6);
  result = lld::scatterBits<int32_t>(result, 0xfff3fff);
  APPLY_RELOC(result);
  return 0;
}

// R_HEX_BN_PCREL_X
int relocHexBNPCRELX(uint8_t *location, uint64_t P, uint64_t S, uint64_t A,
                     int nbits) {
  int32_t result = ((S + A - P) & 0x3f);
  int32_t range = 1 << nbits;
  if (result < range && result > -range) {
    result = lld::scatterBits<int32_t>(result, FINDV4BITMASK(location));
    APPLY_RELOC(result);
    return 0;
  }
  return 1;
}

// R_HEX_6_PCREL_X
int relocHex6PCRELX(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  int32_t result = (S + A - P);
  result = lld::scatterBits<int32_t>(result, FINDV4BITMASK(location));
  APPLY_RELOC(result);
  return 0;
}

// R_HEX_N_X : Word32_U6 : (S + A) : Unsigned Truncate
int relocHex_N_X(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  uint32_t result = (S + A);
  result = lld::scatterBits<uint32_t>(result, FINDV4BITMASK(location));
  APPLY_RELOC(result);
  return 0;
}

// GP REL relocations
int relocHexGPRELN(uint8_t *location, uint64_t P, uint64_t S, uint64_t A,
                   uint64_t GP, int nShiftBits) {
  int32_t result = (int64_t)((S + A - GP) >> nShiftBits);
  int32_t range = 1L << 16;
  if (result <= range) {
    result = lld::scatterBits<uint32_t>(result, FINDV4BITMASK(location));
    APPLY_RELOC(result);
    return 0;
  }
  return 1;
}

/// \brief Word32_LO: 0x00c03fff : (G) : Truncate
int relocHexGOTLO16(uint8_t *location, uint64_t A, uint64_t GOT) {
  int32_t result = (int32_t)(A-GOT);
  result = lld::scatterBits<int32_t>(result, 0x00c03fff);
  APPLY_RELOC(result);
  return 0;
}

/// \brief Word32_LO: 0x00c03fff : (G) >> 16 : Truncate
int relocHexGOTHI16(uint8_t *location, uint64_t A, uint64_t GOT) {
  int32_t result = (int32_t)((A-GOT) >> 16);
  result = lld::scatterBits<int32_t>(result, 0x00c03fff);
  APPLY_RELOC(result);
  return 0;
}

/// \brief Word32: 0xffffffff : (G) : Truncate
int relocHexGOT32(uint8_t *location, uint64_t A, uint64_t GOT) {
  int32_t result = (int32_t)(GOT - A);
  APPLY_RELOC(result);
  return 0;
}

/// \brief Word32_U16 : (G) : Truncate
int relocHexGOT16(uint8_t *location, uint64_t A, uint64_t GOT) {
  int32_t result = (int32_t)(GOT-A);
  int32_t range = 1L << 16;
  if (result <= range) {
    result = lld::scatterBits<int32_t>(result, FINDV4BITMASK(location));
    APPLY_RELOC(result);
    return 0;
  }
  return 1;
}

int relocHexGOT32_6_X(uint8_t *location, uint64_t A, uint64_t GOT) {
  int32_t result = (int32_t)((A-GOT) >> 6);
  result = lld::scatterBits<int32_t>(result, FINDV4BITMASK(location));
  APPLY_RELOC(result);
  return 0;
}

int relocHexGOT16_X(uint8_t *location, uint64_t A, uint64_t GOT) {
  int32_t result = (int32_t)(A-GOT);
  int32_t range = 1L << 6;
  if (result <= range) {
    result = lld::scatterBits<int32_t>(result, FINDV4BITMASK(location));
    APPLY_RELOC(result);
    return 0;
  }
  return 1;
}

int relocHexGOT11_X(uint8_t *location, uint64_t A, uint64_t GOT) {
  uint32_t result = (uint32_t)(A-GOT);
  result = lld::scatterBits<uint32_t>(result, FINDV4BITMASK(location));
  APPLY_RELOC(result);
  return 0;
}

int relocHexGOTRELSigned(uint8_t *location, uint64_t P, uint64_t S, uint64_t A,
                         uint64_t GOT, int shiftBits = 0) {
  int32_t result = (int32_t)((S + A - GOT) >> shiftBits);
  result = lld::scatterBits<int32_t>(result, FINDV4BITMASK(location));
  APPLY_RELOC(result);
  return 0;
}

int relocHexGOTRELUnsigned(uint8_t *location, uint64_t P, uint64_t S,
                           uint64_t A, uint64_t GOT, int shiftBits = 0) {
  uint32_t result = (uint32_t)((S + A - GOT) >> shiftBits);
  result = lld::scatterBits<uint32_t>(result, FINDV4BITMASK(location));
  APPLY_RELOC(result);
  return 0;
}

int relocHexGOTREL_HILO16(uint8_t *location, uint64_t P, uint64_t S, uint64_t A,
                          uint64_t GOT, int shiftBits = 0) {
  int32_t result = (int32_t)((S + A - GOT) >> shiftBits);
  result = lld::scatterBits<int32_t>(result, 0x00c03fff);
  APPLY_RELOC(result);
  return 0;
}

int relocHexGOTREL_32(uint8_t *location, uint64_t P, uint64_t S, uint64_t A,
                      uint64_t GOT) {
  int32_t result = (int32_t)(S + A - GOT);
  APPLY_RELOC(result);
  return 0;
}

} // end anon namespace

ErrorOr<void> HexagonTargetRelocationHandler::applyRelocation(
    ELFWriter &writer, llvm::FileOutputBuffer &buf, const lld::AtomLayout &atom,
    const Reference &ref) const {
  uint8_t *atomContent = buf.getBufferStart() + atom._fileOffset;
  uint8_t *location = atomContent + ref.offsetInAtom();
  uint64_t targetVAddress = writer.addressOfAtom(ref.target());
  uint64_t relocVAddress = atom._virtualAddr + ref.offsetInAtom();

  switch (ref.kind()) {
  case R_HEX_B22_PCREL:
    relocBNPCREL(location, relocVAddress, targetVAddress, ref.addend(), 21);
    break;
  case R_HEX_B15_PCREL:
    relocBNPCREL(location, relocVAddress, targetVAddress, ref.addend(), 14);
    break;
  case R_HEX_B9_PCREL:
    relocBNPCREL(location, relocVAddress, targetVAddress, ref.addend(), 8);
    break;
  case R_HEX_LO16:
    relocLO16(location, relocVAddress, targetVAddress, ref.addend());
    break;
  case R_HEX_HI16:
    relocHI16(location, relocVAddress, targetVAddress, ref.addend());
    break;
  case R_HEX_32:
    reloc32(location, relocVAddress, targetVAddress, ref.addend());
    break;
  case R_HEX_32_6_X:
    reloc32_6_X(location, relocVAddress, targetVAddress, ref.addend());
    break;
  case R_HEX_B32_PCREL_X:
    relocHexB32PCRELX(location, relocVAddress, targetVAddress, ref.addend());
    break;
  case R_HEX_B22_PCREL_X:
    relocHexBNPCRELX(location, relocVAddress, targetVAddress, ref.addend(), 21);
    break;
  case R_HEX_B15_PCREL_X:
    relocHexBNPCRELX(location, relocVAddress, targetVAddress, ref.addend(), 14);
    break;
  case R_HEX_B13_PCREL_X:
    relocHexBNPCRELX(location, relocVAddress, targetVAddress, ref.addend(), 12);
    break;
  case R_HEX_B9_PCREL_X:
    relocHexBNPCRELX(location, relocVAddress, targetVAddress, ref.addend(), 8);
    break;
  case R_HEX_B7_PCREL_X:
    relocHexBNPCRELX(location, relocVAddress, targetVAddress, ref.addend(), 6);
    break;
  case R_HEX_GPREL16_0:
    relocHexGPRELN(location, relocVAddress, targetVAddress, ref.addend(),
                   _targetLayout.getSDataSection()->virtualAddr(), 0);
    break;
  case R_HEX_GPREL16_1:
    relocHexGPRELN(location, relocVAddress, targetVAddress, ref.addend(),
                   _targetLayout.getSDataSection()->virtualAddr(), 1);
    break;
  case R_HEX_GPREL16_2:
    relocHexGPRELN(location, relocVAddress, targetVAddress, ref.addend(),
                   _targetLayout.getSDataSection()->virtualAddr(), 2);
    break;
  case R_HEX_GPREL16_3:
    relocHexGPRELN(location, relocVAddress, targetVAddress, ref.addend(),
                   _targetLayout.getSDataSection()->virtualAddr(), 3);
    break;
  case R_HEX_16_X:
  case R_HEX_12_X:
  case R_HEX_11_X:
  case R_HEX_10_X:
  case R_HEX_9_X:
  case R_HEX_8_X:
  case R_HEX_7_X:
  case R_HEX_6_X:
    relocHex_N_X(location, relocVAddress, targetVAddress, ref.addend());
    break;
  case R_HEX_6_PCREL_X:
    relocHex6PCRELX(location, relocVAddress, targetVAddress, ref.addend());
    break;
  case R_HEX_JMP_SLOT:
  case R_HEX_GLOB_DAT:
    break;
  case R_HEX_GOTREL_32:
    relocHexGOTREL_32(location, relocVAddress, targetVAddress, ref.addend(),
                      _targetHandler.getGOTSymAddr());
    break;
  case R_HEX_GOTREL_LO16:
    relocHexGOTREL_HILO16(location, relocVAddress, targetVAddress, ref.addend(),
                          _targetHandler.getGOTSymAddr());
    break;
  case R_HEX_GOTREL_HI16:
    relocHexGOTREL_HILO16(location, relocVAddress, targetVAddress, ref.addend(),
                          _targetHandler.getGOTSymAddr(), 16);
    break;
  case R_HEX_GOT_LO16:
    relocHexGOTLO16(location, targetVAddress, _targetHandler.getGOTSymAddr());
    break;
  case R_HEX_GOT_HI16:
    relocHexGOTHI16(location, targetVAddress, _targetHandler.getGOTSymAddr());
    break;
  case R_HEX_GOT_32:
    relocHexGOT32(location, targetVAddress, _targetHandler.getGOTSymAddr());
    break;
  case R_HEX_GOT_16:
    relocHexGOT16(location, targetVAddress, _targetHandler.getGOTSymAddr());
    break;
  case R_HEX_GOT_32_6_X:
    relocHexGOT32_6_X(location, targetVAddress, _targetHandler.getGOTSymAddr());
    break;
  case R_HEX_GOT_16_X:
    relocHexGOT16_X(location, targetVAddress, _targetHandler.getGOTSymAddr());
    break;
  case R_HEX_GOT_11_X:
    relocHexGOT11_X(location, targetVAddress, _targetHandler.getGOTSymAddr());
    break;
  case R_HEX_GOTREL_32_6_X:
    relocHexGOTRELSigned(location, relocVAddress, targetVAddress, ref.addend(),
                   _targetHandler.getGOTSymAddr(), 6);
    break;
  case R_HEX_GOTREL_16_X:
  case R_HEX_GOTREL_11_X:
    relocHexGOTRELUnsigned(location, relocVAddress, targetVAddress, ref.addend(),
                   _targetHandler.getGOTSymAddr());
    break;

  case lld::Reference::kindLayoutAfter:
  case lld::Reference::kindLayoutBefore:
  case lld::Reference::kindInGroup:
    break;
  default : {
    std::string str;
    llvm::raw_string_ostream s(str);
    auto name = _targetInfo.stringFromRelocKind(ref.kind());
    s << "Unhandled relocation: "
      << (name ? *name : "<unknown>" ) << " (" << ref.kind() << ")";
    s.flush();
    llvm_unreachable(str.c_str());
  }
  }

  return error_code::success();
}
