//===- lib/ReaderWriter/ELF/Hexagon/HexagonRelocationHandler.cpp ---------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "HexagonEncodings.h"
#include "HexagonLinkingContext.h"
#include "HexagonRelocationHandler.h"
#include "HexagonTargetHandler.h"
#include "llvm/Support/Endian.h"

using namespace lld;
using namespace lld::elf;
using namespace llvm::ELF;
using namespace llvm::support::endian;

// Scatter val's bits as specified by the mask. Example:
//
//  Val:    0bABCDEFG
//  Mask:   0b10111100001011
//  Output: 0b00ABCD0000E0FG
static uint32_t scatterBits(uint32_t val, uint32_t mask) {
  uint32_t result = 0;
  size_t off = 0;
  for (size_t bit = 0; bit < 32; ++bit) {
    if ((mask >> bit) & 1) {
      uint32_t valBit = (val >> off) & 1;
      result |= valBit << bit;
      ++off;
    }
  }
  return result;
}

static void relocBNPCREL(uint8_t *loc, uint64_t P, uint64_t S, uint64_t A,
                         int32_t nBits) {
  int32_t result = (S + A - P) >> 2;
  int32_t range = 1 << nBits;
  if (result < range && result > -range) {
    result = scatterBits(result, findv4bitmask(loc));
    write32le(loc, result | read32le(loc));
  }
}

/// \brief Word32_LO: 0x00c03fff : (S + A) : Truncate
static void relocLO16(uint8_t *loc, uint64_t P, uint64_t S, uint64_t A) {
  uint32_t result = S + A;
  result = scatterBits(result, 0x00c03fff);
  write32le(loc, result | read32le(loc));
}

/// \brief Word32_LO: 0x00c03fff : (S + A) >> 16 : Truncate
static void relocHI16(uint8_t *loc, uint64_t P, uint64_t S, uint64_t A) {
  uint32_t result = (S + A) >> 16;
  result = scatterBits(result, 0x00c03fff);
  write32le(loc, result | read32le(loc));
}

/// \brief Word32: 0xffffffff : (S + A) : Truncate
static void reloc32(uint8_t *loc, uint64_t P, uint64_t S, uint64_t A) {
  uint32_t result = S + A;
  write32le(loc, result | read32le(loc));
}

static void reloc32_6_X(uint8_t *loc, uint64_t P, uint64_t S, uint64_t A) {
  int64_t result = (S + A) >> 6;
  int64_t range = int64_t(1) << 32;
  if (result > range)
  result = scatterBits(result, 0xfff3fff);
  write32le(loc, result | read32le(loc));
}

// R_HEX_B32_PCREL_X
static void relocHexB32PCRELX(uint8_t *loc, uint64_t P, uint64_t S,
                              uint64_t A) {
  int64_t result = (S + A - P) >> 6;
  result = scatterBits(result, 0xfff3fff);
  write32le(loc, result | read32le(loc));
}

// R_HEX_BN_PCREL_X
static void relocHexBNPCRELX(uint8_t *loc, uint64_t P, uint64_t S, uint64_t A,
                             int nbits) {
  int32_t result = (S + A - P) & 0x3f;
  int32_t range = 1 << nbits;
  if (result < range && result > -range) {
    result = scatterBits(result, findv4bitmask(loc));
    write32le(loc, result | read32le(loc));
  }
}

// R_HEX_6_PCREL_X
static void relocHex6PCRELX(uint8_t *loc, uint64_t P, uint64_t S, uint64_t A) {
  int32_t result = S + A - P;
  result = scatterBits(result, findv4bitmask(loc));
  write32le(loc, result | read32le(loc));
}

// R_HEX_N_X : Word32_U6 : (S + A) : Unsigned Truncate
static void relocHex_N_X(uint8_t *loc, uint64_t P, uint64_t S, uint64_t A) {
  uint32_t result = S + A;
  result = scatterBits(result, findv4bitmask(loc));
  write32le(loc, result | read32le(loc));
}

// GP REL relocs
static void relocHexGPRELN(uint8_t *loc, uint64_t P, uint64_t S, uint64_t A,
                           uint64_t GP, int nShiftBits) {
  int32_t result = (S + A - GP) >> nShiftBits;
  int32_t range = 1L << 16;
  if (result <= range) {
    result = scatterBits(result, findv4bitmask(loc));
    write32le(loc, result | read32le(loc));
  }
}

/// \brief Word32_LO: 0x00c03fff : (G) : Truncate
static void relocHexGOTLO16(uint8_t *loc, uint64_t A, uint64_t GOT) {
  int32_t result = A - GOT;
  result = scatterBits(result, 0x00c03fff);
  write32le(loc, result | read32le(loc));
}

/// \brief Word32_LO: 0x00c03fff : (G) >> 16 : Truncate
static void relocHexGOTHI16(uint8_t *loc, uint64_t A, uint64_t GOT) {
  int32_t result = (A - GOT) >> 16;
  result = scatterBits(result, 0x00c03fff);
  write32le(loc, result | read32le(loc));
}

/// \brief Word32: 0xffffffff : (G) : Truncate
static void relocHexGOT32(uint8_t *loc, uint64_t A, uint64_t GOT) {
  int32_t result = GOT - A;
  write32le(loc, result | read32le(loc));
}

/// \brief Word32_U16 : (G) : Truncate
static void relocHexGOT16(uint8_t *loc, uint64_t A, uint64_t GOT) {
  int32_t result = GOT - A;
  int32_t range = 1L << 16;
  if (result <= range) {
    result = scatterBits(result, findv4bitmask(loc));
    write32le(loc, result | read32le(loc));
  }
}

static void relocHexGOT32_6_X(uint8_t *loc, uint64_t A, uint64_t GOT) {
  int32_t result = (A - GOT) >> 6;
  result = scatterBits(result, findv4bitmask(loc));
  write32le(loc, result | read32le(loc));
}

static void relocHexGOT16_X(uint8_t *loc, uint64_t A, uint64_t GOT) {
  int32_t result = A - GOT;
  int32_t range = 1L << 6;
  if (result <= range) {
    result = scatterBits(result, findv4bitmask(loc));
    write32le(loc, result | read32le(loc));
  }
}

static void relocHexGOT11_X(uint8_t *loc, uint64_t A, uint64_t GOT) {
  uint32_t result = A - GOT;
  result = scatterBits(result, findv4bitmask(loc));
  write32le(loc, result | read32le(loc));
}

static void relocHexGOTRELSigned(uint8_t *loc, uint64_t P, uint64_t S,
                                 uint64_t A, uint64_t GOT, int shiftBits) {
  int32_t result = (S + A - GOT) >> shiftBits;
  result = scatterBits(result, findv4bitmask(loc));
  write32le(loc, result | read32le(loc));
}

static void relocHexGOTRELUnsigned(uint8_t *loc, uint64_t P, uint64_t S,
                                   uint64_t A, uint64_t GOT) {
  uint32_t result = S + A - GOT;
  result = scatterBits(result, findv4bitmask(loc));
  write32le(loc, result | read32le(loc));
}

static void relocHexGOTREL_HILO16(uint8_t *loc, uint64_t P, uint64_t S,
                                  uint64_t A, uint64_t GOT, int shiftBits) {
  int32_t result = (S + A - GOT) >> shiftBits;
  result = scatterBits(result, 0x00c03fff);
  write32le(loc, result | read32le(loc));
}

static void relocHexGOTREL_32(uint8_t *loc, uint64_t P, uint64_t S, uint64_t A,
                              uint64_t GOT) {
  int32_t result = S + A - GOT;
  write32le(loc, result | read32le(loc));
}

std::error_code HexagonTargetRelocationHandler::applyRelocation(
    ELFWriter &writer, llvm::FileOutputBuffer &buf, const AtomLayout &atom,
    const Reference &ref) const {
  uint8_t *atomContent = buf.getBufferStart() + atom._fileOffset;
  uint8_t *loc = atomContent + ref.offsetInAtom();
  uint64_t target = writer.addressOfAtom(ref.target());
  uint64_t reloc = atom._virtualAddr + ref.offsetInAtom();

  if (ref.kindNamespace() != Reference::KindNamespace::ELF)
    return std::error_code();
  assert(ref.kindArch() == Reference::KindArch::Hexagon);
  switch (ref.kindValue()) {
  case R_HEX_B22_PCREL:
    relocBNPCREL(loc, reloc, target, ref.addend(), 21);
    break;
  case R_HEX_B15_PCREL:
    relocBNPCREL(loc, reloc, target, ref.addend(), 14);
    break;
  case R_HEX_B9_PCREL:
    relocBNPCREL(loc, reloc, target, ref.addend(), 8);
    break;
  case R_HEX_LO16:
    relocLO16(loc, reloc, target, ref.addend());
    break;
  case R_HEX_HI16:
    relocHI16(loc, reloc, target, ref.addend());
    break;
  case R_HEX_32:
    reloc32(loc, reloc, target, ref.addend());
    break;
  case R_HEX_32_6_X:
    reloc32_6_X(loc, reloc, target, ref.addend());
    break;
  case R_HEX_B32_PCREL_X:
    relocHexB32PCRELX(loc, reloc, target, ref.addend());
    break;
  case R_HEX_B22_PCREL_X:
    relocHexBNPCRELX(loc, reloc, target, ref.addend(), 21);
    break;
  case R_HEX_B15_PCREL_X:
    relocHexBNPCRELX(loc, reloc, target, ref.addend(), 14);
    break;
  case R_HEX_B13_PCREL_X:
    relocHexBNPCRELX(loc, reloc, target, ref.addend(), 12);
    break;
  case R_HEX_B9_PCREL_X:
    relocHexBNPCRELX(loc, reloc, target, ref.addend(), 8);
    break;
  case R_HEX_B7_PCREL_X:
    relocHexBNPCRELX(loc, reloc, target, ref.addend(), 6);
    break;
  case R_HEX_GPREL16_0:
    relocHexGPRELN(loc, reloc, target, ref.addend(),
                   _targetLayout.getSDataSection()->virtualAddr(), 0);
    break;
  case R_HEX_GPREL16_1:
    relocHexGPRELN(loc, reloc, target, ref.addend(),
                   _targetLayout.getSDataSection()->virtualAddr(), 1);
    break;
  case R_HEX_GPREL16_2:
    relocHexGPRELN(loc, reloc, target, ref.addend(),
                   _targetLayout.getSDataSection()->virtualAddr(), 2);
    break;
  case R_HEX_GPREL16_3:
    relocHexGPRELN(loc, reloc, target, ref.addend(),
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
    relocHex_N_X(loc, reloc, target, ref.addend());
    break;
  case R_HEX_6_PCREL_X:
    relocHex6PCRELX(loc, reloc, target, ref.addend());
    break;
  case R_HEX_JMP_SLOT:
  case R_HEX_GLOB_DAT:
    break;
  case R_HEX_GOTREL_32:
    relocHexGOTREL_32(loc, reloc, target, ref.addend(),
                      _targetLayout.getGOTSymAddr());
    break;
  case R_HEX_GOTREL_LO16:
    relocHexGOTREL_HILO16(loc, reloc, target, ref.addend(),
                          _targetLayout.getGOTSymAddr(), 0);
    break;
  case R_HEX_GOTREL_HI16:
    relocHexGOTREL_HILO16(loc, reloc, target, ref.addend(),
                          _targetLayout.getGOTSymAddr(), 16);
    break;
  case R_HEX_GOT_LO16:
    relocHexGOTLO16(loc, target, _targetLayout.getGOTSymAddr());
    break;
  case R_HEX_GOT_HI16:
    relocHexGOTHI16(loc, target, _targetLayout.getGOTSymAddr());
    break;
  case R_HEX_GOT_32:
    relocHexGOT32(loc, target, _targetLayout.getGOTSymAddr());
    break;
  case R_HEX_GOT_16:
    relocHexGOT16(loc, target, _targetLayout.getGOTSymAddr());
    break;
  case R_HEX_GOT_32_6_X:
    relocHexGOT32_6_X(loc, target, _targetLayout.getGOTSymAddr());
    break;
  case R_HEX_GOT_16_X:
    relocHexGOT16_X(loc, target, _targetLayout.getGOTSymAddr());
    break;
  case R_HEX_GOT_11_X:
    relocHexGOT11_X(loc, target, _targetLayout.getGOTSymAddr());
    break;
  case R_HEX_GOTREL_32_6_X:
    relocHexGOTRELSigned(loc, reloc, target, ref.addend(),
                         _targetLayout.getGOTSymAddr(), 6);
    break;
  case R_HEX_GOTREL_16_X:
  case R_HEX_GOTREL_11_X:
    relocHexGOTRELUnsigned(loc, reloc, target, ref.addend(),
                           _targetLayout.getGOTSymAddr());
    break;

  default:
    return make_unhandled_reloc_error();
  }

  return std::error_code();
}
