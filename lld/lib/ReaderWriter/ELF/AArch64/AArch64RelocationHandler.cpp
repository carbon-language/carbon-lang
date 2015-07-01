//===- lib/ReaderWriter/ELF/AArch64/AArch64RelocationHandler.cpp ----------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AArch64TargetHandler.h"
#include "AArch64LinkingContext.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "AArch64"

using namespace lld;
using namespace lld::elf;
using namespace llvm;
using namespace llvm::support::endian;

static int64_t page(int64_t v) { return v & ~int64_t(0xFFF); }

/// \brief Check X is in the interval (-2^(bits-1), 2^bits]
static bool withinSignedUnsignedRange(int64_t X, int bits) {
  return isIntN(bits - 1, X) || isUIntN(bits, X);
}

/// \brief R_AARCH64_ABS64 - word64: S + A
static void relocR_AARCH64_ABS64(uint8_t *location, uint64_t P, uint64_t S,
                                 int64_t A) {
  int64_t result = (int64_t)S + A;
  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
        llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
        llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
        llvm::dbgs() << " result: 0x" << Twine::utohexstr(result) << "\n");
  write64le(location, result | read64le(location));
}

/// \brief R_AARCH64_ABS32 - word32:  S + A
static std::error_code relocR_AARCH64_ABS32(uint8_t *location, uint64_t P,
                                            uint64_t S, int64_t A) {
  int64_t result = S + A;
  if (!withinSignedUnsignedRange(result, 32))
    return make_out_of_range_reloc_error();
  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
        llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
        llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
        llvm::dbgs() << " result: 0x" << Twine::utohexstr(result) << "\n");
  write32le(location, result | read32le(location));
  return std::error_code();
}

/// \brief R_AARCH64_ABS16 - word16:  S + A
static std::error_code relocR_AARCH64_ABS16(uint8_t *location, uint64_t P,
                                            uint64_t S, int64_t A) {
  int64_t result = S + A;
  if (!withinSignedUnsignedRange(result, 16))
    return make_out_of_range_reloc_error();
  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
        llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
        llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
        llvm::dbgs() << " result: 0x" << Twine::utohexstr(result) << "\n");
  write16le(location, result | read16le(location));
  return std::error_code();
}

/// \brief R_AARCH64_PREL64 - word64: S + A - P
static void relocR_AARCH64_PREL64(uint8_t *location, uint64_t P,
                                  uint64_t S, int64_t A) {
  int64_t result = S + A - P;
  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
        llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
        llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
        llvm::dbgs() << " result: 0x" << Twine::utohexstr(result) << "\n");
  write64le(location, result + read64le(location));
}

/// \brief R_AARCH64_PREL32 - word32: S + A - P
static std::error_code relocR_AARCH64_PREL32(uint8_t *location, uint64_t P,
                                             uint64_t S, int64_t A) {
  int64_t result = S + A - P;
  // ELF for the ARM 64-bit architecture manual states the overflow
  // for R_AARCH64_PREL32 to be -2^(-31) <= X < 2^32
  if (!withinSignedUnsignedRange(result, 32))
    return make_out_of_range_reloc_error();
  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
        llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
        llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
        llvm::dbgs() << " result: 0x" << Twine::utohexstr(result) << "\n");
  write32le(location, result + read32le(location));
  return std::error_code();
}

/// \brief R_AARCH64_PREL16 - word16: S + A - P
static std::error_code relocR_AARCH64_PREL16(uint8_t *location, uint64_t P,
                                             uint64_t S, int64_t A) {
  int64_t result = S + A - P;
  if (!withinSignedUnsignedRange(result, 16))
    return make_out_of_range_reloc_error();
  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
        llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
        llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
        llvm::dbgs() << " result: 0x" << Twine::utohexstr(result) << "\n");
  write16le(location, result + read16le(location));
  return std::error_code();
}

/// \brief R_AARCH64_ADR_PREL_PG_HI21 - Page(S+A) - Page(P)
static std::error_code relocR_AARCH64_ADR_PREL_PG_HI21(uint8_t *location,
                                                       uint64_t P, uint64_t S,
                                                       int64_t A) {
  int64_t result = page(S + A) - page(P);
  if (!isInt<32>(result))
    return make_out_of_range_reloc_error();
  result = result >> 12;
  uint32_t immlo = result & 0x3;
  uint32_t immhi = result & 0x1FFFFC;
  immlo = immlo << 29;
  immhi = immhi << 3;
  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: " << Twine::utohexstr(S);
        llvm::dbgs() << " A: " << Twine::utohexstr(A);
        llvm::dbgs() << " P: " << Twine::utohexstr(P);
        llvm::dbgs() << " immhi: " << Twine::utohexstr(immhi);
        llvm::dbgs() << " immlo: " << Twine::utohexstr(immlo);
        llvm::dbgs() << " result: " << Twine::utohexstr(result) << "\n");
  write32le(location, immlo | immhi | read32le(location));
  return std::error_code();
}

/// \brief R_AARCH64_ADR_PREL_LO21 - S + A - P
static std::error_code relocR_AARCH64_ADR_PREL_LO21(uint8_t *location, uint64_t P,
                                                    uint64_t S, int64_t A) {
  uint64_t result = S + A - P;
  if (!isInt<20>(result))
    return make_out_of_range_reloc_error();
  uint32_t immlo = result & 0x3;
  uint32_t immhi = result & 0x1FFFFC;
  immlo = immlo << 29;
  immhi = immhi << 3;
  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: " << Twine::utohexstr(S);
        llvm::dbgs() << " A: " << Twine::utohexstr(A);
        llvm::dbgs() << " P: " << Twine::utohexstr(P);
        llvm::dbgs() << " immhi: " << Twine::utohexstr(immhi);
        llvm::dbgs() << " immlo: " << Twine::utohexstr(immlo);
        llvm::dbgs() << " result: " << Twine::utohexstr(result) << "\n");
  write32le(location, immlo | immhi | read32le(location));
  return std::error_code();
}

/// \brief R_AARCH64_ADD_ABS_LO12_NC
static void relocR_AARCH64_ADD_ABS_LO12_NC(uint8_t *location, uint64_t P,
                                           uint64_t S, int64_t A) {
  int32_t result = (int32_t)((S + A) & 0xFFF);
  result <<= 10;
  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: " << Twine::utohexstr(S);
        llvm::dbgs() << " A: " << Twine::utohexstr(A);
        llvm::dbgs() << " P: " << Twine::utohexstr(P);
        llvm::dbgs() << " result: " << Twine::utohexstr(result) << "\n");
  write32le(location, result | read32le(location));
}

/// \brief R_AARCH64_CALL26 and R_AARCH64_JUMP26
static std::error_code relocJump26(uint8_t *location, uint64_t P, uint64_t S,
                                   int64_t A) {
  int64_t result = S + A - P;
  if (!isInt<27>(result))
    return make_out_of_range_reloc_error();
  result &= 0x0FFFFFFC;
  result >>= 2;
  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: " << Twine::utohexstr(S);
        llvm::dbgs() << " A: " << Twine::utohexstr(A);
        llvm::dbgs() << " P: " << Twine::utohexstr(P);
        llvm::dbgs() << " result: " << Twine::utohexstr(result) << "\n");
  write32le(location, result | read32le(location));
  return std::error_code();
}

/// \brief R_AARCH64_CONDBR19
static std::error_code relocR_AARCH64_CONDBR19(uint8_t *location, uint64_t P,
                                               uint64_t S, int64_t A) {
  int64_t result = S + A - P;
  if (!isInt<20>(result))
    return make_out_of_range_reloc_error();
  result &= 0x01FFFFC;
  result <<= 3;
  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: " << Twine::utohexstr(S);
        llvm::dbgs() << " A: " << Twine::utohexstr(A);
        llvm::dbgs() << " P: " << Twine::utohexstr(P);
        llvm::dbgs() << " result: " << Twine::utohexstr(result) << "\n");
  write32le(location, result | read32le(location));
  return std::error_code();
}

/// \brief R_AARCH64_LDST8_ABS_LO12_NC - S + A
static void relocR_AARCH64_LDST8_ABS_LO12_NC(uint8_t *location, uint64_t P,
                                             uint64_t S, int64_t A) {
  int32_t result = (int32_t)((S + A) & 0xFFF);
  result <<= 10;
  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: " << Twine::utohexstr(S);
        llvm::dbgs() << " A: " << Twine::utohexstr(A);
        llvm::dbgs() << " P: " << Twine::utohexstr(P);
        llvm::dbgs() << " result: " << Twine::utohexstr(result) << "\n");
  write32le(location, result | read32le(location));
}

/// \brief R_AARCH64_LDST16_ABS_LO12_NC
static void relocR_AARCH64_LDST16_ABS_LO12_NC(uint8_t *location, uint64_t P,
                                              uint64_t S, int64_t A) {
  int32_t result = (int32_t)(S + A);
  result &= 0x0FFC;
  result <<= 9;
  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: " << Twine::utohexstr(S);
        llvm::dbgs() << " A: " << Twine::utohexstr(A);
        llvm::dbgs() << " P: " << Twine::utohexstr(P);
        llvm::dbgs() << " result: " << Twine::utohexstr(result) << "\n");
  write32le(location, result | read32le(location));
}

/// \brief R_AARCH64_LDST32_ABS_LO12_NC
static void relocR_AARCH64_LDST32_ABS_LO12_NC(uint8_t *location, uint64_t P,
                                              uint64_t S, int64_t A) {
  int32_t result = (int32_t)(S + A);
  result &= 0x0FFC;
  result <<= 8;
  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: " << Twine::utohexstr(S);
        llvm::dbgs() << " A: " << Twine::utohexstr(A);
        llvm::dbgs() << " P: " << Twine::utohexstr(P);
        llvm::dbgs() << " result: " << Twine::utohexstr(result) << "\n");
  write32le(location, result | read32le(location));
}

/// \brief R_AARCH64_LDST64_ABS_LO12_NC
static void relocR_AARCH64_LDST64_ABS_LO12_NC(uint8_t *location, uint64_t P,
                                              uint64_t S, int64_t A) {
  int32_t result = (int32_t)(S + A);
  result &= 0x0FF8;
  result <<= 7;
  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: " << Twine::utohexstr(S);
        llvm::dbgs() << " A: " << Twine::utohexstr(A);
        llvm::dbgs() << " P: " << Twine::utohexstr(P);
        llvm::dbgs() << " result: " << Twine::utohexstr(result) << "\n");
  write32le(location, result | read32le(location));
}

/// \brief R_AARCH64_LDST128_ABS_LO12_NC
static void relocR_AARCH64_LDST128_ABS_LO12_NC(uint8_t *location, uint64_t P,
                                               uint64_t S, int64_t A) {
  int32_t result = (int32_t)(S + A);
  result &= 0x0FF8;
  result <<= 6;
  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: " << Twine::utohexstr(S);
        llvm::dbgs() << " A: " << Twine::utohexstr(A);
        llvm::dbgs() << " P: " << Twine::utohexstr(P);
        llvm::dbgs() << " result: " << Twine::utohexstr(result) << "\n");
  write32le(location, result | read32le(location));
}

static std::error_code relocR_AARCH64_ADR_GOT_PAGE(uint8_t *location,
                                                   uint64_t P, uint64_t S,
                                                   int64_t A) {
  uint64_t result = page(S + A) - page(P);
  if (!isInt<32>(result))
    return make_out_of_range_reloc_error();
  result = (result >> 12) & 0x3FFFF;
  uint32_t immlo = result & 0x3;
  uint32_t immhi = result & 0x1FFFFC;
  immlo = immlo << 29;
  immhi = immhi << 3;
  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: " << Twine::utohexstr(S);
        llvm::dbgs() << " A: " << Twine::utohexstr(A);
        llvm::dbgs() << " P: " << Twine::utohexstr(P);
        llvm::dbgs() << " immhi: " << Twine::utohexstr(immhi);
        llvm::dbgs() << " immlo: " << Twine::utohexstr(immlo);
        llvm::dbgs() << " result: " << Twine::utohexstr(result) << "\n");
  write32le(location, immlo | immhi | read32le(location));
  return std::error_code();
}

// R_AARCH64_LD64_GOT_LO12_NC
static std::error_code relocR_AARCH64_LD64_GOT_LO12_NC(uint8_t *location,
                                                       uint64_t P, uint64_t S,
                                                       int64_t A) {
  int32_t result = S + A;
  DEBUG(llvm::dbgs() << " S: " << Twine::utohexstr(S);
        llvm::dbgs() << " A: " << Twine::utohexstr(A);
        llvm::dbgs() << " P: " << Twine::utohexstr(P);
        llvm::dbgs() << " result: " << Twine::utohexstr(result) << "\n");
  if ((result & 0x7) != 0)
    return make_unaligned_range_reloc_error();
  result &= 0xFF8;
  result <<= 7;
  write32le(location, result | read32le(location));
  return std::error_code();
}

// ADD_AARCH64_GOTRELINDEX
static void relocADD_AARCH64_GOTRELINDEX(uint8_t *location, uint64_t P,
                                         uint64_t S, int64_t A) {
  int32_t result = S + A;
  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: " << Twine::utohexstr(S);
        llvm::dbgs() << " A: " << Twine::utohexstr(A);
        llvm::dbgs() << " P: " << Twine::utohexstr(P);
        llvm::dbgs() << " result: " << Twine::utohexstr(result) << "\n");
  result &= 0xFFF;
  result <<= 10;
  write32le(location, result | read32le(location));
}

// R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21
static std::error_code relocR_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21(uint8_t *location,
                                                                uint64_t P,
                                                                uint64_t S,
                                                                int64_t A) {
  int64_t result = page(S + A) - page(P);
  if (!isInt<32>(result))
    return make_out_of_range_reloc_error();
  result >>= 12;
  uint32_t immlo = result & 0x3;
  uint32_t immhi = result & 0x1FFFFC;
  immlo = immlo << 29;
  immhi = immhi << 3;
  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: " << Twine::utohexstr(S);
        llvm::dbgs() << " A: " << Twine::utohexstr(A);
        llvm::dbgs() << " P: " << Twine::utohexstr(P);
        llvm::dbgs() << " immhi: " << Twine::utohexstr(immhi);
        llvm::dbgs() << " immlo: " << Twine::utohexstr(immlo);
        llvm::dbgs() << " result: " << Twine::utohexstr(result) << "\n");
  write32le(location, immlo | immhi | read32le(location));
  return std::error_code();
}

// R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC
static void relocR_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC(uint8_t *location,
                                                       uint64_t P, uint64_t S,
                                                       int64_t A) {
  int32_t result = S + A;
  result &= 0xFF8;
  result <<= 7;
  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: " << Twine::utohexstr(S);
        llvm::dbgs() << " A: " << Twine::utohexstr(A);
        llvm::dbgs() << " P: " << Twine::utohexstr(P);
        llvm::dbgs() << " result: " << Twine::utohexstr(result) << "\n");
  write32le(location, result | read32le(location));
}

/// \brief R_AARCH64_TLSLE_ADD_TPREL_HI12
static std::error_code relocR_AARCH64_TLSLE_ADD_TPREL_HI12(uint8_t *location,
                                                           uint64_t P,
                                                           uint64_t S,
                                                           int64_t A) {
  int64_t result = S + A;
  if (!isUInt<24>(result))
    return make_out_of_range_reloc_error();
  result &= 0x0FFF000;
  result >>= 2;
  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: " << Twine::utohexstr(S);
        llvm::dbgs() << " A: " << Twine::utohexstr(A);
        llvm::dbgs() << " P: " << Twine::utohexstr(P);
        llvm::dbgs() << " result: " << Twine::utohexstr(result) << "\n");
  write32le(location, result | read32le(location));
  return std::error_code();
}

/// \brief R_AARCH64_TLSLE_ADD_TPREL_LO12_NC
static void relocR_AARCH64_TLSLE_ADD_TPREL_LO12_NC(uint8_t *location,
                                                   uint64_t P, uint64_t S,
                                                   int64_t A) {
  int32_t result = S + A;
  result &= 0x0FFF;
  result <<= 10;
  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: " << Twine::utohexstr(S);
        llvm::dbgs() << " A: " << Twine::utohexstr(A);
        llvm::dbgs() << " P: " << Twine::utohexstr(P);
        llvm::dbgs() << " result: " << Twine::utohexstr(result) << "\n");
  write32le(location, result | read32le(location));
}

/// \brief R_AARCH64_TLSDESC_ADR_PAGE21 - Page(G(GTLSDESC(S+A))) - Page(P)
static std::error_code relocR_AARCH64_TLSDESC_ADR_PAGE21(uint8_t *location,
                                                         uint64_t P, uint64_t S,
                                                         int64_t A) {
  int64_t result = page(S + A) - page(P);
  if (!isInt<32>(result))
    return make_out_of_range_reloc_error();
  result = result >> 12;
  uint32_t immlo = result & 0x3;
  uint32_t immhi = result & 0x1FFFFC;
  immlo = immlo << 29;
  immhi = immhi << 3;
  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: " << Twine::utohexstr(S);
        llvm::dbgs() << " A: " << Twine::utohexstr(A);
        llvm::dbgs() << " P: " << Twine::utohexstr(P);
        llvm::dbgs() << " immhi: " << Twine::utohexstr(immhi);
        llvm::dbgs() << " immlo: " << Twine::utohexstr(immlo);
        llvm::dbgs() << " result: " << Twine::utohexstr(result) << "\n");
  write32le(location, immlo | immhi | read32le(location));
  return std::error_code();
}

/// \brief R_AARCH64_TLSDESC_LD64_LO12_NC - G(GTLSDESC(S+A)) -> S + A
static std::error_code relocR_AARCH64_TLSDESC_LD64_LO12_NC(uint8_t *location,
                                                           uint64_t P,
                                                           uint64_t S,
                                                           int64_t A) {
  int32_t result = S + A;
  DEBUG(llvm::dbgs() << " S: " << Twine::utohexstr(S);
        llvm::dbgs() << " A: " << Twine::utohexstr(A);
        llvm::dbgs() << " P: " << Twine::utohexstr(P);
        llvm::dbgs() << " result: " << Twine::utohexstr(result) << "\n");
  if ((result & 0x7) != 0)
    return make_unaligned_range_reloc_error();
  result &= 0xFF8;
  result <<= 7;
  write32le(location, result | read32le(location));
  return std::error_code();
}

/// \brief R_AARCH64_TLSDESC_ADD_LO12_NC - G(GTLSDESC(S+A)) -> S + A
static void relocR_AARCH64_TLSDESC_ADD_LO12_NC(uint8_t *location, uint64_t P,
                                               uint64_t S, int64_t A) {
  int32_t result = (int32_t)((S + A) & 0xFFF);
  result <<= 10;
  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: " << Twine::utohexstr(S);
        llvm::dbgs() << " A: " << Twine::utohexstr(A);
        llvm::dbgs() << " P: " << Twine::utohexstr(P);
        llvm::dbgs() << " result: " << Twine::utohexstr(result) << "\n");
  write32le(location, result | read32le(location));
}

std::error_code AArch64TargetRelocationHandler::applyRelocation(
    ELFWriter &writer, llvm::FileOutputBuffer &buf, const AtomLayout &atom,
    const Reference &ref) const {
  uint8_t *atomContent = buf.getBufferStart() + atom._fileOffset;
  uint8_t *loc = atomContent + ref.offsetInAtom();
  uint64_t target = writer.addressOfAtom(ref.target());
  uint64_t reloc = atom._virtualAddr + ref.offsetInAtom();
  int64_t addend = ref.addend();

  if (ref.kindNamespace() != Reference::KindNamespace::ELF)
    return std::error_code();
  assert(ref.kindArch() == Reference::KindArch::AArch64);
  switch (ref.kindValue()) {
  case R_AARCH64_NONE:
    break;
  case R_AARCH64_ABS64:
    relocR_AARCH64_ABS64(loc, reloc, target, addend);
    break;
  case R_AARCH64_ABS32:
    return relocR_AARCH64_ABS32(loc, reloc, target, addend);
  case R_AARCH64_ABS16:
    return relocR_AARCH64_ABS16(loc, reloc, target, addend);
  case R_AARCH64_PREL64:
    relocR_AARCH64_PREL64(loc, reloc, target, addend);
    break;
  case R_AARCH64_PREL32:
    return relocR_AARCH64_PREL32(loc, reloc, target, addend);
  case R_AARCH64_PREL16:
    return relocR_AARCH64_PREL16(loc, reloc, target, addend);
  case R_AARCH64_ADR_PREL_PG_HI21:
    return relocR_AARCH64_ADR_PREL_PG_HI21(loc, reloc, target, addend);
  case R_AARCH64_ADR_PREL_LO21:
    return relocR_AARCH64_ADR_PREL_LO21(loc, reloc, target, addend);
  case R_AARCH64_ADD_ABS_LO12_NC:
    relocR_AARCH64_ADD_ABS_LO12_NC(loc, reloc, target, addend);
    break;
  case R_AARCH64_CALL26:
  case R_AARCH64_JUMP26:
    return relocJump26(loc, reloc, target, addend);
  case R_AARCH64_CONDBR19:
    return relocR_AARCH64_CONDBR19(loc, reloc, target, addend);
  case R_AARCH64_ADR_GOT_PAGE:
    return relocR_AARCH64_ADR_GOT_PAGE(loc, reloc, target, addend);
  case R_AARCH64_LD64_GOT_LO12_NC:
    return relocR_AARCH64_LD64_GOT_LO12_NC(loc, reloc, target, addend);
  case R_AARCH64_LDST8_ABS_LO12_NC:
    relocR_AARCH64_LDST8_ABS_LO12_NC(loc, reloc, target, addend);
    break;
  case R_AARCH64_LDST16_ABS_LO12_NC:
    relocR_AARCH64_LDST16_ABS_LO12_NC(loc, reloc, target, addend);
    break;
  case R_AARCH64_LDST32_ABS_LO12_NC:
    relocR_AARCH64_LDST32_ABS_LO12_NC(loc, reloc, target, addend);
    break;
  case R_AARCH64_LDST64_ABS_LO12_NC:
    relocR_AARCH64_LDST64_ABS_LO12_NC(loc, reloc, target, addend);
    break;
  case R_AARCH64_LDST128_ABS_LO12_NC:
    relocR_AARCH64_LDST128_ABS_LO12_NC(loc, reloc, target, addend);
    break;
  case ADD_AARCH64_GOTRELINDEX:
    relocADD_AARCH64_GOTRELINDEX(loc, reloc, target, addend);
    break;
  case R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21:
    return relocR_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21(loc, reloc, target, addend);
  case R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC:
    relocR_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC(loc, reloc, target, addend);
    break;
  case R_AARCH64_TLSLE_ADD_TPREL_HI12:
  case R_AARCH64_TLSLE_ADD_TPREL_LO12_NC: {
    auto tpoffset = _layout.getTPOffset();
    if (ref.kindValue() == R_AARCH64_TLSLE_ADD_TPREL_HI12)
      return relocR_AARCH64_TLSLE_ADD_TPREL_HI12(loc, reloc, target + tpoffset,
                                                 addend);
    else 
      relocR_AARCH64_TLSLE_ADD_TPREL_LO12_NC(loc, reloc, target + tpoffset,
                                             addend);
  }  break;
  case R_AARCH64_TLSDESC_ADR_PAGE21:
    return relocR_AARCH64_TLSDESC_ADR_PAGE21(loc, reloc, target, addend);
  case R_AARCH64_TLSDESC_LD64_LO12_NC:
    return relocR_AARCH64_TLSDESC_LD64_LO12_NC(loc, reloc, target, addend);
  case R_AARCH64_TLSDESC_ADD_LO12_NC:
    relocR_AARCH64_TLSDESC_ADD_LO12_NC(loc, reloc, target, addend);
    break;
  case R_AARCH64_TLSDESC_CALL:
    // Relaxation only to optimize TLS access. Ignore for now.
    break;
  // Runtime only relocations. Ignore here.
  case R_AARCH64_RELATIVE:
  case R_AARCH64_IRELATIVE:
  case R_AARCH64_JUMP_SLOT:
  case R_AARCH64_GLOB_DAT:
  case R_AARCH64_TLS_TPREL64:
  case R_AARCH64_TLSDESC:
    break;
  default:
    return make_unhandled_reloc_error();
  }
  return std::error_code();
}
