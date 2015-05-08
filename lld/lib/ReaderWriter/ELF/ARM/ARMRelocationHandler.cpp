//===--------- lib/ReaderWriter/ELF/ARM/ARMRelocationHandler.cpp ----------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ARMTargetHandler.h"
#include "ARMLinkingContext.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "ARM"

using namespace lld;
using namespace lld::elf;
using namespace llvm::support::endian;

static Reference::Addend readAddend_THM_MOV(const uint8_t *location) {
  const uint16_t halfHi = read16le(location);
  const uint16_t halfLo = read16le(location + 2);

  const uint16_t imm8 = halfLo & 0xFF;
  const uint16_t imm3 = (halfLo >> 12) & 0x7;

  const uint16_t imm4 = halfHi & 0xF;
  const uint16_t bitI = (halfHi >> 10) & 0x1;

  const auto result = int16_t((imm4 << 12) | (bitI << 11) | (imm3 << 8) | imm8);
  return result;
}

static Reference::Addend readAddend_ARM_MOV(const uint8_t *location) {
  const uint32_t value = read32le(location);

  const uint32_t imm12 = value & 0xFFF;
  const uint32_t imm4 = (value >> 16) & 0xF;

  const auto result = int32_t((imm4 << 12) | imm12);
  return result;
}

static Reference::Addend readAddend_THM_CALL(const uint8_t *location) {
  const uint16_t halfHi = read16le(location);
  const uint16_t halfLo = read16le(location + 2);

  const uint16_t imm10 = halfHi & 0x3FF;
  const uint16_t bitS = (halfHi >> 10) & 0x1;

  const uint16_t imm11 = halfLo & 0x7FF;
  const uint16_t bitJ2 = (halfLo >> 11) & 0x1;
  const uint16_t bitI2 = (~(bitJ2 ^ bitS)) & 0x1;
  const uint16_t bitJ1 = (halfLo >> 13) & 0x1;
  const uint16_t bitI1 = (~(bitJ1 ^ bitS)) & 0x1;

  const auto result = int32_t((bitS << 24) | (bitI1 << 23) | (bitI2 << 22) |
                              (imm10 << 12) | (imm11 << 1));
  return llvm::SignExtend64<25>(result);
}

static Reference::Addend readAddend_ARM_CALL(const uint8_t *location) {
  const uint32_t value = read32le(location);

  const bool isBLX = (value & 0xF0000000) == 0xF0000000;
  const uint32_t bitH = isBLX ? ((value & 0x1000000) >> 24) : 0;

  const auto result = int32_t(((value & 0xFFFFFF) << 2) | (bitH << 1));
  return llvm::SignExtend64<26>(result);
}

static Reference::Addend readAddend_THM_JUMP11(const uint8_t *location) {
  const auto value = read16le(location);
  const uint16_t imm11 = value & 0x7FF;

  return llvm::SignExtend64<12>(imm11 << 1);
}

static Reference::Addend readAddend(const uint8_t *location,
                                    Reference::KindValue kindValue) {
  switch (kindValue) {
  case R_ARM_ABS32:
  case R_ARM_REL32:
  case R_ARM_TARGET1:
  case R_ARM_GOT_BREL:
  case R_ARM_BASE_PREL:
  case R_ARM_TLS_IE32:
  case R_ARM_TLS_LE32:
  case R_ARM_TLS_TPOFF32:
    return (int32_t)read32le(location);
  case R_ARM_PREL31:
    return llvm::SignExtend64<31>(read32le(location) & 0x7FFFFFFF);
  case R_ARM_THM_CALL:
  case R_ARM_THM_JUMP24:
    return readAddend_THM_CALL(location);
  case R_ARM_THM_JUMP11:
    return readAddend_THM_JUMP11(location);
  case R_ARM_CALL:
  case R_ARM_JUMP24:
    return readAddend_ARM_CALL(location);
  case R_ARM_MOVW_ABS_NC:
  case R_ARM_MOVT_ABS:
    return readAddend_ARM_MOV(location);
  case R_ARM_THM_MOVW_ABS_NC:
  case R_ARM_THM_MOVT_ABS:
    return readAddend_THM_MOV(location);
  default:
    return 0;
  }
}

static inline std::error_code make_unsupported_range_group_reloc_error() {
  return make_dynamic_error_code(
      "Negative offsets for group relocations are not implemented");
}

static inline std::error_code applyArmReloc(uint8_t *location, uint32_t result,
                                            uint32_t mask = 0xFFFFFFFF) {
  assert(!(result & ~mask));
  write32le(location, (read32le(location) & ~mask) | (result & mask));
  return std::error_code();
}

static inline std::error_code applyThumb32Reloc(uint8_t *location,
                                                uint16_t resHi, uint16_t resLo,
                                                uint16_t maskHi,
                                                uint16_t maskLo = 0xFFFF) {
  assert(!(resHi & ~maskHi) && !(resLo & ~maskLo));
  write16le(location, (read16le(location) & ~maskHi) | (resHi & maskHi));
  location += 2;
  write16le(location, (read16le(location) & ~maskLo) | (resLo & maskLo));
  return std::error_code();
}

static inline std::error_code
applyThumb16Reloc(uint8_t *location, uint16_t result, uint16_t mask = 0xFFFF) {
  assert(!(result & ~mask));
  write16le(location, (read16le(location) & ~mask) | (result & mask));
  return std::error_code();
}

/// \brief R_ARM_ABS32 - (S + A) | T
static std::error_code relocR_ARM_ABS32(uint8_t *location, uint64_t P,
                                        uint64_t S, int64_t A,
                                        bool addressesThumb) {
  uint64_t T = addressesThumb;
  uint32_t result = (uint32_t)((S + A) | T);

  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
        llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
        llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
        llvm::dbgs() << " T: 0x" << Twine::utohexstr(T);
        llvm::dbgs() << " result: 0x" << Twine::utohexstr(result) << "\n");
  return applyArmReloc(location, result);
}

/// \brief R_ARM_REL32 - ((S + A) | T) - P
static std::error_code relocR_ARM_REL32(uint8_t *location, uint64_t P,
                                        uint64_t S, int64_t A,
                                        bool addressesThumb) {
  uint64_t T = addressesThumb;
  uint32_t result = (uint32_t)(((S + A) | T) - P);

  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
        llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
        llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
        llvm::dbgs() << " T: 0x" << Twine::utohexstr(T);
        llvm::dbgs() << " result: 0x" << Twine::utohexstr(result) << "\n");
  return applyArmReloc(location, result);
}

/// \brief R_ARM_PREL31 - ((S + A) | T) - P
static std::error_code relocR_ARM_PREL31(uint8_t *location, uint64_t P,
                                         uint64_t S, int64_t A,
                                         bool addressesThumb) {
  uint64_t T = addressesThumb;
  uint32_t result = (uint32_t)(((S + A) | T) - P);
  if (!llvm::isInt<31>((int32_t)result))
    return make_out_of_range_reloc_error();

  const uint32_t mask = 0x7FFFFFFF;
  uint32_t rel31 = result & mask;

  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
        llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
        llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
        llvm::dbgs() << " T: 0x" << Twine::utohexstr(T);
        llvm::dbgs() << " result: 0x" << Twine::utohexstr(result);
        llvm::dbgs() << " rel31: 0x" << Twine::utohexstr(rel31) << "\n");

  return applyArmReloc(location, rel31, mask);
}

/// \brief Relocate B/BL instructions. useJs defines whether J1 & J2 are used
static std::error_code relocR_ARM_THM_B_L(uint8_t *location, uint32_t result,
                                          bool useJs) {
  if ((useJs && !llvm::isInt<25>((int32_t)result)) ||
      (!useJs && !llvm::isInt<23>((int32_t)result)))
    return make_out_of_range_reloc_error();

  result = (result & 0x01FFFFFE) >> 1;

  const uint16_t imm10 = (result >> 11) & 0x3FF;
  const uint16_t bitS = (result >> 23) & 0x1;
  const uint16_t resHi = (bitS << 10) | imm10;

  const uint16_t imm11 = result & 0x7FF;
  const uint16_t bitJ2 = useJs ? ((result >> 21) & 0x1) : bitS;
  const uint16_t bitI2 = (~(bitJ2 ^ bitS)) & 0x1;
  const uint16_t bitJ1 = useJs ? ((result >> 22) & 0x1) : bitS;
  const uint16_t bitI1 = (~(bitJ1 ^ bitS)) & 0x1;
  const uint16_t resLo = (bitI1 << 13) | (bitI2 << 11) | imm11;

  return applyThumb32Reloc(location, resHi, resLo, 0x7FF, 0x2FFF);
}

/// \brief R_ARM_THM_CALL - ((S + A) | T) - P
static std::error_code relocR_ARM_THM_CALL(uint8_t *location, uint64_t P,
                                           uint64_t S, int64_t A, bool useJs,
                                           bool addressesThumb) {
  uint64_t T = addressesThumb;
  const bool switchMode = !addressesThumb;

  if (switchMode) {
    P &= ~0x3; // Align(P, 4) by rounding down
  }

  uint32_t result = (uint32_t)(((S + A) | T) - P);

  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
        llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
        llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
        llvm::dbgs() << " T: 0x" << Twine::utohexstr(T);
        llvm::dbgs() << " result: 0x" << Twine::utohexstr(result) << "\n");
  if (auto ec = relocR_ARM_THM_B_L(location, result, useJs))
    return ec;

  if (switchMode) {
    return applyThumb32Reloc(location, 0, 0, 0, 0x1001);
  }
  return std::error_code();
}

/// \brief R_ARM_THM_JUMP24 - ((S + A) | T) - P
static std::error_code relocR_ARM_THM_JUMP24(uint8_t *location, uint64_t P,
                                             uint64_t S, int64_t A,
                                             bool addressesThumb) {
  uint64_t T = addressesThumb;
  uint32_t result = (uint32_t)(((S + A) | T) - P);

  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
        llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
        llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
        llvm::dbgs() << " T: 0x" << Twine::utohexstr(T);
        llvm::dbgs() << " result: 0x" << Twine::utohexstr(result) << "\n");
  return relocR_ARM_THM_B_L(location, result, true);
}

/// \brief R_ARM_THM_JUMP11 - S + A - P
static std::error_code relocR_ARM_THM_JUMP11(uint8_t *location, uint64_t P,
                                             uint64_t S, int64_t A) {
  uint32_t result = (uint32_t)(S + A - P);

  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
        llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
        llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
        llvm::dbgs() << " result: 0x" << Twine::utohexstr(result) << "\n");

  if (!llvm::isInt<12>((int32_t)result))
    return make_out_of_range_reloc_error();

  // we cut off first bit because it is always 1 according to p. 4.5.3
  result = (result & 0x0FFE) >> 1;
  return applyThumb16Reloc(location, result, 0x7FF);
}

/// \brief R_ARM_BASE_PREL - B(S) + A - P => S + A - P
static std::error_code relocR_ARM_BASE_PREL(uint8_t *location, uint64_t P,
                                            uint64_t S, int64_t A) {
  uint32_t result = (uint32_t)(S + A - P);
  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
        llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
        llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
        llvm::dbgs() << " result: 0x" << Twine::utohexstr(result) << "\n");
  return applyArmReloc(location, result);
}

/// \brief R_ARM_GOT_BREL - GOT(S) + A - GOT_ORG => S + A - GOT_ORG
static std::error_code relocR_ARM_GOT_BREL(uint8_t *location, uint64_t P,
                                           uint64_t S, int64_t A,
                                           uint64_t GOT_ORG) {
  uint32_t result = (uint32_t)(S + A - GOT_ORG);
  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
        llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
        llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
        llvm::dbgs() << " result: 0x" << Twine::utohexstr(result) << "\n");
  return applyArmReloc(location, result);
}

/// \brief R_ARM_CALL - ((S + A) | T) - P
static std::error_code relocR_ARM_CALL(uint8_t *location, uint64_t P,
                                       uint64_t S, int64_t A,
                                       bool addressesThumb) {
  uint64_t T = addressesThumb;
  const bool switchMode = addressesThumb;

  uint32_t result = (uint32_t)(((S + A) | T) - P);
  if (!llvm::isInt<26>((int32_t)result))
    return make_out_of_range_reloc_error();

  const uint32_t imm24 = (result & 0x03FFFFFC) >> 2;

  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
        llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
        llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
        llvm::dbgs() << " T: 0x" << Twine::utohexstr(T);
        llvm::dbgs() << " result: 0x" << Twine::utohexstr(result) << "\n");
  if (auto ec = applyArmReloc(location, imm24, 0xFFFFFF))
    return ec;

  if (switchMode) {
    const uint32_t bitH = (result & 0x2) >> 1;
    return applyArmReloc(location, (0xFA | bitH) << 24, 0xFF000000);
  }
  return std::error_code();
}

/// \brief R_ARM_JUMP24 - ((S + A) | T) - P
static std::error_code relocR_ARM_JUMP24(uint8_t *location, uint64_t P,
                                         uint64_t S, int64_t A,
                                         bool addressesThumb) {
  uint64_t T = addressesThumb;
  uint32_t result = (uint32_t)(((S + A) | T) - P);
  if (!llvm::isInt<26>((int32_t)result))
    return make_out_of_range_reloc_error();

  const uint32_t imm24 = (result & 0x03FFFFFC) >> 2;

  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
        llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
        llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
        llvm::dbgs() << " T: 0x" << Twine::utohexstr(T);
        llvm::dbgs() << " result: 0x" << Twine::utohexstr(result) << "\n");
  return applyArmReloc(location, imm24, 0xFFFFFF);
}

/// \brief Relocate ARM MOVW/MOVT instructions
static std::error_code relocR_ARM_MOV(uint8_t *location, uint32_t result) {
  const uint32_t imm12 = result & 0xFFF;
  const uint32_t imm4 = (result >> 12) & 0xF;

  return applyArmReloc(location, (imm4 << 16) | imm12, 0xF0FFF);
}

/// \brief R_ARM_MOVW_ABS_NC - (S + A) | T
static std::error_code relocR_ARM_MOVW_ABS_NC(uint8_t *location, uint64_t P,
                                              uint64_t S, int64_t A,
                                              bool addressesThumb) {
  uint64_t T = addressesThumb;
  uint32_t result = (uint32_t)((S + A) | T);
  const uint32_t arg = result & 0x0000FFFF;

  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
        llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
        llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
        llvm::dbgs() << " T: 0x" << Twine::utohexstr(T);
        llvm::dbgs() << " result: 0x" << Twine::utohexstr(result) << "\n");
  return relocR_ARM_MOV(location, arg);
}

/// \brief R_ARM_MOVT_ABS - S + A
static std::error_code relocR_ARM_MOVT_ABS(uint8_t *location, uint64_t P,
                                           uint64_t S, int64_t A) {
  uint32_t result = (uint32_t)(S + A);
  const uint32_t arg = (result & 0xFFFF0000) >> 16;

  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
        llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
        llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
        llvm::dbgs() << " result: 0x" << Twine::utohexstr(result) << "\n");
  return relocR_ARM_MOV(location, arg);
}

/// \brief Relocate Thumb MOVW/MOVT instructions
static std::error_code relocR_ARM_THM_MOV(uint8_t *location, uint32_t result) {
  const uint16_t imm8 = result & 0xFF;
  const uint16_t imm3 = (result >> 8) & 0x7;
  const uint16_t resLo = (imm3 << 12) | imm8;

  const uint16_t imm4 = (result >> 12) & 0xF;
  const uint16_t bitI = (result >> 11) & 0x1;
  const uint16_t resHi = (bitI << 10) | imm4;

  return applyThumb32Reloc(location, resHi, resLo, 0x40F, 0x70FF);
}

/// \brief R_ARM_THM_MOVW_ABS_NC - (S + A) | T
static std::error_code relocR_ARM_THM_MOVW_ABS_NC(uint8_t *location, uint64_t P,
                                                  uint64_t S, int64_t A,
                                                  bool addressesThumb) {
  uint64_t T = addressesThumb;
  uint32_t result = (uint32_t)((S + A) | T);
  const uint32_t arg = result & 0x0000FFFF;

  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
        llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
        llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
        llvm::dbgs() << " T: 0x" << Twine::utohexstr(T);
        llvm::dbgs() << " result: 0x" << Twine::utohexstr(result) << "\n");
  return relocR_ARM_THM_MOV(location, arg);
}

/// \brief R_ARM_THM_MOVT_ABS - S + A
static std::error_code relocR_ARM_THM_MOVT_ABS(uint8_t *location, uint64_t P,
                                               uint64_t S, int64_t A) {
  uint32_t result = (uint32_t)(S + A);
  const uint32_t arg = (result & 0xFFFF0000) >> 16;

  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
        llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
        llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
        llvm::dbgs() << " result: 0x" << Twine::utohexstr(result) << "\n");
  return relocR_ARM_THM_MOV(location, arg);
}

/// \brief R_ARM_TLS_IE32 - GOT(S) + A - P => S + A - P
static std::error_code relocR_ARM_TLS_IE32(uint8_t *location, uint64_t P,
                                           uint64_t S, int64_t A) {
  uint32_t result = (uint32_t)(S + A - P);

  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
        llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
        llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
        llvm::dbgs() << " result: 0x" << Twine::utohexstr(result) << "\n");
  return applyArmReloc(location, result);
}

/// \brief R_ARM_TLS_LE32 - S + A - tp => S + A + tpoff
static std::error_code relocR_ARM_TLS_LE32(uint8_t *location, uint64_t P,
                                           uint64_t S, int64_t A,
                                           uint64_t tpoff) {
  uint32_t result = (uint32_t)(S + A + tpoff);

  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
        llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
        llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
        llvm::dbgs() << " result: 0x" << Twine::utohexstr(result) << "\n");
  return applyArmReloc(location, result);
}

/// \brief R_ARM_TLS_TPOFF32 - S + A - tp => S + A (offset within TLS block)
static std::error_code relocR_ARM_TLS_TPOFF32(uint8_t *location, uint64_t P,
                                              uint64_t S, int64_t A) {
  uint32_t result = (uint32_t)(S + A);

  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
        llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
        llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
        llvm::dbgs() << " result: 0x" << Twine::utohexstr(result) << "\n");
  return applyArmReloc(location, result);
}

template <uint32_t lshift>
static std::error_code relocR_ARM_ALU_PC_GN_NC(uint8_t *location,
                                               uint32_t result) {
  static_assert(lshift < 32 && lshift % 2 == 0,
                "lshift must be even and less than word size");

  const uint32_t rshift = 32 - lshift;
  result = ((result >> lshift) & 0xFF) | ((rshift / 2) << 8);

  return applyArmReloc(location, result, 0xFFF);
}

/// \brief R_ARM_ALU_PC_G0_NC - ((S + A) | T) - P => S + A - P
static std::error_code relocR_ARM_ALU_PC_G0_NC(uint8_t *location, uint64_t P,
                                               uint64_t S, int64_t A) {
  int32_t result = (int32_t)(S + A - P);
  if (result < 0)
    return make_unsupported_range_group_reloc_error();

  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
        llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
        llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
        llvm::dbgs() << " result: 0x" << Twine::utohexstr((uint32_t)result)
                     << "\n");

  return relocR_ARM_ALU_PC_GN_NC<20>(location, (uint32_t)result);
}

/// \brief R_ARM_ALU_PC_G1_NC - ((S + A) | T) - P => S + A - P
static std::error_code relocR_ARM_ALU_PC_G1_NC(uint8_t *location, uint64_t P,
                                               uint64_t S, int64_t A) {
  int32_t result = (int32_t)(S + A - P);
  if (result < 0)
    return make_unsupported_range_group_reloc_error();

  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
        llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
        llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
        llvm::dbgs() << " result: 0x" << Twine::utohexstr((uint32_t)result)
                     << "\n");

  return relocR_ARM_ALU_PC_GN_NC<12>(location, (uint32_t)result);
}

/// \brief R_ARM_LDR_PC_G2 - S + A - P
static std::error_code relocR_ARM_LDR_PC_G2(uint8_t *location, uint64_t P,
                                            uint64_t S, int64_t A) {
  int32_t result = (int32_t)(S + A - P);
  if (result < 0)
    return make_unsupported_range_group_reloc_error();

  DEBUG(llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
        llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
        llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
        llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
        llvm::dbgs() << " result: 0x" << Twine::utohexstr((uint32_t)result)
                     << "\n");

  const uint32_t mask = 0xFFF;
  return applyArmReloc(location, (uint32_t)result & mask, mask);
}

/// \brief Fixup unresolved weak reference with NOP instruction
static bool fixupUnresolvedWeakCall(uint8_t *location,
                                    Reference::KindValue kindValue) {
  // TODO: workaround for archs without NOP instruction
  switch (kindValue) {
  case R_ARM_THM_CALL:
  case R_ARM_THM_JUMP24:
    // Thumb32 NOP.W
    write32le(location, 0x8000F3AF);
    break;
  case R_ARM_THM_JUMP11:
    // Thumb16 NOP
    write16le(location, 0xBF00);
    break;
  case R_ARM_CALL:
  case R_ARM_JUMP24:
    // A1 NOP<c>, save condition bits
    applyArmReloc(location, 0x320F000, 0xFFFFFFF);
    break;
  default:
    return false;
  }

  return true;
}

std::error_code ARMTargetRelocationHandler::applyRelocation(
    ELFWriter &writer, llvm::FileOutputBuffer &buf, const AtomLayout &atom,
    const Reference &ref) const {
  uint8_t *atomContent = buf.getBufferStart() + atom._fileOffset;
  uint8_t *loc = atomContent + ref.offsetInAtom();
  uint64_t target = writer.addressOfAtom(ref.target());
  uint64_t reloc = atom._virtualAddr + ref.offsetInAtom();

  if (ref.kindNamespace() != Reference::KindNamespace::ELF)
    return std::error_code();
  assert(ref.kindArch() == Reference::KindArch::ARM);

  // Fixup unresolved weak references
  if (!target) {
    bool isCallFixed = fixupUnresolvedWeakCall(loc, ref.kindValue());

    if (isCallFixed) {
      DEBUG(llvm::dbgs() << "\t\tFixup unresolved weak reference '";
            llvm::dbgs() << ref.target()->name() << "'";
            llvm::dbgs() << " at address: 0x" << Twine::utohexstr(reloc);
            llvm::dbgs() << (isCallFixed ? "\n" : " isn't possible\n"));
      return std::error_code();
    }
  }

  // Calculate proper initial addend for the relocation
  const Reference::Addend addend =
      readAddend(loc, ref.kindValue()) + ref.addend();

  // Flags that the relocation addresses Thumb instruction
  bool thumb = false;
  if (const auto *definedAtom = dyn_cast<DefinedAtom>(ref.target())) {
    thumb = isThumbCode(definedAtom);
  }

  switch (ref.kindValue()) {
  case R_ARM_NONE:
    return std::error_code();
  case R_ARM_ABS32:
    return relocR_ARM_ABS32(loc, reloc, target, addend, thumb);
  case R_ARM_REL32:
    return relocR_ARM_REL32(loc, reloc, target, addend, thumb);
  case R_ARM_TARGET1:
    if (_armLayout.target1Rel())
      return relocR_ARM_REL32(loc, reloc, target, addend, thumb);
    else
      return relocR_ARM_ABS32(loc, reloc, target, addend, thumb);
  case R_ARM_THM_CALL:
    // TODO: consider adding bool variable to disable J1 & J2 for archs
    // before ARMv6
    return relocR_ARM_THM_CALL(loc, reloc, target, addend, true, thumb);
  case R_ARM_CALL:
    return relocR_ARM_CALL(loc, reloc, target, addend, thumb);
  case R_ARM_JUMP24:
    return relocR_ARM_JUMP24(loc, reloc, target, addend, thumb);
  case R_ARM_THM_JUMP24:
    return relocR_ARM_THM_JUMP24(loc, reloc, target, addend, thumb);
  case R_ARM_THM_JUMP11:
    return relocR_ARM_THM_JUMP11(loc, reloc, target, addend);
  case R_ARM_MOVW_ABS_NC:
    return relocR_ARM_MOVW_ABS_NC(loc, reloc, target, addend, thumb);
  case R_ARM_MOVT_ABS:
    return relocR_ARM_MOVT_ABS(loc, reloc, target, addend);
  case R_ARM_THM_MOVW_ABS_NC:
    return relocR_ARM_THM_MOVW_ABS_NC(loc, reloc, target, addend, thumb);
  case R_ARM_THM_MOVT_ABS:
    return relocR_ARM_THM_MOVT_ABS(loc, reloc, target, addend);
  case R_ARM_PREL31:
    return relocR_ARM_PREL31(loc, reloc, target, addend, thumb);
  case R_ARM_TLS_IE32:
    return relocR_ARM_TLS_IE32(loc, reloc, target, addend);
  case R_ARM_TLS_LE32:
    return relocR_ARM_TLS_LE32(loc, reloc, target, addend,
                               _armLayout.getTPOffset());
  case R_ARM_TLS_TPOFF32:
    return relocR_ARM_TLS_TPOFF32(loc, reloc, target, addend);
  case R_ARM_GOT_BREL:
    return relocR_ARM_GOT_BREL(loc, reloc, target, addend,
                               _armLayout.getGOTSymAddr());
  case R_ARM_BASE_PREL:
    // GOT origin is used for NULL symbol and when explicitly specified
    if (!target || ref.target()->name().equals("_GLOBAL_OFFSET_TABLE_")) {
      target = _armLayout.getGOTSymAddr();
    } else {
      return make_dynamic_error_code(
          "Segment-base relative addressing is not supported");
    }
    return relocR_ARM_BASE_PREL(loc, reloc, target, addend);
  case R_ARM_ALU_PC_G0_NC:
    return relocR_ARM_ALU_PC_G0_NC(loc, reloc, target, addend);
  case R_ARM_ALU_PC_G1_NC:
    return relocR_ARM_ALU_PC_G1_NC(loc, reloc, target, addend);
  case R_ARM_LDR_PC_G2:
    return relocR_ARM_LDR_PC_G2(loc, reloc, target, addend);
  case R_ARM_JUMP_SLOT:
  case R_ARM_IRELATIVE:
    // Runtime only relocations. Ignore here.
    return std::error_code();
  case R_ARM_V4BX:
    // TODO implement
    return std::error_code();
  default:
    return make_unhandled_reloc_error();
  }

  llvm_unreachable("All switch cases must return directly");
}
