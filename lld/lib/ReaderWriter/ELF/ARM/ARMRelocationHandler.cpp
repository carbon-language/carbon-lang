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
#include "llvm/Support/MathExtras.h"

using namespace lld;
using namespace elf;

static Reference::Addend readAddend_THM_MOV(const uint8_t *location) {
  const auto halfHi = uint16_t(
      *reinterpret_cast<const llvm::support::ulittle16_t *>(location));
  const auto halfLo = uint16_t(
      *reinterpret_cast<const llvm::support::ulittle16_t *>(location + 2));

  const uint16_t imm8 = halfLo & 0xFF;
  const uint16_t imm3 = (halfLo >> 12) & 0x7;

  const uint16_t imm4 = halfHi & 0xF;
  const uint16_t bitI = (halfHi >> 10) & 0x1;

  const auto result = int16_t((imm4 << 12) | (bitI << 11) | (imm3 << 8) | imm8);
  return result;
}

static Reference::Addend readAddend_ARM_MOV(const uint8_t *location) {
  const auto value = uint32_t(
      *reinterpret_cast<const llvm::support::ulittle32_t *>(location));

  const uint32_t imm12 = value & 0xFFF;
  const uint32_t imm4 = (value >> 16) & 0xF;

  const auto result = int32_t((imm4 << 12) | imm12);
  return result;
}

static Reference::Addend readAddend_THM_CALL(const uint8_t *location) {
  const auto halfHi = uint16_t(
      *reinterpret_cast<const llvm::support::ulittle16_t *>(location));
  const auto halfLo = uint16_t(
      *reinterpret_cast<const llvm::support::ulittle16_t *>(location + 2));

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
  const auto value = uint32_t(
      *reinterpret_cast<const llvm::support::ulittle32_t *>(location));

  const bool isBLX = (value & 0xF0000000) == 0xF0000000;
  const uint32_t bitH = isBLX ? ((value & 0x1000000) >> 24) : 0;

  const auto result = int32_t(((value & 0xFFFFFF) << 2) | (bitH << 1));
  return llvm::SignExtend64<26>(result);
}

static Reference::Addend readAddend(const uint8_t *location,
                                    Reference::KindValue kindValue) {
  switch (kindValue) {
  case R_ARM_ABS32:
    return int32_t(
        *reinterpret_cast<const llvm::support::little32_t *>(location));
  case R_ARM_THM_CALL:
  case R_ARM_THM_JUMP24:
    return readAddend_THM_CALL(location);
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

static inline void applyArmReloc(uint8_t *location, uint32_t result,
                                 uint32_t mask = 0xFFFFFFFF) {
  assert(!(result & ~mask));
  *reinterpret_cast<llvm::support::ulittle32_t *>(location) =
      (uint32_t(*reinterpret_cast<llvm::support::ulittle32_t *>(location)) &
       ~mask) | (result & mask);
}

static inline void applyThmReloc(uint8_t *location, uint16_t resHi,
                                 uint16_t resLo, uint16_t maskHi,
                                 uint16_t maskLo = 0xFFFF) {
  assert(!(resHi & ~maskHi) && !(resLo & ~maskLo));
  *reinterpret_cast<llvm::support::ulittle16_t *>(location) =
      (uint16_t(*reinterpret_cast<llvm::support::ulittle16_t *>(location)) &
       ~maskHi) | (resHi & maskHi);
  location += 2;
  *reinterpret_cast<llvm::support::ulittle16_t *>(location) =
      (uint16_t(*reinterpret_cast<llvm::support::ulittle16_t *>(location)) &
       ~maskLo) | (resLo & maskLo);
}

/// \brief R_ARM_ABS32 - (S + A) | T
static void relocR_ARM_ABS32(uint8_t *location, uint64_t P, uint64_t S,
                             int64_t A, bool addressesThumb) {
  uint64_t T = addressesThumb;
  uint32_t result = (uint32_t)((S + A) | T);

  DEBUG_WITH_TYPE(
      "ARM", llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
      llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
      llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
      llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
      llvm::dbgs() << " T: 0x" << Twine::utohexstr(T);
      llvm::dbgs() << " result: 0x" << Twine::utohexstr(result) << "\n");
  applyArmReloc(location, result);
}

/// \brief Relocate B/BL instructions. useJs defines whether J1 & J2 are used
static void relocR_ARM_THM_B_L(uint8_t *location, uint32_t result, bool useJs) {
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

  applyThmReloc(location, resHi, resLo, 0x7FF, 0x2FFF);
}

/// \brief R_ARM_THM_CALL - ((S + A) | T) - P
static void relocR_ARM_THM_CALL(uint8_t *location, uint64_t P, uint64_t S,
                                int64_t A, bool useJs, bool addressesThumb) {
  uint64_t T = addressesThumb;
  const bool switchMode = !addressesThumb;

  if (switchMode) {
    P &= ~0x3; // Align(P, 4) by rounding down
  }

  uint32_t result = (uint32_t)(((S + A) | T) - P);

  DEBUG_WITH_TYPE(
      "ARM", llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
      llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
      llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
      llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
      llvm::dbgs() << " T: 0x" << Twine::utohexstr(T);
      llvm::dbgs() << " result: 0x" << Twine::utohexstr(result) << "\n");
  relocR_ARM_THM_B_L(location, result, useJs);

  if (switchMode) {
    applyThmReloc(location, 0, 0, 0, 0x1001);
  }
}

/// \brief R_ARM_THM_JUMP24 - ((S + A) | T) - P
static void relocR_ARM_THM_JUMP24(uint8_t *location, uint64_t P, uint64_t S,
                                  int64_t A, bool addressesThumb) {
  uint64_t T = addressesThumb;
  uint32_t result = (uint32_t)(((S + A) | T) - P);

  DEBUG_WITH_TYPE(
      "ARM", llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
      llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
      llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
      llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
      llvm::dbgs() << " T: 0x" << Twine::utohexstr(T);
      llvm::dbgs() << " result: 0x" << Twine::utohexstr(result) << "\n");
  relocR_ARM_THM_B_L(location, result, true);
}

/// \brief R_ARM_CALL - ((S + A) | T) - P
static void relocR_ARM_CALL(uint8_t *location, uint64_t P, uint64_t S,
                            int64_t A, bool addressesThumb) {
  uint64_t T = addressesThumb;
  const bool switchMode = addressesThumb;

  uint32_t result = (uint32_t)(((S + A) | T) - P);
  const uint32_t imm24 = (result & 0x03FFFFFC) >> 2;

  DEBUG_WITH_TYPE(
      "ARM", llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
      llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
      llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
      llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
      llvm::dbgs() << " T: 0x" << Twine::utohexstr(T);
      llvm::dbgs() << " result: 0x" << Twine::utohexstr(result) << "\n");
  applyArmReloc(location, imm24, 0xFFFFFF);

  if (switchMode) {
    const uint32_t bitH = (result & 0x2) >> 1;
    applyArmReloc(location, (0xFA | bitH) << 24, 0xFF000000);
  }
}

/// \brief R_ARM_JUMP24 - ((S + A) | T) - P
static void relocR_ARM_JUMP24(uint8_t *location, uint64_t P, uint64_t S,
                              int64_t A, bool addressesThumb) {
  uint64_t T = addressesThumb;
  uint32_t result = (uint32_t)(((S + A) | T) - P);
  const uint32_t imm24 = (result & 0x03FFFFFC) >> 2;

  DEBUG_WITH_TYPE(
      "ARM", llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
      llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
      llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
      llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
      llvm::dbgs() << " T: 0x" << Twine::utohexstr(T);
      llvm::dbgs() << " result: 0x" << Twine::utohexstr(result) << "\n");
  applyArmReloc(location, imm24, 0xFFFFFF);
}

/// \brief Relocate ARM MOVW/MOVT instructions
static void relocR_ARM_MOV(uint8_t *location, uint32_t result) {
  const uint32_t imm12 = result & 0xFFF;
  const uint32_t imm4 = (result >> 12) & 0xF;

  applyArmReloc(location, (imm4 << 16) | imm12, 0xF0FFF);
}

/// \brief R_ARM_MOVW_ABS_NC - (S + A) | T
static void relocR_ARM_MOVW_ABS_NC(uint8_t *location, uint64_t P, uint64_t S,
                                   int64_t A, bool addressesThumb) {
  uint64_t T = addressesThumb;
  uint32_t result = (uint32_t)((S + A) | T);
  const uint32_t arg = result & 0x0000FFFF;

  DEBUG_WITH_TYPE(
      "ARM", llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
      llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
      llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
      llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
      llvm::dbgs() << " T: 0x" << Twine::utohexstr(T);
      llvm::dbgs() << " result: 0x" << Twine::utohexstr(result) << "\n");
  return relocR_ARM_MOV(location, arg);
}

/// \brief R_ARM_MOVT_ABS - S + A
static void relocR_ARM_MOVT_ABS(uint8_t *location, uint64_t P, uint64_t S,
                                int64_t A) {
  uint32_t result = (uint32_t)(S + A);
  const uint32_t arg = (result & 0xFFFF0000) >> 16;

  DEBUG_WITH_TYPE(
      "ARM", llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
      llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
      llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
      llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
      llvm::dbgs() << " result: 0x" << Twine::utohexstr(result) << "\n");
  return relocR_ARM_MOV(location, arg);
}

/// \brief Relocate Thumb MOVW/MOVT instructions
static void relocR_ARM_THM_MOV(uint8_t *location, uint32_t result) {
  const uint16_t imm8 = result & 0xFF;
  const uint16_t imm3 = (result >> 8) & 0x7;
  const uint16_t resLo = (imm3 << 12) | imm8;

  const uint16_t imm4 = (result >> 12) & 0xF;
  const uint16_t bitI = (result >> 11) & 0x1;
  const uint16_t resHi = (bitI << 10) | imm4;

 applyThmReloc(location, resHi, resLo, 0x40F, 0x70FF);
}

/// \brief R_ARM_THM_MOVW_ABS_NC - (S + A) | T
static void relocR_ARM_THM_MOVW_ABS_NC(uint8_t *location, uint64_t P,
                                       uint64_t S, int64_t A,
                                       bool addressesThumb) {
  uint64_t T = addressesThumb;
  uint32_t result = (uint32_t)((S + A) | T);
  const uint32_t arg = result & 0x0000FFFF;

  DEBUG_WITH_TYPE(
      "ARM", llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
      llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
      llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
      llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
      llvm::dbgs() << " T: 0x" << Twine::utohexstr(T);
      llvm::dbgs() << " result: 0x" << Twine::utohexstr(result) << "\n");
  return relocR_ARM_THM_MOV(location, arg);
}

/// \brief R_ARM_THM_MOVT_ABS - S + A
static void relocR_ARM_THM_MOVT_ABS(uint8_t *location, uint64_t P, uint64_t S,
                                    int64_t A) {
  uint32_t result = (uint32_t)(S + A);
  const uint32_t arg = (result & 0xFFFF0000) >> 16;

  DEBUG_WITH_TYPE(
      "ARM", llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
      llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
      llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
      llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
      llvm::dbgs() << " result: 0x" << Twine::utohexstr(result) << "\n");
  return relocR_ARM_THM_MOV(location, arg);
}

std::error_code ARMTargetRelocationHandler::applyRelocation(
    ELFWriter &writer, llvm::FileOutputBuffer &buf, const lld::AtomLayout &atom,
    const Reference &ref) const {
  uint8_t *atomContent = buf.getBufferStart() + atom._fileOffset;
  uint8_t *location = atomContent + ref.offsetInAtom();
  uint64_t targetVAddress = writer.addressOfAtom(ref.target());
  uint64_t relocVAddress = atom._virtualAddr + ref.offsetInAtom();

  if (ref.kindNamespace() != Reference::KindNamespace::ELF)
    return std::error_code();
  assert(ref.kindArch() == Reference::KindArch::ARM);

  // Calculate proper initial addend for the relocation
  const Reference::Addend addend =
      readAddend(location, ref.kindValue());

  // Flags that the relocation addresses Thumb instruction
  bool addressesThumb = false;

  if (const auto *definedAtom = dyn_cast<DefinedAtom>(ref.target())) {
    addressesThumb = (DefinedAtom::codeARMThumb == definedAtom->codeModel());
  }

  switch (ref.kindValue()) {
  case R_ARM_NONE:
    break;
  case R_ARM_ABS32:
    relocR_ARM_ABS32(location, relocVAddress, targetVAddress, addend,
                     addressesThumb);
    break;
  case R_ARM_THM_CALL:
    // TODO: consider adding bool variable to disable J1 & J2 for archs
    // before ARMv6
    relocR_ARM_THM_CALL(location, relocVAddress, targetVAddress, addend, true,
                        addressesThumb);
    break;
  case R_ARM_CALL:
    relocR_ARM_CALL(location, relocVAddress, targetVAddress, addend,
                    addressesThumb);
    break;
  case R_ARM_JUMP24:
    relocR_ARM_JUMP24(location, relocVAddress, targetVAddress, addend,
                      addressesThumb);
    break;
  case R_ARM_THM_JUMP24:
    relocR_ARM_THM_JUMP24(location, relocVAddress, targetVAddress, addend,
                          addressesThumb);
    break;
  case R_ARM_MOVW_ABS_NC:
    relocR_ARM_MOVW_ABS_NC(location, relocVAddress, targetVAddress, addend,
                           addressesThumb);
    break;
  case R_ARM_MOVT_ABS:
    relocR_ARM_MOVT_ABS(location, relocVAddress, targetVAddress, addend);
    break;
  case R_ARM_THM_MOVW_ABS_NC:
    relocR_ARM_THM_MOVW_ABS_NC(location, relocVAddress, targetVAddress, addend,
                               addressesThumb);
    break;
  case R_ARM_THM_MOVT_ABS:
    relocR_ARM_THM_MOVT_ABS(location, relocVAddress, targetVAddress, addend);
    break;
  default:
    return make_unhandled_reloc_error();
  }

  return std::error_code();
}
