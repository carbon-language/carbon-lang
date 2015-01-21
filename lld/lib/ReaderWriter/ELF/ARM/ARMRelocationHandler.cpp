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

static Reference::Addend readAddend_ARM_CALL(const uint8_t *location) {
  const auto value = int32_t(
      *reinterpret_cast<const llvm::support::little32_t *>(location));

  const bool isBLX = (value & 0xF0000000) == 0xF0000000;
  const int32_t bitH = isBLX ? ((value & 0x1000000) >> 24) : 0;

  const int32_t result = ((value & 0xFFFFFF) << 2) | (bitH << 1);
  return llvm::SignExtend64<26>(result);
}

static Reference::Addend readAddend(const uint8_t *location,
                                    Reference::KindValue kindValue) {
  switch (kindValue) {
  case R_ARM_ABS32:
    return int32_t(
        *reinterpret_cast<const llvm::support::little32_t *>(location));
  case R_ARM_CALL:
    return readAddend_ARM_CALL(location);
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

/// \brief R_ARM_ABS32 - (S + A) | T => S + A
static void relocR_ARM_ABS32(uint8_t *location, uint64_t P, uint64_t S,
                             int64_t A) {
  uint32_t result = (uint32_t)(S + A);
  DEBUG_WITH_TYPE(
      "ARM", llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
      llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
      llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
      llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
      llvm::dbgs() << " result: 0x" << Twine::utohexstr(result) << "\n");
  applyArmReloc(location, result);
}

/// \brief R_ARM_CALL - ((S + A) | T) - P => S + A - P
static void relocR_ARM_CALL(uint8_t *location, uint64_t P, uint64_t S,
                            int64_t A) {
  uint32_t result = (uint32_t)((S + A) - P);
  const uint32_t imm24 = (result & 0x03FFFFFC) >> 2;

  DEBUG_WITH_TYPE(
      "ARM", llvm::dbgs() << "\t\tHandle " << LLVM_FUNCTION_NAME << " -";
      llvm::dbgs() << " S: 0x" << Twine::utohexstr(S);
      llvm::dbgs() << " A: 0x" << Twine::utohexstr(A);
      llvm::dbgs() << " P: 0x" << Twine::utohexstr(P);
      llvm::dbgs() << " result: 0x" << Twine::utohexstr(result) << "\n");
  applyArmReloc(location, imm24, 0xFFFFFF);
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

  switch (ref.kindValue()) {
  case R_ARM_NONE:
    break;
  case R_ARM_ABS32:
    relocR_ARM_ABS32(location, relocVAddress, targetVAddress, addend);
    break;
  case R_ARM_CALL:
    relocR_ARM_CALL(location, relocVAddress, targetVAddress, addend);
    break;
  default:
    make_unhandled_reloc_error();
  }

  return std::error_code();
}
