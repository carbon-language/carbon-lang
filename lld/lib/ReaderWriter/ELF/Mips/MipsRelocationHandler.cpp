//===- lib/ReaderWriter/ELF/Mips/MipsRelocationHandler.cpp ----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MipsTargetHandler.h"
#include "MipsLinkingContext.h"
#include "MipsRelocationHandler.h"
#include "lld/ReaderWriter/RelocationHelperFunctions.h"

using namespace lld;
using namespace elf;
using namespace llvm::ELF;

static inline void applyReloc(uint8_t *loc, uint32_t result, uint32_t mask) {
  auto target = reinterpret_cast<llvm::support::ulittle32_t *>(loc);
  *target = (uint32_t(*target) & ~mask) | (result & mask);
}

template <size_t BITS, class T> inline T signExtend(T val) {
  if (val & (T(1) << (BITS - 1)))
    val |= T(-1) << BITS;
  return val;
}

/// \brief R_MIPS_32
/// local/external: word32 S + A (truncate)
static void reloc32(uint8_t *location, uint64_t P, uint64_t S, int64_t A) {
  applyReloc(location, S + A, 0xffffffff);
}

/// \brief R_MIPS_PC32
/// local/external: word32 S + A i- P (truncate)
void relocpc32(uint8_t *location, uint64_t P, uint64_t S, int64_t A) {
  applyReloc(location, S + A - P, 0xffffffff);
}

/// \brief R_MIPS_26
/// local   : ((A | ((P + 4) & 0x3F000000)) + S) >> 2
static void reloc26loc(uint8_t *location, uint64_t P, uint64_t S, int32_t A) {
  uint32_t result = ((A << 2) | ((P + 4) & 0x3f000000)) + S;
  applyReloc(location, result >> 2, 0x03ffffff);
}

/// \brief LLD_R_MIPS_GLOBAL_26
/// external: (sign-extend(A) + S) >> 2
static void reloc26ext(uint8_t *location, uint64_t S, int32_t A) {
  uint32_t result = signExtend<28>(A << 2) + S;
  applyReloc(location, result >> 2, 0x03ffffff);
}

/// \brief R_MIPS_HI16
/// local/external: hi16 (AHL + S) - (short)(AHL + S) (truncate)
/// _gp_disp      : hi16 (AHL + GP - P) - (short)(AHL + GP - P) (verify)
static void relocHi16(uint8_t *location, uint64_t P, uint64_t S, int64_t AHL,
                      bool isGPDisp) {
  int32_t result = 0;

  if (isGPDisp)
    result = (AHL + S - P) - (int16_t)(AHL + S - P);
  else
    result = (AHL + S) - (int16_t)(AHL + S);

  applyReloc(location, result >> 16, 0xffff);
}

/// \brief R_MIPS_LO16
/// local/external: lo16 AHL + S (truncate)
/// _gp_disp      : lo16 AHL + GP - P + 4 (verify)
static void relocLo16(uint8_t *location, uint64_t P, uint64_t S, int64_t AHL,
                      bool isGPDisp) {
  int32_t result = 0;

  if (isGPDisp)
    result = AHL + S - P + 4;
  else
    result = AHL + S;

  applyReloc(location, result, 0xffff);
}

/// \brief R_MIPS_GOT16, R_MIPS_CALL16
/// rel16 G (verify)
static void relocGOT(uint8_t *location, uint64_t P, uint64_t S, int64_t A,
                     uint64_t GP) {
  int32_t G = (int32_t)(S - GP);
  applyReloc(location, G, 0xffff);
}

/// \brief R_MIPS_TLS_DTPREL_HI16, R_MIPS_TLS_TPREL_HI16, LLD_R_MIPS_HI16
/// (S + A) >> 16
static void relocGeneralHi16(uint8_t *location, uint64_t S, int64_t A) {
  int32_t result = S + A + 0x8000;
  applyReloc(location, result >> 16, 0xffff);
}

/// \brief R_MIPS_TLS_DTPREL_LO16, R_MIPS_TLS_TPREL_LO16, LLD_R_MIPS_LO16
/// S + A
static void relocGeneralLo16(uint8_t *location, uint64_t S, int64_t A) {
  int32_t result = S + A;
  applyReloc(location, result, 0xffff);
}

/// \brief R_MIPS_GPREL32
/// local: rel32 A + S + GP0 – GP (truncate)
static void relocGPRel32(uint8_t *location, uint64_t P, uint64_t S, int64_t A,
                         uint64_t GP) {
  int32_t result = A + S + 0 - GP;
  applyReloc(location, result, 0xffffffff);
}

/// \brief LLD_R_MIPS_32_HI16
static void reloc32hi16(uint8_t *location, uint64_t S, int64_t A) {
  applyReloc(location, (S + A + 0x8000) & 0xffff0000, 0xffffffff);
}

std::error_code MipsTargetRelocationHandler::applyRelocation(
    ELFWriter &writer, llvm::FileOutputBuffer &buf, const lld::AtomLayout &atom,
    const Reference &ref) const {
  if (ref.kindNamespace() != lld::Reference::KindNamespace::ELF)
    return std::error_code();
  assert(ref.kindArch() == Reference::KindArch::Mips);

  AtomLayout *gpAtom = _mipsTargetLayout.getGP();
  uint64_t gpAddr = gpAtom ? gpAtom->_virtualAddr : 0;

  AtomLayout *gpDispAtom = _mipsTargetLayout.getGPDisp();
  bool isGpDisp = gpDispAtom && ref.target() == gpDispAtom->_atom;

  uint8_t *atomContent = buf.getBufferStart() + atom._fileOffset;
  uint8_t *location = atomContent + ref.offsetInAtom();
  uint64_t targetVAddress = writer.addressOfAtom(ref.target());
  uint64_t relocVAddress = atom._virtualAddr + ref.offsetInAtom();

  switch (ref.kindValue()) {
  case R_MIPS_NONE:
    break;
  case R_MIPS_32:
    reloc32(location, relocVAddress, targetVAddress, ref.addend());
    break;
  case R_MIPS_26:
    reloc26loc(location, relocVAddress, targetVAddress, ref.addend());
    break;
  case R_MIPS_HI16:
    relocHi16(location, relocVAddress, targetVAddress, ref.addend(), isGpDisp);
    break;
  case R_MIPS_LO16:
    relocLo16(location, relocVAddress, targetVAddress, ref.addend(), isGpDisp);
    break;
  case R_MIPS_GOT16:
    relocGOT(location, relocVAddress, targetVAddress, ref.addend(), gpAddr);
    break;
  case R_MIPS_CALL16:
    relocGOT(location, relocVAddress, targetVAddress, ref.addend(), gpAddr);
    break;
  case R_MIPS_TLS_GD:
    relocGOT(location, relocVAddress, targetVAddress, ref.addend(), gpAddr);
    break;
  case R_MIPS_TLS_LDM:
  case R_MIPS_TLS_GOTTPREL:
    relocGOT(location, relocVAddress, targetVAddress, ref.addend(), gpAddr);
    break;
  case R_MIPS_TLS_DTPREL_HI16:
  case R_MIPS_TLS_TPREL_HI16:
    relocGeneralHi16(location, targetVAddress, ref.addend());
    break;
  case R_MIPS_TLS_DTPREL_LO16:
  case R_MIPS_TLS_TPREL_LO16:
    relocGeneralLo16(location, targetVAddress, ref.addend());
    break;
  case R_MIPS_GPREL32:
    relocGPRel32(location, relocVAddress, targetVAddress, ref.addend(), gpAddr);
    break;
  case R_MIPS_JALR:
    // We do not do JALR optimization now.
    break;
  case R_MIPS_REL32:
  case R_MIPS_JUMP_SLOT:
  case R_MIPS_COPY:
  case R_MIPS_TLS_DTPMOD32:
  case R_MIPS_TLS_DTPREL32:
  case R_MIPS_TLS_TPREL32:
    // Ignore runtime relocations.
    break;
  case R_MIPS_PC32:
    relocpc32(location, relocVAddress, targetVAddress, ref.addend());
    break;
  case LLD_R_MIPS_GLOBAL_GOT:
    // Do nothing.
    break;
  case LLD_R_MIPS_32_HI16:
    reloc32hi16(location, targetVAddress, ref.addend());
    break;
  case LLD_R_MIPS_GLOBAL_26:
    reloc26ext(location, targetVAddress, ref.addend());
    break;
  case LLD_R_MIPS_HI16:
    relocGeneralHi16(location, targetVAddress, 0);
    break;
  case LLD_R_MIPS_LO16:
    relocGeneralLo16(location, targetVAddress, 0);
    break;
  case LLD_R_MIPS_STO_PLT:
    // Do nothing.
    break;
  default: {
    std::string str;
    llvm::raw_string_ostream s(str);
    s << "Unhandled Mips relocation: " << ref.kindValue();
    llvm_unreachable(s.str().c_str());
  }
  }

  return std::error_code();
}
