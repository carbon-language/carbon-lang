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

namespace {

inline
void applyReloc(uint8_t *loc, uint32_t result, uint32_t mask) {
  auto target = reinterpret_cast<llvm::support::ulittle32_t *>(loc);
  *target = (uint32_t(*target) & ~mask) | (result & mask);
}

/// \brief Calculate AHL value combines addends from 'hi' and 'lo' relocations.
inline int64_t calcAHL(int64_t AHI, int64_t ALO) {
  AHI &= 0xffff;
  ALO &= 0xffff;
  return (AHI << 16) + (int16_t)ALO;
}

template <size_t BITS, class T> inline T signExtend(T val) {
  if (val & (T(1) << (BITS - 1)))
    val |= T(-1) << BITS;
  return val;
}

/// \brief R_MIPS_32
/// local/external: word32 S + A (truncate)
void reloc32(uint8_t *location, uint64_t P, uint64_t S, int64_t A) {
  applyReloc(location, S + A, 0xffffffff);
}

/// \brief R_MIPS_26
/// local   : ((A | ((P + 4) & 0x3F000000)) + S) >> 2
void reloc26loc(uint8_t *location, uint64_t P, uint64_t S, int32_t A) {
  uint32_t result = ((A << 2) | ((P + 4) & 0x3f000000)) + S;
  applyReloc(location, result >> 2, 0x03ffffff);
}

/// \brief LLD_R_MIPS_GLOBAL_26
/// external: (sign–extend(A) + S) >> 2
void reloc26ext(uint8_t *location, uint64_t S, int32_t A) {
  uint32_t result = signExtend<28>(A << 2) + S;
  applyReloc(location, result >> 2, 0x03ffffff);
}

/// \brief R_MIPS_HI16
/// local/external: hi16 (AHL + S) - (short)(AHL + S) (truncate)
/// _gp_disp      : hi16 (AHL + GP - P) - (short)(AHL + GP - P) (verify)
void relocHi16(uint8_t *location, uint64_t P, uint64_t S, int64_t AHL,
               uint64_t GP, bool isGPDisp) {
  int32_t result = 0;

  if (isGPDisp)
    result = (AHL + GP - P) - (int16_t)(AHL + GP - P);
  else
    result = (AHL + S) - (int16_t)(AHL + S);

  applyReloc(location, result >> 16, 0xffff);
}

/// \brief R_MIPS_LO16
/// local/external: lo16 AHL + S (truncate)
/// _gp_disp      : lo16 AHL + GP - P + 4 (verify)
void relocLo16(uint8_t *location, uint64_t P, uint64_t S, int64_t AHL,
               uint64_t GP, bool isGPDisp) {
  int32_t result = 0;

  if (isGPDisp)
    result = AHL + GP - P + 4;
  else
    result = AHL + S;

  applyReloc(location, result, 0xffff);
}

/// \brief R_MIPS_GOT16
/// local/external: rel16 G (verify)
void relocGOT16(uint8_t *location, uint64_t P, uint64_t S, int64_t AHL,
                uint64_t GP) {
  // FIXME (simon): for local sym put high 16 bit of AHL to the GOT
  int32_t G = (int32_t)(S - GP);
  applyReloc(location, G, 0xffff);
}

/// \brief R_MIPS_CALL16
/// external: rel16 G (verify)
void relocCall16(uint8_t *location, uint64_t P, uint64_t S, int64_t A,
                 uint64_t GP) {
  int32_t G = (int32_t)(S - GP);
  applyReloc(location, G, 0xffff);
}

/// \brief LLD_R_MIPS_HI16
void relocLldHi16(uint8_t *location, uint64_t S) {
  applyReloc(location, (S + 0x8000) >> 16, 0xffff);
}

/// \brief LLD_R_MIPS_LO16
void relocLldLo16(uint8_t *location, uint64_t S) {
  applyReloc(location, S, 0xffff);
}

} // end anon namespace

MipsTargetRelocationHandler::~MipsTargetRelocationHandler() {
  assert(_pairedRelocations.empty());
}

void
MipsTargetRelocationHandler::savePairedRelocation(const lld::AtomLayout &atom,
                                                  const Reference &ref) const {
  auto pi = _pairedRelocations.find(&atom);
  if (pi == _pairedRelocations.end())
    pi = _pairedRelocations.emplace(&atom, PairedRelocationsT()).first;

  pi->second.push_back(&ref);
}

void MipsTargetRelocationHandler::applyPairedRelocations(
    ELFWriter &writer, llvm::FileOutputBuffer &buf, const lld::AtomLayout &atom,
    int64_t gpAddr, int64_t loAddend) const {
  auto pi = _pairedRelocations.find(&atom);
  if (pi == _pairedRelocations.end())
    return;

  for (auto ri : pi->second) {
    uint8_t *atomContent = buf.getBufferStart() + atom._fileOffset;
    uint8_t *location = atomContent + ri->offsetInAtom();
    uint64_t targetVAddress = writer.addressOfAtom(ri->target());
    uint64_t relocVAddress = atom._virtualAddr + ri->offsetInAtom();

    int64_t ahl = calcAHL(ri->addend(), loAddend);

    if (ri->kindNamespace() != lld::Reference::KindNamespace::ELF)
      continue;
    assert(ri->kindArch() == Reference::KindArch::Mips);
    switch (ri->kindValue()) {
    case R_MIPS_HI16:
      relocHi16(location, relocVAddress, targetVAddress, ahl, gpAddr,
                ri->target()->name() == "_gp_disp");
      break;
    case R_MIPS_GOT16:
      relocGOT16(location, relocVAddress, targetVAddress, ahl, gpAddr);
      break;
    default:
      llvm_unreachable("Unknown type of paired relocation.");
    }
  }

  _pairedRelocations.erase(pi);
}

error_code MipsTargetRelocationHandler::applyRelocation(
    ELFWriter &writer, llvm::FileOutputBuffer &buf, const lld::AtomLayout &atom,
    const Reference &ref) const {
  if (ref.kindNamespace() != lld::Reference::KindNamespace::ELF)
    return error_code::success();
  assert(ref.kindArch() == Reference::KindArch::Mips);

  AtomLayout *gpAtom = _mipsTargetLayout.getGP();
  uint64_t gpAddr = gpAtom ? gpAtom->_virtualAddr : 0;

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
    savePairedRelocation(atom, ref);
    break;
  case R_MIPS_LO16:
    relocLo16(location, relocVAddress, targetVAddress, calcAHL(0, ref.addend()),
              gpAddr, ref.target()->name() == "_gp_disp");
    applyPairedRelocations(writer, buf, atom, gpAddr, ref.addend());
    break;
  case R_MIPS_GOT16:
    savePairedRelocation(atom, ref);
    break;
  case R_MIPS_CALL16:
    relocCall16(location, relocVAddress, targetVAddress, ref.addend(), gpAddr);
    break;
  case R_MIPS_JALR:
    // We do not do JALR optimization now.
    break;
  case R_MIPS_JUMP_SLOT:
    // Ignore runtime relocations.
    break;
  case LLD_R_MIPS_GLOBAL_GOT:
    // Do nothing.
    break;
  case LLD_R_MIPS_GLOBAL_GOT16:
    relocGOT16(location, relocVAddress, targetVAddress, ref.addend(), gpAddr);
    break;
  case LLD_R_MIPS_GLOBAL_26:
    reloc26ext(location, targetVAddress, ref.addend());
    break;
  case LLD_R_MIPS_HI16:
    relocLldHi16(location, targetVAddress);
    break;
  case LLD_R_MIPS_LO16:
    relocLldLo16(location, targetVAddress);
    break;
  default: {
    std::string str;
    llvm::raw_string_ostream s(str);
    s << "Unhandled Mips relocation: " << ref.kindValue();
    llvm_unreachable(s.str().c_str());
  }
  }

  return error_code::success();
}
