//===- lib/ReaderWriter/ELF/Hexagon/HexagonTargetHandler.cpp --------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "HexagonTargetHandler.h"
#include "HexagonTargetInfo.h"
#include "lld/ReaderWriter/RelocationHelperFunctions.h"

using namespace lld;
using namespace elf;

using namespace llvm::ELF;

namespace {
/// \brief Word32_B22: 0x01ff3ffe : (S + A - P) >> 2 : Verify
int relocB22PCREL(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  int32_t result = (uint32_t)(((S + A) - P)>>2);
  if ((result < 0x200000) && (result > -0x200000)) {
    result = lld::scatterBits<int32_t>(result, 0x01ff3ffe);
    *reinterpret_cast<llvm::support::ulittle32_t *>(location) = result |
            (uint32_t)*reinterpret_cast<llvm::support::ulittle32_t *>(location);
    return 0;
  }
  return 1;
}

/// \brief Word32_B15: 0x00df20fe : (S + A - P) >> 2 : Verify
int relocB15PCREL(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  int32_t result = (uint32_t)(((S + A) - P)>>2);
  if ((result < 0x8000) && (result > -0x8000)) {
    result = lld::scatterBits<int32_t>(result, 0x00df20fe);
    *reinterpret_cast<llvm::support::ulittle32_t *>(location) = result |
            (uint32_t)*reinterpret_cast<llvm::support::ulittle32_t *>(location);
    return 0;
  }
  return 1;
}

/// \brief Word32_LO: 0x00c03fff : (S + A) : Truncate
int relocLO16(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  uint32_t result = (uint32_t)(S + A);
  result = lld::scatterBits<int32_t>(result, 0x00c03fff);
  *reinterpret_cast<llvm::support::ulittle32_t *>(location) = result |
            (uint32_t)*reinterpret_cast<llvm::support::ulittle32_t *>(location);
  return 0;
}

/// \brief Word32_LO: 0x00c03fff : (S + A) >> 16 : Truncate
int relocHI16(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  uint32_t result = (uint32_t)((S + A)>>16);
  result = lld::scatterBits<int32_t>(result, 0x00c03fff);
  *reinterpret_cast<llvm::support::ulittle32_t *>(location) = result |
            (uint32_t)*reinterpret_cast<llvm::support::ulittle32_t *>(location);
  return 0;
}

/// \brief Word32: 0xffffffff : (S + A) : Truncate
int reloc32(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  uint32_t result = (uint32_t)(S + A);
  *reinterpret_cast<llvm::support::ulittle32_t *>(location) = result |
            (uint32_t)*reinterpret_cast<llvm::support::ulittle32_t *>(location);
  return 0;
}
} // end anon namespace

ErrorOr<void> HexagonTargetRelocationHandler::applyRelocation(
    ELFWriter &writer, llvm::FileOutputBuffer &buf, const AtomLayout &atom,
    const Reference &ref) const {
  uint8_t *atomContent = buf.getBufferStart() + atom._fileOffset;
  uint8_t *location = atomContent + ref.offsetInAtom();
  uint64_t targetVAddress = writer.addressOfAtom(ref.target());
  uint64_t relocVAddress = atom._virtualAddr + ref.offsetInAtom();

  switch (ref.kind()) {
  case R_HEX_B22_PCREL:
    relocB22PCREL(location, relocVAddress, targetVAddress, ref.addend());
    break;
  case R_HEX_B15_PCREL:
    relocB15PCREL(location, relocVAddress, targetVAddress, ref.addend());
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
  default: {
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

HexagonTargetHandler::HexagonTargetHandler(HexagonTargetInfo &targetInfo)
    : DefaultTargetHandler(targetInfo), _relocationHandler(targetInfo),
      _targetLayout(targetInfo) {
}
