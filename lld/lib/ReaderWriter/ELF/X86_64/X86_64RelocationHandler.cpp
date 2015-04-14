//===- lib/ReaderWriter/ELF/X86_64/X86_64RelocationHandler.cpp ------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "X86_64LinkingContext.h"
#include "X86_64TargetHandler.h"
#include "llvm/Support/Endian.h"

using namespace lld;
using namespace lld::elf;
using namespace llvm::support::endian;

/// \brief R_X86_64_64 - word64: S + A
static void reloc64(uint8_t *location, uint64_t P, uint64_t S, int64_t A) {
  uint64_t result = S + A;
  write64le(location, result | read64le(location));
}

/// \brief R_X86_64_PC32 - word32: S + A - P
static void relocPC32(uint8_t *location, uint64_t P, uint64_t S, int64_t A) {
  uint32_t result = (uint32_t)(S + A - P);
  write32le(location, result + read32le(location));
}

/// \brief R_X86_64_32 - word32:  S + A
static void reloc32(uint8_t *location, uint64_t P, uint64_t S, int64_t A) {
  int32_t result = (uint32_t)(S + A);
  write32le(location, result | read32le(location));
  // TODO: Make sure that the result zero extends to the 64bit value.
}

/// \brief R_X86_64_32S - word32:  S + A
static void reloc32S(uint8_t *location, uint64_t P, uint64_t S, int64_t A) {
  int32_t result = (int32_t)(S + A);
  write32le(location, result | read32le(location));
  // TODO: Make sure that the result sign extends to the 64bit value.
}

/// \brief R_X86_64_16 - word16:  S + A
static void reloc16(uint8_t *location, uint64_t P, uint64_t S, int64_t A) {
  uint16_t result = (uint16_t)(S + A);
  write16le(location, result | read16le(location));
  // TODO: Check for overflow.
}

/// \brief R_X86_64_PC16 - word16: S + A - P
static void relocPC16(uint8_t *location, uint64_t P, uint64_t S, int64_t A) {
  uint16_t result = (uint16_t)(S + A - P);
  write16le(location, result | read16le(location));
  // TODO: Check for overflow.
}

/// \brief R_X86_64_PC64 - word64: S + A - P
static void relocPC64(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  int64_t result = (uint64_t)(S + A - P);
  write64le(location, result | read64le(location));
}

std::error_code X86_64TargetRelocationHandler::applyRelocation(
    ELFWriter &writer, llvm::FileOutputBuffer &buf, const AtomLayout &atom,
    const Reference &ref) const {
  uint8_t *atomContent = buf.getBufferStart() + atom._fileOffset;
  uint8_t *loc = atomContent + ref.offsetInAtom();
  uint64_t target = writer.addressOfAtom(ref.target());
  uint64_t reloc = atom._virtualAddr + ref.offsetInAtom();

  if (ref.kindNamespace() != Reference::KindNamespace::ELF)
    return std::error_code();
  assert(ref.kindArch() == Reference::KindArch::x86_64);
  switch (ref.kindValue()) {
  case R_X86_64_NONE:
    break;
  case R_X86_64_64:
    reloc64(loc, reloc, target, ref.addend());
    break;
  case R_X86_64_PC32:
  case R_X86_64_GOTPCREL:
    relocPC32(loc, reloc, target, ref.addend());
    break;
  case R_X86_64_32:
    reloc32(loc, reloc, target, ref.addend());
    break;
  case R_X86_64_32S:
    reloc32S(loc, reloc, target, ref.addend());
    break;
  case R_X86_64_16:
    reloc16(loc, reloc, target, ref.addend());
    break;
  case R_X86_64_PC16:
    relocPC16(loc, reloc, target, ref.addend());
    break;
  case R_X86_64_TPOFF64:
  case R_X86_64_DTPOFF32:
  case R_X86_64_TPOFF32: {
    _tlsSize = _layout.getTLSSize();
    if (ref.kindValue() == R_X86_64_TPOFF32 ||
        ref.kindValue() == R_X86_64_DTPOFF32) {
      write32le(loc, target - _tlsSize);
    } else {
      write64le(loc, target - _tlsSize);
    }
    break;
  }
  case R_X86_64_TLSGD: {
    relocPC32(loc, reloc, target, ref.addend());
    break;
  }
  case R_X86_64_TLSLD: {
    // Rewrite to move %fs:0 into %rax. Technically we should verify that the
    // next relocation is a PC32 to __tls_get_addr...
    static uint8_t instr[] = { 0x66, 0x66, 0x66, 0x64, 0x48, 0x8b, 0x04, 0x25,
                               0x00, 0x00, 0x00, 0x00 };
    std::memcpy(loc - 3, instr, sizeof(instr));
    break;
  }
  case R_X86_64_PC64:
    relocPC64(loc, reloc, target, ref.addend());
    break;
  case LLD_R_X86_64_GOTRELINDEX: {
    const DefinedAtom *target = cast<const DefinedAtom>(ref.target());
    for (const Reference *r : *target) {
      if (r->kindValue() == R_X86_64_JUMP_SLOT) {
        uint32_t index;
        if (!_layout.getPLTRelocationTable()->getRelocationIndex(*r, index))
          llvm_unreachable("Relocation doesn't exist");
        reloc32(loc, 0, index, 0);
        break;
      }
    }
    break;
  }
  // Runtime only relocations. Ignore here.
  case R_X86_64_RELATIVE:
  case R_X86_64_IRELATIVE:
  case R_X86_64_JUMP_SLOT:
  case R_X86_64_GLOB_DAT:
  case R_X86_64_DTPMOD64:
  case R_X86_64_DTPOFF64:
    break;
  default:
    return make_unhandled_reloc_error();
  }

  return std::error_code();
}
