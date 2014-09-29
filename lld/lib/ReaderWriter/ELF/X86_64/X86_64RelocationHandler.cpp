//===- lib/ReaderWriter/ELF/X86_64/X86_64RelocationHandler.cpp ------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "X86_64TargetHandler.h"
#include "X86_64LinkingContext.h"

using namespace lld;
using namespace elf;

/// \brief R_X86_64_64 - word64: S + A
static void reloc64(uint8_t *location, uint64_t P, uint64_t S, int64_t A) {
  uint64_t result = S + A;
  *reinterpret_cast<llvm::support::ulittle64_t *>(location) =
      result |
      (uint64_t) * reinterpret_cast<llvm::support::ulittle64_t *>(location);
}

/// \brief R_X86_64_PC32 - word32: S + A - P
static void relocPC32(uint8_t *location, uint64_t P, uint64_t S, int64_t A) {
  uint32_t result = (uint32_t)((S + A) - P);
  *reinterpret_cast<llvm::support::ulittle32_t *>(location) =
      result +
      (uint32_t) * reinterpret_cast<llvm::support::ulittle32_t *>(location);
}

/// \brief R_X86_64_32 - word32:  S + A
static void reloc32(uint8_t *location, uint64_t P, uint64_t S, int64_t A) {
  int32_t result = (uint32_t)(S + A);
  *reinterpret_cast<llvm::support::ulittle32_t *>(location) =
      result |
      (uint32_t) * reinterpret_cast<llvm::support::ulittle32_t *>(location);
  // TODO: Make sure that the result zero extends to the 64bit value.
}

/// \brief R_X86_64_32S - word32:  S + A
static void reloc32S(uint8_t *location, uint64_t P, uint64_t S, int64_t A) {
  int32_t result = (int32_t)(S + A);
  *reinterpret_cast<llvm::support::little32_t *>(location) =
      result |
      (int32_t) * reinterpret_cast<llvm::support::little32_t *>(location);
  // TODO: Make sure that the result sign extends to the 64bit value.
}

std::error_code X86_64TargetRelocationHandler::applyRelocation(
    ELFWriter &writer, llvm::FileOutputBuffer &buf, const lld::AtomLayout &atom,
    const Reference &ref) const {
  uint8_t *atomContent = buf.getBufferStart() + atom._fileOffset;
  uint8_t *location = atomContent + ref.offsetInAtom();
  uint64_t targetVAddress = writer.addressOfAtom(ref.target());
  uint64_t relocVAddress = atom._virtualAddr + ref.offsetInAtom();

  if (ref.kindNamespace() != Reference::KindNamespace::ELF)
    return std::error_code();
  assert(ref.kindArch() == Reference::KindArch::x86_64);
  switch (ref.kindValue()) {
  case R_X86_64_NONE:
    break;
  case R_X86_64_64:
    reloc64(location, relocVAddress, targetVAddress, ref.addend());
    break;
  case R_X86_64_PC32:
  case R_X86_64_GOTPCREL:
    relocPC32(location, relocVAddress, targetVAddress, ref.addend());
    break;
  case R_X86_64_32:
    reloc32(location, relocVAddress, targetVAddress, ref.addend());
    break;
  case R_X86_64_32S:
    reloc32S(location, relocVAddress, targetVAddress, ref.addend());
    break;
  case R_X86_64_TPOFF64:
  case R_X86_64_DTPOFF32:
  case R_X86_64_TPOFF32: {
    _tlsSize = _x86_64Layout.getTLSSize();
    if (ref.kindValue() == R_X86_64_TPOFF32 ||
        ref.kindValue() == R_X86_64_DTPOFF32) {
      int32_t result = (int32_t)(targetVAddress - _tlsSize);
      *reinterpret_cast<llvm::support::little32_t *>(location) = result;
    } else {
      int64_t result = (int64_t)(targetVAddress - _tlsSize);
      *reinterpret_cast<llvm::support::little64_t *>(location) = result;
    }
    break;
  }
  case R_X86_64_TLSGD: {
    relocPC32(location, relocVAddress, targetVAddress, ref.addend());
    break;
  }
  case R_X86_64_TLSLD: {
    // Rewrite to move %fs:0 into %rax. Technically we should verify that the
    // next relocation is a PC32 to __tls_get_addr...
    static uint8_t instr[] = { 0x66, 0x66, 0x66, 0x64, 0x48, 0x8b, 0x04, 0x25,
                               0x00, 0x00, 0x00, 0x00 };
    std::memcpy(location - 3, instr, sizeof(instr));
    break;
  }
  case LLD_R_X86_64_GOTRELINDEX: {
    const DefinedAtom *target = cast<const DefinedAtom>(ref.target());
    for (const Reference *r : *target) {
      if (r->kindValue() == R_X86_64_JUMP_SLOT) {
        uint32_t index;
        if (!_x86_64Layout.getPLTRelocationTable()->getRelocationIndex(*r,
                                                                       index))
          llvm_unreachable("Relocation doesn't exist");
        reloc32(location, 0, index, 0);
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
  default: {
    std::string str;
    llvm::raw_string_ostream s(str);
    s << "Unhandled relocation: " << atom._atom->file().path() << ":"
      << atom._atom->name() << "@" << ref.offsetInAtom() << " "
      << "#" << ref.kindValue();
    s.flush();
    llvm_unreachable(str.c_str());
  }
  }

  return std::error_code();
}
