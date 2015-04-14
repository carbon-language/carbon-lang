//===- lib/ReaderWriter/ELF/X86/X86RelocationHandler.cpp ------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "X86LinkingContext.h"
#include "X86TargetHandler.h"
#include "llvm/Support/Endian.h"

using namespace lld;
using namespace lld::elf;
using namespace llvm::support::endian;

/// \brief R_386_32 - word32:  S + A
static int reloc32(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  int32_t result = (uint32_t)(S + A);
  write32le(location, result | read32le(location));
  return 0;
}

/// \brief R_386_PC32 - word32: S + A - P
static int relocPC32(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  uint32_t result = (uint32_t)(S + A - P);
  write32le(location, result + read32le(location));
  return 0;
}

std::error_code X86TargetRelocationHandler::applyRelocation(
    ELFWriter &writer, llvm::FileOutputBuffer &buf, const AtomLayout &atom,
    const Reference &ref) const {
  uint8_t *atomContent = buf.getBufferStart() + atom._fileOffset;
  uint8_t *loc = atomContent + ref.offsetInAtom();
  uint64_t target = writer.addressOfAtom(ref.target());
  uint64_t reloc = atom._virtualAddr + ref.offsetInAtom();

  if (ref.kindNamespace() != Reference::KindNamespace::ELF)
    return std::error_code();
  assert(ref.kindArch() == Reference::KindArch::x86);
  switch (ref.kindValue()) {
  case R_386_32:
    reloc32(loc, reloc, target, ref.addend());
    break;
  case R_386_PC32:
    relocPC32(loc, reloc, target, ref.addend());
    break;
  default:
    return make_unhandled_reloc_error();
  }
  return std::error_code();
}
