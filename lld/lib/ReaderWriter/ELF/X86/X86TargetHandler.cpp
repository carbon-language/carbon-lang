//===- lib/ReaderWriter/ELF/X86/X86TargetHandler.cpp ----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "X86TargetHandler.h"
#include "X86TargetInfo.h"

using namespace lld;
using namespace elf;

using namespace llvm::ELF;

namespace {
/// \brief R_386_32 - word32:  S + A
int reloc32(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  int32_t result = (uint32_t)(S + A);
  *reinterpret_cast<llvm::support::ulittle32_t *>(location) = result |
            (uint32_t)*reinterpret_cast<llvm::support::ulittle32_t *>(location);
  return 0;
}

/// \brief R_386_PC32 - word32: S + A - P
int relocPC32(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  uint32_t result = (uint32_t)((S + A) - P);
  *reinterpret_cast<llvm::support::ulittle32_t *>(location) = result +
            (uint32_t)*reinterpret_cast<llvm::support::ulittle32_t *>(location);
  return 0;
}
} // end anon namespace

ErrorOr<void> X86TargetRelocationHandler::applyRelocation(
    ELFWriter &writer, llvm::FileOutputBuffer &buf, const lld::AtomLayout &atom,
    const Reference &ref) const {
  uint8_t *atomContent = buf.getBufferStart() + atom._fileOffset;
  uint8_t *location = atomContent + ref.offsetInAtom();
  uint64_t targetVAddress = writer.addressOfAtom(ref.target());
  uint64_t relocVAddress = atom._virtualAddr + ref.offsetInAtom();

  switch (ref.kind()) {
  case R_386_32:
    reloc32(location, relocVAddress, targetVAddress, ref.addend());
    break;
  case R_386_PC32:
    relocPC32(location, relocVAddress, targetVAddress, ref.addend());
    break;
  case lld::Reference::kindLayoutAfter:
  case lld::Reference::kindLayoutBefore:
  case lld::Reference::kindInGroup:
    break;
  default : {
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

X86TargetHandler::X86TargetHandler(X86TargetInfo &targetInfo)
    : DefaultTargetHandler(targetInfo), _relocationHandler(targetInfo),
      _targetLayout(targetInfo) {
}
