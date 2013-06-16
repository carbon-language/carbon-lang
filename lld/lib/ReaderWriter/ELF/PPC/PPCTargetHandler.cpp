//===- lib/ReaderWriter/ELF/PPC/PPCTargetHandler.cpp ----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PPCTargetHandler.h"
#include "PPCTargetInfo.h"

using namespace lld;
using namespace elf;

using namespace llvm::ELF;

/// \brief The following relocation routines are derived from the
///  SYSTEM V APPLICATION BINARY INTERFACE: PowerPC Processor Supplement
/// Symbols used:
///  A: Added used to compute the value, r_addend
///  P: Place address of the field being relocated, r_offset
///  S: Value of the symbol whose index resides in the relocation entry.
namespace {
/// \brief low24 (S + A - P) >> 2 : Verify
int relocB24PCREL(uint8_t *location, uint64_t P, uint64_t S, uint64_t A) {
  int32_t result = (uint32_t)(((S + A) - P));
  if ((result < 0x1000000) && (result > -0x1000000)) {
    result &= ~-(0x1000000);
    *reinterpret_cast<llvm::support::ubig32_t *>(location) = result |
               (uint32_t)*reinterpret_cast<llvm::support::ubig32_t *>(location);
    return 0;
  }
  return 1;
}
} // end anon namespace

ErrorOr<void> PPCTargetRelocationHandler::applyRelocation(
    ELFWriter &writer, llvm::FileOutputBuffer &buf, const lld::AtomLayout &atom,
    const Reference &ref) const {
  uint8_t *atomContent = buf.getBufferStart() + atom._fileOffset;
  uint8_t *location = atomContent + ref.offsetInAtom();
  uint64_t targetVAddress = writer.addressOfAtom(ref.target());
  uint64_t relocVAddress = atom._virtualAddr + ref.offsetInAtom();

  switch (ref.kind()) {
  case R_PPC_REL24:
    relocB24PCREL(location, relocVAddress, targetVAddress, ref.addend());
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

PPCTargetHandler::PPCTargetHandler(PPCTargetInfo &targetInfo)
    : DefaultTargetHandler(targetInfo), _relocationHandler(targetInfo),
      _targetLayout(targetInfo) {
}
