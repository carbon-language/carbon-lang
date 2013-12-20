//===- lib/ReaderWriter/ELF/PPC/PPCTargetHandler.cpp ----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PPCTargetHandler.h"
#include "PPCLinkingContext.h"

using namespace lld;
using namespace elf;

using namespace llvm::ELF;

/// \brief The following relocation routines are derived from the
///  SYSTEM V APPLICATION BINARY INTERFACE: PowerPC Processor Supplement
/// Symbols used:
///  A: Added used to compute the value, r_addend
///  P: Place address of the field being relocated, r_offset
///  S: Value of the symbol whose index resides in the relocation entry.

/// \brief low24 (S + A - P) >> 2 : Verify
static int relocB24PCREL(uint8_t *location, uint64_t P, uint64_t S,
                         uint64_t A) {
  int32_t result = (uint32_t)(((S + A) - P));
  if ((result < 0x1000000) && (result > -0x1000000)) {
    result &= ~-(0x1000000);
    *reinterpret_cast<llvm::support::ubig32_t *>(location) = result |
               (uint32_t)*reinterpret_cast<llvm::support::ubig32_t *>(location);
    return 0;
  }
  return 1;
}

error_code PPCTargetRelocationHandler::applyRelocation(
    ELFWriter &writer, llvm::FileOutputBuffer &buf, const lld::AtomLayout &atom,
    const Reference &ref) const {
  uint8_t *atomContent = buf.getBufferStart() + atom._fileOffset;
  uint8_t *location = atomContent + ref.offsetInAtom();
  uint64_t targetVAddress = writer.addressOfAtom(ref.target());
  uint64_t relocVAddress = atom._virtualAddr + ref.offsetInAtom();

  if (ref.kindNamespace() != Reference::KindNamespace::ELF)
    return error_code::success();
  assert(ref.kindArch() == Reference::KindArch::PowerPC);
  switch (ref.kindValue()) {
  case R_PPC_REL24:
    relocB24PCREL(location, relocVAddress, targetVAddress, ref.addend());
    break;

  default : {
    std::string str;
    llvm::raw_string_ostream s(str);
    s << "Unhandled PowerPC relocation: #" << ref.kindValue();
    s.flush();
    llvm_unreachable(str.c_str());
  }
  }

  return error_code::success();
}

PPCTargetHandler::PPCTargetHandler(PPCLinkingContext &targetInfo)
    : DefaultTargetHandler(targetInfo), _relocationHandler(targetInfo),
      _targetLayout(targetInfo) {}

void PPCTargetHandler::registerRelocationNames(Registry &registry) {
  registry.addKindTable(Reference::KindNamespace::ELF,
                        Reference::KindArch::PowerPC, kindStrings);
}

const Registry::KindStrings PPCTargetHandler::kindStrings[] = {
  LLD_KIND_STRING_ENTRY(R_PPC_NONE),
  LLD_KIND_STRING_ENTRY(R_PPC_ADDR32),
  LLD_KIND_STRING_ENTRY(R_PPC_ADDR24),
  LLD_KIND_STRING_ENTRY(R_PPC_ADDR16),
  LLD_KIND_STRING_ENTRY(R_PPC_ADDR16_LO),
  LLD_KIND_STRING_ENTRY(R_PPC_ADDR16_HI),
  LLD_KIND_STRING_ENTRY(R_PPC_ADDR16_HA),
  LLD_KIND_STRING_ENTRY(R_PPC_ADDR14),
  LLD_KIND_STRING_ENTRY(R_PPC_ADDR14_BRTAKEN),
  LLD_KIND_STRING_ENTRY(R_PPC_ADDR14_BRNTAKEN),
  LLD_KIND_STRING_ENTRY(R_PPC_REL24),
  LLD_KIND_STRING_ENTRY(R_PPC_REL14),
  LLD_KIND_STRING_ENTRY(R_PPC_REL14_BRTAKEN),
  LLD_KIND_STRING_ENTRY(R_PPC_REL14_BRNTAKEN),
  LLD_KIND_STRING_ENTRY(R_PPC_GOT16),
  LLD_KIND_STRING_ENTRY(R_PPC_GOT16_LO),
  LLD_KIND_STRING_ENTRY(R_PPC_GOT16_HI),
  LLD_KIND_STRING_ENTRY(R_PPC_GOT16_HA),
  LLD_KIND_STRING_ENTRY(R_PPC_REL32),
  LLD_KIND_STRING_ENTRY(R_PPC_TLS),
  LLD_KIND_STRING_ENTRY(R_PPC_DTPMOD32),
  LLD_KIND_STRING_ENTRY(R_PPC_TPREL16),
  LLD_KIND_STRING_ENTRY(R_PPC_TPREL16_LO),
  LLD_KIND_STRING_ENTRY(R_PPC_TPREL16_HI),
  LLD_KIND_STRING_ENTRY(R_PPC_TPREL16_HA),
  LLD_KIND_STRING_ENTRY(R_PPC_TPREL32),
  LLD_KIND_STRING_ENTRY(R_PPC_DTPREL16),
  LLD_KIND_STRING_ENTRY(R_PPC_DTPREL16_LO),
  LLD_KIND_STRING_ENTRY(R_PPC_DTPREL16_HI),
  LLD_KIND_STRING_ENTRY(R_PPC_DTPREL16_HA),
  LLD_KIND_STRING_ENTRY(R_PPC_DTPREL32),
  LLD_KIND_STRING_ENTRY(R_PPC_GOT_TLSGD16),
  LLD_KIND_STRING_ENTRY(R_PPC_GOT_TLSGD16_LO),
  LLD_KIND_STRING_ENTRY(R_PPC_GOT_TLSGD16_HI),
  LLD_KIND_STRING_ENTRY(R_PPC_GOT_TLSGD16_HA),
  LLD_KIND_STRING_ENTRY(R_PPC_GOT_TLSLD16),
  LLD_KIND_STRING_ENTRY(R_PPC_GOT_TLSLD16_LO),
  LLD_KIND_STRING_ENTRY(R_PPC_GOT_TLSLD16_HI),
  LLD_KIND_STRING_ENTRY(R_PPC_GOT_TLSLD16_HA),
  LLD_KIND_STRING_ENTRY(R_PPC_GOT_TPREL16),
  LLD_KIND_STRING_ENTRY(R_PPC_GOT_TPREL16_LO),
  LLD_KIND_STRING_ENTRY(R_PPC_GOT_TPREL16_HI),
  LLD_KIND_STRING_ENTRY(R_PPC_GOT_TPREL16_HA),
  LLD_KIND_STRING_ENTRY(R_PPC_GOT_DTPREL16),
  LLD_KIND_STRING_ENTRY(R_PPC_GOT_DTPREL16_LO),
  LLD_KIND_STRING_ENTRY(R_PPC_GOT_DTPREL16_HI),
  LLD_KIND_STRING_ENTRY(R_PPC_GOT_DTPREL16_HA),
  LLD_KIND_STRING_ENTRY(R_PPC_TLSGD),
  LLD_KIND_STRING_ENTRY(R_PPC_TLSLD),
  LLD_KIND_STRING_ENTRY(R_PPC_REL16),
  LLD_KIND_STRING_ENTRY(R_PPC_REL16_LO),
  LLD_KIND_STRING_ENTRY(R_PPC_REL16_HI),
  LLD_KIND_STRING_ENTRY(R_PPC_REL16_HA),
  LLD_KIND_STRING_END
};
