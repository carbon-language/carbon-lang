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

std::error_code PPCTargetRelocationHandler::applyRelocation(
    ELFWriter &writer, llvm::FileOutputBuffer &buf, const lld::AtomLayout &atom,
    const Reference &ref) const {
  uint8_t *atomContent = buf.getBufferStart() + atom._fileOffset;
  uint8_t *location = atomContent + ref.offsetInAtom();
  uint64_t targetVAddress = writer.addressOfAtom(ref.target());
  uint64_t relocVAddress = atom._virtualAddr + ref.offsetInAtom();

  if (ref.kindNamespace() != Reference::KindNamespace::ELF)
    return std::error_code();
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

  return std::error_code();
}

PPCTargetHandler::PPCTargetHandler(PPCLinkingContext &context)
    : DefaultTargetHandler(context), _ppcLinkingContext(context),
      _ppcTargetLayout(new PPCTargetLayout<PPCELFType>(context)),
      _ppcRelocationHandler(
          new PPCTargetRelocationHandler(context, *_ppcTargetLayout.get())) {}

void PPCTargetHandler::registerRelocationNames(Registry &registry) {
  registry.addKindTable(Reference::KindNamespace::ELF,
                        Reference::KindArch::PowerPC, kindStrings);
}

std::unique_ptr<Writer> PPCTargetHandler::getWriter() {
  switch (_ppcLinkingContext.getOutputELFType()) {
  case llvm::ELF::ET_EXEC:
    return std::unique_ptr<Writer>(new elf::ExecutableWriter<PPCELFType>(
        _ppcLinkingContext, *_ppcTargetLayout.get()));
  case llvm::ELF::ET_DYN:
    return std::unique_ptr<Writer>(new elf::DynamicLibraryWriter<PPCELFType>(
        _ppcLinkingContext, *_ppcTargetLayout.get()));
  case llvm::ELF::ET_REL:
    llvm_unreachable("TODO: support -r mode");
  default:
    llvm_unreachable("unsupported output type");
  }
}

#define ELF_RELOC(name, value) LLD_KIND_STRING_ENTRY(name),

const Registry::KindStrings PPCTargetHandler::kindStrings[] = {
#include "llvm/Support/ELFRelocs/PowerPC.def"
  LLD_KIND_STRING_END
};

#undef ELF_RELOC
