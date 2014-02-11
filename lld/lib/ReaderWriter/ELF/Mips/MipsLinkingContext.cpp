//===- lib/ReaderWriter/ELF/Mips/MipsLinkingContext.cpp -------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Atoms.h"
#include "MipsLinkingContext.h"
#include "MipsRelocationPass.h"
#include "MipsTargetHandler.h"

using namespace lld;
using namespace lld::elf;

MipsLinkingContext::MipsLinkingContext(llvm::Triple triple)
    : ELFLinkingContext(triple, std::unique_ptr<TargetHandlerBase>(
                                    new MipsTargetHandler(*this))) {}

bool MipsLinkingContext::isLittleEndian() const {
  return Mips32ElELFType::TargetEndianness == llvm::support::little;
}

uint64_t MipsLinkingContext::getBaseAddress() const {
  if (_baseAddress == 0 && getOutputELFType() == llvm::ELF::ET_EXEC)
    return 0x400000;
  return _baseAddress;
}

StringRef MipsLinkingContext::entrySymbolName() const {
  if (_outputELFType == elf::ET_EXEC && _entrySymbolName.empty())
    return "__start";
  return _entrySymbolName;
}

StringRef MipsLinkingContext::getDefaultInterpreter() const {
  return "/lib/ld.so.1";
}

void MipsLinkingContext::addPasses(PassManager &pm) {
  auto pass = createMipsRelocationPass(*this);
  if (pass)
    pm.add(std::move(pass));
  ELFLinkingContext::addPasses(pm);
}

bool MipsLinkingContext::isDynamicRelocation(const DefinedAtom &,
                                             const Reference &r) const {
  if (r.kindNamespace() != Reference::KindNamespace::ELF)
    return false;
  return r.kindValue() == llvm::ELF::R_MIPS_COPY;
}

bool MipsLinkingContext::isPLTRelocation(const DefinedAtom &,
                                         const Reference &r) const {
  if (r.kindNamespace() != Reference::KindNamespace::ELF)
    return false;
  assert(r.kindArch() == Reference::KindArch::Mips);
  switch (r.kindValue()) {
  case llvm::ELF::R_MIPS_JUMP_SLOT:
    return true;
  default:
    return false;
  }
}
