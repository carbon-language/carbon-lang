//===- lib/ReaderWriter/ELF/Mips/MipsLinkingContext.cpp -------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Atoms.h"
#include "MipsCtorsOrderPass.h"
#include "MipsLinkingContext.h"
#include "MipsRelocationPass.h"
#include "MipsTargetHandler.h"

using namespace lld;
using namespace lld::elf;

std::unique_ptr<ELFLinkingContext>
MipsLinkingContext::create(llvm::Triple triple) {
  if (triple.getArch() == llvm::Triple::mipsel ||
      triple.getArch() == llvm::Triple::mips64el)
    return std::unique_ptr<ELFLinkingContext>(new MipsLinkingContext(triple));
  return nullptr;
}

typedef std::unique_ptr<TargetHandlerBase> TargetHandlerBasePtr;

static TargetHandlerBasePtr createTarget(llvm::Triple triple,
                                         MipsLinkingContext &ctx) {
  switch (triple.getArch()) {
  case llvm::Triple::mipsel:
    return TargetHandlerBasePtr(new MipsTargetHandler<Mips32ELType>(ctx));
  case llvm::Triple::mips64el:
    return TargetHandlerBasePtr(new MipsTargetHandler<Mips64ELType>(ctx));
  default:
    llvm_unreachable("Unhandled arch");
  }
}

MipsLinkingContext::MipsLinkingContext(llvm::Triple triple)
    : ELFLinkingContext(triple, createTarget(triple, *this)),
      _flagsMerger(triple.isArch64Bit()) {}

uint32_t MipsLinkingContext::getMergedELFFlags() const {
  return _flagsMerger.getMergedELFFlags();
}

MipsELFFlagsMerger &MipsLinkingContext::getELFFlagsMerger() {
  return _flagsMerger;
}

uint64_t MipsLinkingContext::getBaseAddress() const {
  if (_baseAddress == 0 && getOutputELFType() == llvm::ELF::ET_EXEC)
    return getTriple().isArch64Bit() ? 0x120000000 : 0x400000;
  return _baseAddress;
}

StringRef MipsLinkingContext::entrySymbolName() const {
  if (_outputELFType == elf::ET_EXEC && _entrySymbolName.empty())
    return "__start";
  return _entrySymbolName;
}

StringRef MipsLinkingContext::getDefaultInterpreter() const {
  return getTriple().isArch64Bit() ? "/lib64/ld.so.1" : "/lib/ld.so.1";
}

void MipsLinkingContext::addPasses(PassManager &pm) {
  auto pass = createMipsRelocationPass(*this);
  if (pass)
    pm.add(std::move(pass));
  ELFLinkingContext::addPasses(pm);
  pm.add(llvm::make_unique<elf::MipsCtorsOrderPass>());
}

bool MipsLinkingContext::isDynamicRelocation(const Reference &r) const {
  if (r.kindNamespace() != Reference::KindNamespace::ELF)
    return false;
  assert(r.kindArch() == Reference::KindArch::Mips);
  switch (r.kindValue()) {
  case llvm::ELF::R_MIPS_COPY:
  case llvm::ELF::R_MIPS_REL32:
  case llvm::ELF::R_MIPS_TLS_DTPMOD32:
  case llvm::ELF::R_MIPS_TLS_DTPREL32:
  case llvm::ELF::R_MIPS_TLS_TPREL32:
    return true;
  default:
    return false;
  }
}

bool MipsLinkingContext::isCopyRelocation(const Reference &r) const {
  if (r.kindNamespace() != Reference::KindNamespace::ELF)
    return false;
  assert(r.kindArch() == Reference::KindArch::Mips);
  if (r.kindValue() == llvm::ELF::R_MIPS_COPY)
    return true;
  return false;
}

bool MipsLinkingContext::isPLTRelocation(const Reference &r) const {
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
