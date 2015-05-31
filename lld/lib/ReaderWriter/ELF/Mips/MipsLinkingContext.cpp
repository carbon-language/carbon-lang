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
elf::createMipsLinkingContext(llvm::Triple triple) {
  if (triple.getArch() == llvm::Triple::mipsel ||
      triple.getArch() == llvm::Triple::mips64el)
    return llvm::make_unique<MipsLinkingContext>(triple);
  return nullptr;
}

static std::unique_ptr<TargetHandler> createTarget(llvm::Triple triple,
                                                   MipsLinkingContext &ctx) {
  switch (triple.getArch()) {
  case llvm::Triple::mipsel:
    return llvm::make_unique<MipsTargetHandler<ELF32LE>>(ctx);
  case llvm::Triple::mips64el:
    return llvm::make_unique<MipsTargetHandler<ELF64LE>>(ctx);
  default:
    llvm_unreachable("Unhandled arch");
  }
}

MipsLinkingContext::MipsLinkingContext(llvm::Triple triple)
    : ELFLinkingContext(triple, createTarget(triple, *this)) {}

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
  case llvm::ELF::R_MIPS_TLS_DTPMOD64:
  case llvm::ELF::R_MIPS_TLS_DTPREL64:
  case llvm::ELF::R_MIPS_TLS_TPREL64:
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

bool MipsLinkingContext::isRelativeReloc(const Reference &r) const {
  if (r.kindNamespace() != Reference::KindNamespace::ELF)
    return false;
  assert(r.kindArch() == Reference::KindArch::Mips);
  switch (r.kindValue()) {
  case llvm::ELF::R_MIPS_REL32:
  case llvm::ELF::R_MIPS_GPREL16:
  case llvm::ELF::R_MIPS_GPREL32:
    return true;
  default:
    return false;
  }
}

const Registry::KindStrings kindStrings[] = {
#define ELF_RELOC(name, value) LLD_KIND_STRING_ENTRY(name),
#include "llvm/Support/ELFRelocs/Mips.def"
#undef ELF_RELOC
  LLD_KIND_STRING_ENTRY(LLD_R_MIPS_GLOBAL_GOT),
  LLD_KIND_STRING_ENTRY(LLD_R_MIPS_32_HI16),
  LLD_KIND_STRING_ENTRY(LLD_R_MIPS_64_HI16),
  LLD_KIND_STRING_ENTRY(LLD_R_MIPS_GLOBAL_26),
  LLD_KIND_STRING_ENTRY(LLD_R_MIPS_HI16),
  LLD_KIND_STRING_ENTRY(LLD_R_MIPS_LO16),
  LLD_KIND_STRING_ENTRY(LLD_R_MIPS_STO_PLT),
  LLD_KIND_STRING_ENTRY(LLD_R_MICROMIPS_GLOBAL_26_S1),
  LLD_KIND_STRING_END
};

void MipsLinkingContext::registerRelocationNames(Registry &registry) {
  registry.addKindTable(Reference::KindNamespace::ELF,
                        Reference::KindArch::Mips, kindStrings);
}
