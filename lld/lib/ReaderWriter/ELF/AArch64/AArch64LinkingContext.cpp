//===- lib/ReaderWriter/ELF/AArch64/AArch64LinkingContext.cpp -------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AArch64LinkingContext.h"
#include "AArch64RelocationPass.h"
#include "AArch64TargetHandler.h"

using namespace lld;
using namespace lld::elf;

std::unique_ptr<ELFLinkingContext>
elf::createAArch64LinkingContext(llvm::Triple triple) {
  if (triple.getArch() == llvm::Triple::aarch64)
    return llvm::make_unique<AArch64LinkingContext>(triple);
  return nullptr;
}

AArch64LinkingContext::AArch64LinkingContext(llvm::Triple triple)
    : ELFLinkingContext(triple, std::unique_ptr<TargetHandler>(
                                    new AArch64TargetHandler(*this))) {}

void AArch64LinkingContext::addPasses(PassManager &pm) {
  auto pass = createAArch64RelocationPass(*this);
  if (pass)
    pm.add(std::move(pass));
  ELFLinkingContext::addPasses(pm);
}

static const Registry::KindStrings kindStrings[] = {
#define ELF_RELOC(name, value) LLD_KIND_STRING_ENTRY(name),
#include "llvm/Support/ELFRelocs/AArch64.def"
#undef ELF_RELOC
  LLD_KIND_STRING_END
};

void AArch64LinkingContext::registerRelocationNames(Registry &registry) {
  registry.addKindTable(Reference::KindNamespace::ELF,
                        Reference::KindArch::AArch64, kindStrings);
}
