//===- lib/ReaderWriter/ELF/X86_64/X86_64LinkingContext.cpp ---------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "X86_64LinkingContext.h"
#include "X86_64RelocationPass.h"
#include "X86_64TargetHandler.h"

using namespace lld;
using namespace lld::elf;

X86_64LinkingContext::X86_64LinkingContext(
    llvm::Triple triple, std::unique_ptr<TargetHandler> handler)
    : ELFLinkingContext(triple, std::move(handler)) {}

X86_64LinkingContext::X86_64LinkingContext(llvm::Triple triple)
    : X86_64LinkingContext(triple,
                           llvm::make_unique<X86_64TargetHandler>(*this)) {}

void X86_64LinkingContext::addPasses(PassManager &pm) {
  auto pass = createX86_64RelocationPass(*this);
  if (pass)
    pm.add(std::move(pass));
  ELFLinkingContext::addPasses(pm);
}

std::unique_ptr<ELFLinkingContext>
elf::createX86_64LinkingContext(llvm::Triple triple) {
  if (triple.getArch() == llvm::Triple::x86_64)
    return llvm::make_unique<X86_64LinkingContext>(triple);
  return nullptr;
}

static const Registry::KindStrings kindStrings[] = {
#define ELF_RELOC(name, value) LLD_KIND_STRING_ENTRY(name),
#include "llvm/Support/ELFRelocs/x86_64.def"
#undef ELF_RELOC
  LLD_KIND_STRING_ENTRY(LLD_R_X86_64_GOTRELINDEX),
  LLD_KIND_STRING_END
};

void X86_64LinkingContext::registerRelocationNames(Registry &registry) {
  registry.addKindTable(Reference::KindNamespace::ELF,
                        Reference::KindArch::x86_64, kindStrings);
}
