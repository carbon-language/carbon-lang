//===- lib/ReaderWriter/ELF/Hexagon/HexagonLinkingContext.cpp -------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "HexagonLinkingContext.h"
#include "HexagonTargetHandler.h"
#include "lld/ReaderWriter/ELFTargets.h"

using namespace lld;
using namespace lld::elf;

std::unique_ptr<lld::ELFLinkingContext>
elf::createHexagonLinkingContext(llvm::Triple triple) {
  if (triple.getArch() == llvm::Triple::hexagon)
    return llvm::make_unique<HexagonLinkingContext>(triple);
  return nullptr;
}

HexagonLinkingContext::HexagonLinkingContext(llvm::Triple triple)
    : ELFLinkingContext(triple, std::unique_ptr<TargetHandler>(
                                    new HexagonTargetHandler(*this))) {}

static const Registry::KindStrings kindStrings[] = {
#define ELF_RELOC(name, value) LLD_KIND_STRING_ENTRY(name),
#include "llvm/Support/ELFRelocs/Hexagon.def"
#undef ELF_RELOC
  LLD_KIND_STRING_END
};

void HexagonLinkingContext::registerRelocationNames(Registry &registry) {
  registry.addKindTable(Reference::KindNamespace::ELF,
                        Reference::KindArch::Hexagon, kindStrings);
}
