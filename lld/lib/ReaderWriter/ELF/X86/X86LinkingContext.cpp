//===- lib/ReaderWriter/ELF/X86/X86LinkingContext.cpp ---------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "X86LinkingContext.h"
#include "X86TargetHandler.h"
#include "lld/Core/LLVM.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorOr.h"

using namespace lld;
using namespace lld::elf;

std::unique_ptr<ELFLinkingContext>
elf::createX86LinkingContext(llvm::Triple triple) {
  if (triple.getArch() == llvm::Triple::x86)
    return llvm::make_unique<X86LinkingContext>(triple);
  return nullptr;
}

X86LinkingContext::X86LinkingContext(llvm::Triple triple)
    : ELFLinkingContext(triple, llvm::make_unique<X86TargetHandler>(*this)) {}

static const Registry::KindStrings kindStrings[] = {
#define ELF_RELOC(name, value) LLD_KIND_STRING_ENTRY(name),
#include "llvm/Support/ELFRelocs/i386.def"
#undef ELF_RELOC
  LLD_KIND_STRING_END
};

void X86LinkingContext::registerRelocationNames(Registry &registry) {
  registry.addKindTable(Reference::KindNamespace::ELF, Reference::KindArch::x86,
                        kindStrings);
}
