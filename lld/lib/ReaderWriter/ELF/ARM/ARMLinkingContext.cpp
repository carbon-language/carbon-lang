//===--------- lib/ReaderWriter/ELF/ARM/ARMLinkingContext.cpp -------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ARMLinkingContext.h"
#include "ARMRelocationPass.h"
#include "ARMTargetHandler.h"

namespace lld {
namespace elf {

std::unique_ptr<ELFLinkingContext>
createARMLinkingContext(llvm::Triple triple) {
  if (triple.getArch() == llvm::Triple::arm)
    return llvm::make_unique<ARMLinkingContext>(triple);
  return nullptr;
}

ARMLinkingContext::ARMLinkingContext(llvm::Triple triple)
    : ELFLinkingContext(triple, llvm::make_unique<ARMTargetHandler>(*this)) {}

void ARMLinkingContext::addPasses(PassManager &pm) {
  auto pass = createARMRelocationPass(*this);
  if (pass)
    pm.add(std::move(pass));
  ELFLinkingContext::addPasses(pm);
}

bool isARMCode(const DefinedAtom *atom) {
  return isARMCode(atom->codeModel());
}

bool isARMCode(DefinedAtom::CodeModel codeModel) {
  return !isThumbCode(codeModel);
}

bool isThumbCode(const DefinedAtom *atom) {
  return isThumbCode(atom->codeModel());
}

bool isThumbCode(DefinedAtom::CodeModel codeModel) {
  return codeModel == DefinedAtom::codeARMThumb ||
         codeModel == DefinedAtom::codeARM_t;
}

static const Registry::KindStrings kindStrings[] = {
#define ELF_RELOC(name, value) LLD_KIND_STRING_ENTRY(name),
#include "llvm/Support/ELFRelocs/ARM.def"
#undef ELF_RELOC
  LLD_KIND_STRING_END
};

void ARMLinkingContext::registerRelocationNames(Registry &registry) {
  registry.addKindTable(Reference::KindNamespace::ELF, Reference::KindArch::ARM,
                        kindStrings);
}

} // namespace elf
} // namespace lld
