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

using namespace lld;
using namespace lld::elf;

std::unique_ptr<ELFLinkingContext>
elf::ARMLinkingContext::create(llvm::Triple triple) {
  if (triple.getArch() == llvm::Triple::arm)
    return llvm::make_unique<elf::ARMLinkingContext>(triple);
  return nullptr;
}

elf::ARMLinkingContext::ARMLinkingContext(llvm::Triple triple)
    : ELFLinkingContext(triple, llvm::make_unique<ARMTargetHandler>(*this)) {}

void elf::ARMLinkingContext::addPasses(PassManager &pm) {
  auto pass = createARMRelocationPass(*this);
  if (pass)
    pm.add(std::move(pass));
  ELFLinkingContext::addPasses(pm);
}

bool elf::isARMCode(const DefinedAtom *atom) {
  return isARMCode(atom->codeModel());
}

bool elf::isARMCode(DefinedAtom::CodeModel codeModel) {
  return !isThumbCode(codeModel);
}

bool elf::isThumbCode(const DefinedAtom *atom) {
  return isThumbCode(atom->codeModel());
}

bool elf::isThumbCode(DefinedAtom::CodeModel codeModel) {
  return codeModel == DefinedAtom::codeARMThumb ||
         codeModel == DefinedAtom::codeARM_t;
}
