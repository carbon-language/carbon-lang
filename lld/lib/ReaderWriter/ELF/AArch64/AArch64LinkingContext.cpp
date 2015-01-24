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

std::unique_ptr<ELFLinkingContext>
elf::AArch64LinkingContext::create(llvm::Triple triple) {
  if (triple.getArch() == llvm::Triple::aarch64)
    return std::unique_ptr<ELFLinkingContext>(
             new elf::AArch64LinkingContext(triple));
  return nullptr;
}

elf::AArch64LinkingContext::AArch64LinkingContext(llvm::Triple triple)
    : ELFLinkingContext(triple, std::unique_ptr<TargetHandlerBase>(
                        new AArch64TargetHandler(*this))) {}

void elf::AArch64LinkingContext::addPasses(PassManager &pm) {
  auto pass = createAArch64RelocationPass(*this);
  if (pass)
    pm.add(std::move(pass));
  ELFLinkingContext::addPasses(pm);
}
