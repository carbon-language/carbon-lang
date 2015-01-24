//===- lib/ReaderWriter/ELF/X86_64/X86_64LinkingContext.cpp ---------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "X86_64LinkingContext.h"
#include "X86_64TargetHandler.h"
#include "X86_64RelocationPass.h"

using namespace lld;

std::unique_ptr<ELFLinkingContext>
elf::X86_64LinkingContext::create(llvm::Triple triple) {
  if (triple.getArch() == llvm::Triple::x86_64)
    return std::unique_ptr<ELFLinkingContext>(
             new elf::X86_64LinkingContext(triple));
  return nullptr;
}

elf::X86_64LinkingContext::X86_64LinkingContext(llvm::Triple triple)
    : ELFLinkingContext(triple, std::unique_ptr<TargetHandlerBase>(
                        new X86_64TargetHandler(*this))) {}

void elf::X86_64LinkingContext::addPasses(PassManager &pm) {
  auto pass = createX86_64RelocationPass(*this);
  if (pass)
    pm.add(std::move(pass));
  ELFLinkingContext::addPasses(pm);
}
