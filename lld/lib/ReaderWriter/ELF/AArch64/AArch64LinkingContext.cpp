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

using namespace lld;

void elf::AArch64LinkingContext::addPasses(PassManager &pm) {
  auto pass = createAArch64RelocationPass(*this);
  if (pass)
    pm.add(std::move(pass));
  ELFLinkingContext::addPasses(pm);
}
