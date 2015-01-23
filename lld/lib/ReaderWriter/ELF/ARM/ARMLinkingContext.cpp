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

elf::ARMLinkingContext::ARMLinkingContext(llvm::Triple triple)
    : ELFLinkingContext(triple, std::unique_ptr<TargetHandlerBase>(
                        new ARMTargetHandler(*this))) {}

void elf::ARMLinkingContext::addPasses(PassManager &pm) {
  auto pass = createARMRelocationPass(*this);
  if (pass)
    pm.add(std::move(pass));
  ELFLinkingContext::addPasses(pm);
}
