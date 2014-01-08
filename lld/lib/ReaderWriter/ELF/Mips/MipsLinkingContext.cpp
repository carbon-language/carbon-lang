//===- lib/ReaderWriter/ELF/Mips/MipsLinkingContext.cpp -------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Atoms.h"
#include "MipsLinkingContext.h"
#include "MipsRelocationPass.h"
#include "MipsTargetHandler.h"

using namespace lld;
using namespace lld::elf;

MipsLinkingContext::MipsLinkingContext(llvm::Triple triple)
    : ELFLinkingContext(triple, std::unique_ptr<TargetHandlerBase>(
                                    new MipsTargetHandler(*this))) {}

MipsTargetLayout<Mips32ElELFType> &MipsLinkingContext::getTargetLayout() {
  auto &layout = getTargetHandler<Mips32ElELFType>().targetLayout();
  return static_cast<MipsTargetLayout<Mips32ElELFType> &>(layout);
}

const MipsTargetLayout<Mips32ElELFType> &
MipsLinkingContext::getTargetLayout() const {
  auto &layout = getTargetHandler<Mips32ElELFType>().targetLayout();
  return static_cast<MipsTargetLayout<Mips32ElELFType> &>(layout);
}

bool MipsLinkingContext::isLittleEndian() const {
  return Mips32ElELFType::TargetEndianness == llvm::support::little;
}

void MipsLinkingContext::addPasses(PassManager &pm) {
  auto pass = createMipsRelocationPass(*this);
  if (pass)
    pm.add(std::move(pass));
  ELFLinkingContext::addPasses(pm);
}
