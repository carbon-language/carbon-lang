//===- lib/ReaderWriter/ELF/AMDGPU/AMDGPULinkingContext.cpp ---------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===------------------------------------------------------------------------===//

#include "AMDGPULinkingContext.h"
#include "AMDGPUTargetHandler.h"

namespace lld {
namespace elf {

std::unique_ptr<ELFLinkingContext>
createAMDGPULinkingContext(llvm::Triple triple) {
  if (triple.getArch() == llvm::Triple::amdgcn)
    return llvm::make_unique<AMDGPULinkingContext>(triple);
  return nullptr;
}

AMDGPULinkingContext::AMDGPULinkingContext(llvm::Triple triple)
    : ELFLinkingContext(triple, llvm::make_unique<AMDGPUTargetHandler>(*this)) {
}

static const Registry::KindStrings kindStrings[] = {LLD_KIND_STRING_END};

void AMDGPULinkingContext::registerRelocationNames(Registry &registry) {
  registry.addKindTable(Reference::KindNamespace::ELF,
                        Reference::KindArch::AMDGPU, kindStrings);
}

void setAMDGPUELFHeader(ELFHeader<ELF64LE> &elfHeader) {
  elfHeader.e_ident(llvm::ELF::EI_OSABI, ELFOSABI_AMDGPU_HSA);
}

StringRef AMDGPULinkingContext::entrySymbolName() const { return ""; }

} // namespace elf
} // namespace lld
