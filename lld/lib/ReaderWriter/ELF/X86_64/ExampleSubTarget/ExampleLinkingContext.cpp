//===- lib/ReaderWriter/ELF/X86_64/ExampleTarget/ExampleLinkingContext.cpp ----===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ExampleLinkingContext.h"
#include "ExampleTargetHandler.h"

using namespace lld;
using namespace elf;

std::unique_ptr<ELFLinkingContext>
elf::createExampleLinkingContext(llvm::Triple triple) {
  if (triple.getVendorName() == "example")
    return llvm::make_unique<ExampleLinkingContext>(triple);
  return nullptr;
}

ExampleLinkingContext::ExampleLinkingContext(llvm::Triple triple)
    : X86_64LinkingContext(triple, std::unique_ptr<TargetHandler>(
                                       new ExampleTargetHandler(*this))) {
  _outputELFType = llvm::ELF::ET_LOPROC;
}

StringRef ExampleLinkingContext::entrySymbolName() const {
  return "_start";
}

void ExampleLinkingContext::addPasses(PassManager &p) {
  ELFLinkingContext::addPasses(p);
}
