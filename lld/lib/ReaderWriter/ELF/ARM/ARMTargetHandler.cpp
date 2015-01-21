//===--------- lib/ReaderWriter/ELF/ARM/ARMTargetHandler.cpp --------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Atoms.h"
#include "ARMExecutableWriter.h"
#include "ARMTargetHandler.h"
#include "ARMLinkingContext.h"

using namespace lld;
using namespace elf;

ARMTargetHandler::ARMTargetHandler(ARMLinkingContext &context)
    : _context(context), _armTargetLayout(
          new ARMTargetLayout<ARMELFType>(context)),
      _armRelocationHandler(new ARMTargetRelocationHandler()) {}

void ARMTargetHandler::registerRelocationNames(Registry &registry) {
  registry.addKindTable(Reference::KindNamespace::ELF, Reference::KindArch::ARM,
                        kindStrings);
}

std::unique_ptr<Writer> ARMTargetHandler::getWriter() {
  switch (this->_context.getOutputELFType()) {
  case llvm::ELF::ET_EXEC:
    return std::unique_ptr<Writer>(
        new ARMExecutableWriter<ARMELFType>(_context, *_armTargetLayout.get()));
  default:
    llvm_unreachable("unsupported output type");
  }
}

#define ELF_RELOC(name, value) LLD_KIND_STRING_ENTRY(name),

const Registry::KindStrings ARMTargetHandler::kindStrings[] = {
#include "llvm/Support/ELFRelocs/ARM.def"
    LLD_KIND_STRING_END
};
