//===- lib/ReaderWriter/ELF/AArch64/AArch64TargetHandler.cpp --------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Atoms.h"
#include "AArch64DynamicLibraryWriter.h"
#include "AArch64ExecutableWriter.h"
#include "AArch64LinkingContext.h"
#include "AArch64TargetHandler.h"

using namespace lld;
using namespace elf;

AArch64TargetHandler::AArch64TargetHandler(AArch64LinkingContext &context)
    : DefaultTargetHandler(context), _context(context),
      _AArch64TargetLayout(new AArch64TargetLayout<AArch64ELFType>(context)),
      _AArch64RelocationHandler(
          new AArch64TargetRelocationHandler(context)) {}

void AArch64TargetHandler::registerRelocationNames(Registry &registry) {
  registry.addKindTable(Reference::KindNamespace::ELF,
                        Reference::KindArch::AArch64, kindStrings);
}

std::unique_ptr<Writer> AArch64TargetHandler::getWriter() {
  switch (this->_context.getOutputELFType()) {
  case llvm::ELF::ET_EXEC:
    return std::unique_ptr<Writer>(new AArch64ExecutableWriter<AArch64ELFType>(
        _context, *_AArch64TargetLayout.get()));
  case llvm::ELF::ET_DYN:
    return std::unique_ptr<Writer>(
        new AArch64DynamicLibraryWriter<AArch64ELFType>(
            _context, *_AArch64TargetLayout.get()));
  case llvm::ELF::ET_REL:
    llvm_unreachable("TODO: support -r mode");
  default:
    llvm_unreachable("unsupported output type");
  }
}

#define ELF_RELOC(name, value) LLD_KIND_STRING_ENTRY(name),

const Registry::KindStrings AArch64TargetHandler::kindStrings[] = {
#include "llvm/Support/ELFRelocs/AArch64.def"
    LLD_KIND_STRING_END
};

#undef ELF_RELOC
