//===- lib/ReaderWriter/ELF/X86/X86TargetHandler.cpp ----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "X86TargetHandler.h"
#include "X86DynamicLibraryWriter.h"
#include "X86ExecutableWriter.h"
#include "X86LinkingContext.h"
#include "X86RelocationHandler.h"

using namespace lld;
using namespace elf;

using namespace llvm::ELF;

std::unique_ptr<Writer> X86TargetHandler::getWriter() {
  switch (_x86LinkingContext.getOutputELFType()) {
  case llvm::ELF::ET_EXEC:
    return std::unique_ptr<Writer>(new X86ExecutableWriter<X86ELFType>(
        _x86LinkingContext, *_x86TargetLayout.get()));
  case llvm::ELF::ET_DYN:
    return std::unique_ptr<Writer>(new X86DynamicLibraryWriter<X86ELFType>(
        _x86LinkingContext, *_x86TargetLayout.get()));
  case llvm::ELF::ET_REL:
    llvm_unreachable("TODO: support -r mode");
  default:
    llvm_unreachable("unsupported output type");
  }
}

#define ELF_RELOC(name, value) LLD_KIND_STRING_ENTRY(name),

const Registry::KindStrings X86TargetHandler::kindStrings[] = {
#include "llvm/Support/ELFRelocs/i386.def"
  LLD_KIND_STRING_END
};

#undef ELF_RELOC

void X86TargetHandler::registerRelocationNames(Registry &registry) {
  registry.addKindTable(Reference::KindNamespace::ELF, Reference::KindArch::x86,
                        kindStrings);
}

X86TargetHandler::X86TargetHandler(X86LinkingContext &context)
    : DefaultTargetHandler(context), _x86LinkingContext(context),
      _x86TargetLayout(new X86TargetLayout<X86ELFType>(context)),
      _x86RelocationHandler(
          new X86TargetRelocationHandler(context)) {}
