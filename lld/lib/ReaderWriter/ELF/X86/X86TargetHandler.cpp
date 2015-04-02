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
  switch (_ctx.getOutputELFType()) {
  case llvm::ELF::ET_EXEC:
    return llvm::make_unique<X86ExecutableWriter<X86ELFType>>(_ctx,
                                                              *_targetLayout);
  case llvm::ELF::ET_DYN:
    return llvm::make_unique<X86DynamicLibraryWriter<X86ELFType>>(
        _ctx, *_targetLayout);
  case llvm::ELF::ET_REL:
    llvm_unreachable("TODO: support -r mode");
  default:
    llvm_unreachable("unsupported output type");
  }
}

static const Registry::KindStrings kindStrings[] = {
#define ELF_RELOC(name, value) LLD_KIND_STRING_ENTRY(name),
#include "llvm/Support/ELFRelocs/i386.def"
#undef ELF_RELOC
  LLD_KIND_STRING_END
};

void X86TargetHandler::registerRelocationNames(Registry &registry) {
  registry.addKindTable(Reference::KindNamespace::ELF, Reference::KindArch::x86,
                        kindStrings);
}

X86TargetHandler::X86TargetHandler(X86LinkingContext &ctx)
    : _ctx(ctx), _targetLayout(new TargetLayout<X86ELFType>(ctx)),
      _relocationHandler(new X86TargetRelocationHandler()) {}
