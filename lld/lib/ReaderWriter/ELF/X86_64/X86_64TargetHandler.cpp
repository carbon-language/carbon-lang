//===- lib/ReaderWriter/ELF/X86_64/X86_64TargetHandler.cpp ----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Atoms.h"
#include "X86_64DynamicLibraryWriter.h"
#include "X86_64ExecutableWriter.h"
#include "X86_64LinkingContext.h"
#include "X86_64TargetHandler.h"

using namespace lld;
using namespace elf;

X86_64TargetHandler::X86_64TargetHandler(X86_64LinkingContext &ctx)
    : _ctx(ctx), _targetLayout(new X86_64TargetLayout(ctx)),
      _relocationHandler(new X86_64TargetRelocationHandler(*_targetLayout)) {}

std::unique_ptr<Writer> X86_64TargetHandler::getWriter() {
  switch (this->_ctx.getOutputELFType()) {
  case llvm::ELF::ET_EXEC:
    return llvm::make_unique<X86_64ExecutableWriter>(_ctx, *_targetLayout);
  case llvm::ELF::ET_DYN:
    return llvm::make_unique<X86_64DynamicLibraryWriter>(_ctx, *_targetLayout);
  case llvm::ELF::ET_REL:
    llvm_unreachable("TODO: support -r mode");
  default:
    llvm_unreachable("unsupported output type");
  }
}
