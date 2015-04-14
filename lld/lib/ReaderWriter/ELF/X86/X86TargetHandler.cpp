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
    return llvm::make_unique<X86ExecutableWriter>(_ctx, *_targetLayout);
  case llvm::ELF::ET_DYN:
    return llvm::make_unique<X86DynamicLibraryWriter>(_ctx, *_targetLayout);
  case llvm::ELF::ET_REL:
    llvm_unreachable("TODO: support -r mode");
  default:
    llvm_unreachable("unsupported output type");
  }
}

X86TargetHandler::X86TargetHandler(X86LinkingContext &ctx)
    : _ctx(ctx), _targetLayout(new TargetLayout<ELF32LE>(ctx)),
      _relocationHandler(new X86TargetRelocationHandler()) {}
