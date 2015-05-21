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
#include "ARMDynamicLibraryWriter.h"
#include "ARMTargetHandler.h"
#include "ARMLinkingContext.h"

using namespace lld;
using namespace elf;

ARMTargetHandler::ARMTargetHandler(ARMLinkingContext &ctx)
    : _ctx(ctx), _targetLayout(new ARMTargetLayout(ctx)),
      _relocationHandler(new ARMTargetRelocationHandler(*_targetLayout)) {}

std::unique_ptr<Writer> ARMTargetHandler::getWriter() {
  switch (this->_ctx.getOutputELFType()) {
  case llvm::ELF::ET_EXEC:
    return llvm::make_unique<ARMExecutableWriter>(_ctx, *_targetLayout);
  case llvm::ELF::ET_DYN:
    return llvm::make_unique<ARMDynamicLibraryWriter>(_ctx, *_targetLayout);
  default:
    llvm_unreachable("unsupported output type");
  }
}
