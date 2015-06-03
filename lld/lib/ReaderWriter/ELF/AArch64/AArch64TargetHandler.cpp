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
#include "AArch64SectionChunks.h"

using namespace lld;
using namespace elf;

AArch64TargetLayout::AArch64TargetLayout(ELFLinkingContext &ctx) :
  TargetLayout(ctx),
  _gotSection(new (this->_allocator) AArch64GOTSection(ctx)) {}

AtomSection<ELF64LE> *AArch64TargetLayout::createSection(
    StringRef name, int32_t type, DefinedAtom::ContentPermissions permissions,
    TargetLayout<ELF64LE>::SectionOrder order) {
  if (type == DefinedAtom::typeGOT && name == ".got")
    return _gotSection;
  return TargetLayout<ELF64LE>::createSection(name, type, permissions, order);
}


AArch64TargetHandler::AArch64TargetHandler(AArch64LinkingContext &ctx)
    : _ctx(ctx), _targetLayout(new AArch64TargetLayout(ctx)),
      _relocationHandler(new AArch64TargetRelocationHandler(*_targetLayout)) {}

std::unique_ptr<Writer> AArch64TargetHandler::getWriter() {
  switch (this->_ctx.getOutputELFType()) {
  case llvm::ELF::ET_EXEC:
    return llvm::make_unique<AArch64ExecutableWriter>(_ctx, *_targetLayout);
  case llvm::ELF::ET_DYN:
    return llvm::make_unique<AArch64DynamicLibraryWriter>(_ctx, *_targetLayout);
  case llvm::ELF::ET_REL:
    llvm_unreachable("TODO: support -r mode");
  default:
    llvm_unreachable("unsupported output type");
  }
}
