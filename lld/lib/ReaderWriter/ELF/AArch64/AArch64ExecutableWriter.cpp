//===- lib/ReaderWriter/ELF/AArch64/AArch64ExecutableWriter.cpp -------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AArch64LinkingContext.h"
#include "AArch64ExecutableWriter.h"
#include "AArch64TargetHandler.h"
#include "AArch64SectionChunks.h"

namespace lld {
namespace elf {

AArch64ExecutableWriter::AArch64ExecutableWriter(AArch64LinkingContext &ctx,
                                                 AArch64TargetLayout &layout)
  : ExecutableWriter(ctx, layout), _targetLayout(layout) {}

void AArch64ExecutableWriter::createImplicitFiles(
    std::vector<std::unique_ptr<File>> &result) {
  ExecutableWriter::createImplicitFiles(result);
  auto gotFile = llvm::make_unique<SimpleFile>("GOTFile");
  gotFile->addAtom(*new (gotFile->allocator()) GlobalOffsetTableAtom(*gotFile));
  if (this->_ctx.isDynamic())
    gotFile->addAtom(*new (gotFile->allocator()) DynamicAtom(*gotFile));
  result.push_back(std::move(gotFile));
}

void AArch64ExecutableWriter::buildDynamicSymbolTable(const File &file) {
  for (auto sec : this->_layout.sections()) {
    if (auto section = dyn_cast<AtomSection<ELF64LE>>(sec)) {
      for (const auto &atom : section->atoms()) {
        if (_targetLayout.getGOTSection().hasGlobalGOTEntry(atom->_atom)) {
          this->_dynamicSymbolTable->addSymbol(atom->_atom, section->ordinal(),
                                               atom->_virtualAddr, atom);
          continue;
        }
      }
    }
  }

  ExecutableWriter<ELF64LE>::buildDynamicSymbolTable(file);
}

} // namespace elf
} // namespace lld

