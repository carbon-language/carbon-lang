//===- lib/ReaderWriter/ELF/Hexagon/HexagonDynamicLibraryWriter.h ---------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef HEXAGON_DYNAMIC_LIBRARY_WRITER_H
#define HEXAGON_DYNAMIC_LIBRARY_WRITER_H

#include "DynamicLibraryWriter.h"
#include "HexagonExecutableAtoms.h"
#include "HexagonLinkingContext.h"

namespace lld {
namespace elf {

class HexagonTargetLayout;

class HexagonDynamicLibraryWriter : public DynamicLibraryWriter<ELF32LE> {
public:
  HexagonDynamicLibraryWriter(HexagonLinkingContext &ctx,
                              HexagonTargetLayout &layout);

protected:
  // Add any runtime files and their atoms to the output
  void createImplicitFiles(std::vector<std::unique_ptr<File>> &) override;

  void finalizeDefaultAtomValues() override;

  std::error_code setELFHeader() override {
    DynamicLibraryWriter<ELF32LE>::setELFHeader();
    setHexagonELFHeader(*this->_elfHeader);
    return std::error_code();
  }

private:
  HexagonLinkingContext &_ctx;
  HexagonTargetLayout &_targetLayout;
};

HexagonDynamicLibraryWriter::HexagonDynamicLibraryWriter(
    HexagonLinkingContext &ctx, HexagonTargetLayout &layout)
    : DynamicLibraryWriter<ELF32LE>(ctx, layout), _ctx(ctx),
      _targetLayout(layout) {}

void HexagonDynamicLibraryWriter::createImplicitFiles(
    std::vector<std::unique_ptr<File>> &result) {
  DynamicLibraryWriter<ELF32LE>::createImplicitFiles(result);
  // Add the default atoms as defined for hexagon
  auto file = llvm::make_unique<HexagonRuntimeFile>(_ctx);
  file->addAbsoluteAtom("_GLOBAL_OFFSET_TABLE_");
  file->addAbsoluteAtom("_DYNAMIC");
  result.push_back(std::move(file));
}

void HexagonDynamicLibraryWriter::finalizeDefaultAtomValues() {
  // Finalize the atom values that are part of the parent.
  DynamicLibraryWriter<ELF32LE>::finalizeDefaultAtomValues();
  if (_ctx.isDynamic())
    finalizeHexagonRuntimeAtomValues(_targetLayout);
}

} // namespace elf
} // namespace lld

#endif // HEXAGON_DYNAMIC_LIBRARY_WRITER_H
