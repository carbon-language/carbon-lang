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

template <typename ELFT> class HexagonTargetLayout;

template <class ELFT>
class HexagonDynamicLibraryWriter : public DynamicLibraryWriter<ELFT> {
public:
  HexagonDynamicLibraryWriter(HexagonLinkingContext &ctx,
                              HexagonTargetLayout<ELFT> &layout);

protected:
  // Add any runtime files and their atoms to the output
  bool createImplicitFiles(std::vector<std::unique_ptr<File>> &) override;

  void finalizeDefaultAtomValues() override;

  std::error_code setELFHeader() override {
    DynamicLibraryWriter<ELFT>::setELFHeader();
    setHexagonELFHeader(*this->_elfHeader);
    return std::error_code();
  }

private:
  void addDefaultAtoms() override {
    _hexagonRuntimeFile->addAbsoluteAtom("_GLOBAL_OFFSET_TABLE_");
    _hexagonRuntimeFile->addAbsoluteAtom("_DYNAMIC");
  }

  HexagonLinkingContext &_ctx;
  HexagonTargetLayout<ELFT> &_hexagonTargetLayout;
  std::unique_ptr<HexagonRuntimeFile<ELFT>> _hexagonRuntimeFile;
};

template <class ELFT>
HexagonDynamicLibraryWriter<ELFT>::HexagonDynamicLibraryWriter(
    HexagonLinkingContext &ctx, HexagonTargetLayout<ELFT> &layout)
    : DynamicLibraryWriter<ELFT>(ctx, layout), _ctx(ctx),
      _hexagonTargetLayout(layout),
      _hexagonRuntimeFile(new HexagonRuntimeFile<ELFT>(ctx)) {}

template <class ELFT>
bool HexagonDynamicLibraryWriter<ELFT>::createImplicitFiles(
    std::vector<std::unique_ptr<File>> &result) {
  DynamicLibraryWriter<ELFT>::createImplicitFiles(result);
  // Add the default atoms as defined for hexagon
  addDefaultAtoms();
  result.push_back(std::move(_hexagonRuntimeFile));
  return true;
}

template <class ELFT>
void HexagonDynamicLibraryWriter<ELFT>::finalizeDefaultAtomValues() {
  // Finalize the atom values that are part of the parent.
  DynamicLibraryWriter<ELFT>::finalizeDefaultAtomValues();
  if (_ctx.isDynamic())
    finalizeHexagonRuntimeAtomValues(_hexagonTargetLayout);
}

} // namespace elf
} // namespace lld

#endif // HEXAGON_DYNAMIC_LIBRARY_WRITER_H
