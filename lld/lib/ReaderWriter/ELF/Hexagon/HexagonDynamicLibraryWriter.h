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
class HexagonDynamicLibraryWriter : public DynamicLibraryWriter<ELFT>,
                                    public HexagonELFWriter<ELFT> {
public:
  HexagonDynamicLibraryWriter(HexagonLinkingContext &context,
                              HexagonTargetLayout<ELFT> &layout);

protected:
  // Add any runtime files and their atoms to the output
  virtual bool createImplicitFiles(std::vector<std::unique_ptr<File>> &);

  virtual void finalizeDefaultAtomValues();

  virtual std::error_code setELFHeader() {
    DynamicLibraryWriter<ELFT>::setELFHeader();
    HexagonELFWriter<ELFT>::setELFHeader(*this->_elfHeader);
    return std::error_code();
  }

private:
  void addDefaultAtoms() {
    _hexagonRuntimeFile->addAbsoluteAtom("_GLOBAL_OFFSET_TABLE_");
    _hexagonRuntimeFile->addAbsoluteAtom("_DYNAMIC");
  }

  HexagonLinkingContext &_hexagonLinkingContext;
  HexagonTargetLayout<ELFT> &_hexagonTargetLayout;
  std::unique_ptr<HexagonRuntimeFile<ELFT>> _hexagonRuntimeFile;
};

template <class ELFT>
HexagonDynamicLibraryWriter<ELFT>::HexagonDynamicLibraryWriter(
    HexagonLinkingContext &context, HexagonTargetLayout<ELFT> &layout)
    : DynamicLibraryWriter<ELFT>(context, layout),
      HexagonELFWriter<ELFT>(context, layout), _hexagonLinkingContext(context),
      _hexagonTargetLayout(layout),
      _hexagonRuntimeFile(new HexagonRuntimeFile<ELFT>(context)) {}

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
  HexagonELFWriter<ELFT>::finalizeHexagonRuntimeAtomValues();
}

} // namespace elf
} // namespace lld

#endif // HEXAGON_DYNAMIC_LIBRARY_WRITER_H
